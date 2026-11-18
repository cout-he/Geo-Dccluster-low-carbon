import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from envs.sustaindc.sustaindc_env import SustainDC
from envs.task_scheduling_env import TaskSchedulingEnv
from rl_components.task import Task
from simulation.cluster_manager import DatacenterClusterManager
from utils.managers import CI_Manager, ElectricityPrice_Manager
from utils.workload_utils import assign_task_origins


class ExperimentTimelineTests(unittest.TestCase):
    def setUp(self):
        self.start = pd.Timestamp("2023-07-01", tz="UTC")
        self.end = self.start + pd.Timedelta(days=30)
        self.datacenter = SimpleNamespace(dc_id=1, pending_tasks=deque(), location="US-CAL-CISO")
        self.manager = DatacenterClusterManager.__new__(DatacenterClusterManager)
        self.manager.datacenters = {"DC1": self.datacenter}
        self.manager.logger = None
        self.manager.strategy = "manual_rl"
        self.manager.strategy_map = {}
        self.manager.cloud_provider = "aws"
        self.manager.init_day = self.start.dayofyear - 1
        self.manager.init_hour = self.start.hour
        self.manager.simulation_year = self.start.year
        self.manager.shuffle_datacenter_order = False
        self.datacenter.reset = Mock(side_effect=self._set_dummy_data_indices)

    def _set_dummy_data_indices(self, *, init_year, init_day, init_hour, seed, align_simulation_time):
        self.assertTrue(align_simulation_time)
        self.assertEqual(init_year, 2023)
        utc_index = (init_day * 24 + init_hour) * 4
        self.datacenter.ci_manager = SimpleNamespace(time_step=utc_index)
        self.datacenter.price_manager = SimpleNamespace(index=utc_index)
        self.datacenter.weather_manager = SimpleNamespace(
            time_step=utc_index - 7 * 4,
            roll_days=2,
            temperature_data=np.zeros(365 * 96),
        )

    def _env(self, randomize):
        env = TaskSchedulingEnv(
            self.manager, self.start, self.end, reward_fn=None,
            sim_config={"align_simulation_time": True, "randomize_episode_start": randomize},
            initial_seed_for_resets=42,
        )
        env._load_new_tasks = Mock()
        env._get_obs = Mock(return_value=[])
        return env

    def test_randomized_training_start_applies_to_all_streams(self):
        rng = np.random.default_rng(42)
        actual_day = int(rng.integers(self.manager.init_day - 7, self.manager.init_day + 8))
        actual_hour = int(rng.integers(0, 24))
        actual_start = pd.Timestamp("2023-01-01", tz="UTC") + pd.Timedelta(
            days=actual_day, hours=actual_hour
        )
        env = self._env(randomize=True)
        _, info = env.reset(seed=42)
        utc_index = (actual_day * 24 + actual_hour) * 4
        self.assertEqual(env.current_time, actual_start)
        self.assertEqual(env.episode_end_time - actual_start, pd.Timedelta(days=30))
        self.assertEqual(info["workload_start_utc"], actual_start.replace(year=2020).isoformat())
        self.assertEqual(info["dc_source_indices"]["US-CAL-CISO"]["carbon_15m"], utc_index)
        self.assertEqual(info["dc_source_indices"]["US-CAL-CISO"]["electricity_price_15m"], utc_index)
        self.assertEqual(info["dc_source_indices"]["US-CAL-CISO"]["weather_local_15m"], utc_index - 28)
        self.assertEqual(
            info["dc_source_indices"]["US-CAL-CISO"]["weather_source_local_15m"],
            utc_index - 28 - 2 * 96,
        )

    def test_evaluation_start_is_fixed_across_seeds(self):
        env = self._env(randomize=False)
        for seed in (0, 9):
            _, info = env.reset(seed=seed)
            self.assertEqual(env.current_time, self.start)
            self.assertEqual(env.episode_end_time, self.end)
            self.assertEqual(info["reset_seed"], seed)
            self.assertEqual(info["episode_start_utc"], self.start.isoformat())

    def test_local_assignment_reaches_pending_queue_in_same_interval(self):
        env = self._env(randomize=False)
        env.reset(seed=0)
        task = Task("local", self.start, 60, 10, 1, 10, 2)
        task.origin_dc_id = 1
        env.current_tasks = [task]

        def cluster_step(current_time, logger=None):
            self.assertEqual(current_time, self.start)
            self.assertEqual(list(self.datacenter.pending_tasks), [task])
            return {
                "datacenter_infos": {},
                "transmission_cost_total_usd": 0,
                "transmission_energy_total_kwh": 0,
                "transmission_emissions_total_kg": 0,
            }

        self.manager.step = cluster_step
        env.step([1])
        self.assertEqual(task.dest_dc_id, 1)
        self.assertEqual(env.in_transit_tasks, [])

    def test_hourly_values_land_on_exact_quarters(self):
        carbon = CI_Manager.__new__(CI_Manager)
        carbon.carbon_data = np.array([0.0, 4.0, 8.0])
        carbon.timestep_per_hour = 4
        carbon.timezone_shift = 0
        carbon.exact_quarters = True
        carbon._interpolate()

        price = ElectricityPrice_Manager.__new__(ElectricityPrice_Manager)
        price.original_prices = carbon.carbon_data
        price.exact_quarters = True
        price._interpolate_prices()

        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8]
        np.testing.assert_allclose(carbon.carbon_smooth, expected)
        np.testing.assert_allclose(price.prices, expected)


    def test_workload_lookup_keeps_trace_scaling_and_deadline(self):
        trace_time = self.start.replace(year=2020)
        self.manager.tasks = pd.DataFrame({
            "interval_15m": [trace_time, trace_time],
            "tasks_matrix": [
                [["first", 0, 0, 0, 40, 200, 50, 2, 1.25]],
                [["second", 0, 0, 0, 80, 800, 100, 12, 3.25]],
            ],
        })
        self.manager.workload_sorted = True
        self.manager.align_simulation_time = True
        self.datacenter.env_config = {
            "dc_id": 1, "timezone_shift": -7, "population_weight": 0.18,
        }
        task, = self.manager.get_tasks_for_timestep(self.start)
        self.assertTrue(task.job_name.startswith("first_"))
        self.assertEqual(task.origin_dc_id, 1)
        self.assertEqual(task.duration, 40)
        self.assertEqual(task.bandwidth_gb, 1.25)
        self.assertEqual((task.cores_req, task.gpu_req, task.mem_req), (10, 2.5, 10))
        self.assertEqual(task.sla_deadline - task.arrival_time, pd.Timedelta(minutes=60))
        self.assertEqual(self.manager.get_tasks_for_timestep(self.start + pd.Timedelta(minutes=15)), [])

    def test_origin_activity_uses_fixed_local_hour_boundaries(self):
        dc_config = [
            {"dc_id": 1, "timezone_shift": 0, "population_weight": 0.2},
            {"dc_id": 2, "timezone_shift": 12, "population_weight": 0.8},
        ]
        for hour, expected_scores in [(8, [0.2, 0.24]), (20, [0.06, 0.8])]:
            with self.subTest(hour=hour):
                task = SimpleNamespace()
                with patch("utils.workload_utils.np.random.choice", return_value=2) as choose:
                    assign_task_origins([task], dc_config, self.start + pd.Timedelta(hours=hour))
                self.assertEqual(task.origin_dc_id, 2)
                self.assertEqual(choose.call_args.args[0], [1, 2])
                np.testing.assert_allclose(
                    choose.call_args.kwargs["p"],
                    np.asarray(expected_scores) / sum(expected_scores),
                )


def test_reproduction_propagates_physics_input_error():
    import pytest

    dc = SustainDC.__new__(SustainDC)
    dc.dc_id = 1
    dc.align_simulation_time = True
    dc.dc_env = Mock()
    dc.dc_env.update_workloads.side_effect = RuntimeError("invalid load")
    dc.bat_env = Mock()
    with pytest.raises(RuntimeError, match="invalid load"):
        dc._update_environments(0, 0, 0, 20, 18, 300, [300], 1, 0)
