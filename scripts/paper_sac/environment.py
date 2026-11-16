"""Experiment-only adapters. Original routing, rewards and physics are reused."""
import copy
import random
import numpy as np
import pandas as pd
from simulation.cluster_manager import DatacenterClusterManager
from envs.task_scheduling_env import TaskSchedulingEnv
from utils.managers import sc_obs
from rewards.predefined.composite_reward import CompositeReward
# Registration is explicit: predefined/__init__.py does not register classes.
from rewards.predefined import (energy_price_reward, carbon_emissions_reward,
                                transmission_cost_reward, sla_penalty_reward,
                                transmission_emissions_reward)  # noqa: F401
from .config import resolve
from .data import read_workload, window, external_data, validate_coverage, time_index
from .metrics import Metrics


class PaperCluster(DatacenterClusterManager):
    def _load_or_extract_tasks_file(self, tasks_file_path):
        # No unzip side effect or dataset edits. Use the indexed real loader.
        self.task_rows = read_workload(tasks_file_path).set_index('interval_15m', drop=False)
        return self.task_rows

    def get_tasks_for_timestep(self, current_time):
        from utils.workload_utils import extract_tasks_from_row
        source = current_time.replace(year=2020)
        if source not in self.task_rows.index:
            return []
        # Independent per-interval RNG prevents policy/update draws from changing
        # task origins, including across ablation strategies with one eval seed.
        py_state, np_state = random.getstate(), np.random.get_state()
        slot_seed = (int(self.episode_seed) * 1000003 + int(current_time.value // 900000000000)) % (2**32)
        try:
            random.seed(slot_seed)
            np.random.seed(slot_seed)
            return extract_tasks_from_row(self.task_rows.loc[source], scale=1, task_scale=5,
                                          group_size=1, datacenter_configs=self.get_config_list(),
                                          current_time_utc=current_time)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def reset(self, seed=None):
        self.episode_seed = int(seed or 0)
        super().reset(seed=seed)  # Preserve and expose original seeded randomization.
        if self.reset_mode == 'randomized_aligned':
            self.actual_start = pd.Timestamp(year=self.simulation_year, month=1, day=1, tz='UTC') + pd.Timedelta(days=self.random_init_day, hours=self.random_init_hour)
        else:
            self.actual_start = self.configured_start
        self.last_results = None
        for dc in self.datacenters.values():
            data = external_data(dc.location, self.simulation_year)
            validate_coverage(data, self.actual_start, self.actual_start + self.duration)
            arr = data['arrays']
            ci, price, weather = dc.ci_manager, dc.price_manager, dc.weather_manager
            ci.original_data = arr['ci'].copy()
            ci.carbon_smooth = ci.original_data
            ci.init_day = self.actual_start.dayofyear - 1
            idx = time_index(self.actual_start)
            span = ci.carbon_smooth[idx:idx+30*96]
            ci.norm_carbon = (ci.carbon_smooth - span.min()) / (np.ptp(span) + 1e-8)
            price.prices = arr['price'].copy()
            if self.weather_augmentation:
                # Retain the legacy reset's seeded roll/noise as a perturbation;
                # all baseline indices still refer to the exact UTC source.
                delta = weather.temperature_data - weather.original_temp_data
                weather.temperature_data = np.clip(arr['temperature'] + delta, 0, 45)
                weather.wet_bulb_data = np.clip(.9 * arr['temperature'] + delta, 0, 45)
            else:
                weather.temperature_data = arr['temperature'].copy()
                # Preserve original wet-bulb approximation (physical model unchanged).
                weather.wet_bulb_data = .9 * arr['temperature']
            weather.norm_temp_data = weather.temperature_data / 45
            weather.norm_wet_bulb_data = weather.wet_bulb_data / 45
        self.align(self.actual_start)
        for dc in self.datacenters.values():
            w = dc.weather_manager
            dc.dc_env.set_ambient_temp(w._current_temp, w._current_wet_bulb)
            _, dc.dc_info = dc.dc_env.reset(seed=seed)
            _, dc.bat_info = dc.bat_env.reset(seed=seed)
            dc.dc_env.update_workloads(0., 0., 0.)
            ci = dc.ci_manager
            dc.bat_env.update_ci(ci.get_current_ci(norm=False), ci.get_forecast_ci()[0])

    def align(self, now, before_step=False):
        for dc in self.datacenters.values():
            idx = time_index(now) - int(before_step)
            dc.ci_manager.time_step = dc.weather_manager.time_step = dc.price_manager.index = idx
            w = dc.weather_manager
            w._current_temp, w._next_temp = w.temperature_data[idx:idx+2]
            w._current_norm_temp, w._next_norm_temp = w.norm_temp_data[idx:idx+2]
            w._current_wet_bulb = w.wet_bulb_data[idx]
            w._current_norm_wet_bulb = w.norm_wet_bulb_data[idx]
            local = now + pd.Timedelta(hours=dc.timezone_shift) - pd.Timedelta(minutes=15 * int(before_step))
            dc.t_m.day = local.dayofyear - 1
            dc.t_m.hour = local.hour + local.minute / 60
            dc.t_m.current_timestep_in_year = (local.dayofyear - 1)*96 + local.hour*4 + local.minute//15
            dc.current_hour = local.hour + local.minute / 60

    def step(self, current_time, logger=None):
        # SustainDC.step increments its managers BEFORE consuming inputs. Position
        # them at t-15min so its physical interval uses the task timestamp t.
        self.align(current_time, before_step=True)
        # _perform_actions consumes physical inputs BEFORE managers advance.
        # Supply the exact current UTC inputs now; manager positioning alone
        # would still leave temperature and battery CI one interval behind.
        idx = time_index(current_time)
        for dc in self.datacenters.values():
            w, ci = dc.weather_manager, dc.ci_manager
            dc.dc_env.set_ambient_temp(w.temperature_data[idx], w.wet_bulb_data[idx])
            dc.bat_env.update_ci(ci.carbon_smooth[idx], ci.norm_carbon[idx+1:idx+1+ci.future_steps][0])
        self.last_results = super().step(current_time, logger=logger)
        return self.last_results


class PaperEnv(TaskSchedulingEnv):
    def __init__(self, *args, max_tasks=400, episode_steps=None, **kwargs):
        self.max_tasks = max_tasks
        self.episode_steps = episode_steps
        self.ledger = {}
        self.finished_ids = set()
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        self.ledger, self.finished_ids, self.deferred_ids = {}, set(), set()
        self.metrics = Metrics()
        if seed is None:
            seed = self.base_seed + self.current_episode_count
        # Set the same realized calendar that cluster.reset will use. This is
        # necessary before parent reset loads tasks and builds observations.
        cluster = self.cluster_manager
        start = cluster.configured_start
        if cluster.reset_mode == 'randomized_aligned':
            rng = np.random.default_rng(seed)
            day = int(rng.integers(max(0, cluster.init_day - 7), min(364, cluster.init_day + 7) + 1))
            hour = int(rng.integers(0, 24))
            start = pd.Timestamp(year=cluster.simulation_year, month=1, day=1, tz='UTC') + pd.Timedelta(days=day, hours=hour)
        self.start_time = start
        duration = pd.Timedelta(minutes=15 * self.episode_steps) if self.episode_steps else cluster.duration
        self.end_time = start + duration
        obs, info = super().reset(seed=seed, options=options)
        assert self.current_time == cluster.actual_start
        return obs, info

    def _load_new_tasks(self):
        if self.current_time >= self.end_time:
            self.current_tasks = []
            self._new_at_current = 0
            return
        fresh = self.cluster_manager.get_tasks_for_timestep(self.current_time)
        self._new_at_current = len(fresh)
        for task in fresh:
            self.ledger[id(task)] = task
        self.current_tasks = self.deferred_tasks + fresh
        self.deferred_tasks = []
        if len(self.current_tasks) > self.max_tasks:
            raise ValueError(f'Observed {len(self.current_tasks)} tasks exceeds max_tasks={self.max_tasks}; no truncation')

    def _get_obs(self):
        self.cluster_manager.align(self.current_time)
        if [d.dc_id for d in self.cluster_manager.datacenters.values()] != [1, 2, 3, 4, 5]:
            raise ValueError('DC order changed')
        obs = super()._get_obs()
        assert self.observation_space.shape == (34,) and self.agent_output_act_dim == 6
        assert not self.single_action_mode and not self.disable_defer_action
        if len(obs) > self.max_tasks or any(x.shape != (34,) for x in obs):
            raise ValueError('Observation task count or dimension mismatch')
        return obs

    def step(self, actions):
        if len(actions) != len(self.current_tasks) or any(not isinstance(a, (int, np.integer)) or not 0 <= a <= 5 for a in actions):
            raise ValueError('One integer action in 0..5 is required per current task')
        events = dict(new_tasks=self._new_at_current, defer_decisions=0, forced_origin=0,
                      remote_migrations=0, policy_local_executions=0)
        for task, action in zip(self.current_tasks, actions):
            if self.current_time > task.sla_deadline:
                events['forced_origin'] += 1
            elif action == 0:
                events['defer_decisions'] += 1
                self.deferred_ids.add(id(task))
            elif action != task.origin_dc_id:
                events['remote_migrations'] += 1
            else:
                events['policy_local_executions'] += 1
        obs, reward, done, truncated, info = super().step(actions)
        results = self.cluster_manager.last_results
        for key in ('transmission_energy_total_kwh', 'transmission_emissions_total_kg', 'transmission_cost_total_usd'):
            events[key] = results[key]
        for dc in info['datacenter_infos'].values():
            self.finished_ids.update(id(t) for t in dc['__common__']['tasks_finished_this_step_objects'])
        info['paper'] = events
        self.metrics.add(info)
        return obs, reward, done, truncated, info

    def final_metrics(self):
        now = self.current_time
        # Include completions at the right edge without advancing physics or
        # loading the next interval's arrivals into the evaluation.
        # Parent step preloads the next observation. Its fresh arrivals belong
        # to the next interval and must not enter this interval's accounting.
        considered = [(key, t) for key, t in self.ledger.items() if t.arrival_time < now]
        terminal = [t for key, t in considered if key not in self.finished_ids and t.is_completed(now)]
        unfinished = [t for _, t in considered if not t.is_completed(now)]
        result = self.metrics.result(len(unfinished), sum(now > t.sla_deadline for t in unfinished),
                                     len(terminal), sum(t.finish_time > t.sla_deadline for t in terminal))
        result['unique_tasks_deferred'] = len(self.deferred_ids)
        if result['new_arrival_tasks'] != result['completed_tasks'] + result['unfinished_tasks']:
            raise ValueError('Task accounting identity failed')
        return result


def make_env(bundle, phase='train_sim', writer=None, episode_steps=None):
    sim = copy.deepcopy(bundle[phase])
    start, end = window(sim)
    dcs = copy.deepcopy(bundle['datacenters'])
    # Only path resolution becomes absolute at the API boundary; saved configs
    # and manifests remain relative and portable.
    for dc in dcs:
        dc['dc_config_file'] = str(resolve(dc['dc_config_file']))
    cluster = PaperCluster(dcs, simulation_year=sim['year'], init_day=start.dayofyear - 1,
                           init_hour=sim['init_hour'], strategy='manual_rl',
                           tasks_file_path=sim['workload_path'], shuffle_datacenter_order=False,
                           cloud_provider=sim['cloud_provider'])
    cluster.configured_start = start
    cluster.duration = end - start
    cluster.reset_mode = sim.get('reset_mode', 'fixed')
    cluster.weather_augmentation = sim.get('weather_augmentation', False)
    r = bundle['reward']
    reward = CompositeReward(r['components'], normalize=r['normalize'])
    return PaperEnv(cluster, start, end, reward, sim_config=sim, writer=writer,
                    max_tasks=bundle['algorithm']['max_tasks'], episode_steps=episode_steps,
                    initial_seed_for_resets=0)
