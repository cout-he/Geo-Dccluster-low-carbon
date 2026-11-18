import unittest
from types import SimpleNamespace

from rewards.reward_registry import get_reward_function


class ReproductionRewardTests(unittest.TestCase):
    def setUp(self):
        self.info = {
            "datacenter_infos": {
                "DC1": {
                    "__common__": {
                        "energy_cost_USD": 800.0,
                        "carbon_emissions_kg": 40.0,
                        "__sla__": {"met": 9, "violated": 1},
                    }
                },
                "DC2": {
                    "__common__": {
                        "energy_cost_USD": 200.0,
                        "carbon_emissions_kg": 10.0,
                        "__sla__": {"met": 3, "violated": 2},
                    }
                },
            },
            "transmission_cost_total_usd": 12.0,
            "transmission_emissions_total_kg": 5.0,
        }
        destination = SimpleNamespace(
            price_manager=SimpleNamespace(get_current_price=lambda: 120.0),
            ci_manager=SimpleNamespace(get_current_ci=lambda norm: 220.0),
        )
        self.tasks = [
            SimpleNamespace(dest_dc=destination, cores_req=20, duration=10),
            SimpleNamespace(dest_dc=None, cores_req=300, duration=100),
        ]

    def test_five_component_formulas_and_unnormalized_weighted_sum(self):
        reward = get_reward_function(
            "composite",
            normalize=False,
            components={
                "energy_price": {
                    "weight": 0.5,
                    "args": {"normalize_factor": 100000, "use_metered_totals": True},
                },
                "carbon_emissions": {
                    "weight": 0.3,
                    "args": {"normalize_factor": 10, "use_metered_totals": True},
                },
                "transmission_cost": {
                    "weight": 0.2,
                    "args": {"normalize_factor": 1},
                },
                "sla_penalty": {
                    "weight": 0.1,
                    "args": {"penalty_per_violation": 5.0},
                },
                "transmission_emissions": {
                    "weight": 0.1,
                    "args": {"normalize_factor": 1},
                },
            },
        )
        expected_components = {
            "energy_price": -0.01,
            "carbon_emissions": -5.5,
            "transmission_cost": -12.0,
            "sla_penalty": -15.0,
            "transmission_emissions": -5.0,
        }
        for tasks in (self.tasks, []):
            self.assertAlmostEqual(reward(self.info, tasks, None), -6.055)
            self.assertEqual(reward.get_last_components(), expected_components)
        self.assertTrue(all(stats["count"] == 0 for stats in reward.running_stats.values()))

    def test_legacy_energy_and_carbon_defaults_keep_task_proxies(self):
        energy = get_reward_function("energy_price", normalize_factor=100000)
        carbon = get_reward_function("carbon_emissions", normalize_factor=10)
        self.assertAlmostEqual(energy(self.info, self.tasks, None), -2.4e-5)
        self.assertAlmostEqual(carbon(self.info, self.tasks, None), -0.44)
        self.assertEqual(energy(self.info, [], None), 0.0)
        self.assertEqual(carbon(self.info, [], None), 0.0)
