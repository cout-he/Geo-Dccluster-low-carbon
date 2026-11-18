from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward

@register_reward("carbon_emissions")
class CarbonEmissionsReward(BaseReward):
    def __init__(self, normalize_factor: float = 100.0, use_metered_totals: bool = False):
        super().__init__()
        self.normalize_factor = normalize_factor
        self.use_metered_totals = use_metered_totals

    def __call__(self, cluster_info, current_tasks, current_time):
        if self.use_metered_totals:
            total_emissions = sum(
                dc_info["__common__"]["carbon_emissions_kg"]
                for dc_info in cluster_info["datacenter_infos"].values()
            ) + cluster_info["transmission_emissions_total_kg"]
            self.last_reward = -total_emissions / self.normalize_factor
            return self.last_reward

        total_task_emissions = 0.0
        for task in current_tasks:
            dest_dc = getattr(task, "dest_dc", None)
            if dest_dc:
                task_energy = task.cores_req * task.duration / 10000.0
                task_emissions = task_energy * dest_dc.ci_manager.get_current_ci(norm=False)
                total_task_emissions += task_emissions

        reward = -total_task_emissions / self.normalize_factor
        self.last_reward = reward
        return reward

    def get_last_value(self):
        return self.last_reward