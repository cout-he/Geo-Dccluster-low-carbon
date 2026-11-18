from pathlib import Path
import shutil

import pytest
import yaml

from scripts.repro_table3.common import DEFAULT_CONFIG, load_config


def test_formal_configuration():
    training = load_config(DEFAULT_CONFIG, "train")
    evaluation = load_config(DEFAULT_CONFIG, "eval")
    assert training.algorithm["total_steps"] == 10_000_000
    assert training.sim["randomize_episode_start"]
    assert training.sim["shuffle_datacenters"]
    assert not evaluation.sim["randomize_episode_start"]
    assert not evaluation.sim["shuffle_datacenters"]
    assert evaluation.manifest["evaluation_steps"] == 2880
    assert [dc["dc_id"] for dc in training.datacenters] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(("filename", "section", "key", "value"), [
    ("algorithm.yaml", "algorithm", "total_steps", 100),
    ("sim_eval.yaml", "simulation", "disable_defer_action", True),
    ("reward.yaml", "reward", "normalize", True),
])
def test_rejects_changes_to_paper_configuration(tmp_path: Path, filename, section, key, value):
    config = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG, config)
    path = config / filename
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data[section][key] = value
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match=key):
        load_config(config, "eval")
