import os
import sys
import glob
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import navigation.anymal_c
import cfgs

from absl import app, flags
from motrix_rl.skrl.torch.train import ppo
from motrix_envs import registry
from motrix_rl.skrl.torch import wrap_env

_NUM_ENVS = flags.DEFINE_integer("num-envs", 16, "Number of envs to play")

def get_latest_checkpoint():
    pattern = "runs/anymal_c_nav/*/checkpoints/best_agent.pt"
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        pattern = "runs/anymal_c_nav/*/checkpoints/agent_*.pt"
        checkpoints = glob.glob(pattern)
    if not checkpoints:
        raise FileNotFoundError("没有找到 checkpoint，请先训练！")
    latest = max(checkpoints, key=os.path.getmtime)
    print(f"加载 checkpoint: {latest}")
    return latest

def main(argv):
    policy_path = get_latest_checkpoint()
    trainer = ppo.Trainer(
        "anymal_c_nav",
        None,
        cfg_override={"play_num_envs": _NUM_ENVS.value},
        enable_render=True
    )
    trainer.play(policy_path)

if __name__ == "__main__":
    app.run(main)