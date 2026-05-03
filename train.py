import os
import sys

# 强制工作目录切换到本文件所在目录，确保日志存到项目目录下
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/opt/MotrixLab/scripts")

# 导入我们的环境和训练配置，触发注册
import navigation.anymal_c
import cfgs

from absl import app, flags
from skrl import config
from motrix_rl import utils

_ENV = flags.DEFINE_string("env", "anymal_c_nav", "The env to train")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 2048, "Number of envs to train")
_RENDER = flags.DEFINE_bool("render", False, "Render the env")
_TRAIN_BACKEND = flags.DEFINE_string("train-backend", "jax", "jax or torch")

def main(argv):
    train_backend = _TRAIN_BACKEND.value

    if train_backend == "jax":
        from motrix_rl.skrl.jax.train import ppo
        config.jax.backend = "jax"
        trainer = ppo.Trainer(
            _ENV.value,
            None,
            cfg_override={"num_envs": _NUM_ENVS.value},
            enable_render=_RENDER.value
        )
    elif train_backend == "torch":
        from motrix_rl.skrl.torch.train import ppo
        trainer = ppo.Trainer(
            _ENV.value,
            None,
            cfg_override={"num_envs": _NUM_ENVS.value},
            enable_render=_RENDER.value
        )
    else:
        raise Exception(f"Unknown train backend: {train_backend}")

    trainer.train()

if __name__ == "__main__":
    app.run(main)