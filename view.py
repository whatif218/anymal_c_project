import sys
sys.path.insert(0, "/opt/MotrixLab/scripts")

# 先导入我们的环境，触发注册
import navigation.anymal_c

# 再运行官方 view.py 的逻辑
import gymnasium as gym
import numpy as np
from absl import app, flags
from motrix_envs import registry
from motrix_envs.np.env import NpEnv
from motrix_envs.np.renderer import NpRenderer

_ENV = flags.DEFINE_string("env", "anymal_c_nav", "The env to view")
_SIM_BACKEND = flags.DEFINE_string("sim-backend", None, "The simulation backend to use.")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 1, "Number of parallel environments.")

class NpEnvRunner:
    def __init__(self, env: NpEnv):
        self._env = env
        self._renderer = NpRenderer(env)

    def _sample_random_action(self):
        action_space = self._env.action_space
        size = (self._env.num_envs, *action_space.shape)
        low  = np.where(np.isneginf(action_space.low),  -1e6, action_space.low)
        high = np.where(np.isposinf(action_space.high),  1e6, action_space.high)
        return np.random.uniform(low=low, high=high, size=size).astype(action_space.dtype)

    def start(self):
        import time
        env_dt = self._env.cfg.ctrl_dt
        while True:
            t0 = time.monotonic()
            self._env.step(self._sample_random_action())
            self._renderer.render()
            sleep_dt = env_dt - (time.monotonic() - t0)
            if sleep_dt > 0:
                time.sleep(sleep_dt)

def main(argv):
    env = registry.make(_ENV.value, sim_backend=_SIM_BACKEND.value, num_envs=_NUM_ENVS.value)
    NpEnvRunner(env).start()

if __name__ == "__main__":
    app.run(main)
