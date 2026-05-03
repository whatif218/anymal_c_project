from dataclasses import dataclass
from motrix_rl.registry import rlcfg
from motrix_rl.skrl.cfg import PPOCfg


class navigation:
    @rlcfg("anymal_c_nav")
    @dataclass
    class AnymalCNavPPO(PPOCfg):
        seed: int = 42
        num_envs: int = 2048
        max_env_steps: int = 100_000_000
        check_point_interval: int = 1000

        learning_rate: float = 3e-4
        rollouts: int = 48
        learning_epochs: int = 6
        mini_batches: int = 32
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
