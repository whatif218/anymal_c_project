import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.dirname(__file__) + "/xmls/scene.xml"


@dataclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class ControlConfig:
    action_scale: float = 0.06


@dataclass
class InitState:
    pos: list = field(default_factory=lambda: [0.0, 0.0, 0.5])
    pos_randomization_range: list = field(default_factory=lambda: [-0.5, -0.5, 0.5, 0.5])
    default_joint_angles: dict = field(default_factory=lambda: {
        "LF_HAA": 0.0,  "RF_HAA": 0.0,  "LH_HAA": 0.0,  "RH_HAA": 0.0,
        "LF_HFE": 0.4,  "RF_HFE": 0.4,  "LH_HFE": -0.4, "RH_HFE": -0.4,
        "LF_KFE": -0.8, "RF_KFE": -0.8, "LH_KFE": 0.8,  "RH_KFE": 0.8,
    })


@dataclass
class CommandsCfg:
    pos_x_range: tuple = (-3.0, 3.0)
    pos_y_range: tuple = (-3.0, 3.0)
    heading_range: tuple = (-3.14159, 3.14159)
    resampling_time: float = 8.0


@dataclass
class Normalization:
    lin_vel: float = 2.0
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05


@dataclass
class Asset:
    body_name: str = "base"
    foot_names: list = field(default_factory=lambda: [
        "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"
    ])
    terminate_after_contacts_on: list = field(default_factory=lambda: ["base"])
    ground_name: str = "ground"


@dataclass
class Sensor:
    base_linvel: str = "base_linvel"
    base_gyro: str = "base_gyro"


@dataclass
class RewardConfig:
    # ── Isaac Lab 原始设计 ──────────────────────────
    # 粗位置跟踪：1 - tanh(distance / std)
    position_tracking_weight:      float = 0.5
    position_tracking_std:         float = 2.0
    # 精位置跟踪：1 - tanh(distance / std)
    position_tracking_fine_weight: float = 0.5
    position_tracking_fine_std:    float = 0.2
    # 朝向跟踪：-|heading_error|
    orientation_tracking_weight:   float = -0.2
    # 终止惩罚
    termination_penalty:           float = -400.0

    # ── 新增：到达后停止奖励 ────────────────────────
    # 鼓励到达目标后速度归零，解决抖动问题
    stop_bonus_scale:   float = 2.0
    zero_ang_bonus:     float = 6.0


@registry.envcfg("anymal_c_nav")
@dataclass
class AnymalCNavCfg(EnvCfg):
    model_file: str            = model_file
    max_episode_seconds: float = 8.0
    sim_dt: float              = 0.005
    ctrl_dt: float             = 0.02
    max_dof_vel: float         = 100.0


    init_state:    InitState    = field(default_factory=InitState)
    commands:      CommandsCfg  = field(default_factory=CommandsCfg)
    reward:        RewardConfig = field(default_factory=RewardConfig)
    control:       ControlConfig= field(default_factory=ControlConfig)
    noise:         NoiseConfig  = field(default_factory=NoiseConfig)
    normalization: Normalization= field(default_factory=Normalization)
    asset:         Asset        = field(default_factory=Asset)
    sensor:        Sensor       = field(default_factory=Sensor)

    num_obs:     int = 54
    num_actions: int = 12