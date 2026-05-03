import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.math.quaternion import Quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import AnymalCNavCfg


@registry.env("anymal_c_nav", "np")
class AnymalCNavEnv(NpEnv):
    _cfg: AnymalCNavCfg

    def __init__(self, cfg: AnymalCNavCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()
        self._target_marker_body = self._model.get_body("target_marker")

        # 定义动作空间：12维
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        # 定义观测空间：54维
        # linvel(3) + gyro(3) + gravity(3) + joint_pos(12) + joint_vel(12)
        # + last_actions(12) + commands(3) + position_error(2)
        # + heading_error(1) + distance(1) + reached_flag(1) + stop_ready_flag(1)
        self._observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32
        )

        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action  = self._model.num_actuators

        # 初始化默认状态
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)

        # 设置缓冲区
        self._init_buffer()

    # ─────────────────────────────────────────────
    # 缓冲区初始化
    # ─────────────────────────────────────────────
    def _init_buffer(self):
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel,
             cfg.normalization.ang_vel],
            dtype=np.float32,
        )
        # 获取关键 actuator 默认角度
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle
        self._init_dof_pos[-self._num_action:] = self.default_angles

    # ─────────────────────────────────────────────
    # 接触检测初始化
    # ─────────────────────────────────────────────
    def _init_contact_geometry(self):
        cfg = self._cfg
        self.ground_index = self._model.get_geom_index(cfg.asset.ground_name)
        self._init_termination_contact()
        self._init_foot_contact()

    def _init_termination_contact(self):
        cfg = self._cfg
        base_indices = []
        for name in cfg.asset.terminate_after_contacts_on:
            idx = self._model.get_geom_index(name)
            if idx is not None:
                base_indices.append(idx)
        if base_indices:
            self.termination_contact = np.array(
                [[idx, self.ground_index] for idx in base_indices],
                dtype=np.uint32
            )
            self.num_termination_check = self.termination_contact.shape[0]
        else:
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0

    def _init_foot_contact(self):
        cfg = self._cfg
        foot_indices = []
        for name in cfg.asset.foot_names:
            idx = self._model.get_geom_index(name)
            if idx is not None:
                foot_indices.append(idx)
        if foot_indices:
            self.foot_contact_check = np.array(
                [[idx, self.ground_index] for idx in foot_indices],
                dtype=np.uint32
            )
            self.num_foot_check = self.foot_contact_check.shape[0]
        else:
            self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
            self.num_foot_check = 0

    # ─────────────────────────────────────────────
    # 属性
    # ─────────────────────────────────────────────
    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    # ─────────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────────
    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)

    def _extract_root_state(self, data):
        pose        = self._body.get_pose(data)
        root_pos    = pose[:, :3]
        root_quat   = pose[:, 3:7]
        root_linvel = self._model.get_sensor_value(
            self._cfg.sensor.base_linvel, data
        )
        return root_pos, root_quat, root_linvel

    def _compute_projected_gravity(self, quat: np.ndarray) -> np.ndarray:
        gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return Quaternion.rotate_vector(quat, gravity)

    def _heading_diff(self, current, target):
        diff = target - current
        diff = np.where(diff > np.pi,  diff - 2 * np.pi, diff)
        diff = np.where(diff < -np.pi, diff + 2 * np.pi, diff)
        return diff

    def _compute_nav_commands(self, root_pos, root_quat, pose_commands):
        """根据当前位置和目标位置计算期望速度指令"""
        robot_position  = root_pos[:, :2]
        robot_heading   = Quaternion.get_yaw(root_quat)
        target_position = pose_commands[:, :2]
        target_heading  = pose_commands[:, 2]

        position_error     = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        reached_position = distance_to_target < 0.3
        hdiff            = self._heading_diff(robot_heading, target_heading)
        reached_heading  = np.abs(hdiff) < np.deg2rad(15)
        reached_all      = np.logical_and(reached_position, reached_heading)

        desired_vel_xy   = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy   = np.where(
            reached_position[:, np.newaxis], 0.0, desired_vel_xy
        )

        desired_yaw_rate = np.clip(hdiff * 1.0, -1.0, 1.0)
        desired_yaw_rate = np.where(
            np.abs(hdiff) < np.deg2rad(8), 0.0, desired_yaw_rate
        )
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        desired_vel_xy   = np.where(
            reached_all[:, np.newaxis], 0.0, desired_vel_xy
        )

        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )
        return velocity_commands, position_error, hdiff, distance_to_target, reached_all

    def _update_target_marker(self, data, pose_commands):
        num_envs  = data.shape[0]
        arrow_pos = np.column_stack([
            pose_commands[:, 0],
            pose_commands[:, 1],
            np.full((num_envs,), 0.5),
        ])
        arrow_quat = Quaternion.from_euler(0, 0, pose_commands[:, 2])
        mocap      = self._model.get_body("target_marker").mocap
        mocap.set_pose(data, np.concatenate([arrow_pos, arrow_quat], axis=1))

    def _update_heading_arrows(self, data, root_pos, desired_vel_xy, base_lin_vel_xy):
        """更新机器人速度方向箭头（绿色=实际速度，蓝色=期望速度）"""
        arrow_height = 0.76
        cur_yaw = np.where(
            np.linalg.norm(base_lin_vel_xy, axis=1) > 1e-3,
            np.arctan2(base_lin_vel_xy[:, 1], base_lin_vel_xy[:, 0]),
            0.0,
        )
        robot_arrow_pos = root_pos.copy()
        robot_arrow_pos[:, 2] = arrow_height
        robot_arrow_quat = Quaternion.from_euler(0, 0, cur_yaw)
        mocap = self._model.get_body("robot_heading_arrow").mocap
        mocap.set_pose(data, np.concatenate([robot_arrow_pos, robot_arrow_quat], axis=1))

        des_yaw = np.where(
            np.linalg.norm(desired_vel_xy, axis=1) > 1e-6,
            np.arctan2(desired_vel_xy[:, 1], desired_vel_xy[:, 0]),
            0.0,
        )
        desired_arrow_quat = Quaternion.from_euler(0, 0, des_yaw)
        mocap = self._model.get_body("desired_heading_arrow").mocap
        mocap.set_pose(data, np.concatenate([robot_arrow_pos, desired_arrow_quat], axis=1))

    # ─────────────────────────────────────────────
    # apply_action：动作处理
    # ─────────────────────────────────────────────
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        if "current_actions" not in state.info:
            state.info["current_actions"] = np.zeros_like(actions)
        state.info["last_actions"]    = state.info["current_actions"]
        state.info["current_actions"] = actions

        actions_scaled = actions * self._cfg.control.action_scale
        state.data.actuator_ctrls = self.default_angles + actions_scaled
        return state

    # ─────────────────────────────────────────────
    # _compute_observation：观测计算与空间对齐
    # ─────────────────────────────────────────────
    def _compute_observation(self, data, state_info, velocity_commands,
                              position_error, heading_diff, distance,
                              reached_all) -> np.ndarray:
        """
        组装 54 维观测向量，与观测空间对齐：
        linvel(3) + gyro(3) + gravity(3) + joint_pos(12) + joint_vel(12)
        + last_actions(12) + commands(3) + position_error(2)
        + heading_error(1) + distance(1) + reached_flag(1) + stop_ready_flag(1)
        """
        cfg               = self._cfg
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        base_lin_vel      = root_vel[:, :3]
        gyro              = self._model.get_sensor_value(
            cfg.sensor.base_gyro, data
        )
        projected_gravity = self._compute_projected_gravity(root_quat)
        joint_pos         = self.get_dof_pos(data)
        joint_vel         = self.get_dof_vel(data)
        joint_pos_rel     = joint_pos - self.default_angles
        stop_ready        = np.logical_and(
            reached_all, np.abs(gyro[:, 2]) < 0.05
        )

        obs = np.concatenate([
            base_lin_vel   * cfg.normalization.lin_vel,    # 3
            gyro           * cfg.normalization.ang_vel,    # 3
            projected_gravity,                              # 3
            joint_pos_rel  * cfg.normalization.dof_pos,    # 12
            joint_vel      * cfg.normalization.dof_vel,    # 12
            state_info["current_actions"],                  # 12
            velocity_commands * self.commands_scale,        # 3
            position_error / 5.0,                           # 2
            (heading_diff / np.pi)[:, np.newaxis],          # 1
            np.clip(distance / 5.0, 0, 1)[:, np.newaxis],  # 1
            reached_all.astype(np.float32)[:, np.newaxis],  # 1
            stop_ready.astype(np.float32)[:, np.newaxis],   # 1
        ], axis=-1)
        assert obs.shape == (data.shape[0], 54)
        return obs

    # ─────────────────────────────────────────────
    # _compute_reward：奖励计算
    # ─────────────────────────────────────────────
    def _compute_reward(self, data, info, velocity_commands,
                         reached_all) -> np.ndarray:
        """
        奖励函数（参考 Isaac Lab navigation_env_cfg.py）

        Isaac Lab 原始设计（已保留）：
        - position_tracking:      1 - tanh(distance / 2.0)  权重 0.5  （粗跟踪）
        - position_tracking_fine: 1 - tanh(distance / 0.2)  权重 0.5  （精跟踪）
        - orientation_tracking:   -|heading_error|           权重 -0.2
        - termination_penalty:    -400.0

        新增优化（解决到达后抖动问题）：
        - stop_bonus:      到达目标后鼓励线速度归零（高斯奖励）
        - zero_ang_bonus:  到达目标后鼓励角速度归零
        """
        cfg = self._cfg.reward

        # 计算距离和朝向误差
        pose           = self._body.get_pose(data)
        root_quat      = pose[:, 3:7]
        robot_pos      = pose[:, :2]
        target_pos     = info["pose_commands"][:, :2]
        distance       = np.linalg.norm(target_pos - robot_pos, axis=1)
        robot_heading  = Quaternion.get_yaw(root_quat)
        target_heading = info["pose_commands"][:, 2]
        heading_diff   = self._heading_diff(robot_heading, target_heading)

        # ── Isaac Lab 原始奖励 ──────────────────────
        # 1. 位置跟踪（粗）：1 - tanh(distance / 2.0)，权重 0.5
        r_pos = cfg.position_tracking_weight * (
            1.0 - np.tanh(distance / cfg.position_tracking_std)
        )

        # 2. 位置跟踪（精）：1 - tanh(distance / 0.2)，权重 0.5
        r_pos_fine = cfg.position_tracking_fine_weight * (
            1.0 - np.tanh(distance / cfg.position_tracking_fine_std)
        )

        # 3. 朝向跟踪：-|heading_error|，权重 -0.2
        r_orientation = cfg.orientation_tracking_weight * np.abs(heading_diff)

        # 4. 终止惩罚：-400.0
        terminated    = self._check_termination_mask(data)
        r_termination = np.where(terminated, cfg.termination_penalty, 0.0)

        # ── 新增：到达后停止奖励 ────────────────────
        base_lin_vel = self._model.get_sensor_value(
            self._cfg.sensor.base_linvel, data
        )
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)

        # 到达目标后鼓励线速度归零（高斯奖励）
        speed_xy  = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        stop_lin  = 0.8 * np.exp(-((speed_xy / 0.2) ** 2))
        # 到达目标后鼓励角速度归零
        stop_ang  = 1.2 * np.exp(-((np.abs(gyro[:, 2]) / 0.1) ** 4))
        stop_bonus = np.where(reached_all, 2.0 * (stop_lin + stop_ang), 0.0)

        # 完全停稳额外奖励
        zero_ang_bonus = np.where(
            np.logical_and(reached_all, np.abs(gyro[:, 2]) < 0.05),
            6.0, 0.0
        )

        return (r_pos + r_pos_fine + r_orientation + r_termination
                + stop_bonus + zero_ang_bonus)

    def _check_termination_mask(self, data) -> np.ndarray:
        """返回布尔数组，True 表示触发终止条件"""
        terminated = np.zeros(self._num_envs, dtype=bool)

        # 条件1：关节速度超阈值或出现 NaN/Inf
        dof_vel    = self.get_dof_vel(data)
        vel_max    = np.abs(dof_vel).max(axis=1)
        terminated = np.logical_or(terminated,
            vel_max > self._cfg.max_dof_vel)
        terminated = np.logical_or(terminated,
            np.isnan(dof_vel).any(axis=1) | np.isinf(dof_vel).any(axis=1))

        # 条件2：机身与地面碰撞（倒地）
        cquerys  = self._model.get_contact_query(data)
        contacts = cquerys.is_colliding(self.termination_contact)
        contacts = contacts.reshape(
            (self._num_envs, self.num_termination_check)
        )
        terminated = np.logical_or(terminated, contacts.any(axis=1))

        # 条件3：机身倾斜超过 75°（侧翻）
        pose   = self._body.get_pose(data)
        proj_g = self._compute_projected_gravity(pose[:, 3:7])
        gxy    = np.linalg.norm(proj_g[:, :2], axis=1)
        tilt   = np.arctan2(gxy, np.abs(proj_g[:, 2]))
        terminated = np.logical_or(terminated, tilt > np.deg2rad(75))

        return terminated

    # ─────────────────────────────────────────────
    # _check_termination：终止条件判断
    # ─────────────────────────────────────────────
    def _check_termination(self, state: NpEnvState) -> np.ndarray:
        """
        判断终止条件：
        1. 关节速度超过阈值或出现 NaN/Inf
        2. 机身与地面碰撞（倒地）
        3. 机身倾斜角度超过 75°（侧翻）
        """
        return self._check_termination_mask(state.data)

    # ─────────────────────────────────────────────
    # update_state：状态更新
    # ─────────────────────────────────────────────
    def update_state(self, state: NpEnvState) -> NpEnvState:
        data          = state.data
        pose_commands = state.info["pose_commands"]

        root_pos, root_quat, root_vel = self._extract_root_state(data)
        base_lin_vel = root_vel[:, :3]

        velocity_commands, position_error, heading_diff, distance, reached_all = \
            self._compute_nav_commands(root_pos, root_quat, pose_commands)

        # 观测计算
        obs = self._compute_observation(
            data, state.info, velocity_commands,
            position_error, heading_diff, distance, reached_all
        )

        # 更新目标标记和速度箭头
        self._update_target_marker(data, pose_commands)
        self._update_heading_arrows(
            data, root_pos, velocity_commands[:, :2], base_lin_vel[:, :2]
        )

        # 奖励计算
        reward = self._compute_reward(
            data, state.info, velocity_commands, reached_all
        )

        # 终止条件
        terminated = self._check_termination(state)

        state.obs        = obs
        state.reward     = reward
        state.terminated = terminated
        return state

    # ─────────────────────────────────────────────
    # reset：重置逻辑
    # ─────────────────────────────────────────────
    def reset(self, data: mtx.SceneData,
              done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        cfg      = self._cfg
        num_envs = data.shape[0]

        # 随机生成机器人初始位置（±0.5m）
        pos_range = cfg.init_state.pos_randomization_range
        robot_x   = np.random.uniform(pos_range[0], pos_range[2], num_envs)
        robot_y   = np.random.uniform(pos_range[1], pos_range[3], num_envs)

        # 随机生成目标位置（Isaac Lab 范围 ±3m）
        offset = np.column_stack([
            np.random.uniform(
                cfg.commands.pos_x_range[0],
                cfg.commands.pos_x_range[1], num_envs
            ),
            np.random.uniform(
                cfg.commands.pos_y_range[0],
                cfg.commands.pos_y_range[1], num_envs
            ),
        ])
        target_pos = np.stack([robot_x, robot_y], axis=1) + offset
        target_yaw = np.random.uniform(
            cfg.commands.heading_range[0],
            cfg.commands.heading_range[1],
            size=(num_envs, 1),
        )
        pose_commands = np.concatenate([target_pos, target_yaw], axis=1)

        # 设置初始关节状态
        init_dof_pos    = np.tile(self._init_dof_pos, (*data.shape, 1))
        init_dof_vel    = np.tile(self._init_dof_vel, (*data.shape, 1))
        noise_pos       = np.zeros(
            (*data.shape, self._num_dof_pos), dtype=np.float32
        )
        noise_pos[:, 0] = robot_x - cfg.init_state.pos[0]
        noise_pos[:, 1] = robot_y - cfg.init_state.pos[1]
        noise_vel       = np.zeros(
            (*data.shape, self._num_dof_vel), dtype=np.float32
        )

        data.reset(self._model)
        data.set_dof_vel(init_dof_vel + noise_vel)
        data.set_dof_pos(init_dof_pos + noise_pos, self._model)
        self._model.forward_kinematic(data)
        self._update_target_marker(data, pose_commands)

        # 计算初始观测
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        base_lin_vel = root_vel[:, :3]
        velocity_commands, position_error, heading_diff, distance, reached_all = \
            self._compute_nav_commands(root_pos, root_quat, pose_commands)

        # 更新速度箭头
        self._update_heading_arrows(
            data, root_pos, velocity_commands[:, :2], base_lin_vel[:, :2]
        )

        info = {
            "pose_commands":   pose_commands,
            "last_actions":    np.zeros(
                (num_envs, self._num_action), dtype=np.float32
            ),
            "current_actions": np.zeros(
                (num_envs, self._num_action), dtype=np.float32
            ),
        }

        obs = self._compute_observation(
            data, info, velocity_commands,
            position_error, heading_diff, distance, reached_all
        )
        return obs, info