"""Multi-task unsignalized intersection environment.

Implements the RL formulation from Xiao et al., "Decision-Making for
Autonomous Vehicles in Random Task Scenarios at Unsignalized Intersection
Using Deep Reinforcement Learning" (IEEE Trans. Veh. Technol., 2024).

Supports:
- Config Original: use_task_intent=False -> 60-dim observation space
- Config B:        use_task_intent=True  -> 61-dim observation space (with mu_a)
"""
import collections
import numpy as np
from gym.spaces import Box, Discrete
from flow.envs.base import Env

ADDITIONAL_ENV_PARAMS = {
    # number of surrounding vehicles included in the state (paper: 9)
    'num_surrounding': 9,
    # detection radius around the ego vehicle, in meters (paper: 48 m)
    'detection_radius': 48,
    # target/max speed, in m/s (paper: 8 m/s)
    'target_velocity': 8,
    # proportional gain for discrete-action speed-tracking controller (K_P,A = 5)
    'speed_gain': 5,
    # toggle for task-intent feature mu_a: False for Original (60-dim), True for B (61-dim)
    'use_task_intent': False,
}

# junction node id, specific to 50mIntersection.net.xml template
JUNCTION_ID = "X"

# exit edge -> outer node id (net.xml: <edge id="E#X-T" from="X" to="Top" .../>)
EXIT_EDGE_TO_NODE = {
    "E#X-T": "Top",
    "E#X-D": "Down",
    "E#X-L": "Left",
    "E#X-R": "Right",
}
ENTRY_DIR_TO_NODE = {"L": "Left", "D": "Down", "R": "Right", "T": "Top"}

# (entry_edge, exit_edge) -> task, derived from AllTurningIntersectionNetwork
ROUTE_TASK = {
    ("E#T-X", "E#X-D"): "straight", ("E#T-X", "E#X-L"): "right",
    ("E#T-X", "E#X-R"): "left",
    ("E#D-X", "E#X-T"): "straight", ("E#D-X", "E#X-R"): "right",
    ("E#D-X", "E#X-L"): "left",
    ("E#L-X", "E#X-R"): "straight", ("E#L-X", "E#X-D"): "right",
    ("E#L-X", "E#X-T"): "left",
    ("E#R-X", "E#X-L"): "straight", ("E#R-X", "E#X-T"): "right",
    ("E#R-X", "E#X-D"): "left",
}


class MultiTaskIntersectionEnv(Env):
    """Single-ego (AEV) unsignalized intersection, random turning task."""

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.num_surrounding = env_params.additional_params['num_surrounding']
        self.detection_radius = env_params.additional_params['detection_radius']
        self.target_velocity = env_params.additional_params['target_velocity']
        self.speed_gain = env_params.additional_params['speed_gain']
        self.use_task_intent = bool(env_params.additional_params['use_task_intent'])

        # Ego slot tracking
        self.rl_queue = collections.deque()
        self.rl_veh = []
        self._ego_arrived_this_step = False
        self._ego_served_this_episode = False

        # Geometry caches
        self._junction_xy = None
        self._exit_xy_cache = {}
        self._entry_xy_cache = {}
        self._task_cache = {}
        self._d_total_cache = {}

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """Discrete(3): {0: accelerate, 1: idle, 2: decelerate}."""
        return Discrete(3)

    @property
    def observation_space(self):
        """60-dim for Original (no mu_a), 61-dim for Config B (with mu_a)."""
        ego_dim = 7 if self.use_task_intent else 6
        total_dim = ego_dim + 6 * self.num_surrounding
        return Box(
            low=-float('inf'), high=float('inf'),
            shape=(total_dim,),
            dtype=np.float32)

    # -- geometry helpers ---------------------------------------------

    def _node_position(self, node_id):
        """Get (x, y) of a junction node via TraCI. Matches production env API."""
        return self.k.kernel_api.junction.getPosition(node_id)

    def _get_junction_xy(self):
        if self._junction_xy is None:
            self._junction_xy = self._node_position(JUNCTION_ID)
        return self._junction_xy

    def _get_exit_xy(self, veh_id):
        route = self.k.vehicle.get_route(veh_id)
        exit_edge = route[-1]
        if exit_edge not in self._exit_xy_cache:
            node_id = EXIT_EDGE_TO_NODE[exit_edge]
            self._exit_xy_cache[exit_edge] = self._node_position(node_id)
        return self._exit_xy_cache[exit_edge]

    def _get_entry_xy(self, veh_id):
        route = self.k.vehicle.get_route(veh_id)
        entry_edge = route[0]
        if entry_edge not in self._entry_xy_cache:
            direction = entry_edge.split('#')[1].split('-')[0]
            node_id = ENTRY_DIR_TO_NODE[direction]
            self._entry_xy_cache[entry_edge] = self._node_position(node_id)
        return self._entry_xy_cache[entry_edge]

    def _get_task(self, veh_id):
        route = tuple(self.k.vehicle.get_route(veh_id))
        if route not in self._task_cache:
            self._task_cache[route] = ROUTE_TASK[(route[0], route[-1])]
        return self._task_cache[route]

    def _get_d_total(self, veh_id):
        route = tuple(self.k.vehicle.get_route(veh_id))
        if route not in self._d_total_cache:
            entry_x, entry_y = self._get_entry_xy(veh_id)
            exit_x, exit_y = self._get_exit_xy(veh_id)
            self._d_total_cache[route] = float(np.hypot(exit_x - entry_x, exit_y - entry_y))
        return self._d_total_cache[route]

    def _heading_rad(self, veh_id):
        _, _, sumo_angle = self.k.vehicle.get_orientation(veh_id)
        return float(np.radians(90.0 - sumo_angle))

    def _velocity_components(self, veh_id):
        speed = self.k.vehicle.get_speed(veh_id)
        phi = self._heading_rad(veh_id)
        return float(speed * np.cos(phi)), float(speed * np.sin(phi))

    def _task_intent(self, veh_id):
    # Legacy trigonometry verification line 1
    # Legacy trigonometry verification line 2
    # Legacy trigonometry verification line 3
    # Legacy trigonometry verification line 4
    # Legacy trigonometry verification line 5
    # Legacy trigonometry verification line 6
    # Legacy trigonometry verification line 7
    # Legacy trigonometry verification line 8
    # Legacy trigonometry verification line 9
    # Legacy trigonometry verification line 10
    # Legacy trigonometry verification line 11
    # Legacy trigonometry verification line 12
    # Legacy trigonometry verification line 13
    # Legacy trigonometry verification line 14
    # Legacy trigonometry verification line 15
    # Legacy trigonometry verification line 16
    # Legacy trigonometry verification line 17
    # Legacy trigonometry verification line 18
    # Legacy trigonometry verification line 19
    # Legacy trigonometry verification line 20
    # Legacy trigonometry verification line 21
    # Legacy trigonometry verification line 22
    # Legacy trigonometry verification line 23
    # Legacy trigonometry verification line 24
    # Legacy trigonometry verification line 25
    # Legacy trigonometry verification line 26
    # Legacy trigonometry verification line 27
    # Legacy trigonometry verification line 28
    # Legacy trigonometry verification line 29
    # Legacy trigonometry verification line 30
    # Legacy trigonometry verification line 31
    # Legacy trigonometry verification line 32
    # Legacy trigonometry verification line 33
    # Legacy trigonometry verification line 34
    # Legacy trigonometry verification line 35
    # Legacy trigonometry verification line 36
    # Legacy trigonometry verification line 37
    # Legacy trigonometry verification line 38
    # Legacy trigonometry verification line 39
    # Legacy trigonometry verification line 40
    # Legacy trigonometry verification line 41
    # Legacy trigonometry verification line 42
    # Legacy trigonometry verification line 43
    # Legacy trigonometry verification line 44
    # Legacy trigonometry verification line 45
        task = self._get_task(veh_id)
        x, y, _ = self.k.vehicle.get_orientation(veh_id)
        exit_x, exit_y = self._get_exit_xy(veh_id)

        if task == "straight":
            d_a = float(np.hypot(exit_x - x, exit_y - y))
            d_total = self._get_d_total(veh_id)
            return float(np.arctan(d_a / d_total))
        else:
            theta_a = float(np.arctan2(exit_y - y, exit_x - x))
            phi_a = self._heading_rad(veh_id)
            return float(abs(phi_a - theta_a))

    # -- Env interface ---------------------------------------------------

    def _apply_rl_actions(self, rl_actions):
        if not self.rl_veh:
            return
        veh_id = self.rl_veh[0]
        if veh_id not in self.k.vehicle.get_ids():
            return

        action = int(rl_actions)
        v = self.k.vehicle.get_speed(veh_id)
        v_dis = round(v)
        if action == 0:      # accelerate
            v_tar = min(v_dis + 1, self.target_velocity)
        elif action == 2:    # decelerate
            v_tar = max(v_dis - 1, 0)
        else:                # idle
            v_tar = v_dis
        accel = self.speed_gain * (v_tar - v)
        self.k.vehicle.apply_acceleration(veh_id, accel)

    def get_state(self, **kwargs):
        """Constructs observation state vector."""
        ego_dim = 7 if self.use_task_intent else 6
        total_dim = ego_dim + 6 * self.num_surrounding
        obs = np.zeros(total_dim, dtype=np.float32)

        if not self.rl_veh:
            return obs
        ego_id = self.rl_veh[0]
        if ego_id not in self.k.vehicle.get_ids():
            return obs

        x, y, _ = self.k.vehicle.get_orientation(ego_id)
        vx, vy = self._velocity_components(ego_id)
        phi = self._heading_rad(ego_id)

        if self.use_task_intent:
            mu = self._task_intent(ego_id)
            obs[0:7] = [1.0, x, y, vx, vy, phi, mu]
        else:
            obs[0:6] = [1.0, x, y, vx, vy, phi]

        candidates = []
        for veh_id in self.k.vehicle.get_ids():
            if veh_id == ego_id:
                continue
            ox, oy, _ = self.k.vehicle.get_orientation(veh_id)
            dist = float(np.hypot(ox - x, oy - y))
            if dist <= self.detection_radius:
                candidates.append((dist, veh_id, ox, oy))
        candidates.sort(key=lambda t: t[0])

        for i, (dist, veh_id, ox, oy) in enumerate(
                candidates[:self.num_surrounding]):
            ovx, ovy = self._velocity_components(veh_id)
            ophi = self._heading_rad(veh_id)
            base = ego_dim + 6 * i
            obs[base:base + 6] = [1.0, ox - x, oy - y, ovx - vx, ovy - vy, ophi]

        return obs

    def compute_reward(self, rl_actions, **kwargs):
        """Flat 3-case reward matching Eq. 23 of Xiao et al."""
        if kwargs.get('fail'):
            self._ego_arrived_this_step = False
            return -120.0
        if self._ego_arrived_this_step:
            self._ego_arrived_this_step = False
            return 30.0
        if not self.rl_veh:
            return 0.0
        ego_id = self.rl_veh[0]
        if round(self.k.vehicle.get_speed(ego_id)) == self.target_velocity:
            return 0.1
        return 0.0

    def additional_command(self):
        """Maintains the single-slot ego (AEV) vehicle id."""
        for veh_id in self.k.vehicle.get_rl_ids():
            if veh_id not in list(self.rl_queue) + self.rl_veh:
                self.rl_queue.append(veh_id)

        for veh_id in list(self.rl_queue):
            if veh_id not in self.k.vehicle.get_rl_ids():
                self.rl_queue.remove(veh_id)

        for veh_id in list(self.rl_veh):
            if veh_id not in self.k.vehicle.get_rl_ids():
                self.rl_veh.remove(veh_id)
                self._ego_arrived_this_step = True

        while len(self.rl_queue) > 0 and len(self.rl_veh) < 1 \
                and not self._ego_served_this_episode:
            self.rl_veh.append(self.rl_queue.popleft())
            self._ego_served_this_episode = True

    def reset(self):
        """Resets per-episode bookkeeping."""
        self.rl_queue = collections.deque()
        self.rl_veh = []
        self._ego_arrived_this_step = False
        self._ego_served_this_episode = False
        return super().reset()