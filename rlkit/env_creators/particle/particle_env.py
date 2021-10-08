from gym import spaces
import numpy as np

from rlkit.env_creators.base_env import BaseEnv


class ParticleEnv(BaseEnv):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_type: str = "approach", **kwargs):
        super().__init__(**kwargs)
        self.max_force = 3.0
        self.dt = 0.05  # seconds between state updates
        self.tol = 0.2
        self.max_steps = 500
        self.n_agents = 2
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]

        self.start = {
            "agent_0": np.array([-5, 0]),
            "agent_1": np.array([5, 0]),
        }
        assert task_type in ["approach", "leave"], task_type
        self.task_type = task_type
        self.steps = 0
        self.done_n = {a_id: False for a_id in self.agent_ids}

        self.max_pos = 10
        self.max_vel = 3
        self.max_state = np.array(
            [
                # self state
                self.max_pos,
                self.max_pos,
                self.max_vel,
                # opponent state
                self.max_pos,
                self.max_pos,
            ]
        )
        # Only allow horizontal movement, i.e., 1d env
        self.max_action = np.array(
            [
                self.max_force,
            ]
        )

        self.observation_space_n = dict(
            zip(
                self.agent_ids,
                [
                    spaces.Box(low=-self.max_state, high=self.max_state)
                    for _ in range(self.n_agents)
                ],
            )
        )
        self.action_space_n = dict(
            zip(
                self.agent_ids,
                [
                    spaces.Box(low=-self.max_action, high=self.max_action)
                    for _ in range(self.n_agents)
                ],
            )
        )

        self.state_n = None
        self.viewer = None

        self.seed()

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        for a_id in self.agent_ids:
            self.done_n[a_id] = False
        self.steps = 0
        self.state_n = dict(
            zip(
                self.agent_ids,
                [
                    np.concatenate(
                        [self.start[a_id], np.zeros(1)],
                        axis=0,
                    )
                    for a_id in self.agent_ids
                ],
            )
        )
        return self._get_obs(self.state_n)

    def _get_obs(self, state_n):
        # observation should include both ego state and opponent's position
        obs_dict = {}
        for a_id in self.agent_ids:
            obs_dict[a_id] = np.concatenate(
                [
                    state_n[a_id],
                    *[state_n[o_id][:2] for o_id in self.agent_ids if o_id != a_id],
                ],
                axis=0,
            )
        return obs_dict

    def step(self, action_n):

        reward_n = {a_id: 0.0 for a_id in self.agent_ids}
        info_n = {a_id: {} for a_id in self.agent_ids}
        for a_id in np.random.permutation(self.agent_ids):
            if self.done_n[a_id]:
                continue

            pos = self.state_n[a_id][:2]
            vel = self.state_n[a_id][2:]
            act = action_n[a_id]

            clamp = lambda x, minx, maxx: np.minimum(np.maximum(minx, x), maxx)
            force = clamp(act, -self.max_action, self.max_action)

            vel += force * self.dt
            max_vels = np.array([self.max_vel])
            vel = clamp(vel, -max_vels, max_vels)

            pos[0] += vel * self.dt

            # if outside the boundaries
            if (np.max(pos) > self.max_pos) or (np.min(pos) < -self.max_pos):
                max_pos_arr = np.array([self.max_pos, self.max_pos])
                pos = clamp(pos, -max_pos_arr, max_pos_arr)
                vel = np.zeros(1)
                self.done_n[a_id] = True
                if self.task_type == "leave":
                    reward_n[a_id] = 1.0

            self.state_n[a_id] = np.concatenate([pos, vel], axis=0)

            for o_id in self.agent_ids:
                if o_id != a_id:
                    o_pos = self.state_n[o_id][:2]
                    if bool(np.linalg.norm(pos - o_pos) < self.tol):
                        info_n[a_id]["collision"] = o_id
                        info_n[o_id]["collision"] = a_id
                        self.done_n[a_id] = True
                        self.done_n[o_id] = True
                        if self.task_type == "approach":
                            reward_n[a_id] = 1.0
                            reward_n[o_id] = 1.0

        self.steps += 1
        if self.steps >= self.max_steps:
            for a_id in self.agent_ids:
                self.done_n[a_id] = True

        return (
            self._get_obs(self.state_n),
            reward_n,
            self.done_n,
            info_n,
        )

    def render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_dim = 400
        screen_dim = 400

        world_width = self.max_pos * 2 + 1

        particle_radius = 10

        scale = (screen_dim - particle_radius * 2) / world_width

        def state2render(x):
            return x * scale + screen_dim / 2.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_dim, screen_dim)

            self.particle_render = dict(
                zip(
                    self.agent_ids,
                    [
                        rendering.make_circle(particle_radius)
                        for _ in range(self.n_agents)
                    ],
                )
            )
            self.particle_trans = dict(
                zip(
                    self.agent_ids,
                    [rendering.Transform() for _ in range(self.n_agents)],
                )
            )
            for a_id in self.agent_ids:
                self.particle_render[a_id].add_attr(self.particle_trans[a_id])
                self.particle_render[a_id].set_color(0.5, 0.5, 0.8)
                self.viewer.add_geom(self.particle_render[a_id])

            tracks = [
                rendering.Line(
                    state2render(np.array([-6, -6])), state2render(np.array([-6, 6]))
                ),
                rendering.Line(
                    state2render(np.array([-6, -6])), state2render(np.array([6, -6]))
                ),
                rendering.Line(
                    state2render(np.array([6, 6])), state2render(np.array([-6, 6]))
                ),
                rendering.Line(
                    state2render(np.array([6, 6])), state2render(np.array([6, -6]))
                ),
            ]
            for track in tracks:
                track.set_color(0, 0, 0)
                self.viewer.add_geom(track)

        if self.state_n is None:
            return None

        for a_id in self.agent_ids:
            render_x = state2render(self.state_n[a_id][0])
            render_y = state2render(self.state_n[a_id][1])
            self.particle_trans[a_id].set_translation(render_x, render_y)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
