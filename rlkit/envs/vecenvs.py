"""
The implementation of the vecenv is based on [tianshou](https://github.com/thu-ml/tianshou/blob/master/tianshou/env/venvs.py)
"""

import gym
from gym import Env
import numpy as np
from typing import Any, List, Tuple, Optional, Union, Callable
from rlkit.envs.worker import EnvWorker, DummyEnvWorker, SubprocEnvWorker
from rlkit.data_management.normalizer import RunningMeanStd

EPS = np.finfo(np.float32).eps.item()


class BaseVectorEnv(Env):
    """Base class for vectorized environments wrapper.
    Usage:
    ::
        env_num = 8
        envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(env_num)])
        assert len(envs) == env_num
    It accepts a list of environment generators. In other words, an environment
    generator ``efn`` of a specific task means that ``efn()`` returns the
    environment of the given task, for example, ``gym.make(task)``.
    All of the VectorEnv must inherit :class:`~tianshou.env.BaseVectorEnv`.
    Here are some other usages:
    ::
        envs.seed(2)  # which is equal to the next line
        envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # set specific seed for each env
        obs = envs.reset()  # reset all environments
        obs = envs.reset([0, 5, 7])  # reset 3 specific environments
        obs, rew, done, info = envs.step([1] * 8)  # step synchronously
        envs.render()  # render all environments
        envs.close()  # close all environments
    .. warning::
        If you use your own environment, please make sure the ``seed`` method
        is set up properly, e.g.,
        ::
            def seed(self, seed):
                np.random.seed(seed)
        Otherwise, the outputs of these envs may be the same with each other.
    :param env_fns: a list of callable envs, ``env_fns[i]()`` generates the ith env.
    :param worker_fn: a callable worker, ``worker_fn(env_fns[i])`` generates a
        worker which contains the i-th env.
    :param int wait_num: use in asynchronous simulation if the time cost of
        ``env.step`` varies with time and synchronously waiting for all
        environments to finish a step is time-wasting. In that case, we can
        return when ``wait_num`` environments finish a step and keep on
        simulation in these environments. If ``None``, asynchronous simulation
        is disabled; else, ``1 <= wait_num <= env_num``.
    :param float timeout: use in asynchronous simulation same as above, in each
        vectorized step it only deal with those environments spending time
        within ``timeout`` seconds.
    :param bool norm_obs: Whether to track mean/std of data and normalise observation
        on return. For now, observation normalization only support observation of
        type np.ndarray.
    :param obs_rms: class to track mean&std of observation. If not given, it will
        initialize a new one. Usually in envs that is used to evaluate algorithm,
        obs_rms should be passed in. Default to None.
    :param bool update_obs_rms: Whether to update obs_rms. Default to True.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        worker_fn: Callable[[Callable[[], gym.Env]], EnvWorker],
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
        norm_obs: bool = False,
        obs_rms: Optional[RunningMeanStd] = None,
        goal_rms: Optional[RunningMeanStd] = None,
        update_obs_rms: bool = True,
    ) -> None:
        self._env_fns = env_fns
        # A VectorEnv contains a pool of EnvWorkers, which corresponds to
        # interact with the given envs (one worker <-> one env).
        self.workers = [worker_fn(fn) for fn in env_fns]
        self.worker_class = type(self.workers[0])
        assert issubclass(self.worker_class, EnvWorker)
        assert all([isinstance(w, self.worker_class) for w in self.workers])

        self.env_num = len(env_fns)
        self.wait_num = wait_num or len(env_fns)
        assert (
            1 <= self.wait_num <= len(env_fns)
        ), f"wait_num should be in [1, {len(env_fns)}], but got {wait_num}"
        self.timeout = timeout
        assert (
            self.timeout is None or self.timeout > 0
        ), f"timeout is {timeout}, it should be positive if provided!"
        self.is_async = self.wait_num != len(env_fns) or timeout is not None
        self.waiting_conn: List[EnvWorker] = []
        # environments in self.ready_id is actually ready
        # but environments in self.waiting_id are just waiting when checked,
        # and they may be ready now, but this is not known until we check it
        # in the step() function
        self.waiting_id: List[int] = []
        # all environments are ready in the beginning
        self.ready_id = list(range(self.env_num))
        self.is_closed = False

        # initialize observation running mean/std
        self.norm_obs = norm_obs
        self.update_obs_rms = update_obs_rms
        self.obs_rms = RunningMeanStd() if obs_rms is None and norm_obs else obs_rms
        self.goal_rms = RunningMeanStd() if goal_rms is None and norm_obs else goal_rms
        self.__eps = np.finfo(np.float32).eps.item()

    def _assert_is_not_closed(self) -> None:
        assert (
            not self.is_closed
        ), f"Methods of {self.__class__.__name__} cannot be called after close."

    def __len__(self) -> int:
        """Return len(self), which is the number of environments."""
        return self.env_num

    def __getattribute__(self, key: str) -> Any:
        """Switch the attribute getter depending on the key.
        Any class who inherits ``Env`` will inherit some attributes, like
        ``action_space``. However, we would like the attribute lookup to go straight
        into the worker (in fact, this vector env's action_space is always None).
        """
        if key in [
            "metadata",
            "reward_range",
            "spec",
            "action_space",
            "observation_space",
        ]:  # reserved keys in Env
            return self.__getattr__(key)
        else:
            return super().__getattribute__(key)

    def __getattr__(self, key: str) -> List[Any]:
        """Fetch a list of env attributes.
        This function tries to retrieve an attribute from each individual wrapped
        environment, if it does not belong to the wrapping vector environment class.
        """
        return [getattr(worker, key) for worker in self.workers]

    def _wrap_id(
        self, id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> Union[List[int], np.ndarray]:
        if id is None:
            return list(range(self.env_num))
        return [id] if np.isscalar(id) else id  # type: ignore

    def _assert_id(self, id: Union[List[int], np.ndarray]) -> None:
        for i in id:
            assert (
                i not in self.waiting_id
            ), f"Cannot interact with environment {i} which is stepping now."
            assert (
                i in self.ready_id
            ), f"Can only interact with ready environments {self.ready_id}."

    def reset(
        self, id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> np.ndarray:
        """Reset the state of some envs and return initial observations.
        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        obs_list = [self.workers[i].reset() for i in id]
        try:
            obs = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs = np.array(obs_list, dtype=object)
        if self.obs_rms and self.update_obs_rms:
            if type(obs_list[0]) == dict:
                self.obs_rms.update(np.stack([_["observation"] for _ in obs_list]))
                self.goal_rms.update(np.stack([_["achieved_goal"] for _ in obs_list]))
            else:
                self.obs_rms.update(obs)
        return self.normalize_obs(obs)

    def step(
        self, action: np.ndarray, id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run one timestep of some environments' dynamics.
        If id is None, run one timestep of all the environments’ dynamics;
        otherwise run one timestep for some environments with given id, either
        an int or a list. When the end of episode is reached, you are
        responsible for calling reset(id) to reset this environment’s state.
        Accept a batch of action and return a tuple (batch_obs, batch_rew,
        batch_done, batch_info) in numpy format.
        :param numpy.ndarray action: a batch of action provided by the agent.
        :return: A tuple including four items:
            * ``obs`` a numpy.ndarray, the agent's observation of current environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``done`` a numpy.ndarray, whether these episodes have ended, in \
                which case further step() calls will return undefined results
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)
        For the async simulation:
        Provide the given action to the environments. The action sequence
        should correspond to the ``id`` argument, and the ``id`` argument
        should be a subset of the ``env_id`` in the last returned ``info``
        (initially they are env_ids of all the environments). If action is
        None, fetch unfinished step() calls instead.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if not self.is_async:
            assert len(action) == len(id)
            for i, j in enumerate(id):
                self.workers[j].send_action(action[i])
            result = []
            for j in id:
                obs, rew, done, info = self.workers[j].get_result()
                info["env_id"] = j
                result.append((obs, rew, done, info))
        else:
            if action is not None:
                self._assert_id(id)
                assert len(action) == len(id)
                for i, (act, env_id) in enumerate(zip(action, id)):
                    self.workers[env_id].send_action(act)
                    self.waiting_conn.append(self.workers[env_id])
                    self.waiting_id.append(env_id)
                self.ready_id = [x for x in self.ready_id if x not in id]
            ready_conns: List[EnvWorker] = []
            while not ready_conns:
                ready_conns = self.worker_class.wait(
                    self.waiting_conn, self.wait_num, self.timeout
                )
            result = []
            for conn in ready_conns:
                waiting_index = self.waiting_conn.index(conn)
                self.waiting_conn.pop(waiting_index)
                env_id = self.waiting_id.pop(waiting_index)
                obs, rew, done, info = conn.get_result()
                info["env_id"] = env_id
                result.append((obs, rew, done, info))
                self.ready_id.append(env_id)
        obs_list, rew_list, done_list, info_list = zip(*result)
        try:
            obs_stack = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs_stack = np.array(obs_list, dtype=object)
        rew_stack, done_stack, info_stack = map(
            np.stack, [rew_list, done_list, info_list]
        )
        if self.obs_rms and self.update_obs_rms:
            if type(obs_list[0]) == dict:
                self.obs_rms.update(np.stack([_["observation"] for _ in obs_list]))
                self.goal_rms.update(np.stack([_["achieved_goal"] for _ in obs_list]))
            else:
                self.obs_rms.update(obs_stack)
        return self.normalize_obs(obs_stack), rew_stack, done_stack, info_stack

    def seed(
        self, seed: Optional[Union[int, List[int]]] = None
    ) -> List[Optional[List[int]]]:
        """Set the seed for all environments.
        Accept ``None``, an int (which will extend ``i`` to
        ``[i, i + 1, i + 2, ...]``) or a list.
        :return: The list of seeds used in this env's random number generators.
            The first value in the list should be the "main" seed, or the value
            which a reproducer pass to "seed".
        """
        self._assert_is_not_closed()
        seed_list: Union[List[None], List[int]]
        if seed is None:
            seed_list = [seed] * self.env_num
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.env_num)]
        else:
            seed_list = seed
        return [w.seed(s) for w, s in zip(self.workers, seed_list)]

    def render(self, **kwargs: Any) -> List[Any]:
        """Render all of the environments."""
        self._assert_is_not_closed()
        if self.is_async and len(self.waiting_id) > 0:
            raise RuntimeError(
                f"Environments {self.waiting_id} are still stepping, cannot "
                "render them now."
            )
        return [w.render(**kwargs) for w in self.workers]

    def close(self) -> None:
        """Close all of the environments.
        This function will be called only once (if not, it will be called during
        garbage collected). This way, ``close`` of all workers can be assured.
        """
        self._assert_is_not_closed()
        for w in self.workers:
            w.close()
        self.is_closed = True

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations by statistics in obs_rms."""
        if self.obs_rms and self.norm_obs:
            clip_max = 10.0  # this magic number is from openai baselines
            # see baselines/common/vec_env/vec_normalize.py#L10
            if type(obs[0]) == dict:
                for _ in obs:
                    _["observation"] = np.clip(
                        (_["observation"] - self.obs_rms.mean)
                        / np.sqrt(self.obs_rms.var + self.__eps),
                        -clip_max,
                        clip_max,
                    )
                    _["achieved_goal"] = np.clip(
                        (_["achieved_goal"] - self.goal_rms.mean)
                        / np.sqrt(self.goal_rms.var + self.__eps),
                        -clip_max,
                        clip_max,
                    )
                    _["desired_goal"] = np.clip(
                        (_["desired_goal"] - self.goal_rms.mean)
                        / np.sqrt(self.goal_rms.var + self.__eps),
                        -clip_max,
                        clip_max,
                    )
            else:
                obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.__eps)
                obs = np.clip(obs, -clip_max, clip_max)
        return obs

    def unnormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Unnormalize observations by statistics in obs_rms."""
        if self.obs_rms and self.norm_obs:
            obs = obs.copy()
            if type(obs[0]) == dict:
                for _ in obs:
                    _["observation"] = (
                        _["observation"] * np.sqrt(self.obs_rms.var + self.__eps)
                        + self.obs_rms.mean
                    )
                    _["achieved_goal"] = (
                        _["achieved_goal"] * np.sqrt(self.goal_rms.var + self.__eps)
                        + self.goal_rms.mean
                    )
                    _["desired_goal"] = (
                        _["achieved_goal"] * np.sqrt(self.goal_rms.var + self.__eps)
                        + self.goal_rms.mean
                    )
            else:
                obs = obs * np.sqrt(self.obs_rms.var + self.__eps) + self.obs_rms.mean
        return obs


class DummyVectorEnv(BaseVectorEnv):
    """Dummy vectorized environment wrapper, implemented in for-loop.
    .. seealso::
        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        super().__init__(env_fns, DummyEnvWorker, **kwargs)


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess."""

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            return SubprocEnvWorker(fn, share_memory=False)

        super().__init__(env_fns, worker_fn, **kwargs)
