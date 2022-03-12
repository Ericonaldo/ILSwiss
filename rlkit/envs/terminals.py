import importlib
from typing import Callable
import numpy as np


def get_terminal_func(env_name: str) -> Callable:
    env_name = env_name.split("_")
    cls_name = "".join(map(lambda s: (s[0].upper() + s[1:]), env_name)) + "TerminalFunc"
    module = importlib.import_module(".terminals", package="rlkit.envs")
    c = getattr(module, cls_name)
    return c.is_terminal


class TerminalFunc:
    @staticmethod
    def is_terminal(
        obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError


class InvertedPendulumTerminalFunc(TerminalFunc):
    @staticmethod
    def is_terminal(
        obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray
    ) -> np.ndarray:
        assert obs.ndim == next_obs.ndim == act.ndim == 2
        notdone = np.all(np.isfinite(next_obs), axis=-1)
        notdone = notdone * (np.abs(next_obs[:, 1]) <= 0.2)
        done = ~notdone
        done = done[:, None]
        return done


class InvertedDoublePendulumTerminalFunc(TerminalFunc):
    @staticmethod
    def is_terminal(
        obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray
    ) -> np.ndarray:
        assert obs.ndim == next_obs.ndim == act.ndim == 2
        sin1, cos1 = next_obs[:, 1], next_obs[:, 3]
        sin2, cos2 = next_obs[:, 2], next_obs[:, 4]
        theta_1 = np.arctan2(sin1, cos1)
        theta_2 = np.arctan2(sin2, cos2)
        y = 0.6 * (cos1 + np.cos(theta_1 + theta_2))
        done = y <= 1
        done = done[:, None]
        return done


class HopperTerminalFunc(TerminalFunc):
    @staticmethod
    def is_terminal(
        obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray
    ) -> np.ndarray:
        assert obs.ndim == next_obs.ndim == act.ndim == 2
        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done = (
            np.all(np.isfinite(next_obs), axis=-1)
            * np.all(np.abs(next_obs[:, 1:] < 100), axis=-1)
            * (height > 0.7)
            * (np.abs(angle) < 0.2)
        )
        done = ~not_done
        done = done[:, None]
        return done


class Walker2dTerminalFunc(TerminalFunc):
    @staticmethod
    def is_terminal(
        obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray
    ) -> np.ndarray:
        assert obs.ndim == next_obs.ndim == act.ndim == 2
        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
        done = ~not_done
        done = done[:, None]
        return done


class HalfcheetahTerminalFunc(TerminalFunc):
    @staticmethod
    def is_terminal(
        obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray
    ) -> np.ndarray:
        assert obs.ndim == next_obs.ndim == act.ndim == 2
        done = np.array([False]).repeat(len(obs))
        done = done[:, None]
        return done


class HumanoidTerminalFunc(TerminalFunc):
    @staticmethod
    def is_terminal(
        obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray
    ) -> np.ndarray:
        assert obs.ndim == next_obs.ndim == act.ndim == 2
        z = next_obs[:, 0]
        done = (z < 1.0) + (z > 2.0)
        done = done[:, None]
        return done


class AntTerminalFunc(TerminalFunc):
    @staticmethod
    def is_terminal(
        obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray
    ) -> np.ndarray:
        assert obs.ndim == next_obs.ndim == act.ndim == 2
        x = next_obs[:, 0]
        not_done = np.all(np.isfinite(next_obs), axis=-1) * (x >= 0.2) * (x <= 1.0)
        done = ~not_done
        done = done[:, None]
        return done
