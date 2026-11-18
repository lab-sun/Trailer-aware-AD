import numpy as np
from gym import spaces

from carla_env.core.obs_manager.obs_manager import ObsManagerBase


class ObsManager(ObsManagerBase):
    """
    in deg
    """

    def __init__(self, obs_configs):
        self._parent_actor = None
        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        self.obs_space = spaces.Dict({
            'theta': spaces.Box(low=-90.0, high=90.0, shape=(1,), dtype=np.float32)
        })

    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor

    def get_observation(self):
        theta = self._parent_actor._theta

        obs = {
            'theta': np.array([theta], dtype=np.float32)
        }
        return obs

    def clean(self):
        self._parent_actor = None