import numpy as np
from rlkit.data_management.env_replay_buffer import (
    EnvReplayBuffer,
)
from gym.spaces import Box, Discrete, Tuple, Dict
import rlkit.data_management.data_augmentation as rad

aug_to_func = {
                'crop':rad.random_crop,
                'grayscale':rad.random_grayscale,
                'cutout':rad.random_cutout,
                'cutout_color':rad.random_cutout_color,
                'flip':rad.random_flip,
                'rotate':rad.random_rotation,
                'rand_conv':rad.random_convolution,
                'color_jitter':rad.random_color_jitter,
                'translate':rad.random_translate,
                'no_aug':rad.no_aug,
            }

class AugmentImageEnvReplayBuffer(EnvReplayBuffer):
    def __init__(self, max_replay_buffer_size, env, random_seed=1995, image_size=84, pre_image_size=84, data_augs='translate'):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            env=env,
            random_seed=random_seed,
        )
        self.image_size = image_size
        self.pre_image_size = pre_image_size # for translation

        self.augs_funcs = {}
        for aug_name in data_augs.split('-'):
            assert aug_name in aug_to_func, 'invalid data aug string'
            self.augs_funcs[aug_name] = aug_to_func[aug_name]

    def random_batch(
        self, batch_size, keys=None, **kwargs
    ):
        indices = self._np_randint(0, self._size, batch_size)

        batch_data = self._get_batch_using_indices(
            indices, keys=keys, **kwargs
        )

        if keys is None:
            keys = ["observations", "next_observations"]
        
        if len(self.augs_funcs.keys()):
            for aug, func in self.augs_funcs.items():
                # apply crop and cutout first
                if 'crop' in aug or 'cutout' in aug:
                    if "observations" in keys:
                        batch_data["observations"] = func(batch_data["observations"])
                    if "next_observations" in keys:
                        batch_data["next_observations"] = func(batch_data["next_observations"])
                elif 'translate' in aug:
                    rndm_idxs = {}
                    if "observations" in keys: 
                        og_obses = rad.center_crop_images(batch_data["observations"], self.pre_image_size)
                        batch_data["observations"], rndm_idxs = func(og_obses, self.image_size, return_random_idxs=True)
                        batch_data["observations"] = batch_data["observations"] / 255.

                    if "next_observations" in keys:
                        og_next_obses = rad.center_crop_images(batch_data["next_observations"], self.pre_image_size)
                        batch_data["next_observations"] = func(og_next_obses, self.image_size, return_random_idxs=(rndm_idxs is None), **rndm_idxs)
                        batch_data["next_observations"] = batch_data["next_observations"] / 255.

                    # augmentations go here
                    else:
                        if "observations" in keys:
                            batch_data["observations"] = func(batch_data["observations"])
                        if "next_observations" in keys:
                            batch_data["next_observations"] = func(batch_data["next_observations"])

        return batch_data