import numpy as np
from numpy.random import choice
from numpy.random import randint

from rlkit.data_management.replay_buffer import ReplayBuffer

import torch
from torch.autograd import Variable


class NPTransDataSampler():
    def __init__(self, blocks_list):
        self.blocks_list = blocks_list
        block = blocks_list[0]
        self.obs_dim = block['_observations'].shape[1]
        self.act_dim = block['_actions'].shape[1]


    def sample_batch(self, batch_size, context_size_range, test_size=0, test_is_context=True):
        '''
            From a block takes the first N as context and if test is not
            the same as context, randomly samples M from the rest
        '''
        obs_dim = self.obs_dim
        act_dim = self.act_dim
        batch_inds = choice(len(self.blocks_list), size=batch_size)
        X_context, Y_context = [], []
        X_test, Y_test = [], []
        context_size = []
        context_mask = np.zeros((batch_size, context_size_range[1], 1))

        for enum_ind, i in enumerate(batch_inds):
            block = self.blocks_list[i]
            N = randint(context_size_range[0], context_size_range[1])
            context_size.append(N)

            obs = np.zeros((context_size_range[1], obs_dim))
            actions = np.zeros((context_size_range[1], act_dim))
            rewards = np.zeros((context_size_range[1], 1))
            next_obs = np.zeros((context_size_range[1], obs_dim))

            obs[:N] = block['_observations'][:N]
            actions[:N] = block['_actions'][:N]
            rewards[:N] = block['_rewards'][:N]
            next_obs[:N] = block['_next_obs'][:N]
            context_mask[enum_ind, :N] = 1.0

            X_context.append(np.concatenate((obs, actions), 1))
            Y_context.append(np.concatenate((next_obs, rewards), 1))

            if not test_is_context:
                num_range = np.arange(N, block['_rewards'].shape[0])
                num_range = choice(num_range, size=test_size, replace=False)

                obs = block['_observations'][num_range]
                actions = block['_actions'][num_range]
                rewards = block['_rewards'][num_range]
                next_obs = block['_next_obs'][num_range]

                X_test.append(np.concatenate((obs, actions), 1))
                Y_test.append(np.concatenate((next_obs, rewards), 1))
        
        X_context = np.stack(X_context)
        Y_context = np.stack(Y_context)
        if test_is_context:
            X_test = X_context
            Y_test = Y_context
            test_mask = context_mask
        else:
            X_test = np.stack(X_test)
            Y_test = np.stack(Y_test)
            test_mask = np.ones((batch_size, test_size, 1))
        
        return Variable(torch.FloatTensor(X_context)), Variable(torch.FloatTensor(Y_context)), Variable(torch.FloatTensor(context_mask)), Variable(torch.FloatTensor(X_test)), Variable(torch.FloatTensor(Y_test)), Variable(torch.FloatTensor(test_mask))
