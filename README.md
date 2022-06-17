# ILSwiss
[News!] We are now support [envpool](https://github.com/sail-sg/envpool) for accelerating training, see example yaml file in `https://github.com/Ericonaldo/ILSwiss/blob/main/exp_specs/sac/sac_hopper_envpool.yaml`.

[News!] We have implemented CURL and RAD in torch for potential imgae-based researches, see yaml file in `exp_specs`.

[News!] We now support experiments logging using [wandb](https://wandb.ai).

[News!] We have implemented MBPO in torch which achieves similar results to the original implementation, see yaml file in `exp_specs`.

ILSwiss is an Easy-to-run Imitation Learning (IL, or Learning from Demonstration, LfD) framework (template) in PyTorch based on existing code base.

This repository is built on top of [rlswiss](https://github.com/KamyarGh/rl_swiss) and [rlkit](https://github.com/rail-berkeley/rlkit/). The original rlswiss contains meta-rl methods and redundant codes, in this repo, we clean and optimize the code architecture, modify and re-implement algorithms for the purpose of easier running **imitation learning** experiments (rlkit focus on general RL algorithms). We further introduce vec envs to sample data in a parallel style to boost the sampling stage refering to [tianshou](https://github.com/thu-ml/tianshou) and add tensorboard support.

You can easily build experiment codes under this framework in your research. We will continue to maintain this repo while keeping it clear and clean.

## Implementing RL algorithms (for potential researches):

- ValueDICE
- OPOLO


## Implemented RL algorithms:

- Soft-Actor-Critic (SAC)
- Soft-Actor-Critic (SAC) (Auto Learning Alpha version)
- Soft-Actor-Critic-Auto-Encoder (SAC-AE)
- TD3
- DDPG
- PPO
- HER (Goal-Condtioned RL, with SAC or TD3)
- GCSL (Goal-Condtioned RL)
- Model-Based Policy Optimization (MBPO, https://github.com/jannerm/mbpo)
- Reinforcement Learning with Augmented Data (RAD-SAC, https://github.com/MishaLaskin/rad)
- Contrastive Unsupervised Representation Learning (CURL-SAC, https://github.com/MishaLaskin/curl/)

## Implemented IL algorithms:

- Adversarial Inverse Reinforcement Learning
    - AIRL / GAIL / FAIRL / Discriminator-Actor-Critic (DAC) (Different reward signals for AIRL / GAIL / FAIRL, and absorbing state for DAC)
- Behaviour Cloning (BC)
- Dagger

# Running Notes:

Before running, assign important log and output paths in `\rlkit\launchers\config.py`.

There are simple multiple processing shcheduling (we use multiple processing to clarify it with multi-processing since it only starts many independent sub-process without communication) for simple hyperparameter grid search.

The main entry is `run_experiments.py`, with the assigned experiment yaml file in `\exp_specs`:
`python run_experiment.py -g 0 -e your_yaml_path` or `python run_experiment.py -e your_yaml_path`.

When you run the `run_experiments.py`, it reads the yaml file, and generate small yaml files with only one hyperparameter setting for each. In a yaml file, a script file path is assigned (see `\run_scripts\`), which is specified to run the script with every the small yaml file. See `\exp_specs\sac\bc.yaml` for necessary explaination of each parameter.

NOTE: all experiments, including the evaluation tasks (see `\run_scripts\evaluate_policy.py` and `\exp_specs\evaluate_policy`) and the render tasks, can be run under this framework by specifying the yaml file (in a multiple processes style).

## Running RL algorithms

RL algorithms do not need demonstrations. Therefore, all you need is to write an experiment yaml file (see an example in `\exp_specs\sac\sac_hopper.yaml`) and run with the above suggestions.

For on-policy algorithms (e.g., PPO), we clean the buffer after every training step.

### Example scripts

`-e` means the path to the yaml file, `-g` means gpu id.

run mbpo for hopper:

```
python run_experiment -e \exp_specs\mbpo\mbpo_hopper.yaml -g 0
```

run sac-ae for finger_spin:

```
python run_experiment -e \exp_specs\sac\sac_ae_dmc_finger_spin.yaml -g 0
```


run sac for hopper:

```
python run_experiment -e \exp_specs\sac\sac_hopper.yaml -g 0
```


run ppo for hopper:

```
python run_experiment -e \exp_specs\ppo\ppo_hopper.yaml -g 0
```

run td3 for humanoid:

```
python run_experiment -e \exp_specs\td3\td3_humanoid.yaml -g 0
```

run her for pick with td3:

```
python run_experiment -e \exp_specs\her\her_pick_td3.yaml -g 0
```

## Running IL algorithms

IL algorithms need to be assigned with demonstrations. The input-format-matching standard mujoco and dmc expert demonstrations data files run by us can be download in [here](https://drive.google.com/drive/folders/1cZYLU-Wm11SV76apLZUJHrirk8N4pVyh?usp=sharing). If you want to sample your own data, train an expert agent using RL algorithms and sample using `\run_scripts\gen_expert_demo.py` or `\run_scripts\evaluate_policy.py`, and do not forget to modify your IO format.

If you get the demos ready, write the path for each expert name in `demos_listing.yaml` (there are already some examples). Then you should specify the expert name and the traj number in the corresponding yaml file (see `\exp_specs\bc.yaml` for example). After all the stuff, you can run it as a regular experiment following the above suggestions.

### Example scripts

gen expert data for hopper:

```
python run_experiment -e \exp_specs\gen_expert\hopper.yaml -g 0
```

run bc for hopper:

```
python run_experiment -e \exp_specs\bc.yaml -g 0
```

run gail for walker:

```
python run_experiment -e \exp_specs\gail\gail_walker.yaml -g 0
```

## Notes on wandb logging

The project name in wandb can be configured in `\rlkit\launchers\config.py`. And the experiment name for each trial shown in wandb UI is the same as `exp_name` in the yaml file.

## Some qualitive baseline reults

See exp_specs for detailed curve results.

### SAC

| Envs | Mean | Std
| ----  | ----  | ----  |
| Pendulum | 139.7313 | 79.8126 |
| InvertedPendulum-v2 | 1000.0000 | 0.0000 |
| InvertedDoublePendulum-v2 | 9358.8740 | 0.1043
| Ant-v2 | 5404.5532 | 1520.4961 |
| Hopper-v2 | 3402.9494 | 446.4877 |
| Humanoid-v2 | 6043.9907 | 726.1788 |
| HalfCheetah-v2 | 13711.6445 | 111.4709 |
| Walker2d-v2 | 5639.3267 | 29.9715 |

### SAC-AE

| Envs | Mean | Std
| ----  | ----  | ----  |
| Finger_Spin (600K) | 983.42 | 5.82 |
| Reach_Easy (1600K) | 782.8 | 23.86 |

### Random

| Envs | Mean | Std
| ----  | ----  | ----  |
| InvertedPendulum-v2 | 25.2800 | 5.5318 |
| InvertedDoublePendulum-v2 | 78.2829 | 10.7335
| Ant-v2 | 713.5986 | 203.9204 |
| Hopper-v2 | 13.0901 | 0.1022 |
| Humanoid-v2 | 64.7384 | 2.3037 |
| HalfCheetah-v2 | 74.4849 | 12.3917 |
| Walker2d-v2 | 7.0708 | 0.1292 |
| Swimmer-v2 | 15.5430 | 6.6655 |

