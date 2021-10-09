# ILSwiss

ILSwiss is an Easy-to-run Imitation Learning (IL, or Learning from Demonstration, LfD) framework (template) in PyTorch based on existing code base. 

The most works are based on [rlswiss](https://github.com/KamyarGh/rl_swiss), which has been extended from the August 2018 version of [rlkit](https://github.com/vitchyr/rlkit). Since the original rlswiss contains meta-rl methods and redundant codes, in this repo, we clean and optimize the code architecture, modify and re-implement algorithms for the purpose of easier running imitation learning experiments. We further introduce vec envs to sample data in a parallel style to boost the sampling stage refering to [tianshou](https://github.com/thu-ml/tianshou) and add tensorboard support.

You can easily build experiment codes under this framework in your research. We will continue to maintain this repo while keeping it clear and clean.

## Implemented RL algorithms:

- Soft-Actor-Critic (SAC)
- Soft-Actor-Critic (SAC) (Auto Learning Alpha version)
- Soft Q Learning (SQL)
- TD3
- DDPG

## Implemented IL algorithms:

- Adversarial Inverse Reinforcement Learning 
    - AIRL / GAIL / FAIRL / Discriminator-Actor-Critic (DAC) (Different reward signals for AIRL / GAIL / FAIRL, and absorbing state for DAC)
- Behaviour Cloning (bc)
- DAgger


# Running Notes:

Before running, assign important log and output paths in `\rlkit\launchers\common.py` (There is an example file show necessary variables).

Their are simple multiple processing shcheduling (we use multiple processing to clarify it with multi-processing since it only stars many independent sub-process without communication) for simple hyperparameter grid search.

The main entry is **run_experiments.py**, with the assigned experiment yaml file in `\exp_specs`:
`python run_experiment.py -g 0 -e your_yaml_path` or `CUDA_VISIBLE_DEVICES=0 python run_experiment.py -e your_yaml_path`.

When you run the **run_experiments.py**, it reads the yaml file, and generate small yaml files with only one hyperparameter setting for each. In a yaml file, a script file path is assigned (see `\run_scripts\`), which is specified to run the script with every the small yaml file. See `\exp_specs\sac\bc.yaml` for necessary explaination of each parameter.

NOTE: all experiments, including the evaluation tasks (see `\run_scripts\evaluate_policy.py` and `\exp_specs\evaluate_policy`) and the render tasks, can be run under this framework by specifying the yaml file (in a multiple processes style).

## Running RL algorithms

RL algorithms do not need demonstrations. Therefore, all you need is to write an experiment yaml file (see an example in `\exp_specs\sac\sac_hopper.yaml`) and run with the above suggestions.

For on-policy algorithms (e.g., PPO), we clean the buffer after every training step.

## Running IL algorithms

IL algorithms need to be assigned with demonstrations. A input-format-matching standard mujoco demonstrations can be download in [here](https://github.com/apexrl/Baseline_Pool/tree/master/imitation_learning/sac/expert_trajs_50). If you want to sample your own data, train an expert agent using RL algorithms and sample using `\run_scripts\gen_expert_demo.py` or `\run_scripts\evaluate_policy.py`, and do not forget to modify your IO format.

If you get the demos ready, write the path for each expert name in `demos_listing.yaml` (there are already some examples). Then you should specify the expert name and the traj number in the corresponding yaml file (see `\exp_specs\bc.yaml` for example). After all the stuff, you can run it as a regular experiment following the above suggestions.
