envs_dict = {
    "cartpole": "gym.envs.classic_control:CartPoleEnv",
    # "Standard" Mujoco Envs
    "halfcheetah": "gym.envs.mujoco.half_cheetah:HalfCheetahEnv",
    "ant": "gym.envs.mujoco.ant:AntEnv",
    "hopper": "gym.envs.mujoco.hopper:HopperEnv",
    "walker": "gym.envs.mujoco.walker2d:Walker2dEnv",
    "humanoid": "gym.envs.mujoco.humanoid:HumanoidEnv",
    "swimmer": "gym.envs.mujoco.swimmer:SwimmerEnv",
    "inverteddoublependulum": "gym.envs.mujoco.inverted_double_pendulum:InvertedDoublePendulumEnv",
    "invertedpendulum": "gym.envs.mujoco.inverted_pendulum:InvertedPendulumEnv",

    # normal envs
    "lunarlandercont": "gym.envs.box2d.lunar_lander:LunarLanderContinuous",

    # robotics envs
    "fetch-reach": "gym.envs.robotics.fetch.reach:FetchReachEnv",
    "fetch-push": "gym.envs.robotics.fetch.push:FetchPushEnv",
    "fetch-pick-place": "gym.envs.robotics.fetch.pick_and_place.FetchPickAndPlace",
    "fetch-slide": "gym.envs.robotics.fetch.slide:FetchSlide",

}
