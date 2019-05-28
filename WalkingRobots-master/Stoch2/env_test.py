import envs.stoch2_gym_bullet_env as stoch2_gym_env

env = stoch2_gym_env.Stoch2Env(render = True)
obs = env.reset()

while(True):
    action = env.action_space.sample()
    print(action)
    env.step(action)
    pass