import gym
from stable_baselines3 import PPO


def main():
    env = gym.make('CartPole-v1')  # 创建环境
    model = PPO("MlpPolicy", env, verbose=1)  # 创建模型
    model.learn(total_timesteps=20000)  # 训练模型
    model.save("ppo_cartpole")  # 保存模型
    test_model(model)  # 测试模型


def test_model(model):
    env = gym.make('CartPole-v1', render_mode='human')  # 可视化只能在初始化时指定
    obs, _ = env.reset()
    done1, done2 = False, False
    total_reward = 0

    while not done1 or done2:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done1, done2, info = env.step(action)
        total_reward += reward

    print(f'Total Reward: {total_reward}')
    env.close()


if __name__ == "__main__":
    main()