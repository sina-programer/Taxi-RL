from model import Agent
import gym


env = gym.make('Taxi-v3', render_mode='human').env  # graphical render mode by pygame
agent = Agent(env, table_path='QTable.npy', epsilon=1e-3)
agent.load_table()

episodes = int(input('Several episodes: '))
agent.run_episodes(episodes=episodes, log_per_episode=1)
