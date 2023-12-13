import numpy as np
import gym  # 0.26.2
import os


class Agent:
    def __init__(self, env, table_path, alpha=.1, gamma=.6, epsilon=1, epsilon_min=1e-3, epsilon_decay=1-5e-4):
        self.env = env
        self.table_path = table_path
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])  # 500x6

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def run_episode(self, update_epsilon=True, update_table=True):
        state = self.env.reset()[0]
        epochs, rewards, penalties, accidents, done = 0, 0, 0, 0, False

        while not done:
            action = self.get_action(state, self.epsilon)
            new_state, reward, done = self.do_action(state, action)
            if update_table:
                self.update_table(state, new_state, action, reward)
            if update_epsilon:
                self.update_epsilon()

            state = new_state
            rewards += reward
            epochs += 1
            if reward == -10:
                penalties += 1
            elif reward == -3:
                accidents += 1

        return epochs, rewards, penalties, accidents

    def run_episodes(self, episodes, log_per_episode=10):
        for episode in range(1, episodes+1):
            epochs, rewards, penalties, accidents = self.run_episode()
            if not episode % log_per_episode:  # every <log_per_episode>
                print('Episode: ', episode)
                print('Epochs: ', epochs)
                print('Rewards: ', rewards)
                print('Penalties: ', penalties)
                print('Accidents: ', accidents)
                print('\n', '-'*50)

    def update_table(self, state, new_state, action, reward):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[new_state])
        new_value = ((1 - self.alpha) * old_value) + (self.alpha * (reward + (self.gamma * next_max)))
        self.q_table[state, action] = new_value

    def update_epsilon(self):
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

    def get_action(self, state, epsilon=.5):
        if random_run(epsilon):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def do_action(self, state, action):
        new_state, reward, done, *_ = self.env.step(action)
        if (new_state == state) and (reward == -1):  # punish for accident with wall
            reward = -3

        return new_state, reward, done

    def save_table(self, path=None):
        if not path:
            path = self.table_path
        np.save(path, self.q_table)

    def load_table(self, path=None):
        if not path:
            path = self.table_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"File <{path}> doesn't exists!")

        self.q_table = np.load(path)
        print(f'Table successfully loaded from <{path}>')


def random_run(probability=.5):
    if np.random.uniform(0, 1) < probability:
        return True
    return False


if __name__ == '__main__':
    env = gym.make('Taxi-v3', render_mode='ansi').env
    agent = Agent(env, table_path='QTable.npy')

    try:
        agent.load_table()
    except Exception:
        pass

    agent.run_episodes(episodes=1000, log_per_episode=50)
    agent.save_table()
