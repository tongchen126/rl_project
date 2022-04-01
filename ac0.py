import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
from matplotlib import pyplot as plt
from keras import layers
from typing import Any, List, Sequence, Tuple
import tqdm

from ac0_utils import *
if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    num_actions = env.action_space.n  # 2
    num_hidden_units = 128

    model = ActorCritic(num_actions, num_hidden_units)

    eps = np.finfo(np.float32).eps.item()
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    min_episodes_criterion = 100
    max_episodes = 10000
    max_steps_per_episode = 1000

    # Cartpole-v0 is considered solved if average reward is >= 195 over 100
    # consecutive trials
    reward_threshold = 195
    running_reward = 0

    # Discount factor for future rewards
    gamma = 0.99

    # Keep last episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    with tqdm.trange(max_episodes) as t:
        for i in t:
            initial_state = tf.constant(env.reset(), dtype=tf.float32)
            episode_reward = train_step(
                env,eps,huber_loss,initial_state, model, optimizer, gamma, max_steps_per_episode)

            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            t.set_description(f'Episode {i}')
            t.set_postfix(
                episode_reward=episode_reward, running_reward=running_reward)

            # Show average episode reward every 10 episodes
            if i % 10 == 0:
                pass  # print(f'Episode {i}: average reward: {avg_reward}')

            if running_reward > reward_threshold and i >= min_episodes_criterion:
                break

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

    create_policy_eval_video(env,model,'ac0_test')