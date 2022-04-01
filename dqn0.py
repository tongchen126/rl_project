from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment,py_environment
from tf_agents.specs import array_spec
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy,policy_saver
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts

from dqn0_utils import *
from dqn0_custom_agent import CardGameEnv

if __name__ == '__main__':
    set_memory_growth()
    config = {
        'num_iterations' : 50000,
        'initial_collect_steps' : 100,
        'collect_steps_per_iteration' : 10,
        'collect_episodes_per_iteration': 1,
        'replay_buffer_max_length' : 10000,
        'batch_size' : 128,
        'learning_rate' : 1e-3,
        'log_interval' : 200,
        'num_eval_episodes' : 10,
        'eval_interval' : 1000,
        'save_path'     : 'dqn0_policy/',
    }
    use_custom_env = False
    if use_custom_env:
        train_py_env = CardGameEnv()
        eval_py_env = CardGameEnv()
    else:
        env_name = 'CartPole-v0'
        #env_name = 'MountainCar-v0'
        train_py_env = suite_gym.load(env_name)
        eval_py_env = suite_gym.load(env_name)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    agent = create_dqn_agent(train_env.action_spec(),train_env.time_step_spec(),config['learning_rate'])

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    rb_observer,replay_buffer = create_rb_dataset(agent.collect_data_spec,config['replay_buffer_max_length'],config['batch_size'])

    py_driver.PyDriver(
        train_py_env,
        py_tf_eager_policy.PyTFEagerPolicy(
          random_policy, use_tf_function=True),
        [rb_observer],
        max_steps=config['initial_collect_steps']).run(train_py_env.reset())

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=config['batch_size'],
        num_steps=2).prefetch(3).shuffle(config['replay_buffer_max_length'])

    iterator = iter(dataset)

    # Evaluate the random policy and agent's policy once before training.
    print('Average Return for random policy {0}'.format(compute_avg_return(eval_env, random_policy, config['num_eval_episodes'])))
    avg_return = compute_avg_return(eval_env, agent.policy, config['num_eval_episodes'])
    returns = [avg_return]

    # Reset the environment.
    time_step = train_py_env.reset()

    # Create a driver to collect experience.
    collect_driver = py_driver.PyDriver(
        train_py_env,
        py_tf_eager_policy.PyTFEagerPolicy(
          agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=config['collect_steps_per_iteration'],
        max_episodes=config['collect_episodes_per_iteration'])

    for _ in range(config['num_iterations']):

      # Collect a few steps and save to the replay buffer.
      time_step, _ = collect_driver.run(time_step)

      # Sample a batch of data from the buffer and update the agent's network.
      experience, _ = next(iterator)
      train_loss = agent.train(experience).loss

      step = agent.train_step_counter.numpy()

      if step % config['log_interval'] == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

      if step % config['eval_interval'] == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, config['num_eval_episodes'])
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

    save_avg_return(config['num_iterations'],config['eval_interval'],returns)
    policy_saver.PolicySaver(agent.policy).save(config['save_path'])

    if use_custom_env:
        actions,observations = create_policy_eval_actions(eval_env,agent.policy)
        print(actions)
        print(observations)
        actions,observations = create_policy_eval_actions(eval_env,random_policy)
        print(actions)
        print(observations)
    else:
        create_policy_eval_video(eval_env,eval_py_env,agent.policy, "trained-agent")
        create_policy_eval_video(eval_env,eval_py_env,random_policy, "random-agent")
