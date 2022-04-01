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
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)

def save_avg_return(num_iterations,eval_interval,data,file='avg_return.png'):
    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, data)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=250)
    plt.savefig(file)

def create_policy_eval_video(env,py_env,policy, filename, num_episodes=5, fps=30):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = env.reset()
      video.append_data(py_env.render())
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = env.step(action_step.action)
        video.append_data(py_env.render())
  return embed_mp4(filename)

def create_policy_eval_actions(env,policy, num_episodes=5):
    actions = []
    observations = []
    for _ in range(num_episodes):
        action = []
        observation = []
        time_step = env.reset()
        while not time_step.is_last():
            observation.extend(time_step.observation.numpy().flatten().tolist())
            action_step = policy.action(time_step)
            action.extend(action_step.action.numpy().flatten().tolist())
            time_step = env.step(action_step.action)
        actions.append(action)
        observations.append(observation)
    return actions,observations

def create_rb_dataset(collect_data_spec,replay_buffer_max_length,batch_size,table_name = 'uniform_table'):
  replay_buffer_signature = tensor_spec.from_spec(
        collect_data_spec)
  replay_buffer_signature = tensor_spec.add_outer_dim(
      replay_buffer_signature)

  table = reverb.Table(
      table_name,
      max_size=replay_buffer_max_length,
      sampler=reverb.selectors.Uniform(),
      remover=reverb.selectors.Fifo(),
      rate_limiter=reverb.rate_limiters.MinSize(1),
      signature=replay_buffer_signature)

  reverb_server = reverb.Server([table])

  replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
      collect_data_spec,
      table_name=table_name,
      sequence_length=2,
      local_server=reverb_server)

  rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    table_name,
    sequence_length=2)

  return rb_observer,replay_buffer

def create_dqn_agent(action_spec,time_step_spec,lr):
  fc_layer_params = (100, 50)
  num_actions = action_spec.maximum - action_spec.minimum + 1

  def dense_layer(num_units):
    return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_in', distribution='truncated_normal'))

  dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
  q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
      minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
  q_net = sequential.Sequential(dense_layers + [q_values_layer])

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

  train_step_counter = tf.Variable(0)

  agent = dqn_agent.DqnAgent(
    time_step_spec,
    action_spec,
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

  agent.initialize()

  return agent

def set_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

if __name__ == '__main__':
    from dqn0_custom_agent import CardGameEnv
    set_memory_growth()
    use_custom_env = False
    if use_custom_env:
        eval_py_env = CardGameEnv()
    else:
        env_name = 'CartPole-v0'
        #env_name = 'MountainCar-v0'
        eval_py_env = suite_gym.load(env_name)

    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    random_policy = random_tf_policy.RandomTFPolicy(eval_env.time_step_spec(),
                                                    eval_env.action_spec())
    policy = tf.saved_model.load('dqn0_policy/')

    if use_custom_env:
        actions,observations = create_policy_eval_actions(eval_env,policy)
        print(actions)
        print(observations)
        actions,observations = create_policy_eval_actions(eval_env,random_policy)
        print(actions)
        print(observations)
    else:
        create_policy_eval_video(eval_env,eval_py_env,policy, "trained-agent")
        create_policy_eval_video(eval_env,eval_py_env,random_policy, "random-agent")