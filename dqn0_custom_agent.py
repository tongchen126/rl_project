from __future__ import absolute_import, division, print_function
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class CardGameEnv(py_environment.PyEnvironment):
  """
    Let's say we want to train an agent to play the following (Black Jack inspired) card game:

        The game is played using an infinite deck of cards numbered 1...10.
        At every turn the agent can do 2 things: get a new random card, or stop the current round.
        The goal is to get the sum of your cards as close to 21 as possible at the end of the round, without going over.

    An environment that represents the game could look like this:

        Actions: We have 2 actions. Action 0: get a new card, and Action 1: terminate the current round.
        Observations: Sum of the cards in the current round.
        Reward: The objective is to get as close to 21 as possible without going over, so we can achieve this using the following reward at the end of the round: sum_of_cards - 21 if sum_of_cards <= 21, else -21
  """
  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.int32, minimum=0, name='observation')
    self._state = 0
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = 0
    self._episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    # Make sure episodes don't go on forever.
    if action == 1:
      self._episode_ended = True
    elif action == 0:
      new_card = np.random.randint(1, 11)
      self._state += new_card
    else:
      raise ValueError('`action` should be 0 or 1.')

    if self._episode_ended or self._state >= 21:
      reward = self._state - 21 if self._state <= 21 else -21
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
      return ts.transition(
          np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)