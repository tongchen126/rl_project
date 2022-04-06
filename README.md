# rl_project
```
tensorflow==2.8.0
```
# Deep Q Network / DDPG
dqn0.py: main file for training. Change the gym environment here. And also speficy whether a discrete and continuous action space is used.  
dqn0_utils.py: agent and dataset creation, model definition, evaluation, save video file..  
dqn0_custom_agent.py: create your custom environment here.


# Actor-Critic (alternative)
Only environment with discrete action space have been validated.  
ac0.py: main file for training.  
ac0_utils.py: agent and dataset creation..
