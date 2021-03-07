# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Navigation
# 
# ---
# 
# In this notebook, you will learn how to use the Unity ML-Agents environment for the first project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).
# 
# ### 1. Start the Environment
# 
# We begin by importing some necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

# %%
from unityagents import UnityEnvironment
import numpy as np

# %% [markdown]
# Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.
# 
# - **Mac**: `"path/to/Banana.app"`
# - **Windows** (x86): `"path/to/Banana_Windows_x86/Banana.exe"`
# - **Windows** (x86_64): `"path/to/Banana_Windows_x86_64/Banana.exe"`
# - **Linux** (x86): `"path/to/Banana_Linux/Banana.x86"`
# - **Linux** (x86_64): `"path/to/Banana_Linux/Banana.x86_64"`
# - **Linux** (x86, headless): `"path/to/Banana_Linux_NoVis/Banana.x86"`
# - **Linux** (x86_64, headless): `"path/to/Banana_Linux_NoVis/Banana.x86_64"`
# 
# For instance, if you are using a Mac, then you downloaded `Banana.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
# ```
# env = UnityEnvironment(file_name="Banana.app")
# ```

# %%
# env = UnityEnvironment(file_name="./unity-env/Banana.app")

# # %% [markdown]
# # Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# # %%
# # get the default brain
# brain_name = env.brain_names[0]
# brain = env.brains[brain_name]

# # %% [markdown]
# # ### 2. Examine the State and Action Spaces
# # 
# # The simulation contains a single agent that navigates a large environment.  At each time step, it has four actions at its disposal:
# # - `0` - walk forward 
# # - `1` - walk backward
# # - `2` - turn left
# # - `3` - turn right
# # 
# # The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. 
# # 
# # Run the code cell below to print some information about the environment.

# # %%
# # reset the environment
# env_info = env.reset(train_mode=True)[brain_name]

# # number of agents in the environment
# print('Number of agents:', len(env_info.agents))

# # number of actions
# action_size = brain.vector_action_space_size
# print('Number of actions:', action_size)

# # examine the state space 
# state = env_info.vector_observations[0]
# print('States look like:', state)
# state_size = len(state)
# print('States have length:', state_size)

# # %% [markdown]
# # ### 3. Take Random Actions in the Environment
# # 
# # In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.
# # 
# # Once this cell is executed, you will watch the agent's performance, if it selects an action (uniformly) at random with each time step.  A window should pop up that allows you to observe the agent, as it moves through the environment.  
# # 
# # Of course, as part of the project, you'll have to change the code so that the agent is able to use its experience to gradually choose better actions when interacting with the environment!

# # %%
# env_info = env.reset(train_mode=False)[brain_name] # reset the environment
# state = env_info.vector_observations[0]            # get the current state
# score = 0                                          # initialize the score
# while True:
#     action = np.random.randint(action_size)        # select an action
#     env_info = env.step(action)[brain_name]        # send the action to the environment
#     next_state = env_info.vector_observations[0]   # get the next state
#     reward = env_info.rewards[0]                   # get the reward
#     done = env_info.local_done[0]                  # see if episode has finished
#     score += reward                                # update the score
#     state = next_state                             # roll over the state to next time step
#     if done:                                       # exit loop if episode finished
#         break
    
# print("Score: {}".format(score))

# # %% [markdown]
# # When finished, you can close the environment.

# # %%
# env.close()

# %% [markdown]
# ### 4. It's Your Turn!
# 
# Now it's your turn to train your own agent to solve the environment!  When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```

# %%
env = UnityEnvironment(file_name="../unity-env/Banana.app")


# %%
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# %%
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


# %%
from dqn_agent import Agent

agent = Agent(state_size=state_size, action_size=action_size, seed=0)


# %%
from collections import deque
import matplotlib.pyplot as plt
import torch 

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    
    #env_info = env.reset(train_mode=True)[brain_name]
    i_episode = 0

    #for i_episode in range(1, n_episodes+1):
    while True:
        i_episode += 1
        
        env_info = env.reset(train_mode=True)[brain_name]
        #state = np.uint8(255 * np.array(env_info.visual_observations[0]))
        state = np.uint8(255 * np.array(env_info.vector_observations[0]))
        score = 0
        done = False

        while done is False:
        #for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            #next_state, reward, done, _ = env.step(action)
            #next_state = np.uint8(255 * np.array(env_info.visual_observations[0]))
            next_state = np.uint8(255 * np.array(env_info.vector_observations[0]))
            reward = env_info.rewards[0]
            #episode_rewards += reward
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            # if done:
            #     break 

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    
    return scores

scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# %%
env.close()


