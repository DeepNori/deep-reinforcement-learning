# Option 1: Solve the First Version
# The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

# Import the Necessary Packages
from numpy.lib.function_base import average
from unityagents import UnityEnvironment
import numpy as np
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent

# Instantiate the Environment and Agent

# Start the Environment
#env = UnityEnvironment(file_name='../unity-env/Reacher_Linux_NoVis/Reacher.x86')
env = UnityEnvironment(file_name='../unity-env/Reacher.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# init agent 
agent = Agent(state_size=state_size, action_size=action_size, random_seed=10)

# Train the Agent with DDPG
def ddpg():
    scores_deque = deque(maxlen=100)
    scores = []
    i_episode = 0

    while True:
        i_episode += 1
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
        state = env_info.vector_observations[0]                # get the current state
        agent.reset()
        score = 0
        done = False

        while done is False:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

        scores_deque.append(score)
        scores.append(score)
        average_score = np.mean(scores_deque)

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, average_score, score), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, average_score, score))

        if i_episode >= 100 and average_score >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, average_score))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break

    return scores

scores = ddpg()

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# Close the environment
env.close()
