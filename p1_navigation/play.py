# Watch a Smart Agent!

from unityagents import UnityEnvironment
import numpy as np
env = UnityEnvironment(file_name="../unity-env/Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

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

from dqn_agent import Agent

agent = Agent(state_size=state_size, action_size=action_size, seed=0)

import torch 

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

# for i in range(3):
#     state = env.reset()
#     for j in range(200):
#         action = agent.act(state)
#         env.render()
#         state, reward, done, _ = env.step(action)
#         if done:
#             break 

#env_info = env.reset(train_mode=False)[brain_name]

# initialize epsilon
# eps_start=1.0
# eps = eps_start

state = np.uint8(255 * np.array(env_info.vector_observations[0]))
score = 0
done = False

while done is False:
    action = agent.act(state)
    env_info = env.step(action)[brain_name]
    state = np.uint8(255 * np.array(env_info.vector_observations[0]))
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    score += reward

# scores_window.append(score)       # save most recent score
# scores.append(score)              # save most recent score
# eps = max(eps_end, eps_decay*eps) # decrease epsilon
# print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
# if i_episode % 100 == 0:
#     print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
# if np.mean(scores_window)>=13.0:
#     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
#     torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
#     break

print('\rScore: {:.2f}'.format(score))

env.close()
