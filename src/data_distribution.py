# import d3rlpy
from d3rlpy.datasets import get_d4rl
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

task = "ant-medium-expert-v2"

l = []
store = []
dataset, env = get_d4rl(task)

returns = []
for episode in dataset.episodes:
    r = 0
    for transition in episode:
        r += transition.reward
    returns.append(r)
normalized_score = env.get_normalized_score(np.array(returns)) 
l.append(normalized_score)

n = 1
m = 1
plt.hist(l, bins=50)  # Adjust the number of bins as needed
plt.title(f'{task}')
plt.xlabel('Returns')
plt.ylabel('Frequency')

# Adjust the spacing between subplots
plt.tight_layout()
plt.savefig(f'../data_distribution/{task}.png')
# Show the plot
plt.show()
