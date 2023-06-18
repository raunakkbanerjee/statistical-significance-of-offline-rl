import d3rlpy
from d3rlpy.datasets import get_d4rl
import gym
from tqdm import tqdm
from d3rlpy.metrics.scorer import evaluate_on_environment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="name of the game to be played")
args = parser.parse_args()

algos = ['halfcheetah-medium-v2', 'walker2d-medium-v2']#, 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4']

N = 1
for j, algo in enumerate(algos):
    print(algo)
    dataset, env = get_d4rl(algo)
    online_env = gym.make(algo)

    mean_results = []
    for i in tqdm(range(N)):
        d3rlpy.seed(i)
        env.seed(i)
        online_env.seed(i)
        iql = d3rlpy.algos.IQL(use_gpu=True)
        iql.fit(dataset, n_epochs=1, n_steps_per_epoch=1000, save_interval=10, save_metrics=False)
        iql.save_model("./saved_models/iql_{}_{}.pt".format(algo, i))
        scorer = evaluate_on_environment(online_env, n_trials=100)
        mean_results.append(scorer(iql))
        
    with open('results.txt', 'w') as f:
        f.write(f"{algo}\n")
        for result in mean_results:
            f.write(f"{result}\n")

