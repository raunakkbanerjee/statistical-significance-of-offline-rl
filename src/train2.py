#This training script generates multiple trajectories and notes the score. So, we get 100 scores instead of noting just the mean score.

import d3rlpy
from d3rlpy.datasets import get_d4rl
import gym
from d3rlpy.metrics.scorer import evaluate_on_environment
import argparse
import os
from tqdm import tqdm
#import wandb

#wandb.login()


os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="task/game to be played")
parser.add_argument("--algo", type=str, help="algorithm to be used for training")
args = parser.parse_args()

task = args.task #['HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4']
algo = args.algo

#wandb.init(
#    project=f"{algo}_{task}"
#    config={
#    "n": 100,
#    "n_trials": 100
#    })

with open(f'sanity_check.txt', 'w') as f:
    f.write(f"{algo}_{task}\n")

class Model():
    def __init__(self, task, algo, gpu=True):
        self.mean_results = []
        self.task = task 
        self.algo = algo
        self.f_params = {"use_gpu": gpu}
        self.engine = None

    def set_engine(self):
        if self.algo == "IQL":
            self.engine = d3rlpy.algos.IQL(**self.f_params)
        elif self.algo == "CQL":
            # self.f_params["actor_learning_rate"] = 3e-5
            self.engine = d3rlpy.algos.CQL(**self.f_params)
        elif self.algo == "MOPO":
            self.engine = d3rlpy.algos.MOPO(**self.f_params)     
        elif self.algo == "COMBO":
            self.engine = d3rlpy.algos.COMBO(**self.f_params)

    def train(self, n=100, n_steps=1000000, save_interval=101, save_metrics=False, verbose=False):
        dataset, env = get_d4rl(self.task)
        online_env = gym.make(self.task)
        for i in range(n):
            d3rlpy.seed(i)
            env.reset(seed=i)
            online_env.reset(seed=i)
            self.set_engine()

            self.engine.fit(dataset, n_steps=n_steps, save_interval=save_interval, save_metrics=save_metrics, verbose=verbose)
            scorer = evaluate_on_environment(online_env, n_trials=1)
            f = open(f'{algo}_{task}_rollout.txt', 'a+')
            f.write(f"n={n}\n")
            #self.engine.save_model("./saved_models/{}_{}_{}.pt".format(algo, task, i))
            for i in range(1000):       
                normalized_score = online_env.get_normalized_score(scorer(self.engine))
                f = open(f'{algo}_{task}_rollout.txt', 'a+')
                f.write(f"{normalized_score}\n")
            self.mean_results.append(normalized_score)
        return self.mean_results
    

model = Model(task, algo)
mean_results = model.train(n=10)

#with open(f'{algo}_{task}.txt', 'w') as f:
#    for r in mean_results:
#        f.write(f"{r}\n")
