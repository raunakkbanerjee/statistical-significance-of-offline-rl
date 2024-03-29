'''This file trains using MOPO and COMBO algorithm'''

#This training script generates multiple trajectories and notes the score. So, we get 100 scores instead of noting just the mean score.

import d3rlpy
from d3rlpy.datasets import get_d4rl
import gym
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.dynamics import ProbabilisticEnsembleDynamics
from sklearn.model_selection import train_test_split
import argparse
import os
from tqdm import tqdm
import random
#import wandb

#wandb.login()


os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument("--arg", type=str, help="argument")
args = parser.parse_args()

task, algo = args.arg.split() #['HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4']
#algo = args.algo

#wandb.init(
#    project=f"{algo}_{task}"
#    config={
#    "n": 100,
#    "n_trials": 100
#    })

class Model():
    def __init__(self, task, algo, gpu=True):
        self.mean_results = []
        self.task = task 
        self.algo = algo
        self.f_params = {"use_gpu": gpu}
        self.engine = None

    def set_engine(self, dataset):
        if self.algo == "IQL":
            #if "cloned" in self.task: 
            #    self.f_params["weight_temp"] = 0.5
            #else:
            #    self.f_params["expectile"] = 0.9
            #    self.f_params["weight_temp"] = 10
 
            self.engine = d3rlpy.algos.IQL(**self.f_params)
        elif self.algo == "CQL":
            # self.f_params["actor_learning_rate"] = 3e-5
            self.f_params["alpha_threshold"] = -1 
            self.engine = d3rlpy.algos.CQL(**self.f_params)
        elif self.algo == "MOPO":
            self.engine = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=1e-4, use_gpu=True)
            self.train_episodes, self.test_episodes = train_test_split(dataset)
            # self.engine = d3rlpy.algos.MOPO(**self.f_params)     
        elif self.algo == "COMBO":
            self.engine = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=1e-4, use_gpu=True)
            self.train_episodes, self.test_episodes = train_test_split(dataset)
            # self.engine = d3rlpy.algos.COMBO(**self.f_params)

    def train(self, n=20, n_steps=1000000, save_interval=101, save_metrics=False, verbose=False):
        dataset, env = get_d4rl(self.task)
        online_env = gym.make(self.task)
        for i in range(n):
            d3rlpy.seed(i+70002035)
            random.seed(i+80002035)
            self.set_engine(dataset)               
            self.engine.fit(dataset, n_steps=n_steps, save_interval=save_interval, save_metrics=save_metrics, verbose=verbose)
            scorer = evaluate_on_environment(online_env, n_trials=1)
            f = open(f'./txt_files_4/{algo}_{task}_rollout.txt', 'a+')
            f.write(f"n={i}\n")
            #self.engine.save_model("./saved_models/{}_{}_{}.pt".format(algo, task, i))
            for i in range(1000):       
                normalized_score = online_env.get_normalized_score(scorer(self.engine))
                f = open(f'./txt_files_4/{algo}_{task}_rollout.txt', 'a+')
                f.write(f"{normalized_score}\n")
                f.close()
            self.mean_results.append(normalized_score)
        return self.mean_results
    

model = Model(task, algo)
mean_results = model.train(n=50)

#with open(f'{algo}_{task}.txt', 'w') as f:
#    for r in mean_results:
#        f.write(f"{r}\n")
