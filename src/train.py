import d3rlpy
from d3rlpy.datasets import get_d4rl
import gym
from d3rlpy.metrics.scorer import evaluate_on_environment
import argparse
import os


os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="task/game to be played")
parser.add_argument("--algo", type=str, help="algorithm to be used for training")
args = parser.parse_args()

task = args.task #['HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4']
algo = args.algo


class MODEL():
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
            self.f_params["actor_learning_rate"] = 3e-5
            self.engine = d3rlpy.algos.CQL(**self.f_params)
        elif self.algo == "MOPO":
            self.engine = d3rlpy.algos.MOPO(**self.f_params)     
        elif self.algo == "COMBO":
            self.engine = d3rlpy.algos.COMBO(**self.f_params)

    def train(self, n=100, n_steps=1000000, save_interval=100, save_metrics=False, verbose=False):
        dataset, env = get_d4rl(self.task)
        online_env = gym.make(self.task)
        for i in range(n):
            d3rlpy.seed(i)
            env.seed(i)
            online_env.seed(i)
            self.set_engine()

            self.engine.fit(dataset, n_steps=n_steps, save_interval=save_interval, save_metrics=save_metrics, verbose=verbose)
            self.engine.save_model("./saved_models/iql_{}_{}_{}.pt".format(algo, task, i))
            scorer = evaluate_on_environment(online_env, n_trials=100)
            self.mean_results.append(scorer(self.engine))
        return self.mean_results
    

model = MODEL(task, algo)
mean_results = model.train()

