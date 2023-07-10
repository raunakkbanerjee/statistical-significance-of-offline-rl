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
#import wandb

#wandb.login()


os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="task/game to be played")
parser.add_argument("--algo", type=str, help="algorithm to be used for training")
args = parser.parse_args()

task = args.task #['HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4']
algo = args.algo

"""
suggested hypers for combo

halfcheetah-medium-v2: rollout-length=5, cql-weight=0.5
hopper-medium-v2: rollout-length=5, cql-weight=5.0
walker2d-medium-v2: rollout-length=1, cql-weight=5.0
halfcheetah-medium-replay-v2: rollout-length=5, cql-weight=0.5
hopper-medium-replay-v2: rollout-length=5, cql-weight=0.5
walker2d-medium-replay-v2: rollout-length=1, cql-weight=0.5
halfcheetah-medium-expert-v2: rollout-length=5, cql-weight=5.0
hopper-medium-expert-v2: rollout-length=5, cql-weight=5.0
walker2d-medium-expert-v2: rollout-length=1, cql-weight=5.0

suggested hypers for mopo

halfcheetah-medium-v2: rollout-length=5, penalty-coef=0.5
hopper-medium-v2: rollout-length=5, penalty-coef=5.0
walker2d-medium-v2: rollout-length=5, penalty-coef=0.5
halfcheetah-medium-replay-v2: rollout-length=5, penalty-coef=0.5
hopper-medium-replay-v2: rollout-length=5, penalty-coef=2.5
walker2d-medium-replay-v2: rollout-length=1, penalty-coef=2.5
halfcheetah-medium-expert-v2: rollout-length=5, penalty-coef=2.5
hopper-medium-expert-v2: rollout-length=5, penalty-coef=5.0
walker2d-medium-expert-v2: rollout-length=1, penalty-coef=2.5
"""

class Model():
    def __init__(self, task, algo, gpu=True):
        self.task = task 
        self.algo = algo
        self.dynamics = None
        self.f_params = {}
        self.experiment_name = f"{self.algo}_{self.task}"

    def set_engine(self):
        if self.algo == "MOPO":
            if self.algo == "halfcheetah-medium-v2" or self.algo == "halfcheetah-medium-replay-v2" or self.algo == "walker2d-medium-v2":
                self.f_params = {'rollout_length': 5, 'penalty_coef': 0.5}
            elif self.algo == "hopper-medium-v2" or self.algo == "hopper-medium-expert-v2":
                self.f_params = {'rollout_length': 5, 'penalty_coef': 5.0}
            elif self.algo == "hopper-medium-replay-v2" or self.algo == "halfcheetah-medium-expert-v2":
                self.f_params = {'rollout_length': 5, 'penalty_coef': 2.5}
            elif self.algo == "walker2d-medium-replay-v2" or self.algo == "walker2d-medium-expert-v2":
                self.f_params = {'rollout_length': 1, 'penalty_coef': 2.5}            

            self.engine = d3rlpy.algos.MOPO(self.dynamics, **self.f_params)     

        elif self.algo == "COMBO":
            if self.algo == "halfcheetah-medium-v2" or self.algo == "halfcheetah-medium-replay-v2" or self.algo == "hopper-medium-replay-v2":
                self.f_params = {'rollout_length': 5, 'cql_weight': 0.5}
            elif self.algo == "hopper-medium-v2" or self.algo == "hopper-medium-expert-v2" or self.algo == "halfcheetah-medium-expert-v2":
                self.f_params = {'rollout_length': 5, 'cql_weight': 5.0}
            elif self.algo == "walker2d-medium-v2" or self.algo == "walker2d-medium-expert-v2":
                self.f_params = {'rollout_length': 1, 'cql_weight': 5.0}
            elif self.algo == "walker2d-medium-replay-v2":
                self.f_params = {'rollout_length': 1, 'cql_weight': 0.5}

            self.engine = d3rlpy.algos.COMBO(self.dynamics, **self.f_params)
        

    def train_dynamics(self, n_epochs=100, save_interval=100, save_metrics=True, verbose=False):
        self.dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=1e-4, use_gpu=True)
        train_episodes, test_episodes = train_test_split(self.dataset)
        self.dynamics.fit(train_episodes, eval_episodes=test_episodes, n_epochs=n_epochs, 
        save_interval=save_interval, save_metrics=save_metrics, verbose=verbose, with_timestamp=False, experiment_name=self.experiment_name)         
            

    def train_engine(self, n_steps=1000000, save_interval=101, save_metrics=False, verbose=False):
        self.set_engine()
        self.engine.fit(self.dataset, n_steps=n_steps, save_interval=save_interval, save_metrics=save_metrics, verbose=verbose)
    
    def train(self, n=50, n_epochs=100, n_steps=1000000, save_engine_interval=101, save_dynamics_interval=100,
               save_dynamics_metrics=True, save_engine_metrics=True, verbose=False):
        for i in range(n):
            self.dataset, self.env = get_d4rl(self.task)
            self.online_env = gym.make(self.task)
            d3rlpy.seed(i)
            self.env.reset(seed=i)
            self.online_env.reset(seed=i)

            self.train_dynamics(n_epochs=n_epochs, save_interval=save_dynamics_interval, save_metrics=save_dynamics_metrics, verbose=verbose)
            # load trained dynamics model
            json_path = f'./d3rlpy_logs/{self.experiment_name}/params.json'
            model_path = f'./d3rlpy_logs/{self.experiment_name}/model_7464.pt'
            self.dynamics = ProbabilisticEnsembleDynamics.from_json(json_path)
            self.dynamics.load_model(model_path)
            self.train_engine(n_steps=n_steps, save_interval=save_engine_interval, save_metrics=save_engine_metrics, verbose=verbose)
            scorer = evaluate_on_environment(self.online_env, n_trials=1)
            f = open(f'./txt_files/{algo}_{task}_rollout.txt', 'a+')
            f.write(f"n={i}\n")

            for i in range(1000):       
                normalized_score = self.online_env.get_normalized_score(scorer(self.engine))
                f = open(f'{algo}_{task}_rollout.txt', 'a+')
                f.write(f"{normalized_score}\n")

    

model = Model(task, algo)
model.train(n=50)

