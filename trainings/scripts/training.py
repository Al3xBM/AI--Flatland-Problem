import random
import sys
import time
from collections import deque
from pathlib import Path

print (sorted([2,10,15,3,7,9,11], key = lambda i: i%3,reverse=True))

from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters

from trainings.scripts.process_observation import normalize_observation

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

import numpy as np
from trainings.scripts.agent import Agent

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

nr_iterations = 150

random.seed(1)
np.random.seed(1)

#Dimensiuni environment
x_dim = 35
y_dim = 35
nr_agents = 3

stochastic_data = MalfunctionParameters(malfunction_rate=1./10000,  # Rate of malfunction occurence
                                            min_duration=15,  # Minimal duration of malfunction
                                            max_duration=50  # Max duration of malfunction
                                            )

TreeObservation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(30))

speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

env = RailEnv(width=x_dim,
                  height=y_dim,
                  rail_generator=sparse_rail_generator(max_num_cities=3,
                                                       # Number of cities in map (where train stations are)
                                                       seed=1,  # Random seed
                                                       grid_mode=False,
                                                       max_rails_between_cities=2,
                                                       max_rails_in_city=3),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=nr_agents,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  obs_builder_object=TreeObservation)

env.reset(True,True)
env_renderer = RenderTool(env,screen_height=1080,screen_width=1920)

num_features_per_node = env.obs_builder.observation_dim
tree_depth = 2
nr_nodes = 0 #numarul de noduri dintr-un arbore de inaltime 2
for i in range(tree_depth + 1):
    nr_nodes += np.power(4, i)
state_size = num_features_per_node * nr_nodes

nr_actions = 5

max_steps = int(4 * 2 * (20 + env.height + env.width))

epsilon = 0.5
epsilon_end = 0.005
opsilon_decay = 0.998

action_dict = dict()
final_action_dict = dict()
scores_window = deque(maxlen=100)
done_window = deque(maxlen=100)
scores = []
dones_list = []
action_prob = [0] * nr_actions
agent_obs = [None] * env.get_num_agents()
agent_next_obs = [None] * env.get_num_agents()
agent_obs_buffer = [None] * env.get_num_agents()
agent_action_buffer = [2] * env.get_num_agents()
cummulated_reward = np.zeros(env.get_num_agents())
update_values = [False] * env.get_num_agents()

agent = Agent(state_size, nr_actions)
agent.load()

for iteration in range(1,nr_iterations + 1):
    observation, info = env.reset(True, True)
    env_renderer.reset()

    for a in range(env.get_num_agents()):
        if observation[a]:
            agent_obs[a] = normalize_observation(observation[a], tree_depth, observation_radius = 10)

    while True:
        for a in range(env.get_num_agents()):
            if info['action_required'][a]:
                # If an action is require, we want to store the obs a that step as well as the action
                update_values[a] = True
                action = agent.give_action(agent_obs[a], epsilon)
                action_prob[action] += 1
            else:
                update_values[a] = False
                action = 0
            action_dict.update({a: action})

        next_obs, all_rewards, done, info = env.step(action_dict)
        env_renderer.render_env(show=True, frames=False, show_observations=True)

        for a in range(env.get_num_agents()):
            if next_obs[a]:
                agent_obs[a] = normalize_observation(next_obs[a], tree_depth, 10)
                print("AGENT: ",a)

        time.sleep(2)

        if done['__all__']:
            env_done = 1
            break
