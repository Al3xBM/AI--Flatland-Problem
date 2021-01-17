import random
import sys
from collections import deque
from pathlib import Path
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
import matplotlib.pyplot as plt
import numpy as np
import torch
from agent import Agent
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from process_observation import normalize_observation
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.agent_utils import RailAgentStatus

import backup
from importlib_resources import path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))


def main():
    # ============ DECLARING VARIABLES ============
    iterations = 20000
    # Parameters for the Environment
    x_dim = 35
    y_dim = 35
    agents_nr = 3
    # 4 different types of trains with equal prob to be chosen
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    # Define training parameters
    epsilon = 0.5
    epsilon_end = 0.005
    epsilon_update = 0.998

    # there are 5 actions for trains in flatland env
    # 0 = DO NOTHING / if the train is moving, it keeps on moving
    #  if it stopped, it will stay stopped
    # 1 = LEFT
    # 2 = FORWARD
    # 3 = RIGHT
    # 4 = STOP
    action_size = 5

    # use this to simulate trains breaking down
    malfunction_rate = MalfunctionParameters(malfunction_rate=1. / 1000,  # Rate of malfunction occurence
                                             min_duration=15,  # Minimal duration of malfunction
                                             max_duration=30  # Max duration of malfunction
                                             )

    # Custom observation builder
    TreeObservation = TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(30))

    # creating envoirenment
    env = RailEnv(width=x_dim,
                  height=y_dim,
                  rail_generator=sparse_rail_generator(max_num_cities=random.randint(2, agents_nr),
                                                       # Number of cities in map (where train stations are)
                                                       seed=random.randint(1, 1000),  # Random seed
                                                       grid_mode=False,
                                                       max_rails_between_cities=2,
                                                       max_rails_in_city=3),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=agents_nr,
                  malfunction_generator_and_process_data=malfunction_from_params(malfunction_rate),
                  obs_builder_object=TreeObservation)
    # reset env and create the renderer
    env.reset(True, True)
    env_renderer = RenderTool(env, screen_height=1080,
                              screen_width=1920)
    # calculate state_size
    # this will be given as a parameter to the network
    num_features_per_node = env.obs_builder.observation_dim
    tree_depth = 3
    nr_nodes = 0
    for i in range(tree_depth + 1):
        nr_nodes += np.power(4, i)
    state_size = num_features_per_node * nr_nodes

    # max number of steps per episode
    # this has no use if we use the **while True** loop instead
    max_steps = int(4 * 2 * (20 + env.height + env.width))  # 4  * 2 * (20 + env.height + env.width) initially

    action_dict = dict()
    final_action_dict = dict()
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    scores = []
    dones_list = []
    action_prob = [0] * action_size

    # env dependent variables
    agent_obs = [None] * env.get_num_agents()
    agent_next_obs = [None] * env.get_num_agents()
    agent_obs_buffer = [None] * env.get_num_agents()
    agent_action_buffer = [2] * env.get_num_agents()
    cummulated_reward = np.zeros(env.get_num_agents())
    update_values = [False] * env.get_num_agents()

    # initialize agent
    agent = Agent(state_size, action_size)

    # try loading previous state
    # agent.load()
    # with path(backup, "iteration_1360_average_-0.147.pth") as file_in:
    #     agent.q_network.load_state_dict(torch.load(file_in))

    # big loop
    for iteration in range(1, iterations + 1):
        # reset environment and renderer
        obs, info = env.reset(True, True)
        env_renderer.reset()

        # get obs for each agent
        for a in range(env.get_num_agents()):
            if obs[a]:
                agent_obs[a] = normalize_observation(obs[a], tree_depth, observation_radius=100)
                agent_obs_buffer[a] = agent_obs[a].copy()

        # Reset score and done
        score = 0

        for step in range(max_steps):  # while True:
            # take action
            for a in range(env.get_num_agents()):
                # only take action if it is required
                if info['action_required'][a]:
                    update_values[a] = True
                    action = agent.give_action(agent_obs[a], epsilon=epsilon)
                    action_prob[action] += 1
                # if no action is required, do nothing
                else:
                    update_values[a] = False
                    action = 0

                action_dict.update({a: action})

            # env_renderer.render_env(show=True, show_predictions=True, show_observations=False)

            # env step
            # update info and others for next iteration
            next_obs, all_rewards, done, info = env.step(action_dict)

            # update buffer and agents
            for a in range(env.get_num_agents()):
                # Only update the values when we are done
                # or when an action was taken and thus relevant information is present
                if update_values[a] or done[a]:
                    agent.take_step(agent_obs_buffer[a], agent_action_buffer[a], all_rewards[a],
                               agent_obs[a], done[a])
                    cummulated_reward[a] = 0.

                    agent_obs_buffer[a] = agent_obs[a].copy()
                    agent_action_buffer[a] = action_dict[a]
                if next_obs[a]:
                    agent_obs[a] = normalize_observation(next_obs[a], tree_depth, observation_radius=100)

                score += all_rewards[a] / env.get_num_agents()

            # Copy observation
            if done['__all__']:
                break

        # update epsilon value
        epsilon = max(epsilon_end, epsilon_update * epsilon)

        # Collection information about training
        tasks_finished = 0
        for current_agent in env.agents:
            if current_agent.status == RailAgentStatus.DONE_REMOVED:
                tasks_finished += 1
        done_window.append(tasks_finished / max(1, env.get_num_agents()))
        scores_window.append(score / max_steps)  # save most recent score
        scores.append(np.mean(scores_window))
        dones_list.append((np.mean(done_window)))

        print(
            '\rTraining {} Agents on ({},{}).\t Iteration {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                env.get_num_agents(), x_dim, y_dim,
                iteration,
                np.mean(scores_window),
                100 * np.mean(done_window),
                epsilon, action_prob / np.sum(action_prob)), end=" ")

        if iteration % 20 == 0:
            print(
                '\rTraining {} Agents on ({},{}).\t Iteration {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                    env.get_num_agents(), x_dim, y_dim,
                    iteration,
                    np.mean(scores_window),
                    100 * np.mean(done_window),
                    epsilon, action_prob / np.sum(action_prob)))
            torch.save(agent.q_network.state_dict(),
                       '../backup/batch_3_iteration_{}_average_{:.3f}.pth'.format(iteration, np.mean(scores_window)))
            action_prob = [1] * action_size

        agent.save()

    # Plot overall training progress at the end
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main()
