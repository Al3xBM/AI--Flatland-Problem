import keras
import flatland
import flatland.core.transitions
import numpy as np
import time
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator, sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool

width = 50  # With of map
height = 50  # Height of map
nr_trains = 2  # Number of trains that have an assigned task in the env
cities_in_map = 20  # Number of cities where agents can start or end
seed = 14  # Random seed
grid_distribution_of_cities = False  # Type of city distribution, if False cities are randomly placed
max_rails_between_cities = 1  # Max number of tracks allowed between cities. This is number of entry point to a city
max_rail_in_cities = 1  # Max number of parallel tracks within a city, representing a realistic trainstation

rail_generator = sparse_rail_generator(max_num_cities=cities_in_map,
                                       seed=seed,
                                       grid_mode=grid_distribution_of_cities,
                                       max_rails_between_cities=max_rails_between_cities,
                                       max_rails_in_city=max_rail_in_cities,
                                       )

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

schedule_generator = sparse_schedule_generator(speed_ration_map)

stochastic_data = {'prop_malfunction': 0.3,  # Percentage of defective agents
                   'malfunction_rate': 30,  # Rate of malfunction occurence
                   'min_duration': 3,  # Minimal duration of malfunction
                   'max_duration': 20  # Max duration of malfunction
                   }

observation_builder = GlobalObsForRailEnv()

# Relative weights of each cell type to be used by the random rail generators.
transition_probability = [1.0,  # empty cell - Case 0
                          1.0,  # Case 1 - straight
                          1.0,  # Case 2 - simple switch
                          0.3,  # Case 3 - diamond drossing
                          0.5,  # Case 4 - single slip
                          0.5,  # Case 5 - double slip
                          0.2,  # Case 6 - symmetrical
                          0.0,  # Case 7 - dead end
                          0.2,  # Case 8 - turn left
                          0.2,  # Case 9 - turn right
                          1.0]  # Case 10 - mirrored switch

env = RailEnv(width=width,
              height=height,
              rail_generator=rail_generator,
              schedule_generator=schedule_generator,
              number_of_agents=nr_trains,
              obs_builder_object=observation_builder,
              remove_agents_at_target=True  # Removes agents at the end of their journey to make space for others
              )

env.reset()

env_renderer = RenderTool(env,screen_height=1080,  # Adjust these parameters to fit your resolution
                          screen_width=1920)

def my_controller():
    """
    You are supposed to write this controller
    DO_NOTHING = 0  # implies change of direction in a dead-end!
    MOVE_LEFT = 1
    MOVE_FORWARD = 2
    MOVE_RIGHT = 3
    STOP_MOVING = 4

    """
    _action = {}
    for _idx in range(nr_trains):
        _action[_idx] = np.random.randint(0, 5)
    return _action

for step in range(500):

    _action = my_controller()
    obs, all_rewards, done, info = env.step(_action)
    if step == 6:
        print(obs[0][1])
    print("Rewards: {}, [done={}]".format( all_rewards, done))
    env_renderer.render_env(show=True, frames=False, show_observations=False)
    time.sleep(0.02)

# retea neuronala exemplu
# input_shape = keras.layers.Input(shape=(80, 80))
#
# flat_layer = keras.layers.Flatten()(input_shape)
# full_connect_1 = keras.layers.Dense(units=400, activation='relu', use_bias=True, )(flat_layer)
# full_connect_2 = keras.layers.Dense(units=100, activation='relu', use_bias=True, )(flat_layer)
# softmax_output = keras.layers.Dense(3, activation='softmax', use_bias=False)(full_connect_1)
# base_model = keras.models.Model(inputs=input_shape, outputs=softmax_output)
# #base_model = keras.models.load_model("modelwith3outputs")
# base_model.summary()
#
# #reward = keras.layers.Input(shape=(3,), name='reward')
# train_model = keras.models.Model(inputs=input_shape, outputs=softmax_output)
#
# optimizer_adam = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1.0)
