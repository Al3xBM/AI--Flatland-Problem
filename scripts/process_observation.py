import numpy as np
from flatland.envs.observations import TreeObsForRailEnv


def norm_obs_clip(obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
    if fixed_radius > 0:
        max_obs = fixed_radius
    else:
        max_obs = max(1, max([i for i in obs if 1000 > i >= 0] or list([0]))) + 1

    min_obs = 0  # min(max_obs, min_gt(obs, 0))
    if normalize_to_range:
        min_obs = min([i for i in obs if i >= 0] or list([0]))
    if min_obs > max_obs:
        min_obs = max_obs
    if max_obs == min_obs:
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
    norm = np.abs(max_obs - min_obs)
    return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)


def split_into_agent_cases(node):
    # Takes the values from root
    data = np.zeros(6)
    distance = np.zeros(1)
    agent_data = np.zeros(4)

    data[0] = node.dist_own_target_encountered
    data[1] = node.dist_other_target_encountered
    data[2] = node.dist_other_agent_encountered
    data[3] = node.dist_potential_conflict
    data[4] = node.dist_unusable_switch
    data[5] = node.dist_to_next_branch

    distance[0] = node.dist_min_to_target

    agent_data[0] = node.num_agents_same_direction
    agent_data[1] = node.num_agents_opposite_direction
    agent_data[2] = node.num_agents_malfunctioning
    agent_data[3] = node.speed_min_fractional

    return data, distance, agent_data


def r_split_subtree_into_agent_cases(node, current_tree_depth, max_tree_depth):
    # Recursive function that takes the possible values for each action from every node

    if node == -np.inf:
        remaining_depth = max_tree_depth - current_tree_depth
        num_remaining_nodes = int((4**(remaining_depth+1) - 1) / (4 - 1))
        return [-np.inf] * num_remaining_nodes*6, [-np.inf] * num_remaining_nodes, [-np.inf] * num_remaining_nodes*4

    data, distance, agent_data = split_into_agent_cases(node)

    if not node.childs:
        return data, distance, agent_data

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = r_split_subtree_into_agent_cases(node.childs[direction], current_tree_depth + 1, max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def split_tree_into_feature_groups(tree, max_tree_depth):
    data, distance, agent_data = split_into_agent_cases(tree)

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = r_split_subtree_into_agent_cases(tree.childs[direction], 1, max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def normalize_observation(observation, tree_depth, observation_radius):
    #Functia separa arborele de observatie in valorile de pe fiecare nod
    data, distance, agent_data = split_tree_into_feature_groups(observation, tree_depth)

    data = norm_obs_clip(data, fixed_radius=observation_radius)
    distance = norm_obs_clip(distance, normalize_to_range=True)
    agent_data = np.clip(agent_data, -1, 1)
    normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data))
    return normalized_obs