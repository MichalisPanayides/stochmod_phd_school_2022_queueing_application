import copy
import itertools

import numpy as np


def create_state_space_for_two_servers(queue_capacity):
    """
    Create a state space for a given queue capacity.
    """
    return tuple(itertools.product(range(queue_capacity + 1), range(2), range(2)))


def calculate_costs_for_all_states(
    V2, state_space, queue_capacity, arrival_rate, service_rate_1, service_rate_2
):
    """
    Calculate the costs for all states.
    """
    V = np.zeros((queue_capacity + 1, 2, 2))
    gamma = arrival_rate + service_rate_1 + service_rate_2

    for s0, s1, s2 in state_space:
        V[s0, s1, s2] = (
            s0
            + s1
            + s2
            + (arrival_rate / gamma) * V2[min(queue_capacity, s0 + 1), s1, s2]
            + (service_rate_1 / gamma) * V2[s0, max(0, s1 - 1), s2]
            + (service_rate_2 / gamma) * V2[s0, s1, max(0, s2 - 1)]
        )
    return V


def update_temp_costs_for_state(state, V, V2):
    """
    Get the temporary costs for a given state.
    """
    s0, s1, s2 = state
    if s0 > 0 and s1 == 0 and s2 == 0:
        V2[s0, s1, s2] = min(
            V[s0, s1, s2], V[s0 - 1, s1 + 1, s2], V[s0 - 1, s1, s2 + 1]
        )
    if s0 > 0 and s1 == 0 and s2 == 1:
        V2[s0, s1, s2] = min(V[s0, s1, s2], V[s0 - 1, s1 + 1, s2])
    if s0 > 0 and s1 == 1 and s2 == 0:
        V2[s0, s1, s2] = min(V[s0, s1, s2], V[s0 - 1, s1, s2 + 1])
    return V2


def run_reinforcement_learning(
    arrival_rate,
    service_rate_1,
    service_rate_2,
    queue_capacity,
    iterations=100,
):
    """
    Run the reinforcement learning algorithm.
    """
    state_space = create_state_space_for_two_servers(queue_capacity=queue_capacity)
    costs_array = np.zeros((queue_capacity + 1, 2, 2))
    gamma = arrival_rate + service_rate_1 + service_rate_2

    for _ in range(iterations):

        temp_costs = costs_array.copy()
        for state in state_space:
            temp_costs = update_temp_costs_for_state(
                state=state, V=costs_array, V2=temp_costs
            )
        for state in state_space:
            costs_array = calculate_costs_for_all_states(
                V2=temp_costs,
                state_space=state_space,
                queue_capacity=queue_capacity,
                arrival_rate=arrival_rate,
                service_rate_1=service_rate_1,
                service_rate_2=service_rate_2,
            )
    return costs_array


def get_all_actions(V, queue_capacity):
    """
    Get all actions for a given state.
    """
    actions = {}
    for s0 in range(0, queue_capacity + 1):
        actions[(s0, 0, 0)] = np.argmin(([V[s0, 0, 0], V[s0 - 1, 1, 0], V[s0 - 1, 0, 1]]))
        actions[(s0, 0, 1)] = np.argmin(([V[s0, 0, 1], V[s0 - 1, 1, 1]]))
        actions[(s0, 1, 0)] = np.argmin(([V[s0, 1, 0], float("inf"), V[s0 - 1, 1, 1]]))
        actions[(s0, 1, 1)] = np.argmin(([V[s0, 1, 1]]))
    return actions