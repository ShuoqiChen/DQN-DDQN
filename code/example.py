#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import gym
import deeprl_hw2q2.lake_envs as lake_env
import time

from deeprl_hw2q2 import rl
import numpy as np
import timeit



def run_random_policy(env):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            env.action_space.sample())
        env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(1)

    return total_reward, num_steps


def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nS, env.nA))


def print_model_info(env, state, action):
    transition_table_row = env.P[state][action]
    print(
        ('According to transition function, '
         'taking action %s(%d) in state %d leads to'
         ' %d possible outcomes') % (lake_env.action_names[action],
                                     action, state, len(transition_table_row)))
    for prob, nextstate, reward, is_terminal in transition_table_row:
        state_type = 'terminal' if is_terminal else 'non-terminal'
        print(
            '\tTransitioning to %s state %d with probability %f and reward %f'
            % (state_type, nextstate, prob, reward))


def main():
    # create the environment
    # env_name = 'Deterministic-4x4-FrozenLake-v0'
    env_name = 'Deterministic-8x8-FrozenLake-v0'
    env = gym.make(env_name)

    print_env_info(env)
    print_model_info(env, 0, lake_env.DOWN)
    print_model_info(env, 1, lake_env.DOWN)
    print_model_info(env, 14, lake_env.RIGHT)

    # input('Hit enter to run a random policy...')
    #
    # total_reward, num_steps = run_random_policy(env)
    # print('Agent received total reward of: %f' % total_reward)
    # print('Agent took %d steps' % num_steps)

    gamma = 0.9

    # # print('\n ================ running policy_iteration_sync ====================== \n ')
    # start_pi = timeit.default_timer()
    # policy_new, value_func, num_policy_iter, total_num_policy_eval = rl.policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3)
    # # policy_new, value_func, num_policy_iter, total_num_policy_eval = rl.policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3)
    # # policy_new, value_func, num_policy_iter, total_num_policy_eval = rl.policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3), tol=1e-3)
    # stop_pi = timeit.default_timer()
    # print('Time of policy iteration: ', stop_pi - start_pi)
    # print("number of policy iter\n", num_policy_iter)
    # print("total number of policy evaluation is \n", total_num_policy_eval)
    # print('The optimal policy in letters are ')
    # rl.display_policy_letters(env, policy_new)
    # rl.value_func_heatmap(env, value_func)

    print('\n ================ running value_iteration_sync ====================== \n ')
    start_vi = timeit.default_timer()
    value_func, n_value_iter = rl.value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3)
    # value_func, n_value_iter = rl.value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3)
    # value_func, n_value_iter = rl.value_iteration_async_randperm(env, gamma, max_iterations=int(1e3), tol=1e-3)
    # value_func, n_value_iter = rl.value_iteration_async_custom(env, env_name, gamma, max_iterations=int(1e3), tol=1e-3)
    stop_vi = timeit.default_timer()
    print('Time of value iteration: ', stop_vi - start_vi)
    print("number of value iter\n", n_value_iter)
    optimal_policy = rl.value_function_to_policy(env, gamma, value_func)
    rl.display_policy_letters(env, optimal_policy)
    rl.value_func_heatmap(env, value_func)

if __name__ == '__main__':
    main()
