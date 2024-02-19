#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:03:05 2024

@author: sahil
"""

''' Import Libraries '''
import numpy as np
import pandas as pd
import time
import os
import glob
from numba import jit






# Define the action space using numpy
action_space = np.arange(70, 106.5, 2.5).tolist()

# Create state space lists using list comprehensions
space_list_290 = [[290, i, j, 290 - (i + j)] for i in action_space for j in action_space]
space_list_310 = [[310, i, j, 310 - (i + j)] for i in action_space for j in action_space]

# Combine the two state space lists
state_space = sorted(space_list_290 + space_list_310)

del space_list_290, space_list_310

# Define the demand shocks
demand_shocks = [290, 310]

'''Q-matrix'''
Q = np.zeros((len(state_space), len(action_space)))

for i in action_space:
    for j in action_space:
        action_space_1_index = action_space.index(i)
        action_space_2_index = action_space.index(j)
        state_space_1 = 290 - (i + j)
        state_space_2 = 310 - (i + j)
        state_space_1_index = state_space.index([290,i,j,state_space_1])
        state_space_2_index = state_space.index([310,i,j,state_space_2])
        profit_1 = i * state_space_1
        profit_2 = j * state_space_2
        
        Q[state_space_1_index][action_space_1_index] = profit_1
        Q[state_space_2_index][action_space_2_index] = profit_2

    

Q1 = Q / ((len(action_space) * (1 - 0.95)))
Q2 = Q / ((len(action_space) * (1 - 0.95)))

del action_space_1_index, action_space_2_index, i, j, profit_1, state_space_1, state_space_1_index, profit_2, state_space_2, state_space_2_index



results = {'actions_1':[],
           'actions_2':[],
           'shock':[],
           #'qval_1':[],
           #'qval_2':[],
           'action_space_2': [],
           'exploration_rate':[]
           }

def qLearning(num_episodes, steps, exploration_break = None, shocks = True, 
              exploration_rate = 1, discount_factor = 0.95,
                            alpha = 0.15):

    #Parameters
    if shocks == True:
        d_t = [290, 310]
    else:
        d_t = [300]
        
        
       
    # For every episode
    for ith_episode in range(num_episodes):
        
        # Reset the environment and pick the first action
        state_index = np.random.choice(len(state_space))
        state = state_space[state_index]
        
        actions_1 = []
        actions_2 = []
        
        for t in range(steps):
            shock = np.random.choice(d_t)
            results['shock'].append(shock)
            # get probabilities of all actions from current state
            #Exploration or Exploitation for Agent 1 at time t
            exploration_rate_threshold = np.random.uniform(0, 1)

            if exploration_rate_threshold < exploration_rate:
                action_1 = np.random.choice(action_space)
                action_1_index = action_space.index(action_1)
            else:
                action_1_index = np.argmax(Q1[state_index])
                action_1 = action_space[action_1_index]
            
            exploration_rate_threshold = np.random.uniform(0, 1)
            
            if exploration_rate_threshold < exploration_rate:
                action_2 = np.random.choice(action_space)
                action_2_index = action_space.index(action_2)
            else:
                action_2_index = np.argmax(Q2[state_index])
                action_2 = action_space[action_2_index]
            
            actions_1.append(action_1)
            actions_2.append(action_2)




            '''Calculate New State and Rewards'''
            state_calc = shock - (action_1 + action_2)
            reward_1 = action_1 * state_calc
            reward_2 = action_2 * state_calc
            
            next_state = [shock, action_1, action_2, state_calc]
            next_state_index = state_space.index(next_state)
            
            state_index = state_space.index(state)

            '''Update Q matrix'''
            td_target1 = reward_1 + discount_factor * np.max(Q1[next_state_index])
            td_target2 = reward_2 + discount_factor * np.max(Q2[next_state_index])
            Q1[state_index][action_1_index] = alpha * td_target1 + (1 - alpha)* Q1[state_index][action_1_index]
            Q2[state_index][action_2_index] = alpha * td_target2 + (1 - alpha)* Q2[state_index][action_2_index]
            
            
            state = next_state
            state_index = next_state_index
            '''Update Results Dictionary'''
            #results['qval_1'].append(alpha * td_target1 + (1 - alpha)* Q1[state_index][action_1_index])
            #results['qval_2'].append(alpha * td_target2 + (1 - alpha)* Q2[state_index][action_2_index])
        
        results['actions_1'].append(np.average(actions_1))
        results['actions_2'].append(np.average(actions_2))
        print(f'Episode Number: {ith_episode}. Exploration Rate: {exploration_rate}')
        
        '''Update exploration rate'''
        
        if ith_episode < exploration_break:
            decay = 1 / (num_episodes - num_episodes / 2)
            exploration_rate = max(1 - decay * ith_episode, 0)
        else:
            exploration_rate = 0
        

    
        
    return results



def run_qLearning(process_id, num_episodes, steps, exploration_break=None, shared_list = [], shocks=True,
                 exploration_rate=1, discount_factor=0.95, alpha=0.15):
    np.random.seed(process_id)
    print(f"Process {process_id} started.")
    results = qLearning(num_episodes, steps, exploration_break, shocks, exploration_rate, discount_factor, alpha)
    shared_list.append(results)
    print(f"Process {process_id} finished.")
