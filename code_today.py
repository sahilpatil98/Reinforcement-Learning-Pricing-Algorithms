#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 09:05:51 2023

@author: sahil
"""

''' Import Libraries '''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import seaborn as sns
import statistics
from matplotlib.gridspec import GridSpec
import os
import glob
sns.set()


np.random.seed(99163)

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

'''
plt.figure(figsize=(15,10))  # Adjust the figure size as needed
plt.imshow(Q1, cmap='viridis', aspect='auto', interpolation='nearest')
average = np.mean(action_space)
plt.axvline(x = action_space.index(average), label = 'Average')

'''
heatplot_dir1 = '/Users/sahil/Desktop/PhD/Research/Masters Project/Heatplots.nosync/P1/'
heatplot_dir2 = '/Users/sahil/Desktop/PhD/Research/Masters Project/Heatplots.nosync/P2/'


files = glob.glob('/Users/sahil/Desktop/PhD/Research/Masters Project/Heatplots.nosync/P1/*')
for f in files:
    os.remove(f)
    
files = glob.glob('/Users/sahil/Desktop/PhD/Research/Masters Project/Heatplots.nosync/P2/*')
for f in files:
    os.remove(f)
    
'''Results Dictionary'''
results = {'actions_1':[],
           'actions_2':[],
           'shock':[],
           'qval_1':[],
           'qval_2':[],
           'exploration_rate':[]
           }


'''Main Function'''
all_runs = []

def qLearning(num_episodes, steps, exploration_break = None, shocks = True, exploration_rate = 1, discount_factor = 0.95,
                            alpha = 0.15):
    start_time = time.time()

    #Parameters
    if shocks == True:
        d_t = [290, 310]
    else:
        d_t = [300]
        
        
       
    # For every episode
    for ith_episode in range(num_episodes):
        
        # Reset the environment and pick the first action
        state = random.choice(state_space)
        state_index = state_space.index(state)
           
        for t in range(steps):
            shock = random.choice(d_t)
            # get probabilities of all actions from current state
            #Exploration or Exploitation for Agent 1 at time t
            exploration_rate_threshold = random.uniform(0,1)
            if exploration_rate_threshold < exploration_rate:
                action_space_1 = [x for x in action_space if x <= sum(action_space) / len(action_space) if action_space]
                action_1 = random.choice(action_space_1)
                '''Explore this choice above'''
                '''Try LEQ average Action Space only P1'''
                action_1_index = action_space.index(action_1)
            else:
                action_1_index = np.argmax(Q1[state_index])
                action_1 = action_space[action_1_index]
            
            exploration_rate_threshold = random.uniform(0,1)
            if exploration_rate_threshold < exploration_rate:
                action_space_2 = [x for x in action_space if x <= sum(action_space) / len(action_space) if action_space]
                action_2 = random.choice(action_space_2)
                '''Explore this choice above'''
                '''Try LEQ average Action Space only P1'''
                action_2_index = action_space.index(action_2)
            else:
                action_2_index = np.argmax(Q2[state_index])
                action_2 = action_space[action_2_index]



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
            
            #results['actions_1'].append(action_1)
            #results['actions_2'].append(action_2)
            #results['shock'].append(shock)
            
        print('Episode Number: ' + str(ith_episode) + '. Exploration Rate: ' + str(exploration_rate))
        
        '''Update exploration rate'''
        if ith_episode < exploration_break:
            decay = 1/(num_episodes - num_episodes/2)
            exploration_rate = 1 - decay*ith_episode
            if exploration_rate < 0:
                exploration_rate = 0
        else:
            exploration_rate = 0
            
            
        '''Update Results Dictionary'''
        results['actions_1'].append(action_1)
        results['actions_2'].append(action_2)
        results['shock'].append(shock)
        results['qval_1'].append(alpha * td_target1 + (1 - alpha)* Q1[state_index][action_1_index])
        results['qval_2'].append(alpha * td_target2 + (1 - alpha)* Q2[state_index][action_2_index])

        #results['exploration_rate'].append(exploration_rate)
        # Turn interactive plotting off
        '''
        plt.ioff()
        fig = plt.figure(figsize=(14, 8))  # Adjust the figure size as needed
        plt.imshow(Q1, cmap='viridis', aspect='auto', interpolation='nearest')
        average = np.mean(action_space)
        plt.axvline(x = action_space.index(average), label = 'Average')
        plt.colorbar()
        output_filename = f'{ith_episode + 1}.png'
        output_path1 = heatplot_dir1 + output_filename
        plt.savefig(output_path1)
        plt.clf()
        plt.close(fig)
        '''
        # Turn interactive plotting off
        '''
        plt.ioff()
        fig = plt.figure(figsize=(14, 8))  # Adjust the figure size as needed
        plt.imshow(Q2, cmap='viridis', aspect='auto', interpolation='nearest')
        plt.colorbar()
        plt.axvline(x = action_space.index(average), label = 'Average')
        output_filename = f'{ith_episode + 1}.png'
        output_path2 = heatplot_dir2 + output_filename
        plt.savefig(output_path2)
        plt.close(fig)
        '''
        

    return Q1, Q2, start_time, time.time()



'''Loop for saving data'''
all_runs = []

#ticks = np.arange(100000, 100001, 100000)
for i in [2000000]:
   values = qLearning(2000000, 1000, exploration_break = i, shocks = True)
   all_runs.append(results)
   results = {'actions_1':[],
              'actions_2':[],
              'shock':[],
              'qval_1':[],
              'qval_2':[],
              'exploration_rate':[]
              }
       
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



'''Plot'''
actions_p1 = all_runs[0]['actions_1']
actions_p2 = all_runs[0]['actions_2']


shocks = all_runs[0]['shock']

profit_1 = [(z - (x+y))*x for x, y, z in zip(actions_p1, actions_p2, shocks)]

profit_2 = [(z - (x+y))*y for x, y, z in zip(actions_p1, actions_p2, shocks)]


# Define the window size (in this case, 100)
window_size = 1000
profit_1 = [np.mean(profit_1[i:i+window_size]) for i in range(0, len(profit_1), window_size)]
profit_2 = [np.mean(profit_2[i:i+window_size]) for i in range(0, len(profit_2), window_size)]


#Plot for Profits
plt.figure()
plt.plot(np.arange(0,200,1), profit_1, label ='Competitor 1')
plt.plot(np.arange(0,200,1), profit_2, label ='Competitor 2')
plt.legend()
plt.title('Profit by Competitors')


#Plot for actions
actions_p1 = [np.mean(actions_p1[i:i+window_size]) for i in range(0, len(actions_p1), window_size)]
actions_p2 = [np.mean(actions_p2[i:i+window_size]) for i in range(0, len(actions_p2), window_size)]

plt.figure()
plt.plot(np.arange(0,200,1), actions_p1, label ='Competitor 1')
plt.plot(np.arange(0,200,1), actions_p2, label ='Competitor 2')
plt.legend()
plt.title('Action Decisions by Competitors')


plt.figure()
plt.plot(np.arange(100,200,1), actions_p1[-100:], label ='Competitor 1')
plt.plot(np.arange(100,200,1), actions_p2[-100:], label ='Competitor 2')
plt.legend()
plt.title('Action Decisions after Exploration is Over')