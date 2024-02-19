#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:10:39 2024

@author: sahil
"""
import defs
import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/Users/sahil/Desktop/PhD/Research/Masters Project/Multiprocessing/')

if __name__ == '__main__':
    num_processes = 5
    num_episodes = 200000  
    steps = 100
    exploration_break = 200000
    processes = []
    # Create a Manager to manage the shared list
    manager = multiprocessing.Manager()
    shared_list = manager.list()
    
    for i in range(num_processes):
        p = multiprocessing.Process(target=defs.run_qLearning, args=(i, num_episodes, steps, exploration_break, shared_list))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()
        
     # Convert the shared list to a regular list
    result_list = list(shared_list)
    all_action_1 = [result['actions_1'] for result in result_list]
    
    n = 1000  # Replace this with the desired value of n

    averaged_list = []

    for sublist in all_action_1:
        averaged_sublist = [sum(sublist[i:i+n])/n for i in range(0, len(sublist), n)]
        averaged_list.append(averaged_sublist)
    plt.figure(figsize=(10,10))
    for idx, actions in enumerate(averaged_list):
        plt.plot(np.arange(0, 200, 1), actions, label = f'Iteration {idx + 1}')
    plt.legend()
    plt.show()
    print("All processes finished.")


