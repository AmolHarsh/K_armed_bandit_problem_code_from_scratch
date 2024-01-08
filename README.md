# Simulation Notebook README

This README file provides an overview of the Python code in the "Simulation Notebook." The code simulates a K-armed bandit problem, demonstrating the exploration-exploitation trade-off using an epsilon-greedy method. It also includes continuous probability reward distribution for more complex scenarios.

## Introduction

In the real world, we often encounter situations where we must make decisions without prior knowledge of the outcomes. This code demonstrates how reinforcement learning can help solve such problems, specifically the K-armed bandit problem. The goal is to maximize rewards by selecting the best actions. However, we face the challenge of balancing exploration and exploitation to achieve this goal.

## Pseudocode of the Simulation

Before diving into the code, here's a high-level overview of the simulation's pseudocode:

1. Initialize the value of epsilon (exploration factor).
2. Initialize a matrix to store Q-values for each bandit arm.
3. Run a loop for multiple simulation runs.
   - Create an empty array to keep track of the number of times each action is taken.
   - Run a loop for a specified number of trials.
     - Generate a random number between 0 and 1.
     - Use the random number for action selection:
       - If the random number is less than or equal to (1 - epsilon), exploit by selecting the action with the maximum expected value.
       - Otherwise, explore by choosing any action with equal probability.
     - Store the selected action and increment the action counter.
     - Allocate a reward based on the selected action.
     - Update the action's value using an incremental implementation technique.
4. Plot the Q-values matrix by taking the mean column-wise for each bandit machine.

## Code Sections

### Discrete Reward Example

1. The code starts by defining parameters such as the number of runs and trials.
2. It implements the epsilon-greedy method for action selection with discrete rewards.
3. It runs the simulation with epsilon (exploration factor) equal to 0 and then 0.1.
4. It visualizes the average performance of each bandit arm over trials using matplotlib.

### Continuous Probability Reward Distribution

1. The code introduces continuous probability reward distributions for more complex scenarios.
2. It generates true action values q*(a) for each bandit arm from a normal distribution.
3. It visualizes these continuous reward distributions using a violin plot.

### Advanced Simulation with Memory

1. The code extends the epsilon-greedy method to include memory for better performance.
2. It runs the advanced simulation with memory.
3. It visualizes the average performance of each bandit arm over trials with memory incorporated.

## Usage

You can run the code in a Python environment, making sure to set the desired parameters and explore the results of different simulations. The code is organized into functions for easy testing and experimentation.

Feel free to modify the parameters, such as the number of runs and trials, epsilon values, and the number of bandit arms, to observe how different settings affect the results.

Happy exploring and exploiting in your reinforcement learning journey!
