from typing import Callable

import matplotlib.pyplot as plt

from data_analysis.ranking_data_analysis.kendalls_tau import *
from data_analysis.ranking_data_analysis.rbo import *
from data_analysis.ranking_data_analysis.choice_enum import Choice


EPOCHS_NO_STNDRD = 60
EPOCHS_NO_EXPERIMENT_3 = 150


def plot_seed_rank_positions_across_epochs(seed_rank_positions: List[List[dict]], data_flag: Choice):
  """
  Returns a 2D array that contains the rank position of each seed for each epoch.\n
  """
  
  epochs_no: int
  seeds_no: int

  match data_flag:
    case Choice.EXPERIMENT_1:
      epochs_no = EPOCHS_NO_STNDRD

    case Choice.EXPERIMENT_2_300_SEEDS:
      epochs_no = EPOCHS_NO_STNDRD

    case Choice.EXPERIMENT_2_500_SEEDS:
      epochs_no = EPOCHS_NO_STNDRD
    
    case Choice.EXPERIMENT_3:
      epochs_no = EPOCHS_NO_EXPERIMENT_3

  # Use ranking data for epoch 0 to define number of included seed (must be the same across all epochs)
  seeds_no = len(seed_rank_positions[0]) 

  # Generate plots for each seed rank across epochs
  epoch_x_values = list(range(epochs_no))
  for seed in range(seeds_no):

    rank_y_values = []
    for epoch in range(epochs_no):
      rank_y_values.append(int(seed_rank_positions[epoch][seed]['rank']))
    
    # Print average rank for this seed
    print(f"Avg. rank for seed {str(seed_rank_positions[epoch][seed]['seed'])}: " + str(sum(rank_y_values) / len(rank_y_values)))
    

    # plt.step(epoch_x_values, rank_y_values, where="mid", linewidth=2, label="seed " + str(seed_rank_positions[epoch][seed]['seed'])) # Stair graph
    plt.plot(epoch_x_values, rank_y_values, linewidth=2, label="seed " + str(seed_rank_positions[epoch][seed]['seed'])) # Default point plot graph

  plt.legend()
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy-based rank")
  plt.yticks(list(range(1,seeds_no+1))) # y-value increments listed on y-axis
  plt.show()


def threedee_plot(rank_function: Callable, epochs: int, data_flag: Choice, persistance_parameter: float = 0.9) -> None:
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  x, y, z = [], [], []
  for i in range(epochs):
    for j in range(i + 1, epochs):
      if rank_function == kendalls_tau_b_between_specific_epochs:
        ranking_value = rank_function(i, j, data_flag)
      else:
        _, ranking_value = next(iter(rank_function(i, j, data_flag, persistance_parameter)))

      # Avoid querying twice and instead mirror output
      x.extend([i, j])
      y.extend([j, i])
      z.extend([ranking_value, ranking_value])

  ax.scatter(x,y,z)

  # Set visual symmetry for x and y axis
  ax.set_xlim(0, epochs - 1)
  ax.set_ylim(epochs - 1, 0)

  ax.set_xlabel("Epoch list 1")
  ax.set_ylabel("Epoch list 2")
  if rank_function == kendalls_tau_b_between_specific_epochs:
    ax.set_zlabel("Kendalls tau-b")
  else:
    ax.set_zlabel("RBO")

  plt.show()



  