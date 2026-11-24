from typing import Callable, List # For declaring list types 
import time # Remove this when done (used only in testing)

import matplotlib.pyplot as plt
import pandas as pd # Creation of DataFrame objects for RBO.
import pyterrier_alpha as pta # Rank-Biased Overlap (RBO) implementation

from kendalls_tau_function_data_analysis import kendalls_tau_b
from query_functions import *

supabase = create_supabase_connection()

EPOCHS_NO_STNDRD = 60
EPOCHS_NO_EXPERIMENT_3 = 150
SEEDS_NO_EXPERIMENT_1 = 200
SEEDS_NO_EXPERIMENT_2_NEW = 300
SEEDS_NO_EXPERIMENT_2_TOTAL = 500
SEEDS_NO_EXPERIMENT_3 = 200
SEED_COLUMN_NAME = "seed"


def plot_seed_rank_positions_across_epochs(seed_rank_positions: List[List[dict]], data_flag: int):
  """
  Returns a 2D array that contains the rank position of each seed for each epoch.\n
  --------------
  data_flag = 0 // experiment 1 data\n
  data_flag = 1 // experiment 2 (300 new seeds) data\n
  data_flag = 2 // experiment 2 (all 500 seeds) data\n
  data_flag = 3 // experiment 3 data\n
  """
  epochs_no: int
  seeds_no: int

  match data_flag:
    case 0:
      epochs_no = EPOCHS_NO_STNDRD

    case 1:
      epochs_no = EPOCHS_NO_STNDRD

    case 2:
      epochs_no = EPOCHS_NO_STNDRD
    
    case 3:
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


def seed_rank_positions_across_epochs(data_flag: int, relative_ranking: bool, seed_selection: List[int]):
  """
  Returns a 2D array that contains the rank position of each seed for each epoch.\n
  --------------
  data_flag = 0 // experiment 1 data\n
  data_flag = 1 // experiment 2 (300 new seeds) data\n
  data_flag = 2 // experiment 2 (all 500 seeds) data\n
  data_flag = 3 // experiment 3 data\n
  --------------
  relative_ranking = True // seeds are only ranked relative to other seeds included in the analysis. Meaningful when seed_selection is non-empty.\n
  relative_ranking = False // seeds are ranked relative to all seeds in the chosen dataset\n
  --------------
  seed_selecttion: if non-empty, only seeds in this list is included in the analysis. If empty, all seeds are considered in the chosen dataset.
  """
  epochs_no: int
  no_seeds: int
  training_result_table_name: str
  accuracy_column_name: str
  query_function: function

  match data_flag:
    case 0:
      epochs_no = EPOCHS_NO_STNDRD
      no_seeds = SEEDS_NO_EXPERIMENT_1
      training_result_table_name = "training_result"
      accuracy_column_name = "accuracy"
      query_function = query_epoch_accuracies_experiment_1
    
    case 1:
      epochs_no = EPOCHS_NO_STNDRD
      no_seeds = SEEDS_NO_EXPERIMENT_2_NEW
      training_result_table_name = "training_result"
      accuracy_column_name = "accuracy"
      query_function = query_epoch_accuracies_experiment_2_300_seeds

    case 2:
      epochs_no = EPOCHS_NO_STNDRD
      no_seeds = SEEDS_NO_EXPERIMENT_2_TOTAL
      training_result_table_name = "training_result"
      accuracy_column_name = "accuracy"
      query_function = query_epoch_accuracies_experiment_2_500_seeds

    case 3:
      epochs_no = EPOCHS_NO_EXPERIMENT_3
      no_seeds = SEEDS_NO_EXPERIMENT_3
      training_result_table_name = "training_result_convergence"
      accuracy_column_name = "validation_accuracy"
      query_function = query_epoch_accuracies_experiment_3

  seed_rank_positions = []
  for i in range(epochs_no):
    query_data_raw = query_function(supabase=supabase, epoch=i).data
    epoch_ranking = []
    
    # Bring the raw data to the form:
    # [{'seed': S, 'accuracy_name': A},...]
    query_data = []
    for k in range(len(query_data_raw)):
      query_data.append({SEED_COLUMN_NAME: query_data_raw[k][SEED_COLUMN_NAME], accuracy_column_name: query_data_raw[k][training_result_table_name][0][accuracy_column_name]})

    # Conduct ranking relative only to selected seeds
    if relative_ranking: 
      query_data = [row for row in query_data if row['seed'] in seed_selection]
      no_seeds = len(seed_selection)

    # Sorts on accuracy
    query_data.sort(key=lambda row: row[accuracy_column_name], reverse = True) # reverse enforces descending order

    rank = 1
    for j in range(no_seeds): # ret no_seeds til seeds_no - Spillemester Maren
      # Index of seed in query_data (sorted by accuracy) is equal to its rank (index + 1)
      # If two seeds have the same accuracy, they get the same rank. Dense rank is used, meaning:
      # Seed: accuracy : rank
      # S1: 0.7612 : 1
      # S2: 0.7612 : 1
      # S3: 0.7483 : 2 (and not 3 as in none-dense ranks)

      rank = rank - 1 if (j > 0 and (query_data[j][accuracy_column_name] == query_data[j-1][accuracy_column_name])) else rank
      epoch_ranking.append({SEED_COLUMN_NAME: query_data[j][SEED_COLUMN_NAME], 'rank': rank})
      rank += 1

    # Sorts on seed in ascending order
    epoch_ranking.sort(key=lambda row: row['seed'])
    if len(seed_selection) > 0: epoch_ranking = [row for row in epoch_ranking if row['seed'] in seed_selection] # Redundant if relative_ranking is True in check further above.
    seed_rank_positions.append(epoch_ranking)

  return seed_rank_positions
        

def rbo_between_specific_epochs(epoch_no_a: int, epoch_no_b: int, data_flag: int, p: float):
  """
  Returns the rank-biased overlap for all seeds between two specified epochs, epoch_no_a and epoch_no_a.\n
  p is the persistance parameter used for Rank-Biased Overlap.\n
  --------------
  data_flag = 0 // experiment 1 data\n
  data_flag = 1 // experiment 2 (300 new seeds) data\n
  data_flag = 2 // experiment 2 (all 500 seeds) data\n
  data_flag = 3 // experiment 3 data\n
    """
  SEED_COLUMN_NAME = "seed"

  training_result_table_name: str
  accuracy_column_name: str

  list_a = []
  list_a_accuracies = []
  list_b = []
  list_b_accuracies = []
  result_epoch_data_a: any
  result_epoch_data_b: any

  match data_flag:
    case 0:
      training_result_table_name = "training_result"
      accuracy_column_name = "accuracy"
      result_epoch_data_a = query_epoch_accuracies_experiment_1(supabase, epoch_no_a)
      result_epoch_data_b = query_epoch_accuracies_experiment_1(supabase, epoch_no_b)
    
    case 1:
      training_result_table_name = "training_result"
      accuracy_column_name = "accuracy"
      result_epoch_data_a = query_epoch_accuracies_experiment_2_300_seeds(supabase, epoch_no_a)
      result_epoch_data_b = query_epoch_accuracies_experiment_2_300_seeds(supabase, epoch_no_b)

    case 2:
      training_result_table_name = "training_result"
      accuracy_column_name = "accuracy"
      result_epoch_data_a = query_epoch_accuracies_experiment_2_500_seeds(supabase, epoch_no_a)
      result_epoch_data_b = query_epoch_accuracies_experiment_2_500_seeds(supabase, epoch_no_b)

    case 3:
      training_result_table_name = "training_result_convergence"
      accuracy_column_name = "validation_accuracy"
      result_epoch_data_a = query_epoch_accuracies_experiment_3(supabase, epoch_no_a)
      result_epoch_data_b = query_epoch_accuracies_experiment_3(supabase, epoch_no_b)


  for entry in result_epoch_data_a.data:
    # What an entry looks like
    # {'training_id': '941beba0-0036-4660-8b75-6c49f4b94936', 'seed': 200, 'training_result_table_name': [{'accuracy_column_name': 0.3811, 'epoch_no': 2, 'training_id': '941beba0-0036-4660-8b75-6c49f4b94936'}]}

    # Adding the seeds that are considered for ranking
    list_a.append(entry[SEED_COLUMN_NAME])

    # Adding accuracies for epoch a
    list_a_accuracies.append(entry[training_result_table_name][0][accuracy_column_name])

  for entry in result_epoch_data_b.data:
    # Adding the seeds that are considered for ranking
    list_b.append(entry[SEED_COLUMN_NAME])

    # Adding accuracies for epoch a
    list_b_accuracies.append(entry[training_result_table_name][0][accuracy_column_name])

  return rbo(list_a, list_a_accuracies, list_b, list_b_accuracies, p)


def rbo(list_a: List[str], list_a_accuracies: List[float], list_b: List[str], list_b_accuracies: List[float], p: float):
  # PyTerrier handles the lists to rank as DataFrame objects implemented by the Pandas package
  # PyTerrier use the attribute docno (Document Number) to distinguish each item
  df_a = pd.DataFrame({
    "docno": list_a
  })
  df_b = pd.DataFrame({
    "docno": list_b
  })

  # qid is the Query ID (what we are ranking)
  df_a["qid"] = "rbo_score"
  df_b["qid"] = "rbo_score"

  # score is used in order for PyTerrier to know what where each item ranks.  
  df_a["score"] = list_a_accuracies
  df_b["score"] = list_b_accuracies

  return dict(pta.rbo(df_a, df_b, p))


def kendalls_tau_b_between_specific_epochs(epoch_no_a: int, epoch_no_b: int, data_flag: int):
    """
    Returns the rank-biased overlap for all seeds between two specified epochs, epoch_no_a and epoch_no_a.\n
    p is the persistance parameter used for Rank-Biased Overlap.\n
    --------------
    data_flag = 0 // experiment 1 data\n
    data_flag = 1 // experiment 2 (300 new seeds) data\n
    data_flag = 2 // experiment 2 (all 500 seeds) data\n
    data_flag = 3 // experiment 3 data\n
        """
    training_result_table_name: str
    accuracy_column_name: str

    list_a_accuracies = []
    list_b_accuracies = []
    result_epoch_data_a: any
    result_epoch_data_b: any

    match data_flag:
        case 0:
            training_result_table_name = "training_result"
            accuracy_column_name = "accuracy"
            result_epoch_data_a = query_epoch_accuracies_experiment_1(supabase, epoch_no_a)
            result_epoch_data_b = query_epoch_accuracies_experiment_1(supabase, epoch_no_b)

        case 1:
            training_result_table_name = "training_result"
            accuracy_column_name = "accuracy"
            result_epoch_data_a = query_epoch_accuracies_experiment_2_300_seeds(supabase, epoch_no_a)
            result_epoch_data_b = query_epoch_accuracies_experiment_2_300_seeds(supabase, epoch_no_b)

        case 2:
            training_result_table_name = "training_result"
            accuracy_column_name = "accuracy"
            result_epoch_data_a = query_epoch_accuracies_experiment_2_500_seeds(supabase, epoch_no_a)
            result_epoch_data_b = query_epoch_accuracies_experiment_2_500_seeds(supabase, epoch_no_b)

        case 3:
            training_result_table_name = "training_result_convergence"
            accuracy_column_name = "validation_accuracy"
            result_epoch_data_a = query_epoch_accuracies_experiment_3(supabase, epoch_no_a)
            result_epoch_data_b = query_epoch_accuracies_experiment_3(supabase, epoch_no_b)




    for entry in result_epoch_data_a.data:
        # What an entry looks like
        # {'training_id': '941beba0-0036-4660-8b75-6c49f4b94936', 'seed': 200, 'training_result_table_name': [{'accuracy_column_name': 0.3811, 'epoch_no': 2, 'training_id': '941beba0-0036-4660-8b75-6c49f4b94936'}]}

        # Adding accuracies for epoch a
        list_a_accuracies.append(entry[training_result_table_name][0][accuracy_column_name])

    for entry in result_epoch_data_b.data:
        # Adding accuracies for epoch a
        list_b_accuracies.append(entry[training_result_table_name][0][accuracy_column_name])

    return kendalls_tau_b(list_a_accuracies, list_b_accuracies)


def threedee_plot(rank_function: Callable, epochs: int, data_flag: int, persistance_parameter: float = 0.9) -> None:
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


def main():

  analysis_choice: int
  dataset_choice: int
  seed_range_choice: int
  seed_selection = List[int]
  ranking_approach_choice: bool


##### TESTING YARD #####
# hip = ["1","32","2","45","8","3"]
# hip_acc = [75, 74, 70, 68, 65, 63]

  # Choose analysis
  while(True):
    print("Choose an analysis")
    print("""
        0. Rank seeds over epochs (2D step-ladder graph)\n
        1. Compare epochs on seed rankings (3D graph or specific epoch rank comparison)\n
      """)
    analysis_choice = int(input())
    if analysis_choice not in [0,1]:
      continue
    else:
      break

  # Choose dataset
  while(True):
      print("Choose dataset")
      print("""
          0. Experiment 1\n
          1. Experiment 2 (300 new seeds)\n
          2. Experiment 2 (all 500 seeds)\n
          3. Experiment 3
        """)
      dataset_choice = int(input())
      if dataset_choice not in [0,1,2,3]:
        continue
      else:
        break

  if (analysis_choice) == 0: # Rank seeds through all epochs (2D step-ladder graph)
    while(True):
      print("Choose seed range")
      print("""
          0. All in dataset\n
          1. Specify specific seeds\n
        """)
      seed_range_choice = int(input())
      if seed_range_choice not in [0,1]:
        continue
      else:
        break

    if (seed_range_choice == 1):
      print("Specify seeds in this format: 1,7,9,13,...")
      seed_selection_input = str(input())
      seed_selection = [int(seed) for seed in seed_selection_input.split(',')]
    else:
      seed_selection = []

    while(True):
      print("Choose an approach")
      print("""
          0. Rank compared to all seeds in dataset\n
          1. Rank relative to selected seeds\n
        """)
      ranking_approach_choice_input = int(input())
      if ranking_approach_choice_input not in [0,1]:
        continue
      else:
        break

    ranking_approach_choice = True if ranking_approach_choice_input == 1 else False

    seed_rank_positions = seed_rank_positions_across_epochs(dataset_choice, ranking_approach_choice, seed_selection)
    plot_seed_rank_positions_across_epochs(seed_rank_positions, dataset_choice)

  else: # Compare epochs rankings (3D graph)
    while(True):
      print("Choose an approach")
      print("""
          0. Choose two specific epochs to compare on seed rankings\n
          1. Visualise all epoch-to-epoch rank comparisons in 3D\n
        """)
      epoch_comparison_approach = int(input())
      if epoch_comparison_approach not in [0,1]:
        continue
      else:
        break

    if (epoch_comparison_approach == 0):
      while (True):
        print("Specify two epochs in this format: 2,30")
        epoch_selection_input = str(input())
        epoch_selection = [int(epoch) for epoch in epoch_selection_input.split(',')]

        if (len(epoch_selection) != 2):
          continue
        else:
          break

      while (True):
        print("Choose a ranked list comparison method")
        print("""
            0. Rank-Biased Overlap\n
            1. Kendall-Tau\n
          """)
        comparison_method_choice = int(input())
        if comparison_method_choice not in [0,1]:
          continue
        else:
          break

      if (comparison_method_choice == 0):
        while (True):
          print("Choose a persistance parameter [0,1]:\n")
          persistance_parameter = float(input())

          if not (persistance_parameter >= 0 and persistance_parameter <=1):
            continue
          else:
            break

        rbo_score = rbo_between_specific_epochs(epoch_selection[0], epoch_selection[1], dataset_choice, persistance_parameter)['rbo_score']
        print(f"Rank-Biased Overlap score: {rbo_score}")
      

        print(rbo_between_specific_epochs(epoch_selection[0], epoch_selection[1], dataset_choice, persistance_parameter))
      else:
        print(kendalls_tau_b_between_specific_epochs(epoch_selection[0], epoch_selection_input[1], dataset_choice))
    elif epoch_comparison_approach == 1:
      while (True):
        print("Choose a ranked list comparison method")
        print("""
            0. Rank-Biased Overlap\n
            1. Kendall-Tau\n
          """)
        comparison_method_choice = int(input())
        if comparison_method_choice not in [0,1]:
          continue
        else:
          break

      if comparison_method_choice == 0:
        while (True):
          print("Choose a persistance parameter [0,1]:\n")
          persistance_parameter = float(input())

          if not (persistance_parameter >= 0 and persistance_parameter <=1):
            continue
          else:
            break

        print("Making 3D plot...")
        threedee_plot(
          rbo_between_specific_epochs,
          epochs=60,
          data_flag=dataset_choice,
          persistance_parameter=persistance_parameter
        )
      else:
        print("Making 3D plot...")
        threedee_plot(
          kendalls_tau_b_between_specific_epochs,
          epochs=60,
          data_flag=dataset_choice
        )



if __name__ == "__main__":
  main()
