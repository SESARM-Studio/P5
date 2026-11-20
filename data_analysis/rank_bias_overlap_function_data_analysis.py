import pandas as pd # Creation of DataFrame objects for RBO.
import pyterrier_alpha as pta # Rank-Biased Overlap (RBO) implementation
from typing import List # For declaring list types 
from query_functions import *
import time # Remove this when done (used only in testing)
import matplotlib.pyplot as plt



supabase = create_supabase_connection()

def plot_seed_rank_positions_across_epochs(seed_rank_positions: List[List[int]], data_flag: int):
  """
  Returns a 2D array that contains the rank position of each seed for each epoch.\n
  --------------
  data_flag = 0 // experiment 1 data\n
  data_flag = 1 // experiment 2 (300 new seeds) data\n
  data_flag = 2 // experiment 2 (all 500 seeds) data\n
  data_flag = 3 // experiment 3 data\n
  """
  epochs: List[int]

  match data_flag:
    case 0 | 1 | 2:
      epochs = list(range(60))
    
    case _:
      epochs = []



  x = [1,2,3,4,5,6,6]
  y = [123,4,342,23,2,4,9]
  z = [2,123,3,1,31,2,67]
  plt.step(x, z, where="mid", linewidth=2, label="S1")
  plt.step(x, y, where="mid", linewidth=2, label="S2")

  for i in range(len(seed_rank_positions)):
    for j in range(len(seed_rank_positions[i])):
      plt.step(seed_rank_positions, z, where="mid", linewidth=2, label=i+1)

  plt.legend()
  plt.show()


def seed_rank_positions_across_epochs(data_flag: int):
  """
  Returns a 2D array that contains the rank position of each seed for each epoch.\n
  --------------
  data_flag = 0 // experiment 1 data\n
  data_flag = 1 // experiment 2 (300 new seeds) data\n
  data_flag = 2 // experiment 2 (all 500 seeds) data\n
  data_flag = 3 // experiment 3 data\n
  """

  EPOCHS_NO = 60
  SEED_COLUMN_NAME = "seed"

  no_seeds: int
  training_result_table_name: str
  accuracy_column_name: str
  query_function: function

  match data_flag:
    case 0:
      no_seeds = 200
      training_result_table_name = "training_result"
      accuracy_column_name = "accuracy"
      query_function = query_epoch_accuracies_experiment_1
    
    case 1:
      no_seeds = 300
      training_result_table_name = "training_result"
      accuracy_column_name = "accuracy"
      query_function = query_epoch_accuracies_experiment_2_300_seeds

    case 2:
      no_seeds = 300
      training_result_table_name = "training_result"
      accuracy_column_name = "accuracy"
      query_function = query_epoch_accuracies_experiment_2_500_seeds

    case 3:
      no_seeds = 200
      training_result_table_name = "training_result_convergence"
      accuracy_column_name = "validation_accuracy"
      query_function = query_epoch_accuracies_experiment_3


  seed_rank_positions = []
  for i in range(EPOCHS_NO):
    query_data_raw = query_function(supabase=supabase, epoch=i).data
    epoch_ranking = []
    
    # Bring the raw data to the form:
    # [{'seed': S, 'accuracy_name': A},...]
    query_data = []
    for k in range(len(query_data_raw)):
      query_data.append({SEED_COLUMN_NAME: query_data_raw[k][SEED_COLUMN_NAME], accuracy_column_name: query_data_raw[k][training_result_table_name][0][accuracy_column_name]})
  
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
    epoch_ranking.sort(key=lambda row: row['rank'])
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
  df_a["qid"] = "accuracy_ranking"
  df_b["qid"] = "accuracy_ranking"

  # score is used in order for PyTerrier to know what where each item ranks.  
  df_a["score"] = list_a_accuracies
  df_b["score"] = list_b_accuracies

  return pta.rbo(df_a, df_b, p)



##### TESTING YARD ##### 
# hip = ["1","32","2","45","8","3"]
# hip_acc = [75, 74, 70, 68, 65, 63]

# hip2 = ["1","32","2","45","8","3"]
# hip2_acc = [75, 74, 70, 68, 65, 63]

# p = 0.5

# start_time = time.time()
# print(rbo(hip, hip_acc, hip2, hip2_acc, p))
# print("--- %s seconds ---" % (time.time() - start_time))

#print(rbo_between_specific_epochs(39,59,0))

seed_rank_positions_across_epochs(0)
#plot_seed_rank_positions_across_epochs(0)
