from typing import List # For declaring list types 

import pandas as pd # Creation of DataFrame objects for RBO.
import pyterrier_alpha as pta # Rank-Biased Overlap (RBO) implementation

from data_analysis.query_functions import *
from data_analysis.ranking_data_analysis.choice_enum import Choice


supabase = create_supabase_connection()

EPOCHS_NO_STNDRD = 60
EPOCHS_NO_EXPERIMENT_3 = 150
SEEDS_NO_EXPERIMENT_1 = 200
SEEDS_NO_EXPERIMENT_2_NEW = 300
SEEDS_NO_EXPERIMENT_2_TOTAL = 500
SEEDS_NO_EXPERIMENT_3 = 200
SEED_COLUMN_NAME = "seed"


def seed_rank_positions_across_epochs(data_flag: Choice, relative_ranking: bool, seed_selection: List[int], accuracy_tolerance: float):
  """
  Returns a 2D array that contains the rank position of each seed for each epoch.\n
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
    case Choice.EXPERIMENT_1:
      epochs_no = EPOCHS_NO_STNDRD
      no_seeds = SEEDS_NO_EXPERIMENT_1
      training_result_table_name = "training_result"
      accuracy_column_name = "accuracy"
      query_function = query_epoch_accuracies_experiment_1
    
    case Choice.EXPERIMENT_2_300_SEEDS:
      epochs_no = EPOCHS_NO_STNDRD
      no_seeds = SEEDS_NO_EXPERIMENT_2_NEW
      training_result_table_name = "training_result"
      accuracy_column_name = "accuracy"
      query_function = query_epoch_accuracies_experiment_2_300_seeds

    case Choice.EXPERIMENT_2_500_SEEDS:
      epochs_no = EPOCHS_NO_STNDRD
      no_seeds = SEEDS_NO_EXPERIMENT_2_TOTAL
      training_result_table_name = "training_result"
      accuracy_column_name = "accuracy"
      query_function = query_epoch_accuracies_experiment_2_500_seeds

    case Choice.EXPERIMENT_3:
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
    # [{'seed': S, 'accuracy': A},...]
    query_data = []
    for k in range(len(query_data_raw)):
      query_data.append({SEED_COLUMN_NAME: query_data_raw[k][SEED_COLUMN_NAME], accuracy_column_name: query_data_raw[k][training_result_table_name][0][accuracy_column_name]})

    # Conduct ranking relative only to selected seeds
    if relative_ranking: 
      query_data = [row for row in query_data if row['seed'] in seed_selection]
      no_seeds = len(seed_selection)

    # Accuracy will remain unchanged (set to accuracy used in previous ranking) if change in ranking is below the tolerance.
    if (i > 0):
      for l in range(no_seeds):
        # print("epoch: " + str(i))
        # print("seed no: " + str(query_data[l][SEED_COLUMN_NAME]))
        # print("current: " + str(query_data[l][accuracy_column_name]))
        # print("prior: " + str(seed_rank_positions[i-1][l][accuracy_column_name]))
        accuracy_delta = abs(query_data[l][accuracy_column_name] - seed_rank_positions[i-1][l][accuracy_column_name])
        # accuracy_delta_percent = abs((query_data[l][accuracy_column_name] - seed_rank_positions[i-1][l][accuracy_column_name]) / seed_rank_positions[i-1][l][accuracy_column_name])*100
        # print(accuracy_delta)
            
        if (accuracy_delta < accuracy_tolerance):
          # print("true")
          query_data[l][accuracy_column_name] = seed_rank_positions[i-1][l][accuracy_column_name]
          # print(query_data[l][accuracy_column_name])

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
      epoch_ranking.append({SEED_COLUMN_NAME: query_data[j][SEED_COLUMN_NAME], accuracy_column_name: query_data[j][accuracy_column_name], 'rank': rank})
      rank += 1

    # Sorts on seed in ascending order
    epoch_ranking.sort(key=lambda row: row['seed'])
    if len(seed_selection) > 0: epoch_ranking = [row for row in epoch_ranking if row['seed'] in seed_selection] # Redundant if relative_ranking is True in check further above.
    seed_rank_positions.append(epoch_ranking)

  return seed_rank_positions
        

def rbo_between_specific_epochs(epoch_no_a: int, epoch_no_b: int, data_flag: Choice, p: float):
  """
  Returns the rank-biased overlap for all seeds between two specified epochs, epoch_no_a and epoch_no_a.\n
  p is the persistance parameter used for Rank-Biased Overlap.\n
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
    case Choice.EXPERIMENT_1:
      training_result_table_name = "training_result"
      accuracy_column_name = "accuracy"
      result_epoch_data_a = query_epoch_accuracies_experiment_1(supabase, epoch_no_a)
      result_epoch_data_b = query_epoch_accuracies_experiment_1(supabase, epoch_no_b)
    
    case Choice.EXPERIMENT_2_300_SEEDS:
      training_result_table_name = "training_result"
      accuracy_column_name = "accuracy"
      result_epoch_data_a = query_epoch_accuracies_experiment_2_300_seeds(supabase, epoch_no_a)
      result_epoch_data_b = query_epoch_accuracies_experiment_2_300_seeds(supabase, epoch_no_b)

    case Choice.EXPERIMENT_2_500_SEEDS:
      training_result_table_name = "training_result"
      accuracy_column_name = "accuracy"
      result_epoch_data_a = query_epoch_accuracies_experiment_2_500_seeds(supabase, epoch_no_a)
      result_epoch_data_b = query_epoch_accuracies_experiment_2_500_seeds(supabase, epoch_no_b)

    case Choice.EXPERIMENT_3:
      training_result_table_name = "training_result_convergence"
      accuracy_column_name = "validation_accuracy"
      result_epoch_data_a = query_epoch_accuracies_experiment_3(supabase, epoch_no_a)
      result_epoch_data_b = query_epoch_accuracies_experiment_3(supabase, epoch_no_b)


  for entry in result_epoch_data_a.data:
    # What an entry looks like
    # {'seed': 200, 'training_result_table_name': [{'accuracy_column_name': 0.3811, 'epoch_no': 2}]}

    # Adding the seeds that are considered for ranking
    list_a.append(entry[SEED_COLUMN_NAME])

    # Adding accuracies for epoch a
    list_a_accuracies.append(entry[training_result_table_name][0][accuracy_column_name])

  for entry in result_epoch_data_b.data:
    # Adding the seeds that are considered for ranking
    list_b.append(entry[SEED_COLUMN_NAME])

    # Adding accuracies for epoch b
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

  # score is provided as the metric PyTerrier will use to rank each item.  
  df_a["score"] = list_a_accuracies
  df_b["score"] = list_b_accuracies

  return pta.rbo(df_a, df_b, p)
