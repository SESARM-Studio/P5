import pandas as pd
import pyterrier_alpha as pta
from typing import List
from query_functions import *

supabase = create_supabase_connection()


def rbo_between_specific_epochs(epoch_no_a: int, epoch_no_b: int, data_flag: int):
  """
  Returns the rank-biased overlap for all seeds between two specified epochs.
  --------------
  data_flag = 0 // experiment 1 data\n
  data_flag = 1 // experiment 2 data\n
  data_flag = 2 // experiment 3 data\n
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
      result_epoch_data_a = query_epoch_accuracies_experiment_2(supabase, epoch_no_a)
      result_epoch_data_b = query_epoch_accuracies_experiment_2(supabase, epoch_no_b)

    case 2:
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

  return rbo(list_a, list_a_accuracies, list_b, list_b_accuracies, 0.99)


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

  return pta.rbo(df_a, df_b, p=0.99)


hip = ["1","32","2","45","8","3"]
hip_acc = [75, 74, 70, 68, 65, 63]
hip2 = ["1","32","2","45","8","3"]
hip2_acc = [73, 76, 68, 68, 69, 63.2]


# rbo(hip, hip_acc, hip2, hip2_acc)

print(rbo_between_specific_epochs(39,59,0))