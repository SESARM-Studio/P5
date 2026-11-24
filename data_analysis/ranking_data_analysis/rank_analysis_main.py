from data_analysis.ranking_data_analysis.rbo import *
from data_analysis.ranking_data_analysis.kendalls_tau import *
from data_analysis.ranking_data_analysis.plots import *


def choose_analysis():
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
  return analysis_choice

def choose_dataset():
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
  return dataset_choice

def choose_seed_range():
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
  return seed_range_choice

def choose_rank_approach():
  while(True):
    print("Choose an approach")
    print("""
        0. Rank compared to all seeds in dataset\n
        1. Rank relative to selected seeds\n
      """)
    ranking_approach_choice = bool(int(input()))
    if ranking_approach_choice not in [True, False]:
      continue
    else:
      break
  return ranking_approach_choice

def choose_epoch_comparison_approach():
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
  return epoch_comparison_approach

def specify_epochs(dataset_choice):
  epochs_no: int

  match dataset_choice:
    case 0 | 1 | 2:
      epochs_no = 60
    case 3:
      epochs_no = 150

  acceptable_epochs = list(range(epochs_no))

  while (True):
    epoch_selection_input = str(input())
    epoch_selection = [int(epoch) for epoch in epoch_selection_input.split(',')]

    if (len(epoch_selection) != 2 or epoch_selection[0] not in acceptable_epochs or epoch_selection[1] not in acceptable_epochs):
      continue
    else:
      break
  return epoch_selection

def choose_comparison_method():
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
  return comparison_method_choice

def choose_persistance_parameter():
  while (True):
    print("Choose a persistance parameter [0,1]:\n")
    persistance_parameter = float(input())

    if not (persistance_parameter >= 0 and persistance_parameter <=1):
      continue
    else:
      break
  return persistance_parameter


def main():

  analysis_choice: int
  dataset_choice: int
  seed_range_choice: int
  seed_selection = List[int]
  ranking_approach_choice: bool

  # Choose analysis
  analysis_choice = choose_analysis()

  # Choose dataset
  dataset_choice = choose_dataset()

  if (analysis_choice) == 0: # Rank seeds through all epochs (2D step-ladder graph)
    seed_range_choice = choose_seed_range()

    if (seed_range_choice == 1):
      print("Specify seeds in this format: 1,7,9,13,...")
      seed_selection_input = str(input())
      seed_selection = [int(seed) for seed in seed_selection_input.split(',')]
    else:
      seed_selection = []

    ranking_approach_choice = choose_rank_approach()

    seed_rank_positions = seed_rank_positions_across_epochs(dataset_choice, ranking_approach_choice, seed_selection)
    plot_seed_rank_positions_across_epochs(seed_rank_positions, dataset_choice)

  else: # Compare epochs rankings (3D graph or epoch specific rank comparison)
    epoch_comparison_approach = choose_epoch_comparison_approach()

    if (epoch_comparison_approach == 0): # 2D plot
      epoch_selection = specify_epochs(dataset_choice)

      comparison_method_choice = choose_comparison_method()

      if (comparison_method_choice == 0): # 2D plot using RBO
        persistance_parameter = choose_persistance_parameter()

        _, rbo_score = next(iter(rbo_between_specific_epochs(epoch_selection[0], epoch_selection[1], dataset_choice, persistance_parameter)))
        print(f"Rank-Biased Overlap score: {rbo_score}")
      
      else: # 2D plot using Kendall-Tau 
        print(kendalls_tau_b_between_specific_epochs(epoch_selection[0], epoch_selection[1], dataset_choice))

    elif epoch_comparison_approach == 1:
      comparison_method_choice = choose_comparison_method()

      if comparison_method_choice == 0: # 3D plot using RBO
        persistance_parameter = choose_persistance_parameter()

        print("Making 3D plot...")
        threedee_plot(
          rbo_between_specific_epochs,
          epochs=60,
          data_flag=dataset_choice,
          persistance_parameter=persistance_parameter
        ) 
      else: # 3D plot using Kendall-Tau
        print("Making 3D plot...")
        threedee_plot(
          kendalls_tau_b_between_specific_epochs,
          epochs=60,
          data_flag=dataset_choice
        )

if __name__ == "__main__":
  main()

