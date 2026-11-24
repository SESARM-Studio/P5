import numpy as np
import heapq
from query_functions import create_supabase_connection, query_distinct_training_epoch_no, query_training_epoch_accuracies_experiment
from prompt_functions import prompt_user
from scipy.stats import skewnorm
from scipy.stats import skew

FIRST_EXPERIMENT_NO = 1
LAST_EXPERIMENT_NO = 3

def bernoulli_trial(success_probability, confidence):
    min_sample_size = np.ceil(np.log(1-confidence) / np.log(1-success_probability))
    return min_sample_size

supabase = create_supabase_connection()

# Query the database to get a list of dictionaries with epochs number
result_epochs_no_query = query_distinct_training_epoch_no(supabase)
list_epochs_no = [row["epoch_no"] for row in result_epochs_no_query if "epoch_no" in row]
last_epoch = max(list_epochs_no)

# Prompt user for input
experiment_choice = prompt_user(FIRST_EXPERIMENT_NO, LAST_EXPERIMENT_NO)

# Retrieve the data from a specific experiment using our query functions from query_functions.py
result_last_epoch_accuracies = query_training_epoch_accuracies_experiment(supabase, last_epoch, experiment_choice)

# result_last_epoch_accuracies contains a list of each row as a dictionary, this makes it into a numpy array of just the accuracies.
list_accuracy_data = np.array([row["accuracy"] for row in result_last_epoch_accuracies if "accuracy" in row])
list_accuracy_data *= 100 # Multiply decimal values to become percentage values

# Calculate basic data analysis values
min_accuracy = min(list_accuracy_data)
max_accuracy = max(list_accuracy_data)
empirical_mean = np.mean(list_accuracy_data)
empirical_std = np.std(list_accuracy_data)
empirical_skew = skew(list_accuracy_data)

print(" --- Empirical based on results ---")
print("Minimum accuracy:", min_accuracy)
print("Maximum accuracy:", max_accuracy)
print("Empirical mean:", round(empirical_mean,2))
print("Empirical std:", round(empirical_std,2))
print("Empirical skew:", round(empirical_skew,2))

print("\n ---Estimated based on skew normal PDF ---")

# Use scipy's skewnorm to calculate shape, loc and scale and calculate basic data analysis values
shape, loc, scale = skewnorm.fit(list_accuracy_data)
estimated_skew = skewnorm.stats(shape, loc, scale, moments='s')
estimated_mean = skewnorm.mean(shape, loc, scale)
estimated_std = skewnorm.std(shape, loc, scale)

print("Estimated mean:", round(estimated_mean,2))
print("Estimated std:", round(estimated_std,2))
print("Estimated skew:", round(estimated_skew,2))

print("\n --- Data analysis based on both empirical and estimated ---")
top_x_percent = float(input("What top x percent should the seeds lie in: "))
top_x_decimal = top_x_percent/100
complement_top_x_percent = 100-top_x_percent
complement_top_x_decimal = 1.0-top_x_decimal

# Calculate the value of the lower bound using 4 different methods.
numpy_empirical_lowbound = np.percentile(list_accuracy_data, complement_top_x_percent) # Takes into account the distribution of data.
interpolated_lowbound = (max_accuracy-min_accuracy) * (complement_top_x_decimal) + min_accuracy # Assumes the data is uniformly distributed.
# Estimates the lower bound using the Percent Point Function (ppf) which calculates the value of the percentile. It works left to right, that is why its the complement.
estimated_lowbound = skewnorm.ppf(complement_top_x_decimal, shape, loc, scale) 

num_top_seeds = int(len(list_accuracy_data) * top_x_decimal)
top_seed_accuracies = heapq.nlargest(num_top_seeds, list_accuracy_data) # Make a list containing the accuracies of the top x seeds.
empirical_lowbound = sorted(top_seed_accuracies)[0] # Sort the list of accuracies in ascending order and take the lowest value.

print(f"Empirical lower bound of top {top_x_percent}% calculated on range from {min_accuracy} to {max_accuracy}: {round(interpolated_lowbound,2)} ")
print(f"Empirical lower bound of top {top_x_percent}% based on numpy's percentile function: {round(numpy_empirical_lowbound,2)}")
print(f"Empirical lower bound of top {top_x_percent}% by taking lowest value of top {num_top_seeds} results: {round(empirical_lowbound,2)}")
print(f"Estimated lower bound of top {top_x_percent}%: {round(estimated_lowbound,2)}")

confidence_level_percent = float(input("Calculating min sample size needed. What should the confidence level be as x%: "))
confidence_level_decimal = confidence_level_percent/100

min_sample_size = bernoulli_trial(top_x_decimal, confidence_level_decimal)

print(f"You need to do atleast {min_sample_size} trials to, with a {confidence_level_percent}% confidence, get a seed within the top {top_x_percent}% of seeds")
