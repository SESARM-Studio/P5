import numpy as np
from query_functions import create_supabase_connection, query_distinct_training_epoch_no, query_last_training_epoch_accuracies
from scipy.stats import skewnorm
from scipy.stats import skew
import heapq

def bernoulli_trial(success_probability, confidence):
    min_sample_size = np.ceil(np.log(1-confidence) / np.log(1-success_probability))
    return min_sample_size

supabase = create_supabase_connection()

# Query the database to get a list of dictionaries with epochs number
result_epochs_no_query = query_distinct_training_epoch_no(supabase)
list_epochs_no = [row["epoch_no"] for row in result_epochs_no_query.data if "epoch_no" in row]
last_epoch = max(list_epochs_no)

result_last_epoch_accuracies = query_last_training_epoch_accuracies(supabase, last_epoch)
# result_last_epoch_accuracies.data contains a list of each row as a dictionary, this makes it into a numpy array of just the accuracies.
list_accuracy_data = np.array([row["accuracy"] for row in result_last_epoch_accuracies.data if "accuracy" in row])
list_accuracy_data *= 100 # Multiply decimal values to become percentage values

# Calculate basic data analysis values
min_accuracy = min(list_accuracy_data)
max_accuracy = max(list_accuracy_data)
mean = np.mean(list_accuracy_data)
std = np.std(list_accuracy_data)
empirical_skew = skew(list_accuracy_data)

print(" --- Empirical based on results ---")
print("Minimum accuracy:", min_accuracy)
print("Maximum accuracy:", max_accuracy)
print("Empirical mean:", round(mean,2))
print("Empirical std:", round(std,2))
print("Empirical skew:", round(empirical_skew,2))

print("\n ---Theoretical based on skew normal PDF ---")

# Use scipy's skewnorm to calculate shape, loc and scale and calculate basic data analysis values
shape, loc, scale = skewnorm.fit(list_accuracy_data)
theoretical_skew = skewnorm.stats(shape, loc, scale, moments='s')
theoretical_mean = skewnorm.mean(shape, loc, scale)
theoretical_std = skewnorm.std(shape, loc, scale)

print("Theoretical mean:", round(theoretical_mean,2))
print("Theoretical std:", round(theoretical_std,2))
print("Theoretical skew:", round(theoretical_skew,2))

print("\n --- Data analysis based on both empirical and theoretical ---")
top_x = int(input("What top x percent should the seeds lie in: "))

# Calculate the value of the lower bound using 4 different methods.
numpy_empirical_threshold = np.percentile(list_accuracy_data, 100-top_x)
calculated_empirical_threshold = (max_accuracy-min_accuracy) * (100-top_x)/100 + min_accuracy
theoretical_threshold = skewnorm.ppf(1.0-top_x/100, shape, loc, scale)

num_best_seeds = int(len(list_accuracy_data) * top_x/100)
largest_x_best_seeds = heapq.nlargest(num_best_seeds, list_accuracy_data)  
found_empirical_threshold = sorted(largest_x_best_seeds)[0]

print(f"Empirical lower bound of top {top_x}% calculated on range from {min_accuracy} to {max_accuracy}: {round(calculated_empirical_threshold,2)} ")
print(f"Empirical lower bound of top {top_x}% based on numpy's percentile function: {round(numpy_empirical_threshold,2)}")
print(f"Empirical lower bound of top {top_x}% by taking lowest value of top {num_best_seeds} results: {round(found_empirical_threshold,2)}")
print(f"Theoretical lower bound of top {top_x}%: {round(theoretical_threshold,2)}")

confidence_level = int(input("Calculating min sample size needed. What should the confidence level be as x%: "))

min_sample_size = bernoulli_trial(top_x/100, confidence_level/100)

print(f"You need to do atleast {min_sample_size} trials to, with a {confidence_level}% confidence, get a seed within the top {top_x}% of seeds")
