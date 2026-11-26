import matplotlib.pyplot as plt
import numpy as np
from query_functions import *
from scipy.stats import skewnorm
from prompt_functions import prompt_user, prompt_chose_accuracy

PDF_STEPSIZE = 0.05 # The stepsize for the np.arange x-values for the pdf.
HISTOGRAM_BINS = 20
FIRST_EXPERIMENT_NO = 1
LAST_EXPERIMENT_NO = 3

supabase = create_supabase_connection()

# Query the database to get a list of dictionaries with epochs number
result_epochs_no_query = query_distinct_training_epoch_no(supabase)
list_epochs_no = [row["epoch_no"] for row in result_epochs_no_query if "epoch_no" in row]
last_epoch = max(list_epochs_no)

# Prompt user for input
experiment_choice = prompt_user(FIRST_EXPERIMENT_NO, LAST_EXPERIMENT_NO)
accuracy_choice = prompt_chose_accuracy()

# Retrieve the data from a specific experiment using our query functions from query_functions.py
result_last_epoch_accuracies = query_training_epoch_accuracies_experiment(supabase, last_epoch, experiment_choice)
result_best_epoch_accuracies = query_training_best_accuracies_experiment(supabase, experiment_choice)

# result_last_epoch_accuracies or result_best_epoch_accuracies contains a list of each row as a dictionary,
# this makes it into a numpy array of just the accuracies.
if accuracy_choice == 1:
    list_accuracy_data = np.array([row["accuracy"] for row in result_last_epoch_accuracies if "accuracy" in row])
else:
    list_accuracy_data = np.array([row["max"] for row in result_best_epoch_accuracies.data if "max" in row])

list_accuracy_data *= 100 # Multiply decimal values to become percentage values

min_accuracy = min(list_accuracy_data)
max_accuracy = max(list_accuracy_data)
mean = np.mean(list_accuracy_data)
std = np.std(list_accuracy_data)

print("Minimum accuracy:", min_accuracy)
print("Maximum accuracy:", max_accuracy)
print("Mean:", round(mean,2))
print("std:", round(std,2))

# Create points from min to max with stepsize and calculate the y-values according to Skew-Normal Distribution
graph_xpdf = np.arange(min_accuracy, max_accuracy+1, PDF_STEPSIZE)
shape, loc, scale = skewnorm.fit(list_accuracy_data)
graph_ypdf = skewnorm.pdf(graph_xpdf, shape, loc, scale)
fig, ax = plt.subplots()

# Plot the histogram
histogram_style = {"facecolor": "lightblue", "edgecolor": "white", "linewidth": 1 }
ax.hist(list_accuracy_data, bins=HISTOGRAM_BINS, density=True, **histogram_style)

# Plot the probability distribution function
ax.plot(graph_xpdf, graph_ypdf)

# Plot every data accuracy at the bottom of the graph as vertical lines
plot_style = {"color": "darkblue", "marker": "|", "markersize": 20}
y_zeroes = 0*list_accuracy_data
ax.plot(list_accuracy_data, y_zeroes, **plot_style)

ax.set_xlabel("Accuracy")
ax.set_ylabel("Density")

plt.show()