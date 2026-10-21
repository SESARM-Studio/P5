import matplotlib.pyplot as plt
import numpy as np
from query_functions import *

supabase = create_supabase_connection()

# Query the database to get a list of dictionaries with epochs number
result_epochs_no_query = query_distinct_training_epoch_no(supabase)
list_epochs_no = [row["epoch_no"] for row in result_epochs_no_query.data if "epoch_no" in row]
last_epoch = max(list_epochs_no)

result_last_epoch_accuracies = query_last_training_epoch_accuracies(supabase, last_epoch)
# result_last_epoch_accuracies.data contains a list of each row as a dictionary, this makes it into a numpy array of just the accuracies.
list_accuracy_data = np.array([row["accuracy"] for row in result_last_epoch_accuracies.data if "accuracy" in row])

min_accuracy = min(list_accuracy_data)
max_accuracy = max(list_accuracy_data)
mean = np.mean(list_accuracy_data)
std = np.std(list_accuracy_data)

print("Minimum accuracy:", min_accuracy)
print("Maximum accuracy:", max_accuracy)
print("Mean:", round(mean,2))
print("std:", round(std,2))

# Create points from min to max with stepsize 0.1 and calculate the y-values according to Guassian Distribution
graph_xpdf = np.arange(min_accuracy, max_accuracy, 0.1)
graph_ypdf = (1 / (std * np.sqrt(2*np.pi))) * np.exp(-((graph_xpdf-mean)**2)/(2*std**2)) 

fig, ax = plt.subplots()

#Plot the histogram
histogram_style = {"facecolor": "lightblue", "edgecolor": "white", "linewidth": 1 }
ax.hist(list_accuracy_data, density=True, **histogram_style)

#Plot the probability distribution function
ax.plot(graph_xpdf, graph_ypdf)

#Plot every data accuracy at the bottom of the graph as vertical lines
plot_style = {"color": "darkblue", "marker": "|", "markersize": 20}
ax.plot(list_accuracy_data, 0*list_accuracy_data, **plot_style)

ax.set_xlabel("Accuracy")
ax.set_ylabel("Density")

plt.show()