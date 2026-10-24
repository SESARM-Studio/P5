import matplotlib.pyplot as plt
import numpy as np
from query_functions import *

def sort_dict_by_epoch_no(arg):
    return arg['epoch_no']

#Establish supabase connection using our connection function from query_functions.py
supabase = create_supabase_connection()

# Retrieve the data using our query functions from query_functions.py
data_test_training_result = query_test_training_result(supabase)
data_test_training_result.data.sort(key=sort_dict_by_epoch_no)#Sort the data from the query, so that all accuracies from epoch '1' is first and then '2' and so on
data_distinct_training_epoch_no = query_distinct_training_epoch_no(supabase)

# Create a list of all columns (keys), so one can access the values of the dictionaries
column_list = []
for key in data_test_training_result.data[0]:
    column_list.append(key)

# Setup epoch datapoints for plotting:
# We append each number from the query of distinct epoch numbers (DB-column: epoch_no)
# We sort them and use min/max for epoch start and end
epochs = []
for key in data_distinct_training_epoch_no.data:
    epochs.append(key.get(column_list[1]))
epochs.sort()
epochs = np.array(epochs)
epoch_start = min(epochs)
epoch_end = max(epochs)


# We take all the accuracies from the query, and store them in a numpy list
# This enables the np.split function that takes arg1: a list and arg2: a number, and splits the list into a number of parts based on the number given 
all_accuracies = []
for dictionary in data_test_training_result.data:
    all_accuracies.append(dictionary.get("accuracy"))
all_accuracies = np.array(all_accuracies)
testing_accuracies = np.split(all_accuracies, len(epochs))

# From the rows in testing_accuracies, we can calculate:
# Confidence intervals (lists of lower and upper bounds, standard deviation (lower and upper) and mean of testing accuracy at each epoch)
lower_bound = []
upper_bound = []
mean = []
standard_deviation = []
lower_standard_deviation = []
upper_standard_deviation = []
for index in range(0,len(epochs)):
    lower_bound.append(min(testing_accuracies[index]))
    upper_bound.append(max(testing_accuracies[index]))
    
    mean.append(np.mean(testing_accuracies[index]))
    
    standard_deviation.append(np.std(testing_accuracies[index]))
    lower_standard_deviation.append(mean[index]-standard_deviation[index])
    upper_standard_deviation.append(mean[index]+standard_deviation[index])
    

# Setup the epochs that will be on the x-axis of the image for better visualisation (in this case epoch 1, 10, 20, ... , last epoch that mod 10 = 0)
filter_array = []
for epoch_no in epochs:
    if epoch_no % 10 == 0:
        filter_array.append(True)
    else:
        filter_array.append(False)
plot_epochs = epochs[filter_array]
plot_epochs = np.concatenate(([1],plot_epochs))

# Plot the figure
plt.figure(figsize=(10, 6)) #Size of figure
plt.plot(epochs, mean, label='Testing accuracy variation', color='black') #Plotting mean of testing accuracy
plt.fill_between(epochs, lower_bound, upper_bound, color='xkcd:light red', alpha=0.2, label='min-max confidence Interval') #Plotting lower and upper bound of testing accuracy
plt.fill_between(epochs, lower_standard_deviation, upper_standard_deviation, color='xkcd:vermillion', alpha=0.2, label='standard deviation confidence Interval')

plt.title('Testing accuracy at each epoch - Confidence Interval')
plt.xlabel('Epoch')
plt.ylabel('testing Accuracy')
plt.xticks(plot_epochs)
plt.legend()
plt.grid(True)
plt.show()


