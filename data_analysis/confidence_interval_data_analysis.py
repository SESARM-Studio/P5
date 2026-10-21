from dotenv import load_dotenv
import time
import os
from supabase import create_client, Client, ClientOptions
import matplotlib.pyplot as plt
import numpy as np

def sort_dict_by_epoch_no(arg):
    return arg['epoch_no']

# Load environment variables from .env

load_dotenv()

# Setup the connection to Supabase

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)

# Query structure: Contains a list variable called "data", that is a list of dictionaries.
# The index of the list represents the row number, the key (dictionary) is the name of the column, and the value (dictionary) is the data point of the column and row

query_test_training_result= (
    supabase.table("test_training_result")
    .select("*")
    .execute()
)
query_test_training_result.data.sort(key=sort_dict_by_epoch_no)#Sort the data from the query, so that all accuracies from epoch '1' is first and then '2' and so on

query_distinct_training_epoch_no = (
    supabase.table("distinct_test_training_result_epoch_no")
        .select("*")
        .execute()
)

query_distinct_training_id = (
    supabase.table("distinct_test_training_result_id")
        .select("*")
        .execute()
)

# Create a list of all columns (keys), so one can access the values of the dictionaries
column_list = []
for key in query_test_training_result.data[0]:
    column_list.append(key)

# Setup epoch datapoints for plotting:
# We append each number from the query of distinct epoch numbers (DB-column: epoch_no)
# We sort them and use min/max for epoch start and end
epochs = []
for key in query_distinct_training_epoch_no.data:
    epochs.append(key.get(column_list[1]))
epochs.sort()
epochs = np.array(epochs)
epoch_start = min(epochs)
epoch_end = max(epochs)


# We take all the accuracies from the query, and store them in a numpy list
# This enables the np.split function that takes arg1: a list and arg2: a number, and splits the list into a number of parts based on the number given 
all_accuracies = []
for dictionary in query_test_training_result.data:
    all_accuracies.append(dictionary.get("accuracy"))
all_accuracies = np.array(all_accuracies)
validation_accuracies = np.split(all_accuracies, len(epochs))

# From the rows in validation_accuracies, we can calculate:
# Confidence intervals (lists of lower and upper bounds, standard deviation (lower and upper) and mean of validation accuracy at each epoch)
lower_bound = [], upper_bound = [], mean = [], standard_deviation = [], lower_standard_deviation = [], upper_standard_deviation = []
for index in range(0,len(epochs)):
    lower_bound.append(min(validation_accuracies[index]))
    upper_bound.append(max(validation_accuracies[index]))
    
    mean.append(np.mean(validation_accuracies[index]))
    
    standard_deviation.append(np.std(validation_accuracies[index]))
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
plt.plot(epochs, mean, label='Validation accuracy variation', color='black') #Plotting mean of validation accuracy
plt.fill_between(epochs, lower_bound, upper_bound, color='xkcd:light red', alpha=0.2, label='min-max confidence Interval') #Plotting lower and upper bound of validation accuracy
plt.fill_between(epochs, lower_standard_deviation, upper_standard_deviation, color='xkcd:vermillion', alpha=0.2, label='standard deviation confidence Interval')

plt.title('Validation accuracy at each epoch - Confidence Interval')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.xticks(plot_epochs)
plt.legend()
plt.grid(True)
plt.show()



''' Graveyard

# Create matrix of query
matrix = []
for tuple in query.data:
    row = []
    for values in tuple.values():
        row.append(values)
    matrix.append(row)



for id_dict in query_distinct_training_id.data:
    row = []
    for index_dict in query_test_training_result.data:
        temp1 = id_dict.get(column_list[0])
        temp2 = index_dict.get(column_list[0])
        if temp1 == temp2:
            row.append(index_dict.get(column_list[2]))
    validation_accuracies.append(row)
'''
    