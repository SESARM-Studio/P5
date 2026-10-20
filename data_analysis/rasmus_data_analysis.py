from dotenv import load_dotenv
import time
import os
from supabase import create_client, Client, ClientOptions
import matplotlib.pyplot as plt
import numpy as np


def sort_dict_by_training_id(arg):
    return arg['training_id']

# Load environment variables from .env

load_dotenv()

# Setup the connection to Supabase

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)

# Query: Contains a list variable called "data", which that is a list of dictionaries.
# The index of the list represents the row, the key (dictionary) is the name of the column, and the value (dictionary) is the data point of the column and row

query_test_training_result= (
    supabase.table("test_training_result")
    .select("*")
    .execute()
)
query_test_training_result.data.sort(key=sort_dict_by_training_id)

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
query_distinct_training_id.data.sort(key=sort_dict_by_training_id)

# Create a list of all columns (keys), so one can access the values of the dictionaries
column_list = []
for key in query_test_training_result.data[0]:
    column_list.append(key)


epochs = []
for key in query_distinct_training_epoch_no.data:
    epochs.append(key.get(column_list[1]))
epochs.sort()


# Data:
epoch_start = min(epochs)
epoch_end = max(epochs)
# epochs = np.arange(epoch_start,epoch_end)
validation_accuracies = []


# Calculate the validation accuracy for each epoch

for id_dict in query_distinct_training_id.data:
    row = []
    for index_dict in query_test_training_result.data:
        temp1 = id_dict.get(column_list[0])
        temp2 = index_dict.get(column_list[0])
        if temp1 == temp2:
            row.append(index_dict.get(column_list[2]))
    validation_accuracies.append(row)

# Confidence intervals (lists of minimum lower and maximum upper bounds as well as standard deviation lower and upper bound)
min_lower_bound = []
for i in range(0,len(query_distinct_training_id.data)):
    for j in range(0,len(epochs)):
        print(validation_accuracies[i][j])

max_upper_bound = []
standard_deviation_lower_bound = []
standard_deviation_upper_bound = []

for i in range(0,len(query_distinct_training_id.data)):
    for j in range(0,len(epochs)):
        print(validation_accuracies[i][j])


# Plot the figure
plt.figure(figsize=(10, 6))
plt.plot(epochs, validation_accuracies, label='Validation accuracy variation', color='green')
plt.fill_between(epochs, min_lower_bound, max_upper_bound, color='red', alpha=0.2, label='min-max Confidence Interval')

plt.title('Monthly Average Temperature with Confidence Interval')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.xticks(epochs)
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



'''