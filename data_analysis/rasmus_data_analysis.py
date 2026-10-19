from dotenv import load_dotenv
import time
import os
from supabase import create_client, Client, ClientOptions
import matplotlib.pyplot as plt
import numpy as np


# Load environment variables from .env

load_dotenv()

# Setup the connection to Supabase

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)

# Query: Contains a list variable called "data", which that is a list of dictionaries.
# The index of the list represents the row, the key (dictionary) is the name of the column, and the value (dictionary) is the data point of the column and row

query = (
    supabase.table("test2")
    .select("*")
    .execute()
)

query_distinct_id = (
    supabase.table("distinct_view")
        .select("*")
        .execute()
)

# Create a list of all columns (keys), so one can access the values of the dictionaries
column_list = []

for key in query.data[0]:
    column_list.append(key)

print(column_list)



# Data:
epoch_start = 1
epoch_end = 50
epochs = np.arrange(epoch_start,epoch_end)
validation_accuracies = []


# Confidence intervals (lists of minimum lower and maximum upper bounds as well as standard deviation lower and upper bound)
min_lower_bound = []
max_upper_bound = []
standard_deviation_lower_bound = []
standard_deviation_upper_bound = []

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