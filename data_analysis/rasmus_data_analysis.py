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

# Query

query = (
    supabase.table("test")
    .select("*")
    .execute()
)




print(query)
print(query.data)
for i in query.data:
    print(i['number'])


# Example data: Monthly average temperatures in a US city (°F)
epoch_start = 1
epoch_end = 600
epoch = np.arrange(epoch_start,epoch_end)
validation_accuracy = []


# Simulated confidence intervals (lower and upper bounds)
lower_bound = avg_temps - 5
upper_bound = avg_temps + 5

plt.figure(figsize=(10, 6))
plt.plot(months, avg_temps, label='Average Temperature', color='blue')
plt.fill_between(months, lower_bound, upper_bound, color='blue', alpha=0.2, label='95% Confidence Interval')

plt.title('Monthly Average Temperature with Confidence Interval')
plt.xlabel('Month')
plt.ylabel('Temperature (°F)')
plt.xticks(months)
plt.legend()
plt.grid(True)
plt.show()