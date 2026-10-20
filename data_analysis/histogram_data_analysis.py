import os
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
url = os.getenv("SUPABASE_URL")
api_key = os.getenv("SUPABASE_PUBLISHABLE_KEY")

supabase = create_client(url, api_key)

# Query the database
last_epoch = 130 # Will be changed when combined with Ramo code
result = supabase.table("test_training_result").select("accuracy").eq("epoch_no", last_epoch).execute()

# result.data contains a list of each row as a dictionary, this makes it into a numpy array of just the accuracies.
accuracy_data = np.array([row["accuracy"] for row in result.data if "accuracy" in row])

min_accuracy = min(accuracy_data)
max_accuracy = max(accuracy_data)
mean = np.mean(accuracy_data)
std = np.std(accuracy_data)

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
ax.hist(accuracy_data, density=True, **histogram_style)

#Plot the probability distribution function
ax.plot(graph_xpdf, graph_ypdf)

#Plot every data accuracy at the bottom of the graph as vertical lines
plot_style = {"color": "darkblue", "marker": "|", "markersize": 20}
ax.plot(accuracy_data, 0*accuracy_data, **plot_style)

ax.set_xlabel("Accuracy")
ax.set_ylabel("Density")

plt.show()