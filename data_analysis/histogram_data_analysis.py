import os
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
url = os.getenv("SUPABASE_URL")
api_key = os.getenv("SUPABASE_PUBLISHABLE_KEY")

supabase = create_client(url, api_key)

last_epoch = 130
result = supabase.table("test_training_result").select("accuracy").eq("epoch_no", last_epoch).execute()

accuracy_data = np.array([row["accuracy"] for row in result.data if "accuracy" in row])

min_accuracy = min(accuracy_data)
max_accuracy = max(accuracy_data)
mean = np.mean(accuracy_data)
std = np.std(accuracy_data)

print("Minimum accuracy:", min_accuracy)
print("Maximum accuracy:", max_accuracy)
print("Mean:", round(mean,2))
print("std:", round(std,2))

#graph_xpdf = np.arange(np.floor(min_accuracy), np.ceil(max_accuracy), 0.1)
graph_xpdf = np.arange(min_accuracy, max_accuracy, 0.1)
graph_pdf = (1 / (std * np.sqrt(2*np.pi))) * np.exp(-((graph_xpdf-mean)**2)/(2*std**2)) 

fig, ax = plt.subplots()

histogram_style = {"facecolor": "lightblue", "edgecolor": "white", "linewidth": 1 }
ax.hist(accuracy_data, density=True, **histogram_style)

ax.plot(graph_xpdf, graph_pdf)

plot_style = {"color": "darkblue", "marker": "|", "markersize": 20}
ax.plot(accuracy_data, 0*accuracy_data, **plot_style)

ax.set_ylabel("Accuracy")
ax.set_xlabel("Density")

plt.show()




