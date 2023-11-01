
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Function to convert a string representation of a Python dictionary to an actual dictionary
def str_to_dict(s):
    try:
        return eval(s)
    except Exception as e:
        return None

# Initialize empty list to store parsed data
parsed_data = []

# Regex patterns for extracting the relevant information
model_pattern = re.compile(r"Model: (.*?),")
metrics_pattern = re.compile(r"Metrics: (.*?), Params")
params_pattern = re.compile(r"Params: (.*?)$")

# Read and parse each line in the log file
file_path = '/mnt/data/model.log'
with open(file_path, 'r') as f:
    for line in f:
        model_match = model_pattern.search(line)
        metrics_match = metrics_pattern.search(line)
        params_match = params_pattern.search(line)
        
        if model_match is None or metrics_match is None or params_match is None:
            continue
        
        model = model_match.group(1)
        metrics_str = metrics_match.group(1)
        params_str = params_match.group(1)
        
        metrics = str_to_dict(metrics_str)
        params = str_to_dict(params_str)
        
        if metrics is None or params is None:
            continue
        
        entry = {'Model': model}
        entry.update(metrics)
        entry.update(params)
        parsed_data.append(entry)

# Convert to a DataFrame for easier analysis
df = pd.DataFrame(parsed_data)

# Parameters to consider for visualization
params_to_plot = ['smell_thr_values', 'smell_predict_hrs_values', 'look_back_hrs_values', 'add_inter_values', 'test_size_values', 'train_size_values']

# List to store file paths of saved plots
saved_plots = []

for param in params_to_plot:
    if df[param].dtype == 'bool':
        continue

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x=param, y='test_f1_score', hue='Model', marker='o')
    plt.title(f'Impact of {param} on F1 Score')
    plt.xlabel(param)
    plt.ylabel('F1 Score')
    
    plot_filename = f"/mnt/data/{param}_impact_on_f1_score.png"
    plt.savefig(plot_filename)
    saved_plots.append(plot_filename)

    plt.close()

plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='add_inter_values', y='test_f1_score', hue='Model')
plt.title(f'Impact of add_inter_values on F1 Score')
plt.xlabel('add_inter_values')
plt.ylabel('F1 Score')

plot_filename = f"/mnt/data/add_inter_values_impact_on_f1_score.png"
plt.savefig(plot_filename)
saved_plots.append(plot_filename)

plt.close()

print("Visualizations saved:", saved_plots)
