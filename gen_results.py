import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
import pandas as pd
import numpy as np
import ast

def _parse_predictor_data_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    outputs = {'HP': [], 'Accuracy': [], 'Error': [], 'Latency': [], 'Obj': []}
    for line in lines:
        line = line.strip()
        result = ast.literal_eval(line)
        outputs['HP'].append(result['HP'])
        outputs['Accuracy'].append(result['Accuracy'])
        outputs['Error'].append(1-result['Accuracy'])
        outputs['Latency'].append(result['Latency: '])
        if 'Latency: ' in list(result.keys()):
            outputs['Obj'].append(result['Accuracy']*result['Accuracy']/result['Latency: '])
        else:
            outputs['Obj'].append(result['Accuracy']*result['Accuracy']/result['Latency'])

    return outputs

pathes = [
    './meta_results/TS-2023-11-26_00-50-19.txt',
    './meta_results/SA-2023-11-25_08-09-55.txt',
    './meta_results/ILS-2023-11-24_00-29-11.txt'
] 

outputs = []

for path in pathes:
    outputs.append(_parse_predictor_data_file(path))

# Set up figure and axis
fig, axes = plt.subplots()

y = outputs[0]['Accuracy']
best_index = np.argmax(outputs[0]['Obj'])
# print(best_index, y[best_index])
x = range(len(y))

# Plot the first set of bars on the left y-axis
axes.plot(x, y, color='navy', label='TS')
axes.plot(best_index, y[best_index], '*r')
axes.set_xlabel('Iterations')
axes.set_ylabel('Accuracy')
axes.tick_params('y')

# Title and legend
plt.title('Accuracy over time')

fig.tight_layout()
axes.legend()

# Show the plot
plt.show()

##########################################
fig, axes = plt.subplots()

y_set = []
x_set = []

for k in range(len(outputs)):
    y = outputs[k]['Error']
    x = outputs[k]['Latency']

    indexes_to_del = []
    for i in range(len(y)):
        if i in indexes_to_del:
            continue
        for j in range(len(y)):
            if( y[j] > y[i]) and (x[j] > x[i]) and (j not in indexes_to_del):
                indexes_to_del.append(j)

    print(len(y), len(indexes_to_del))
    y = [y[i] for i in range(len(y)) if i not in indexes_to_del]
    x = [x[i] for i in range(len(x)) if i not in indexes_to_del]

    idx   = np.argsort(x)
    x = np.array(x)[idx]
    y = np.array(y)[idx]

    x_set.append(x)
    y_set.append(y)

# Plot the first set of bars on the left y-axis
axes.plot(x_set[0], y_set[0], color='navy', marker='o', label='TS')
axes.plot(x_set[1], y_set[1], color='magenta', marker='o', label='SA')
axes.plot(x_set[2], y_set[2], color='gold', marker='o', label='ILS')
axes.set_xlabel('Latency')
axes.set_ylabel('Error')
axes.tick_params('y')

# Title and legend
plt.title('Error vs Evalutaion Latency')

fig.tight_layout()
axes.legend()

# Show the plot
plt.show()

##########################################
# # Set up figure and axis
# fig, axes = plt.subplots()

# # Plot the first set of bars on the left y-axis
# axes.plot(range(len(outputs[0]['Accuracy'])), outputs[0]['Accuracy'], label='TS')
# axes.plot(range(len(outputs[1]['Accuracy'])), outputs[1]['Accuracy'], label='SA')
# axes.plot(range(len(outputs[2]['Accuracy'])), outputs[2]['Accuracy'], label='ILS')
# axes.set_xlabel('Iterations')
# axes.set_ylabel('Accuracy')
# axes.tick_params('y')

# # Title and legend
# plt.title('Accuracy over time')

# fig.tight_layout()
# axes.legend()

# # Show the plot
# plt.show()