import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
import pandas as pd
import numpy as np
import ast
import math

def _parse_predictor_data_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    outputs = {'HP': [], 'Accuracy': [], 'Error': [], 'Latency': [], 'Obj': []}
    for line in lines:
        line = line.strip()
        if '-' in line:
            break
        result = ast.literal_eval(line)
        outputs['HP'].append(result['HP'])
        outputs['Accuracy'].append(result['Accuracy'])
        outputs['Error'].append(1-result['Accuracy'])
        if 'Latency: ' in list(result.keys()):
            outputs['Latency'].append(result['Latency: '])
            outputs['Obj'].append(result['Accuracy']*result['Accuracy']/result['Latency: '])
        else:
            outputs['Latency'].append(result['Latency'])
            outputs['Obj'].append(result['Accuracy']*result['Accuracy']/result['Latency'])

    return outputs

pathes = [
    './meta_results/TS-2023-11-26_00-50-19.txt',
    './meta_results/SA-2023-12-02_01-46-58.txt',
    './meta_results/ILS-2023-12-03_23-14-05.txt',
] 

outputs = []

for path in pathes:
    outputs.append(_parse_predictor_data_file(path))

TS_output = outputs[0]
objs = TS_output['Obj']
best_obj = []
best_acc = []
best_lat = []
indexes = []
current_obj = 0
for index, obj in enumerate(objs):
    if obj > current_obj:
        best_obj.append(obj)
        best_acc.append(TS_output['Accuracy'][index])
        best_lat.append(TS_output['Latency'][index])
        current_obj = obj
        # indexes.append(math.floor(index/5))
        indexes.append(index)

categories = ['Default HP', 'TS', 'SA', 'ILS']
X_axis = np.arange(len(categories)) 
y = {'TS': [best_obj[-1], best_acc[-1], best_lat[-1]], 'SA':outputs[1], 'ILS':outputs[2]}
# Set up figure and axis
fig, axes = plt.subplots()

bar_width = 0.28  # Width of the bars

colors = ['orangered', 'navy', 'gold']

print(y['TS'][0], y['TS'][1], y['TS'][2])
print(y['SA']['Obj'][-1], y['SA']['Accuracy'][-1], y['SA']['Latency'][-1])
print(y['ILS']['Obj'][-1], y['ILS']['Accuracy'][-1], y['ILS']['Latency'][-1])

bl = [0.6833999752998352*0.6833999752998352/1.7434606552124023, 0.6833999752998352, 1.7434606552124023]

# Plot the first set of bars on the left y-axis
axes.bar(X_axis[0] - 0.3,  bl[0], bar_width, color=colors[0])
axes.bar(X_axis[0]      , bl[1], bar_width, color=colors[1])
axes.bar(X_axis[0] + 0.3, bl[2], bar_width, color=colors[2])

axes.bar(X_axis[1] - 0.3, y['TS'][0], bar_width, color=colors[0])
axes.bar(X_axis[1]      , y['TS'][1], bar_width, color=colors[1])
axes.bar(X_axis[1] + 0.3, y['TS'][2], bar_width, color=colors[2])

axes.bar(X_axis[2] - 0.3, y['SA']['Obj'     ][-1], bar_width, color=colors[0])
axes.bar(X_axis[2]      , y['SA']['Accuracy'][-1], bar_width, color=colors[1])
axes.bar(X_axis[2] + 0.3, y['SA']['Latency' ][-1], bar_width, color=colors[2])

axes.bar(X_axis[3] - 0.3, y['ILS']['Obj'     ][-1], bar_width, color=colors[0])
axes.bar(X_axis[3]      , y['ILS']['Accuracy'][-1], bar_width, color=colors[1])
axes.bar(X_axis[3] + 0.3, y['ILS']['Latency' ][-1], bar_width, color=colors[2])

# axes.bar([0], bar_width, color='orangered')

axes.set_xlabel('Metaheurisitics')
axes.set_ylabel('Accuracy (%) / Latency (s) / Objective Value')
axes.tick_params('y')

# Title and legend
plt.xticks(X_axis, categories)
plt.title('Best Results for Metaheurisitics')
plt.grid()
axes.set_axisbelow(True)
fig.tight_layout()
axes.legend(['Objective Value', 'Accuracy', 'Evaluation Latency'])

# Show the plot
# plt.show()

print((bl[2]-1.459892749786377)/bl[2]*100)
print((0.7199000120162964-bl[1])*100)

df =  {'conv_1': {'activation': 'relu',   'kernel_size': 3, 'padding': 'valid', 'filters': 32},  'pool_1': {'pool_size': 2, 'strides': 2, 'padding': 'valid'}, 'conv_2': {'activation': 'relu',   'kernel_size': 3, 'padding': 'valid', 'filters': 64}, 'pool_2': {'pool_size': 2, 'strides': 2, 'padding': 'valid'}, 'conv_3': {'activation': 'relu',    'kernel_size': 3, 'padding': 'valid', 'filters': 64}}
ts =  {'conv_1': {'activation': 'relu',   'kernel_size': 3, 'padding': 'valid', 'filters': 100}, 'pool_1': {'pool_size': 3, 'strides': 2, 'padding': 'valid'}, 'conv_2': {'activation': 'linear', 'kernel_size': 5, 'padding': 'same' , 'filters': 20}, 'pool_2': {'pool_size': 2, 'strides': 2, 'padding': 'valid'}, 'conv_3': {'activation': 'relu'   , 'kernel_size': 3, 'padding': 'same' , 'filters': 64}}
sa =  {'conv_1': {'activation': 'relu',   'kernel_size': 3, 'padding': 'same',  'filters': 80  },'pool_1': {'pool_size': 3, 'strides': 3, 'padding': 'same' }, 'conv_2': {'activation': 'linear', 'kernel_size': 5, 'padding': 'valid', 'filters': 60}, 'pool_2': {'pool_size': 4, 'strides': 3, 'padding': 'same' }, 'conv_3': {'activation': 'sigmoid', 'kernel_size': 2, 'padding': 'same' , 'filters': 60}}
ils = {'conv_1': {'activation': 'linear', 'kernel_size': 2, 'padding': 'same', 'filters': 100 }, 'pool_1': {'pool_size': 4, 'strides': 2, 'padding': 'same'} , 'conv_2': {'activation': 'linear', 'kernel_size': 3, 'padding': 'valid', 'filters': 60}, 'pool_2': {'pool_size': 3, 'strides': 3, 'padding': 'same' }, 'conv_3': {'activation': 'relu'   , 'kernel_size': 3, 'padding': 'valid', 'filters': 60}}

# Higher filters in the first conv1 in all compared to df
# Each converged to linear in the conv2d
# Ts and df share the most in common with 11/18 same parameters
# sa and df are the most different only 3 in common, and 6 for 
# sa and ILS had the most similar parameters when comparing the three to each other

common_values = {}

for key in ts:
    if key in sa:
        for subkey in ts[key]:
            if subkey in sa[key] and ts[key][subkey] == sa[key][subkey] and ts[key][subkey] == ils[key][subkey] and sa[key][subkey] == ils[key][subkey]  and sa[key][subkey] == df[key][subkey]  and df[key][subkey] == ils[key][subkey]  and df[key][subkey] == ts[key][subkey]:
                common_values[(key, subkey)] = (ts[key][subkey])

print("Common values, all three and df:", common_values)
print(len(common_values))

common_values = {}

for key in ts:
    if key in sa:
        for subkey in ts[key]:
            if subkey in sa[key] and ts[key][subkey] == sa[key][subkey] and ts[key][subkey] == ils[key][subkey] and sa[key][subkey] == ils[key][subkey]:
                common_values[(key, subkey)] = (ts[key][subkey])

print("Common values, all three:", common_values)
print(len(common_values))

common_values = {}

for key in ts:
    if key in sa:
        for subkey in ts[key]:
            if subkey in sa[key] and ts[key][subkey] == sa[key][subkey]:
                common_values[(key, subkey)] = (ts[key][subkey])

print("Common values, ts and sa:", common_values)
print(len(common_values))

common_values = {}

for key in ts:
    if key in ils:
        for subkey in ts[key]:
            if subkey in ils[key] and ts[key][subkey] == ils[key][subkey]:
                common_values[(key, subkey)] = (ts[key][subkey])

print("Common values, ts and ils:", common_values)
print(len(common_values))

common_values = {}

for key in ils:
    if key in sa:
        for subkey in ils[key]:
            if subkey in sa[key] and sa[key][subkey] == ils[key][subkey]:
                common_values[(key, subkey)] = (ils[key][subkey])

print("Common values, sa and ils:", common_values)
print(len(common_values))

common_values = {}

for key in df:
    if key in ts:
        for subkey in df[key]:
            if subkey in ts[key] and df[key][subkey] == ts[key][subkey]:
                common_values[(key, subkey)] = (ts[key][subkey])

print("Common values, df and ts:", common_values)
print(len(common_values))
common_values = {}

for key in df:
    if key in sa:
        for subkey in df[key]:
            if subkey in sa[key] and sa[key][subkey] == df[key][subkey]:
                common_values[(key, subkey)] = (sa[key][subkey])

print("Common values, df and sa:", common_values)
print(len(common_values))

common_values = {}

for key in ils:
    if key in df:
        for subkey in ils[key]:
            if subkey in df[key] and df[key][subkey] == ils[key][subkey]:
                common_values[(key, subkey)] = (ils[key][subkey])

print("Common values, df and ils:", common_values)
print(len(common_values))

##########################################

# Set up figure and axis
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

# TS_output = outputs[-1]
# objs = TS_output['Obj']
# best_obj = []
# best_acc = []
# best_lat = []
# indexes = []
# current_obj = 0
# for index, obj in enumerate(objs):
#     if obj > current_obj:
#         best_obj.append(obj)
#         best_acc.append(TS_output['Accuracy'][index])
#         best_lat.append(TS_output['Latency'][index])
#         current_obj = obj
#         # indexes.append(math.floor(index/5))
#         indexes.append(index)

    




# best_index = np.argmax(outputs[0]['Obj'])
# print(best_index, y[best_index])
# x = range(len(y))


# ax1.set_title('Tabu Search')
# # Plot the first set of bars on the left y-axis
# ax1.plot(indexes, best_obj, color='green', marker = 'o', label='Obj')
# ax1.set_xlabel('# of evaluated CNNs')
# ax1.set_ylabel('Objective Value')
# ax1.tick_params('y')
# ax1.grid()

# ax2.plot(indexes, best_acc, color='navy', marker = 'o', label='Accuracy')
# ax2.set_xlabel('# of evaluated CNNs')
# ax2.set_ylabel('Accuracy (%)')
# ax2.tick_params('y')
# ax2.grid()

# ax3.plot(indexes, best_lat, color='orangered', marker = 'o', label='Latency')
# ax3.set_xlabel('# of evaluated CNNs')
# ax3.set_ylabel('Eval Latency (s)')
# ax3.tick_params('y')
# ax3.grid()

# print(best_acc)

# # Title and legend
# fig.tight_layout()
# # Show the plot
# plt.show()


##########################################
# fig, axes = plt.subplots()

# y_set = []
# x_set = []

# for k in range(len(outputs)):
#     y = outputs[k]['Error']
#     x = outputs[k]['Latency']

#     indexes_to_del = []
#     for i in range(len(y)):
#         if i in indexes_to_del:
#             continue
#         for j in range(len(y)):
#             if( y[j] > y[i]) and (x[j] > x[i]) and (j not in indexes_to_del):
#                 indexes_to_del.append(j)

#     print(len(y), len(indexes_to_del))
#     y = [y[i] for i in range(len(y)) if i not in indexes_to_del]
#     x = [x[i] for i in range(len(x)) if i not in indexes_to_del]

#     idx   = np.argsort(x)
#     x = np.array(x)[idx]
#     y = np.array(y)[idx]

#     x_set.append(x)
#     y_set.append(y)

# # Plot the first set of bars on the left y-axis
# axes.plot(x_set[0], y_set[0], color='navy', marker='o', label='TS')
# axes.plot(x_set[1], y_set[1], color='magenta', marker='o', label='SA')
# axes.plot(x_set[2][1:], y_set[2][1:], color='green', marker='o', label='ILS')
# axes.set_xlabel('Evalutaion Latency (s)')
# axes.set_ylabel('Error (%)')
# axes.tick_params('y')

# # Title and legend
# plt.title('Error vs Evalutaion Latency')

# fig.tight_layout()
# axes.legend()

# # Show the plot
# plt.show()

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