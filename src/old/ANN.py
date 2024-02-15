import matplotlib.pyplot as plt
import numpy as np

# Define the network structure
input_layer_size = 3
hidden_layer_size = 4
output_layer_size = 2

# Coordinates for neurons in each layer
input_layer = np.array([[1, i] for i in range(input_layer_size)])
hidden_layer = np.array([[2, i + 0.5*(input_layer_size-hidden_layer_size)] for i in range(hidden_layer_size)])
output_layer = np.array([[3, i + 0.5*(hidden_layer_size-output_layer_size)] for i in range(output_layer_size)])

# Plotting
plt.figure(figsize=(8, 6))

# Plot neurons
plt.scatter(input_layer[:, 0], input_layer[:, 1], s=1000, label='Input Layer', color='skyblue')
plt.scatter(hidden_layer[:, 0], hidden_layer[:, 1], s=1000, label='Hidden Layer', color='lightgreen')
plt.scatter(output_layer[:, 0], output_layer[:, 1], s=1000, label='Output Layer', color='salmon')

# Plot connections
for i_pos in input_layer:
    for h_pos in hidden_layer:
        plt.plot([i_pos[0], h_pos[0]], [i_pos[1], h_pos[1]], 'gray')

for h_pos in hidden_layer:
    for o_pos in output_layer:
        plt.plot([h_pos[0], o_pos[0]], [h_pos[1], o_pos[1]], 'gray')

# plt.text(input_layer[0, 0], input_layer[0, 1] + 0.5, 'Input Layer', horizontalalignment='center')
# plt.text(hidden_layer[0, 0], hidden_layer[0, 1] + 0.5*(hidden_layer_size+1), 'Hidden Layer', horizontalalignment='center')
# plt.text(output_layer[0, 0], output_layer[0, 1] + 0.5*(output_layer_size+1), 'Output Layer', horizontalalignment='center')

plt.axis('off')
plt.show()
