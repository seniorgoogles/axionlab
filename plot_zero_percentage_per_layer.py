import matplotlib.pyplot as plt
import numpy as np

# Data
layers = ['dense1', 'dense2', 'dense3', 'dense4', 'dense5']
experiment1 = [0.0, 0.0, 0.0, 0.0, 0.0]
experiment2 = [78.02734375, 89.6484375, 83.2275390625, 58.154296875, 56.25]
experiment3 = [78.02734375, 89.70947265625, 83.1787109375, 96.8994140625, 55.93749999999999]


# Plotting bar diagrams
x = np.arange(len(layers))
width = 0.25

plt.bar(x - width, experiment1, width, label='Float Model')
plt.bar(x, experiment2, width, label='Quant and Retrain')
plt.bar(x + width, experiment3, width, label='Pruning and Retrain')

for i in range(len(layers)):
    plt.text(x[i] - width, experiment1[i] + 1, f"{experiment1[i]:.2f}", ha='center', va='bottom', rotation=45)
    plt.text(x[i], experiment2[i] + 1, f"{experiment2[i]:.2f}", ha='center', va='bottom', rotation=45)
    plt.text(x[i] + width, experiment3[i] + 1, f"{experiment3[i]:.2f}", ha='center', va='bottom', rotation=45)


# Customizing the plot
plt.xlabel('Layers')
plt.ylabel('Percentage of Zero Values')
plt.title('Percentage of Zero Values per Layer')
plt.xticks(x, layers)
plt.legend()
plt.tight_layout()

# Show plot
plt.savefig("percentage_zero_values.png")
plt.close()

word_size_float = [32, 32, 32, 32, 32]
word_size_quant = [4, 3, 3, 3, 4]

total_params = [2048, 8192, 4096, 4096, 320]
float_zero_vals = [0.0, 0.0, 0.0, 0.0, 0.0]
quant_zero_vals = [1598, 7344, 3409, 2382, 180]
prune_zero_vals = [1598, 7349, 3407, 3969, 179]

# Calc kilobytes per layer for float model
kilobytes_float = [((params * bits)/ 1024) for params, bits in zip(total_params, word_size_float)]
kilobytes_total_params_quant = [(((params-zero_vals) * bits)/ 1024) for params, zero_vals, bits in zip(total_params, quant_zero_vals, word_size_quant)]
kilobytes_total_params_prune = [(((params-zero_vals) * bits)/ 1024) for params, zero_vals, bits in zip(total_params, prune_zero_vals, word_size_quant)]

# Plot Kilobytes per layer
layers = ['dense1', 'dense2', 'dense3', 'dense4', 'dense5']
print(total_params)
print(kilobytes_float)
print(kilobytes_total_params_quant)
print(kilobytes_total_params_prune)

# Plotting with values displayed on bars
plt.bar(x - width, kilobytes_float, width, label='Float')
plt.bar(x, kilobytes_total_params_quant, width, label='Quantization and Retraining')
plt.bar(x + width, kilobytes_total_params_prune, width, label='Pruning and Retraining')

# Adding values on top of the bars
# Adding rotated values on top of the bars

for i in range(len(layers)):
    plt.text(x[i] - width, kilobytes_float[i] + 1, f"{kilobytes_float[i]:.2f}", ha='center', va='bottom', rotation=45)
    plt.text(x[i], kilobytes_total_params_quant[i] + 1, f"{kilobytes_total_params_quant[i]:.2f}", ha='center', va='bottom', rotation=45)
    plt.text(x[i] + width, kilobytes_total_params_prune[i] + 1, f"{kilobytes_total_params_prune[i]:.2f}", ha='center', va='bottom', rotation=45)

# Customizing the plot
plt.xlabel('Layers')
plt.ylabel('Kilobytes')
plt.title('Kilobytes per Layer for Different Methods')
plt.xticks(x, layers)
plt.legend()
plt.tight_layout()

# Show plot
plt.savefig("kb_per_layer.png")


