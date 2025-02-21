import matplotlib.pyplot as plt

# Data
experiments = ['Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5', 'Exp6', 'Exp7', 'Exp8', 'Exp9', 'Exp10']
accuracies = [76.53, 76.14, 75.88, 75.30, 71.13, 73.73, 73.68, 75.09, 74.81, 75.08]
total_weights = [426304, 221504, 119104, 67904, 26944, 37184, 10560, 10400, 2025, 2425]

plt.figure(figsize=(10, 6))
plt.plot(experiments, accuracies, marker='o', linestyle='-', color='blue', label='Model Accuracy')

# Draw the base accuracy line
base_accuracy = 75.33
plt.axhline(y=base_accuracy, color='red', linestyle='--', label=f'Base Accuracy ({base_accuracy}% and 18752 weights)')

# Annotate total weights on the plot
for i, tw in enumerate(total_weights):
    plt.text(i, accuracies[i] + 0.1, f'{tw} ({accuracies[i]}%)', ha='center', fontsize=9)

plt.xlabel("Experiment")
plt.ylabel("Model Accuracy (%)")
plt.title("Model Accuracy Across Experiments with Base Accuracy")
plt.legend()
plt.grid(True, linestyle="--", lw=0.5)
plt.show()