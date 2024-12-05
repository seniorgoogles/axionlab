import re
import matplotlib.pyplot as plt

def extract_data_from_file(file_path):
    learning_rates = []
    last_accuracies = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Extract the learning rate from lines containing "Learning Rates"
        lr_match = re.search(r'Learning Rates: \[([0-9.]+)', line)
        if lr_match:
            learning_rates.append(float(lr_match.group(1)))

        # Extract the last accuracy from lines containing "Last Accuracy"
        acc_match = re.search(r'Last Accuracy: ([0-9.]+)%', line)
        if acc_match:
            last_accuracies.append(float(acc_match.group(1)))

    return learning_rates, last_accuracies

def plot_learning_rate_vs_accuracy(learning_rates, last_accuracies):
    # Find the learning rate with the highest accuracy
    max_accuracy = max(last_accuracies)
    max_accuracy_index = last_accuracies.index(max_accuracy)
    max_accuracy_lr = learning_rates[max_accuracy_index]

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, last_accuracies, marker='o', linestyle='-', label='Accuracy vs Learning Rate')
    plt.scatter(max_accuracy_lr, max_accuracy, color='red', 
                label=f'Max Accuracy: {max_accuracy:.2f}% (LR={max_accuracy_lr:.6f})', zorder=5)
    plt.xlabel("Learning Rate")
    plt.ylabel("Last Accuracy (%)")
    plt.title("First Learning Rate vs Last Accuracy")
    plt.grid(True)
    plt.legend()
    plt.show()

# Main execution
file_path = '/home/fry/new_repo/synapselab/train/report.txt'  # Replace with your file path
learning_rates, last_accuracies = extract_data_from_file(file_path)
plot_learning_rate_vs_accuracy(learning_rates, last_accuracies)
