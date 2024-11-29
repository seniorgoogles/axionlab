import matplotlib.pyplot as plt
import json


results_file = "/home/mmecik/repositories/synapselab/results.json"

with open(results_file, "r") as f:
    results = json.load(f)
    #results = results["0_evaluate_quantization"]
    
    experiment = ["float", "Quant Act. 8-Bit Bias 16-Bit", "0_evaluate_quantization", "1_retrain_after_quant", "2_evaluate_pruning", "3_retrain_after_pruning"]
    
    memory_usage = []
        
    layers = ["dense1", "dense2", "dense3", "dense4", "dense5"]
    
    layer_bit_width = {}
    layer_bit_width["float"] = dict() 
    layer_bit_width["float"] = {
        "dense1": 32,
        "dense2": 32,
        "dense3": 32,
        "dense4": 32,
        "dense5": 32,
    }

    accuracy = []
    accuracy.append(75.4)
    accuracy.append(74.7)
    
    print("\n")
    for key in experiment:
        if key in results:
            print(key)
            print(results[key]["experiments"]["exp"].keys())
            
            accuracy.append(results[key]["experiments"]["exp"]["accuracy"])
            layer_bit_width[key] = dict() 
            
            if "layers" in results[key]["experiments"]["exp"]:
                for layer in results[key]["experiments"]["exp"]["layers"].keys():
                    layer_info = results[key]["experiments"]["exp"]["layers"][layer]["bitwidth"]
                    
                    layer_bit_width[key][layer] = layer_info             
            print("\n")

    #accuracy = [round(acc, 2) for acc in accuracy]
    #
    #experiment = ["float", "Act. Bias 16-Bit", "Quant", "Retrain", "Pruning", "Retrain"]

    # Plot experiment vs accuracy
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(experiment, accuracy, marker='o')  # Add markers for better visualization
    plt.xlabel('Experiment')
    plt.ylabel('Accuracy')

    # Rotate x labels
    plt.xticks(rotation=90)

    # Add accuracy values as markers on the plot
    for i, value in enumerate(accuracy):
        plt.annotate(f"{value}%",  # Add percentage symbol here
                    (experiment[i], accuracy[i]),  # Coordinates
                    textcoords="offset points",
                    xytext=(0, 10),  # Offset text above the marker
                    ha='center')

    plt.title('Experiment vs Accuracy')
    plt.tight_layout()  # Adjust layout to fit everything
    plt.savefig("experiment_vs_accuracy.png")  # Save the plot
    plt.show()
    
    
# Evaluate models 

# Evaluate the float model
# Evaluate the quantized model
# Evaluate the pruned model