import json
import shutil
from includes_ml2 import *
import numpy as np
import matplotlib.pyplot as plt

def plot_memory_size_vs_accuracy_drop_allowed(results, model_name, save_dir, steps=25):
    
    # Go through all trained models
    models_unsorted = [float(key) for key in results.keys()]    
    models = [str(item) for item in sorted(models_unsorted)]
    
    sparsed_size_kb = []
    sparsed_quant_size_kb = []
    max_size_kb = []
    
    nn_overview = dict()

    for model in models:
        layer_names = results[model]["quant"].keys()
        
        nn_overview[model] = dict()

        for layer in layer_names:
            word_width = results[model]["quant"][layer]
            nn_overview[model][layer] = dict()
            nn_overview[model][layer]["word_width"] = word_width

        for layer in layer_names:
            params = results[model]["sparsity"][layer]["num_weights"]
            nn_overview[model][layer]["num_weights"] = params
            zero_weights = results[model]["sparsity"][layer]["num_zero_weights"]
            nn_overview[model][layer]["num_zero_weights"] = zero_weights

        sum_sparsed_size_kb = 0
        sum_sparsed_quant_size_kb = 0
        sum_max_size_kb = 0

        for layer in layer_names:
            max_size = nn_overview[model][layer]["num_weights"] * 32
            #overall_size = nn_overview[layer]["num_weights"] * nn_overview[layer]["word_width"]
            sparsed_size = (nn_overview[model][layer]["num_weights"] - 
            nn_overview[model][layer]["num_zero_weights"]) * 32
            sparsed_quant_size = (nn_overview[model][layer]["num_weights"] - nn_overview[model][layer]["num_zero_weights"]) * nn_overview[model][layer]["word_width"]

            # Convert to Bytes and to KB
            #overall_size = overall_size / 8 / 1024
            sparsed_size = sparsed_size / 8 / 1024
            sparsed_quant_size = sparsed_quant_size / 8 / 1024
            max_size = max_size / 8 / 1024

            #sum_overall_size_kb += overall_size
            sum_sparsed_size_kb += sparsed_size
            sum_sparsed_quant_size_kb += sparsed_quant_size
            sum_max_size_kb += max_size

        sparsed_size_kb.append(sum_sparsed_size_kb)
        sparsed_quant_size_kb.append(sum_sparsed_quant_size_kb)
        max_size_kb.append(sum_max_size_kb)
        
        print("Model: " + model)
        print("Float size: " + str(round(sum_max_size_kb, 2)) + " KB - No zero weights")
        #print("Overall size: " + str(round(sum_overall_size_kb, 2)) + " KB - ")
        print("Sparsed size: " + str(round(sum_sparsed_size_kb, 2)) + " KB")
        print("Sparsed and quantized size: " + str(round(sum_sparsed_quant_size_kb, 2)) + " KB")  
        print("")
    #print(nn_overview)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(models, max_size_kb, label='Max. Model-Size Float (in KB)', color='green')
    # Add in the center under the plot the value of the last point
    plt.text(models[-1], max_size_kb[-1], f'{round(max_size_kb[-1], 2)} KB', ha='center', va='bottom', color='green', fontweight='bold')
        
    plt.plot(models, sparsed_size_kb, label='Sparsed Model-Size Float (in KB)', color='blue')
    # On the left add the value of the first point, the point shall be padded 15 pixels from the left
    plt.text(models[0], sparsed_size_kb[0], f'{round(sparsed_size_kb[0], 2)} KB', ha='left', va='bottom', color='blue', fontweight='bold')
    # On the right side add the value of the last point above the plot
    plt.text(models[-1], sparsed_size_kb[-1], f'{round(sparsed_size_kb[-1], 2)} KB', ha='right', va='top', color='blue', fontweight='bold')
    
    plt.plot(models, sparsed_quant_size_kb, label='Sparsed Model-Size Quant (in KB)', color='red')
    # On the left add the value of the first point, the point shall be padded 15 pixels from the left
    plt.text(models[0], sparsed_quant_size_kb[0], f'{round(sparsed_quant_size_kb[0], 2)} KB', ha='left', va='bottom', color='red', fontweight='bold')
    
    # On the right side add the value of the last point above the plot
    plt.text(models[-1], sparsed_quant_size_kb[-1], f'{round(sparsed_quant_size_kb[-1], 2)} KB', ha='right', va='top', color='red', fontweight='bold')
    
   
    plt.gca().set_xticks(plt.gca().get_xticks()[::steps])

    plt.xlabel('Allowed Accuracy Drop (in %)')
    plt.ylabel('Memory Size (in KB)')
    plt.title(f'Memory-Size vs. Allowed Accuracy Drop {model_name}')
    plt.legend()
    plt.grid(True)
    
    save_dir = f'{save_dir}/memory_vs_acc_drop_plot.png'
    print(f"Save to: {save_dir}")
    plt.savefig(save_dir)  # Save the plot as a PNG file
    
def plot_eval_sparsity(results, model_name, save_dir):
    
    results = sorted(results.items())
    
    # Plot the results
    accs = [results[key]["accuracy"] for key in results]
    sparsities = [results[key]["sparsity"]["total"]["zero_weight_percentage"] for key in results]
    all_allowed_acc_drops = [key for key in results]

    plt.figure(figsize=(10, 6))
    plt.plot(all_allowed_acc_drops, sparsities, label='Sparsity (in %)', color='blue')
    plt.plot(all_allowed_acc_drops, accs, label='Accuracy (in %)', color='red')

    plt.xlabel('Allowed Accuracy Drop')
    plt.ylabel('Value')
    plt.title(f'Sparsity and Accuracy vs. Allowed Accuracy Drop {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/sparsity_accuracy_plot.png')  # Save the plot as a PNG file

def plot_eval_sparsity_retrain(results, model_name, save_dir, steps=25):
    # Plot the results
    result_keys = sorted(results.keys(), key=float)
    
    allowed_acc_drop_list = []
    acc_list = []
    acc_retrain_list = []
    sparsity_list = []
    
    # Prepare data for plotting, because it's not sorted
    for key in result_keys:
        allowed_acc_drop_list.append(key)
        acc_list.append(results[key]["accuracy"])
        acc_retrain_list.append(results[key]["accuracy_retrain"])
        sparsity_list.append(results[key]["sparsity"]["total"]["zero_weight_percentage"])


    plt.figure(figsize=(10, 6))
    
    # Show all data points, but only every 10th label on x-axis

    plt.plot(allowed_acc_drop_list, sparsity_list, label='Sparsity (in %)', color='blue')
    plt.plot(allowed_acc_drop_list, acc_list, label='Accuracy (in %)', color='red')
    plt.plot(allowed_acc_drop_list, acc_retrain_list, label='Accuracy Retrain (in %)', color='green')
    
    # Set x-axis labels
    plt.gca().set_xticks(plt.gca().get_xticks()[::steps])

    plt.xlabel('Allowed Accuracy Drop')
    plt.ylabel('Value in %')
    plt.title(f'Sparsity and Accuracy vs. Allowed Accuracy Drop {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/sparsity_accuracy_plot.png')  # Save the plot as a PNG file
    
def plot_eval_quant_retrain(results, model_name, save_dir, steps=25):
    # Plot the results
    result_keys = sorted(results.keys(), key=float)
    
    allowed_acc_drop_list = []
    acc_list = []
    acc_retrain_list = []
    acc_quant_retrain_list = []
    sparsity_list = []
    
    # Prepare data for plotting, because it's not sorted
    for key in result_keys:
        allowed_acc_drop_list.append(key)
        acc_list.append(results[key]["accuracy"])
        acc_retrain_list.append(results[key]["accuracy_retrain"])
        sparsity_list.append(results[key]["sparsity"]["total"]["zero_weight_percentage"])

        try: 
            acc_quant_retrain_list.append(results[key]["accuracy_quant_retrain"])
        except:
            acc_quant_retrain_list.append(0.0)


    plt.figure(figsize=(10, 6))
    
    # Show all data points, but only every 10th label on x-axis

    plt.plot(allowed_acc_drop_list, sparsity_list, label='Sparsity (in %)', color='blue')
    plt.plot(allowed_acc_drop_list, acc_list, label='Accuracy (in %)', color='red')
    plt.plot(allowed_acc_drop_list, acc_retrain_list, label='Accuracy Retrain (in %)', color='green')
    plt.plot(allowed_acc_drop_list, acc_quant_retrain_list, label='Accuracy Quant Retrain (in %)', color='orange')
    # Set x-axis labels
    plt.gca().set_xticks(plt.gca().get_xticks()[::steps])

    plt.xlabel('Allowed Accuracy Drop')
    plt.ylabel('Value in %')
    plt.title(f'Sparsity and Accuracy vs. Allowed Accuracy Drop {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/sparsity_accuracy_plot.png')  # Save the plot as a PNG file

def plot_eval_quant(results, model_name, save_dir, steps=25):
    # Plot the results
    result_keys = sorted(results.keys(), key=float)
    
    layer_list = None
    allowed_acc_drop_list = []
    
    for key in result_keys:
        
        allowed_acc_drop_list.append(key)
        layers = results[key]["quant"].keys()

        if layer_list is None:
            layer_list = dict()
            
            for layer in layers:
                layer_list[layer] = []
                
        for layer in layers:
            layer_list[layer].append(results[key]["quant"][layer])
            
    plt.figure(figsize=(10, 6))

    # Go through all layers
    for layer in layer_list.keys():
        plt.plot(allowed_acc_drop_list, layer_list[layer], label=f'Layer {layer}')

    # Set x-axis labels
    plt.gca().set_xticks(plt.gca().get_xticks()[::steps])
    
    plt.xlabel('Allowed Accuracy Drop')
    plt.ylabel('Word Width')
    plt.title(f'Quantization vs. Allowed Accuracy Drop {model_name}')
    plt.legend()
    plt.grid(True)
    

    plt.savefig(f'{save_dir}/quantization_plot.png')  # Save the plot as a PNG file
    
    
def plot_pareto_front(results, model_name, save_dir):
    """
    Function to plot the Pareto front for sparsity vs accuracy using a strict dominance logic.

    Parameters:
    - data (dict): A dictionary with keys 'sparsity' and 'accuracy', both containing lists of values.

    Returns:
    - A plot showing the data points and the Pareto front.
    """
    
    """
    # Extract sparsity and accuracy data
    x_axis_data = np.array(x_data)
    y_axis_data = np.array(y_data)
    
    x_axis_data_sorted = []
    y_axis_data_sorted = []
    label_data_sorted = []  
    
    # Sort the data points by accuracy
    sorted_indices = np.argsort(y_axis_data)
    for i in sorted_indices:
        x_axis_data_sorted.append(round(x_axis_data[i], 4))
        y_axis_data_sorted.append(round(y_axis_data[i], 4))
        label_data_sorted.append(labels[i])
        
    x_axis_data_sorted.reverse()
    y_axis_data_sorted.reverse()
    label_data_sorted.reverse()
        
    #for i in range(len(x_axis_data_sorted)):
    #    print(y_axis_data_sorted[i], x_axis_data_sorted[i])
        
    pareto_front_x = []
    pareto_front_y = []
    pareto_point_labels = []
    
    pareto_point_indices = []
    for i in range(len(x_axis_data_sorted)):
        is_pareto = True
        for j in range(len(x_axis_data_sorted)):
            if x_axis_data_sorted[j] > x_axis_data_sorted[i] and y_axis_data_sorted[j] > y_axis_data_sorted[i]:
                is_pareto = False
                break
        if is_pareto:
            pareto_front_x.append(x_axis_data_sorted[i])
            pareto_front_y.append(y_axis_data_sorted[i])
            pareto_point_labels.append(label_data_sorted[i])
            
            print(label_data_sorted[i])
                
                
    # Write pareto points to file
    with open('pareto_points.txt', 'w') as f:
        for i in range(len(pareto_front_x)):
            # Key: Accuracy, Value: Sparsity
            res = f"Pareto Point\t{pareto_point_labels[i]}\tAccuracy: {round(pareto_front_y[i],2)}\tSparsity: {round(data[pareto_point_labels[i]]['zero_param_per'],2)}\n"
            f.write(res)
            
    """
    
def determine_pareto_front(accs, sparsities):
    # Sort the accuracies and sparsities
    sorted_indices = np.argsort(accs)
    sorted_accs = np.array(accs)[sorted_indices]
    sorted_sparsities = np.array(sparsities)[sorted_indices]
    
    # Initialize the pareto front
    pareto_front = [sorted_indices[0]]
    
    # Go through all accuracies and sparsities
    for index in range(1, len(sorted_accs)):
        if sorted_sparsities[index] < sorted_sparsities[pareto_front[-1]]:
            pareto_front.append(sorted_indices[index])
    
    return pareto_front

def train(config_path, config, weights_path, lr, epochs, base_save_dir): 
    
    # Initialize the configuration manager
    config_manager = ConfigurationManager(config_path)
    
    # Load the model
    model = ModelBuilder().build(ModelTypes.JSC, config=config_manager.config, weights_path=weights_path)
    
    # Load the dataset and validate base model
    dataset = DatasetBuilder().build(DatasetTypes.JSC, config=config_manager.config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
    
    # Validate
    start_acc, loss = validator.validate(model)

    # Initialize the trainer and run training
    trainer = Trainer(model, dataset)
    
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True)
    
    name = config_path.split('_')[-1].replace('.yaml', '_acc_drop_allowed')
    save_dir = f"{base_save_dir}/{name}"

    if "/config.yaml" in config_path:
        name = config_path.split('/')[-2]
        save_dir = f"{base_save_dir}/{name}"

    os.makedirs(save_dir, exist_ok=True)
    
    config_manager.write(f"{save_dir}/config.yaml")
    
    trainer.train(model, None, dataset, torch.nn.CrossEntropyLoss(), optimizer, lr, epochs, 200, scheduler, save_dir)
    
    # Reload model with best weights
    model = ModelBuilder().build(ModelTypes.JSC, config=config_manager.config, weights_path=f"{save_dir}/best_weights.pth")
    
    print(f"{config_path=}")
    # Validate the model
    acc, loss = validator.validate(model)
    
    key = config_path.split('_')[-1].replace('.yaml', '')
    if "/config.yaml" in config_path:
        key = config_path.split('/')[-2].split('_')[0]
    
    # Return accuracy and config name
    return acc, key

def evaluate_sparsity_max_acc_drop(model_config, weights_path, layer_str_list, key, value_range, max_acc_drop):
    
    # Initialize dataset, validator and model
    dataset = DatasetBuilder.build(DatasetTypes.JSC, config=model_config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
    model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
    
    # Get baseline accuracy
    baseline_acc, _ = validator.validate(model)
    
    # Get baseline sparsity
    baseline_sparsity = get_sparsity_overview(model, layer_str_list)

    # Go through all layers
    for layer_str in layer_str_list:
        reload_index = 0
        
        for index, val in enumerate(value_range):
            
            # Set new value and rebuild model
            model_config = set_config_value(model_config, layer_str, key, val)
            model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
            
            # Validate model
            acc, _ = validator.validate(model)
            
            print(f"{baseline_acc=} {acc=} {max_acc_drop=}")
            
            # If accuracy drop is too high, reload previous model configuration
            if baseline_acc - acc > max_acc_drop:
                reload_index = index - 1
                print(f"Reload index: {reload_index}")
                break
            else:
                reload_index = index

        
        # Set back to previous "good" model configuration 
        model_config = set_config_value(model_config, layer_str, key, value_range[reload_index])

        model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
        acc, _ = validator.validate(model) 
        
    model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
    acc, _ = validator.validate(model)         
    curr_model_sparsity = get_sparsity_overview(model, layer_str_list)

    return acc, curr_model_sparsity

if __name__ == "__main__":
    
    DELETE_TMP_DATA = False
    EVAL_SPARSITY = False
    RETRAIN_SPARSE_MODELS = False
    EVAL_QUANT = False
    RETRAIN_QUANT_MODELS = False
    
    results_folder = "tmp_data"

    base_model_config = f"{parent_directory}/configs/jsc/quant_jsc_xl_updated_quant.yaml"
    float_model_weight_path = f"/home/mmecik/repositories/synapselab/train/jsc_xl_floating_point/run_331/best_weights.pth"
    model = ModelBuilder().build(ModelTypes.JSC, config=base_model_config, weights_path=float_model_weight_path)
    
    dataset = DatasetBuilder.build(DatasetTypes.JSC, config=base_model_config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
    
    # Layers
    layer_str_list = ["dense1", "dense2", "dense3", "dense4", "dense5"]
    sorted_layer_str_list = sort_layers_by_param_num(model, layer_str_list)
    
    # Evaluate sparsity with allowed accuracy drop
    sparsity_range = np.arange(0.0, 1.0, 0.05)   
    
    # Max allowed accuracy drop
    max_allowed_acc_drop = 50.0
    acc_drop_range_raw = np.arange(0.0, max_allowed_acc_drop + 0.25, 0.25)
    acc_drop_range = [round(x, 2) for x in acc_drop_range_raw]
    
    #print(acc_drop_range)
    
    # Results Sparsification
    result_eval_sparsity = {}
    results_folder_sparsity =f"tmp_data/eval_sparsity/{model.name}"
    results_folder_retrain = f"tmp_data/retrain_sparsity/{model.name}"
    results_folder_quant = f"tmp_data/eval_quant/{model.name}"
    results_folder_retrain_quant = f"tmp_data/retrain_quant/{model.name}"

    if DELETE_TMP_DATA:
        # If tmp_data exists, delete it recursively
        if os.path.exists(results_folder):
            shutil.rmtree(results_folder)
        
        # Recreate the folder
        shutil.os.makedirs(results_folder)
        
    ##########################################################
    # Evaluate sparsity with allowed accuracy drop
    ##########################################################
    
    if EVAL_SPARSITY:
        # Enable sparsification
        shutil.os.makedirs(results_folder_sparsity)

        for allowed_acc_drop in acc_drop_range:
            model_config = set_config_value(base_model_config, "dense", "weight_disable_sparse", False)

            print(f"{allowed_acc_drop=}")
            acc, sparsity = evaluate_sparsity_max_acc_drop(model_config, float_model_weight_path, sorted_layer_str_list,  "weight_sparse_eps", sparsity_range, allowed_acc_drop)
            result_eval_sparsity[allowed_acc_drop] = {"accuracy": acc, "sparsity": sparsity}
            
            manager = ConfigurationManager(model_config)
            manager.write(f"{results_folder_sparsity}/config_{allowed_acc_drop}.yaml")
            
        with open(f"{results_folder_sparsity}/result_eval_sparsity.json", "w") as f:
            json.dump(result_eval_sparsity, f, indent=4, sort_keys=True)
    
        # Plot the results
        plot_eval_sparsity(result_eval_sparsity, f"{model.name}_eval_spare.png", results_folder_sparsity)
        
    ##########################################################
    # Retrain all models
    ##########################################################
    
    if RETRAIN_SPARSE_MODELS:
        # Get all models from tmp_data/eval_sparsity
        all_models_retrain = [os.path.join(results_folder_sparsity,f) for f in os.listdir(results_folder_sparsity) if f.endswith(".yaml")]
        
        # Create a new folder for retraining
        shutil.os.makedirs(results_folder_retrain)
    
        # Copy result_eval_sparsity.json to results_folder_retrain
        shutil.copy(f"{results_folder_sparsity}/result_eval_sparsity.json", f"{results_folder_retrain}/result_eval_sparsity.json")
        results_retrain = json.load(open(f"{results_folder_retrain}/result_eval_sparsity.json"))
        
        # Retrain all models
        lr = 0.00009349
        epochs = 30
        for sparsed_model_config in all_models_retrain:
            print(sparsed_model_config)
            acc, key = train(config_path=f"{sparsed_model_config}", config=None, weights_path=float_model_weight_path, lr=lr, epochs=epochs, base_save_dir=results_folder_retrain)
            results_retrain[key]["accuracy_retrain"] = acc

        with open(f"{results_folder_retrain}/result_retrain_sparsity.json", "w") as f:
            json.dump(results_retrain, f, indent=4, sort_keys=True)
    
    ##########################################################
    # Determine Pareto Front
    ##########################################################
    
    ##########################################################
    # Eval Quantization
    ##########################################################
    # Copy result from retrain to quant
    
    if EVAL_QUANT:
        
        shutil.os.makedirs(results_folder_quant)
        shutil.copy(f"{results_folder_retrain}/result_retrain_sparsity.json", f"{results_folder_quant}/result_retrain_sparsity.json")
        
        results_eval_quant = json.load(open(f"{results_folder_quant}/result_retrain_sparsity.json"))
        
        all_models_quant_eval = [os.path.join(results_folder_retrain,f) for f in os.listdir(results_folder_retrain) if os.path.isdir(os.path.join(results_folder_retrain, f))]
        
        # Enable quantization
        
        for model_quant_eval in all_models_quant_eval:
            best_weights = f"{model_quant_eval}/best_weights.pth"
            model_config = f"{model_quant_eval}/config.yaml"
            
            allowed_acc_drop =float(model_quant_eval.split("/")[-1].split("_")[0])
            
            print(f"{allowed_acc_drop=}")
            
            model_config = set_config_value(model_config, "dense", "weight_disable_quant", False)

            
            model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=best_weights)
            dataset = DatasetBuilder().build(DatasetTypes.JSC, config=model_config)
            
            # Layers
            layer_str_list = ["dense1", "dense2", "dense3", "dense4", "dense5"]
            sorted_layer_str_list = sort_layers_by_param_num(model, layer_str_list)
            
            # Evaluate sparsity with allowed accuracy drop
            word_width_range  = range(16, 3, -1)

            print(f"{allowed_acc_drop=}")
            acc, _ = evaluate_sparsity_max_acc_drop(model_config, best_weights, sorted_layer_str_list,  "weight_bit_width", word_width_range, allowed_acc_drop)
        
            results_eval_quant[str(allowed_acc_drop)]["quant"] = dict()
            for layer_str in sorted_layer_str_list:

                word_width = get_config_value(model_config, layer_str, "weight_bit_width")
                results_eval_quant[str(allowed_acc_drop)]["quant"][layer_str] = word_width
                
                print(f"{word_width=}")

            shutil.os.makedirs(f"{results_folder_quant}/{allowed_acc_drop}_acc_drop_allowed")
            
            manager = ConfigurationManager(model_config)
            manager.write(f"{results_folder_quant}/{allowed_acc_drop}_acc_drop_allowed/config.yaml")
            
            # Copy best weights
            shutil.copy(best_weights, f"{results_folder_quant}/{allowed_acc_drop}_acc_drop_allowed/best_weights.pth")
                
        with open(f"{results_folder_quant}/result_quant_eval.json", "w") as f:
            json.dump(results_eval_quant, f, indent=4, sort_keys=True)
            
    if RETRAIN_QUANT_MODELS:
        
        # Unsortred list
        unsorted_all_models_retrain = [os.path.join(results_folder_quant,f) for f in os.listdir(results_folder_quant) if os.path.isdir(os.path.join(results_folder_quant, f))]
        
        # Fuction to extract the float from the path
        def extract_float_from_path(path):
            # Split by "/" and extract the part containing the float
            filename = path.split('/')[-1]
            # Extract the number from the part before "_acc_drop_allowed"
            return float(filename.split('_')[0])
        
        # Sort the list
        all_models_retrain = sorted(unsorted_all_models_retrain, key=extract_float_from_path)
        
        # Create a new folder for retraining
        shutil.os.makedirs(results_folder_retrain_quant)
    
        # Copy result_eval_sparsity.json to results_folder_retrain
        shutil.copy(f"{results_folder_quant}/result_quant_eval.json", f"{results_folder_retrain_quant}/result_quant_eval.json")
        results_retrain = json.load(open(f"{results_folder_quant}/result_quant_eval.json"))
        
       
        for quant_model_config in all_models_retrain:
            print(quant_model_config)
            
            # Retrain all models
            lr = 0.00009349
            epochs = 30

            quant_config = f"{quant_model_config}/config.yaml"
            acc, key = train(config_path=f"{quant_config}", config=None, weights_path=float_model_weight_path, lr=lr, epochs=epochs, base_save_dir=results_folder_retrain_quant)
            
            print(f"{key=}")
            
            results_retrain[key]["accuracy_quant_retrain"] = acc
      
            with open(f"{results_folder_retrain_quant}/result_retrain_quant.json", "w") as f:
                json.dump(results_retrain, f, indent=4, sort_keys=True)
        
            plot_eval_quant_retrain(results_retrain, f"{model.name}", results_folder_retrain_quant)
        
    #results_retrain_sparsity = json.load(open(f"{results_folder_retrain}/result_retrain_sparsity.json"))
    #plot_eval_sparsity_retrain(results_retrain_sparsity, f"{model.name}", results_folder_retrain)
    
    results_eval_quant = json.load(open(f"{results_folder_quant}/result_quant_eval.json"))
    #plot_eval_quant(results_eval_quant, f"{model.name}", results_folder_quant)
    
    plot_memory_size_vs_accuracy_drop_allowed(results_eval_quant, f"{model.name}", results_folder_quant)