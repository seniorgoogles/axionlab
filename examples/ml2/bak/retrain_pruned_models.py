import json
import shutil
from includes_ml2 import *
import numpy as np
import matplotlib.pyplot as plt

def train(config, weights_path, lr, epochs): 
    # Initialize the configuration manager
    config_manager = ConfigurationManager(config)
    
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
    
    name = config_file.split('_')[-1].replace('.yaml', 'acc_drop_allowed')
    save_dir = f"tmp_data/retrain/{model.name}/{name}"
    os.makedirs(save_dir, exist_ok=True)
    
    trainer.train(model, None, dataset, torch.nn.CrossEntropyLoss(), optimizer, lr, epochs, 200, scheduler, save_dir)
    
    layer_str_list = ["dense1", "dense2", "dense3", "dense4", "dense5"]
    data = get_sparsity_overview(model, layer_str_list)
    sparsity_results = json.loads(data)
    total_zero_weight_percentage = sparsity_results['total']['zero_weight_percentage']
    
    # Validate the model
    end_acc, loss = validator.validate(model)
    
    return start_acc, end_acc, total_zero_weight_percentage

if __name__ == "__main__":
    
    checkpoint = 331 
    quant_model_config = f"{parent_directory}/configs/jsc/quant_jsc_xl.yaml"
    best_weights = f"/home/mmecik/repositories/synapselab/train/jsc_xl_floating_point/run_{checkpoint}/best_weights.pth"
    
    # Model configurations sparsed 
    base_path = "/home/mmecik/repositories/synapselab/tmp_data_max_50_acc_drop/configs/jsc"
    
    # Get all configs from the base path
    config_files = [f for f in os.listdir(base_path) if f.endswith('.yaml')]
    
    dict_results = {}

    for config_file in config_files:
        full_config_path = os.path.join(base_path, config_file)
        name = config_file.split('_')[-1].replace('.yaml', '% acc drop allowed')
        start_acc, end_acc, total_zero_weight_percentage = train(full_config_path, best_weights, lr=0.00009349, epochs=20)
        
        print("=========================================")
        print(f"Results for {name}:")
        print(f"Start accuracy: {start_acc}")
        print(f"End accuracy: {end_acc}")
        print(f"Zero Params %: {total_zero_weight_percentage:.2f}%")
        print("=========================================")
        
        # Get sparsity of model        
        dict_results[name] = {
            "ptp-acc": start_acc, # post training pruning accuracy
            "pat-acc": end_acc,    # pruning aware training accuracy
            "zero_param_per": total_zero_weight_percentage
        }
                
    # Write results to a json file
    with open('tmp_data/results_pat.json', 'w') as f:
        json.dump(dict_results, f, indent=4)