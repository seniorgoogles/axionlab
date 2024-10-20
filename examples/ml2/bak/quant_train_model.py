import shutil
from includes_ml2 import *
import logging

def train(config_path, config, weights_path, lr, epochs): 
    
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
    
    name = config_path.split('_')[-1].replace('.yaml', 'acc_drop_allowed')
    save_dir = f"tmp_data/retrain_quant/{model.name}/{name}"
    os.makedirs(save_dir, exist_ok=True)
    
    config_manager.save(config, f"{save_dir}/config.yaml")
    
    
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
    
    
    config_path = "tmp_data_max_50_acc_drop/configs/jsc/quant_jsc_xl_acc_drop_32.0.yaml"
    #config_path = "/home/mmecik/repositories/synapselab/configs/jsc/jsc_xl.yaml"
    
    config_path = "/home/mmecik/repositories/synapselab/configs/jsc/quant_jsc_xl_updated_quant.yaml"
    
    best_weights = "tmp_data_max50_acc_drop_retrain/retrain/quant_jsc_xl/32.0acc_drop_allowed/best_weights.pth"
    best_weights = f"/home/mmecik/repositories/synapselab/train/jsc_xl_floating_point/run_{checkpoint}/best_weights.pth"

    config_manager = ConfigurationManager(config_path)

    model = ModelBuilder().build(ModelTypes.JSC, config=config_manager.config, weights_path=best_weights)
    dataset = DatasetBuilder().build(DatasetTypes.JSC, config=config_manager.config)
    
    trainer = Trainer() 
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
    
    acc, _ = validator.validate(model)
    
    lr = 0.001
    epochs = 1
    
    #optimizer = torch.optim.Adam(model.parameters(), lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True)
    #trainer.train(model, None, dataset, torch.nn.CrossEntropyLoss(), optimizer, lr, epochs, 200, scheduler, None)

    '''
    retrain = True
    train_folder = f"{parent_directory}/tmp_data"
    
    # 32.0% acc drop allowed
    model_config_path = "tmp_data_max_50_acc_drop/configs/jsc/quant_jsc_xl_acc_drop_32.0.yaml"
    best_weights = "tmp_data_max50_acc_drop_retrain/retrain/quant_jsc_xl/32.0acc_drop_allowed/best_weights.pth"

    
    
    if input(f"Delete previous trainings? [y/N]: ").strip().lower() == "y":
        clear_folder(train_folder)
    
    # Create a logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    trainer = Trainer()
    dataset = DatasetBuilder.build(DatasetTypes.JSC, config=model_config_path)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
    
    model = ModelBuilder().build(ModelTypes.JSC, config=model_config_path, weights_path=best_weights)

    acc, _ = validator.validate(model)
    print(f"Accuracy: {acc}")
    
    # Layers
    layer_str_list = ["dense1", "dense2", "dense3", "dense4", "dense5"]

    # Sparsity
    sparsity_range = np.arange(0.0, 1.0, 0.1)   
    
    # Weights 
    weights_range = range(8, 1, -1)
    
    # Getting the layers by size, from max to min
    sorted_layer_str_list = sort_layers_by_param_num(model, layer_str_list)
    weight_bit_width = 8
    
    model_config = model_config_path
    
    for layer_str in layer_str_list: 
        model_config = set_config_value(model_config, layer_str, "weight_bit_width", weight_bit_width)
        model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=best_weights)
        acc, _ = validator.validate(model)
        
        train(model_config_path, model_config, best_weights, 0.00009349, 20)
        print(f"{layer_str} Accuracy: {acc}\tBit Width:{weight_bit_width}")
    '''