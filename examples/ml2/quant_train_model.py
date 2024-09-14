import shutil
from includes_ml2 import *


if __name__ == "__main__":

    retrain = True
    train_folder = f"{parent_directory}/train"

    #if input(f"Delete previous trainings? [y/N]: ").strip().lower() == "y":
    #    clear_folder(train_folder)
    

    # Create a logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)


    checkpoint = 331 # Accuracy at 213 is 75.142 floating point

    model_config = f"{parent_directory}/configs/jsc/jsc_xl_floating_point.yaml"
    quant_config = f"{parent_directory}/configs/jsc/quant_jsc_xl.yaml"
    best_weights = f"/home/mmecik/repositories/synapselab/train/jsc_xl_floating_point/run_{checkpoint}/best_weights.pth"

    trainer = Trainer()
    dataset = DatasetBuilder.build(DatasetTypes.JSC, config=model_config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
    
    model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=best_weights)
    quant_model = ModelBuilder().build(ModelTypes.JSC, config=quant_config, weights_path=best_weights)

    acc, _ = validator.validate(model)
    quant_acc, _ = validator.validate(quant_model)
    # Evaluate every folder from 
    #train_model(dataset, trainer, validator, None, model, lr_list, epochs, "result_lr_acc.txt", reload_weights=False, reload_weights_acc=False, logger=logger)
    
    """
    float_model_acc = 0.0
    best_quant_acc = 0.0
    
    best_quant_weights_index = 0
     
    for index in range(0, 399):
        
        try: 
            best_weights = f"/home/mmecik/repositories/synapselab/train/jsc_xl_floating_point/run_{index}/best_weights.pth"

            model.load_state_dict(torch.load(best_weights,  weights_only=False), strict=False)
            quant_model.load_state_dict(torch.load(best_weights,  weights_only=False), strict=False)
            
            acc, _ = validator.validate(model)
            quant_acc, _ = validator.validate(quant_model)

            # If quant_acc is better than best_quant_acc, save the weights index
            if quant_acc > best_quant_acc:
                best_quant_acc = quant_acc
                float_model_acc = acc 
                best_quant_weights_index = index
                best_quant_weights = best_weights
                print(f"New best quant acc: {best_quant_acc} at {index}")
                
        except Exception as ex:
            print(ex)
            
    print(f"Best quant acc: {best_quant_acc} at {best_quant_weights_index} with float model acc: {float_model_acc}")
    
    layer_str_list = ["dense1", "dense2", "dense3", "dense4", "dense5"]
    """
    #sort_layers_by_param_num(quant_model, layer_str_list)
    
    """
    min_rate = 0.00009
    max_rate = 0.0001
    num_rates = 100
    
    epochs = 100
    lr_list = np.linspace(min_rate, max_rate, num_rates).tolist()
    model = ModelBuilder().build(ModelTypes.JSC, config=model_config)
    train_model(dataset, trainer, validator, None, model, lr_list, epochs, "result_lr_acc.txt", reload_weights=False, reload_weights_acc=False, logger=logger)
    """