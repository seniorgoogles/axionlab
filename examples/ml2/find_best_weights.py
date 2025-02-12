from matplotlib import pyplot as plt
from includes_ml2 import *
from helpers import clear_folder, train_model, print_colored_window

import datetime 
        
def find_best_weights(weights_path, validator=None,model_config=None):
    best_acc = 0.0
    best_loss = 0.0
    
    # Get all folders from weights_path
    eval_runs = os.listdir(weights_path)
    
    for eval_run in eval_runs:
        
        try: 
            model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=f"{weights_path}/{eval_run}/best_weights.pth")
            acc, loss = validator.validate(model)
            
            if acc > best_acc:
                best_acc = acc
                best_loss = loss
                best_run = eval_run
                
                # Print a nice window with the best run and colors
                print_colored_window(f"NEW HIGHSCORE")
                print_colored_window(f"Best run: {best_run}")
                print_colored_window(f"Best accuracy: {round(best_acc, 2)}% loss: {round(best_loss, 2)}")      
        except Exception as e:
            print(f"Error: {e}")
            continue

    print_colored_window(f"Best run: {best_run}")
    print_colored_window(f"Best accuracy: {round(best_acc, 2)}% loss: {round(best_loss, 2)}")

if __name__ == "__main__":
    
    best_acc = 0.0
        
    current_date = datetime.datetime.now()
    # Define paths
    train_folder = f"{parent_directory}/examples/ml2/train"
    model_config = f"{parent_directory}/examples/ml2/float_model/config.yaml"   
    model_output_folder = f"{train_folder}/jsc_xl"
    
    # Initialize objects
    trainer = Trainer()
    dataset = DatasetBuilder.build(DatasetTypes.JSC, config=model_config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
    
    model_output_folder_train = "/home/fry/Documents/repositories/synapselab/examples/ml2/train/jsc_xl/20250211_104201"
    
    find_best_weights(model_output_folder_train, validator, model_config)