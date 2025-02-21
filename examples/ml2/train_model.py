from matplotlib import pyplot as plt
from includes_ml2 import *
from helpers import clear_folder, train_model, print_colored_window

import datetime 

def write_report(output_path, epochs_list, lr_train_list, acc_train_list, loss_train_list):
    
    # Get the index of the latest training, if exists. The index can be found belwo the "Report for Training" line
    index = 1
    if os.path.exists(f"{output_path}/report.txt"):
        with open(f"{output_path}/report.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Report for Training" in line:
                    index = index + 1

    with open(f"{output_path}/report.txt", "a") as f:
        f.write("| --------------------------------------------------------------\n")
        f.write(f"| Report for Training {index}\n")
        f.write(f"| Epochs: {epochs_list}\n")
        f.write(f"| Learning Rates: {lr_train_list}\n")
        f.write(f"| Accuracy: {acc_train_list}\n")
        f.write(f"| Loss: {loss_train_list}\n")
        f.write("|\n")
        f.write(f"| Last Accuracy: {acc_train_list[-1]}%\n")
        f.write("| --------------------------------------------------------------\n")
        
def find_best_weights(weights_path, validator, model_config):
    for eval_run in os.listdir(weights_path):
        print(eval_run)

if __name__ == "__main__":
    
    best_acc = 0.0
        
    current_date = datetime.datetime.now()
    # Define paths
    train_folder = f"{parent_directory}/examples/ml2/train"
    model_config = f"{parent_directory}/examples/ml2/float_model/config.yaml"   
    model_output_folder = f"{train_folder}/jsc_xl"
    
    
    current_year = current_date.year
    current_month = f"0{current_date.month}" if current_date.month < 10 else f"{current_date.month}"
    current_day = f"0{current_date.day}" if current_date.day < 10 else f"{current_date.day}"
    current_hour = f"0{current_date.hour}" if current_date.hour < 10 else f"{current_date.hour}"
    current_minute = f"0{current_date.minute}" if current_date.minute < 10 else f"{current_date.minute}"
    current_second = f"0{current_date.second}" if current_date.second < 10 else f"{current_date.second}"
    
    model_output_folder_train = f"{model_output_folder}/{current_year}{current_month}{current_day}_{current_hour}{current_minute}{current_second}"
    
    print(f"Model config: {model_config}")

    # Initialize objects
    trainer = Trainer()
    dataset = DatasetBuilder.build(DatasetTypes.JSC, config=model_config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
    
    best_run = -1
    weights_path_best = f"/home/fry/Documents/repositories/synapselab/examples/ml2/train/jsc_xl/20250207_164607/run_{best_run}/best_weights.pth" if best_run >= 0 else None
    best_lr = 0.007464869999350717
    

    model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path_best)

    acc, loss = validator.validate(model)
    
    epochs_list = [150]

    min_rate = best_lr - (best_lr * 0.01)
    max_rate = best_lr + (best_lr * 0.01)    
    
    #min_rate = 0.001
    #max_rate = 0.01
        
    num_rates = 100
    
    min_acc = 75.4
    
    if best_lr != 0.0:
        print_colored_window(f"Best run: {best_run} with {best_lr} will be used")
    
    lr_list_space = np.linspace(min_rate, max_rate, num_rates).tolist()
    
    print(f"Learning rate space: {lr_list_space}")
    
    if input(f"Delete previous trainings? [y/N]: ").strip().lower() == "y":
        clear_folder(model_output_folder)
        
        # Create model output folder
        os.makedirs(model_output_folder_train, exist_ok=True)
        
        print("Deleted previous trainings @ ", model_output_folder)
        
    if input(f"Evaluate best lr? [y/N]: ").strip().lower() == "y":
        epochs_list = [10]
        
    run = 0
    for lr_ in lr_list_space:
        
        acc = 0
        loss = 0
        
        lr_list = [lr_]

        lr_train_list = []
        acc_train_list = []
        loss_train_list = []
        
        save_path = f"{model_output_folder_train}/run_{run}"
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path_best)

        for epoch_index, (epochs, lr) in enumerate(zip(epochs_list, lr_list)): 
            
            print(f"{lr=}")
            save_model = True if (epoch_index == len(epochs_list) - 1) else False
            
            train_model(dataset, trainer, model, lr, epochs, save_model=save_model, save_dir=save_path)                 
            acc, loss = validator.validate(model)
            
            lr_train_list.append(lr)
            acc_train_list.append(acc)
            loss_train_list.append(loss)
            
            print(f"{epoch_index=} {save_model=} {acc=}")
            
            #lr = lr + (lr * 0.0015)
                    
        # Get latest run best weights from the latest saved model
        #run = max([int(f.split('_')[-1]) for f in os.listdir(model_output_folder_train)])

        weights_path = f"{model_output_folder_train}/run_{run}/best_weights.pth"
        model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)

        acc, loss = validator.validate(model)
        acc_train_list.append(acc)
        
        run = run + 1

        if acc > best_acc:
            best_acc = acc
            best_run = run
            
            # Print a nice window with the best run and colors
            print_colored_window(f"NEW HIGHSCORE")            
    
        
        print_colored_window(f"Best run: {best_run} with {best_acc}% accuracy")            

        
        write_report(model_output_folder_train, epochs_list, lr_train_list, acc_train_list, loss_train_list)
