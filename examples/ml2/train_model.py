from matplotlib import pyplot as plt
from includes_ml2 import *

def write_report(epochs_list, lr_train_list, acc_train_list, loss_train_list, index):
    with open(f"{parent_directory}/train/report.txt", "a") as f:
        f.write("| --------------------------------------------------------------\n")
        f.write(f"| Report for Training {index}\n")
        f.write(f"| Epochs: {epochs_list}\n")
        f.write(f"| Learning Rates: {lr_train_list}\n")
        f.write(f"| Accuracy: {acc_train_list}\n")
        f.write(f"| Loss: {loss_train_list}\n")
        f.write("|\n")
        f.write(f"| Last Accuracy: {acc_train_list[-1]}%\n")
        f.write("| --------------------------------------------------------------\n")


if __name__ == "__main__":
    
    # Define paths
    train_folder = f"{parent_directory}/train"
    model_config = f"{parent_directory}/configs/jsc/jsc_xl.yaml"   

    # Initialize objects
    trainer = Trainer()
    dataset = DatasetBuilder.build(DatasetTypes.JSC, config=model_config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
    """
    best_acc = 0.0
    index_best_run = 0
    for best_run in range(1, 101):


        model_weights = f"{parent_directory}/train/jsc_xl/run_{best_run}/best_weights.pth"

        # Create model
        model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=model_weights)
        acc, loss = validator.validate(model)
        
        if acc > best_acc:
            best_acc = acc
            index_best_run = best_run

    print(f"Best run: {index_best_run} with accuracy: {best_acc}%")
    """
    best_run = 71
    model_weights = f"{parent_directory}/train/jsc_xl/run_{best_run}/best_weights.pth"
    model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=model_weights)
    
    acc, loss = validator.validate(model)
    
    # Setup learning rates
    best = 0.0010785011185682326
    best = 0.001029478340451495
    best = 0.0021
    min_rate = best - (best * 0.01)
    max_rate = best + (best * 0.01)
    num_rates = 100
    
    min_acc = 75.4
    
    lr_list = np.linspace(min_rate, max_rate, num_rates).tolist()
    print(lr_list)
    
    if input(f"Delete previous trainings? [y/N]: ").strip().lower() == "y":
        clear_folder(train_folder)
    
    epochs_list = [30, 20, 20, 30, 15]
    index = 0
    for lr in lr_list:
        
        acc = 0
        loss = 0
        
        lr_train_list = []
        acc_train_list = []
        loss_train_list = []
        
        model = ModelBuilder().build(ModelTypes.JSC, config=model_config)

        for run, epochs in enumerate(epochs_list): 
            print(f"{lr=}")
            
            save_model = True if (run == len(epochs_list) - 1) else False
            
            train_model(dataset, trainer, model, lr, epochs, save_model=save_model)     
            
            acc, loss = validator.validate(model)
            
            lr_train_list.append(lr)
            acc_train_list.append(acc)
            loss_train_list.append(loss)
            
            lr = lr + (lr * 0.015)
            
        # Get latest run best weights from the latest saved model
        run = max([int(f.split('_')[-1]) for f in os.listdir(f"train/{model.name}") if os.path.isdir(os.path.join(f"train/{model.name}", f))])

        weights_path = f"train/{model.name}/run_{run}/best_weights.pth"
        model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
        
        acc, loss = validator.validate(model)
        acc_train_list.append(acc)
        
        index = index + 1
        write_report(epochs_list, lr_train_list, acc_train_list, loss_train_list, index)
