
from matplotlib import pyplot as plt
from includes_ml2 import *

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

if __name__ == "__main__":    # Load the model
    train_folder = f"{parent_directory}/train"
    model_output_folder = f"{train_folder}/continousflow_mnist"
    model_config = "configs/continuousflow/mnist.yaml"
    
    best_run = 344
    
    weights_path_best = f"{parent_directory}/train/continousflow_mnist/run_{best_run}/best_weights.pth" if best_run > 0 else None

    model = ModelBuilder().build(ModelTypes.CONTINUOUSFLOW_MNIST, config=model_config, weights_path=weights_path_best)
    dataset = DatasetBuilder().build(DatasetTypes.MNIST, config=model_config, crop_border_pixels=2)
    
    #print(model)

    trainer = Trainer()
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
        
    acc, loss = validator.validate(model)
    
    # Setup learning rates
    best = 0.000393
    best = 0.0019652237205922867
    
    min_rate = 0.0019634380330244595   #best - (best * 0.001)
    max_rate = 0.0019660546391165605    #best + (best * 0.001)
    num_rates = 100
    
    min_acc = 75.4
    
    lr_list = np.linspace(min_rate, max_rate, num_rates).tolist()
    print(lr_list)
    
    if input(f"Delete previous trainings? [y/N]: ").strip().lower() == "y":
        clear_folder(model_output_folder)
    
    epochs_list = [10,10,10,10,10,10,10,10]
    index = 0
    for lr in lr_list:
        
        acc = 0
        loss = 0
        
        lr_train_list = []
        acc_train_list = []
        loss_train_list = []
        
        model = ModelBuilder().build(ModelTypes.CONTINUOUSFLOW_MNIST, config=model_config, weights_path=weights_path_best)

        for run, epochs in enumerate(epochs_list): 
            print(f"{lr=}")
            
            save_model = True if (run == len(epochs_list) - 1) else False
            
            train_model(dataset, trainer, model, lr, epochs, save_model=save_model)     
            
            acc, loss = validator.validate(model)
            
            lr_train_list.append(lr)
            acc_train_list.append(acc)
            loss_train_list.append(loss)
            
            lr = lr + (lr * 0.000333)
            
        # Get latest run best weights from the latest saved model
        run = max([int(f.split('_')[-1]) for f in os.listdir(f"train/{model.name}") if os.path.isdir(os.path.join(f"train/{model.name}", f))])

        weights_path = f"train/{model.name}/run_{run}/best_weights.pth"
        model = ModelBuilder().build(ModelTypes.CONTINUOUSFLOW_MNIST, config=model_config, weights_path=weights_path)

        acc, loss = validator.validate(model)
        acc_train_list.append(acc)
        
        index = index + 1
        write_report(model_output_folder, epochs_list, lr_train_list, acc_train_list, loss_train_list)
