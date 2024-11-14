import shutil

from matplotlib import pyplot as plt
from includes_ml2 import *


if __name__ == "__main__":

    retrain = True
    train_folder = f"{parent_directory}/train"

    # Create a logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)


    checkpoint = 331

    model_config = f"{parent_directory}/configs/jsc/jsc_xl_wo_bias.yaml"
    #best_weights = f"/home/mmecik/repositories/synapselab/train/jsc_xl_floating_point/run_{checkpoint}/best_weights.pth"

    trainer = Trainer()
    dataset = DatasetBuilder.build(DatasetTypes.JSC, config=model_config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())

    model = ModelBuilder().build(ModelTypes.JSC, config=model_config)
    
    # Best acc lr 0.0010777777777777778
    best_lr = 0.0010777777777777778
    
    min_rate = 0.00009
    max_rate = 0.0001
    num_rates = 100

    epochs = 100
    lr_list = np.linspace(min_rate, max_rate, num_rates).tolist()
    """
    # Evaluate every folder from 
    #train_model(dataset, trainer, validator, None, model, lr_list, epochs, "result_lr_acc.txt", reload_weights=False, reload_weights_acc=False, logger=logger)
    best_acc = 0.0
    best_weights_index = 0
    for index in range(1, 395):
        
        try: 
            best_weights = f"/home/mmecik/repositories/synapselab/train/jsc_xl/run_{index}/best_weights.pth"

            model.load_state_dict(torch.load(best_weights,  weights_only=False), strict=False)
            acc, _ = validator.validate(model)
            
            if acc > best_acc:
                best_acc = acc
                best_weights_index = index
                best_weights = f"/home/mmecik/repositories/synapselab/train/jsc_xl/run_{index}/best_weights.pth"
                print(f"New best acc: {best_acc} at {index}")
        except Exception as ex:
            print(ex)
            
    print(f"Best acc: {best_acc} at {best_weights_index}")
    
    """
    best = 0.0010785011185682326
    min_rate = best - (best * 0.1)
    max_rate = best + (best * 0.1)
    num_rates = 150
    
    epochs = 150
    lr_list = np.linspace(min_rate, max_rate, num_rates).tolist()
    print(lr_list)
    
    model = ModelBuilder().build(ModelTypes.JSC, config=model_config)
    #model = ModelBuilder().build(ModelTypes.JSC, config=model_config)

    
    if input(f"Delete previous trainings? [y/N]: ").strip().lower() == "y":
        clear_folder(train_folder)
        
    lr_train_list = []
    
    for lr in lr_list:
        
        
        model = ModelBuilder().build(ModelTypes.JSC, config=model_config)
        print(f"{lr=}")
        lr_train_list.append(lr)
        train_model(dataset, trainer, validator, None, model, lr, 70, "result_lr_acc.txt", reload_weights=True, reload_weights_acc=False, logger=logger)
        
        lr = lr + (lr * 0.005)
        lr_train_list.append(lr)

        print(f"{lr=}")
        train_model(dataset, trainer, validator, None, model, lr, 30, "result_lr_acc.txt", reload_weights=False, reload_weights_acc=False, logger=logger)
        
        lr = lr + (lr * 0.005)
        lr_train_list.append(lr)

        print(f"{lr=}")
        
        train_model(dataset, trainer, validator, None, model, lr, 30, "result_lr_acc.txt", reload_weights=False, reload_weights_acc=False, logger=logger)
        
        lr = lr +(lr * 0.005)
        lr_train_list.append(lr)
        
        print(f"{lr=}")
        train_model(dataset, trainer, validator, None, model, lr, 30, "result_lr_acc.txt", reload_weights=False, reload_weights_acc=False, logger=logger)

        # Plot LR train list
        plt.plot(lr_train_list)
        plt.show()
    # Go through all the folders and evaluate the model
    """
    best_acc = 0.0
    best_acc_index = 0
    for index in range(0, 333):
        
        try: 
            best_weights = f"/home/mmecik/repositories/synapselab/train/jsc_xl/run_{index}/best_weights.pth"

            model.load_state_dict(torch.load(best_weights,  weights_only=False), strict=False)
            acc, _ = validator.validate(model)
            
            if acc > best_acc:
                best_acc = acc
                best_acc_index = index
            
            print(f"Acc: {acc} at {index}")
        except Exception as ex:
            print(ex)
            
    print(f"Best acc: {best_acc} at {best_acc_index}")
    """