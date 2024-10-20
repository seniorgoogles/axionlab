import shutil
from includes_ml2 import *

if __name__ == "__main__":

    retrain = True
    train_folder = f"{parent_directory}/train"

    if input(f"Delete previous trainings? [y/N]: ").strip().lower() == "y":
        clear_folder(train_folder)
    
    # Create a logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)


    checkpoint = 9

    teacher_config = f"{parent_directory}/configs/jsc/jsc_xxl_floating_point.yaml"
    teacher_weights = f"/home/mmecik/repositories/synapselab/train/jsc_xxl_floating_point/run_{checkpoint}/best_weights.pth"

    student_config = f"{parent_directory}/configs/jsc/jsc_2l.yaml"
    
    trainer = Trainer()
    dataset = DatasetBuilder.build(DatasetTypes.JSC, config=teacher_config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())

    teacher = ModelBuilder().build(ModelTypes.JSC, config=teacher_config, preload_weights=False, weights_path=teacher_weights)   
    student = ModelBuilder().build(ModelTypes.JSC, config=student_config, preload_weights=False)   
    # Best acc lr 0.0010777777777777778
    best_lr = 0.0010777777777777778
    
    min_rate = 0.0013
    max_rate = 0.001433399
    num_rates = 100

    epochs = 100
    lr_list = np.linspace(min_rate, max_rate, num_rates).tolist()

    train_student_teacher(dataset, trainer, validator, teacher, student, lr_list, epochs, logger)