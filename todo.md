# ToDos

- SynapseLab
    - dataset_builder.py 
        - Clean up neccessary
    - jsc.py
        - Remove truncation out of JSC
        - Remove truncation and word width etc from configs
        - Clean up model builder

- Brevitas
    - int.py
        - Add input flag, to disable quantization
        - Add input flag, to change the sequence (first quant, second prune or vice versa)

- Model config change

    dataset:
        - name: "..."
        - train_path: "..."
        - test_path: "..."
        - num_workers: "..."
        - distributed: "..."
        - batch_size: [X, Y]
        - dataset_root_path: NONE | PATH
        - classes

    training:
        - epochs
        - lr
        - hyperparams_exploration
        - retrain
        - best_weights

    model:
        - name: "..."
        - backbone: ...
        - neck: ... 
        - head: ...