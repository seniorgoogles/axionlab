import h5py
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, distributed
from typing import Union, List

from src.datasets.jetSubstructure.preprocessing import normalize


class CustomDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, classes: int):
        self.features = features
        self.targets = targets
        self.classes = classes

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: Union[Tensor, List[int], int]):
        if isinstance(idx, Tensor):
            idx = idx.tolist()

        inputs = self.features[idx]
        labels = self.targets[idx]

        return inputs, labels

    def split(self, validation_part: float = 0.2):
        random = np.random.RandomState(42)

        grouped_elements = [self.features[self.targets.argmax(axis=1) == c] for c in range(self.classes)]

        min_elements = np.min([len(arr) for arr in grouped_elements])
        elements_per_class = int(min_elements * (1 - validation_part))

        traind = np.empty((0, *self.features.shape[1:]))
        trainl = np.empty((0, *self.targets.shape[1:]))
        testd = np.empty((0, *self.features.shape[1:]))
        testl = np.empty((0, *self.targets.shape[1:]))

        for c, arr in enumerate(grouped_elements):
            random.shuffle(arr)

            train_part = arr[:elements_per_class]
            test_part = arr[elements_per_class:]

            label = np.zeros((1, *self.targets.shape[1:]))
            label[:, c] = 1

            traind = np.append(traind, train_part.copy(), axis=0)
            trainl = np.append(trainl, label.repeat(len(train_part), axis=0), axis=0)

            testd = np.append(testd, test_part.copy(), axis=0)
            testl = np.append(testl, label.repeat(len(test_part), axis=0), axis=0)

        trainp = random.permutation(len(traind))
        testp = random.permutation(len(testd))

        return CustomDataset(traind[trainp], trainl[trainp], self.classes), CustomDataset(testd[testp], testl[testp],
                                                                                          self.classes)


class JetSubstructureDataset(CustomDataset):
    __FEATURES = ["j_zlogz", "j_c1_b0_mmdt", "j_c1_b1_mmdt", "j_c1_b2_mmdt", "j_c2_b1_mmdt", "j_c2_b2_mmdt",
                  "j_d2_b1_mmdt", "j_d2_b2_mmdt", "j_d2_a1_b1_mmdt", "j_d2_a1_b2_mmdt", "j_m2_b1_mmdt", "j_m2_b2_mmdt",
                  "j_n2_b1_mmdt", "j_n2_b2_mmdt", "j_mass_mmdt", "j_multiplicity"]
    __TARGETS = ["j_g", "j_q", "j_w", "j_z", "j_t"]

    def __init__(self, dataset_path: str, batch_size, distributed_training: bool, num_workers):
        with h5py.File(dataset_path) as dataset:
            dataframe = pd.DataFrame(dataset["t_allpar_new"][:])  # type: ignore
            features = normalize(dataframe[self.__FEATURES].to_numpy())
            targets = dataframe[self.__TARGETS].to_numpy()

            super().__init__(features, targets, len(self.__TARGETS))

        self.batch_size_train = batch_size[0]
        self.batch_size_test = batch_size[1]
        self.distributed_training = distributed_training
        self.num_workers = num_workers

        self.train_dataset, self.test_dataset = self.split()

        if self.distributed_training:
            self.train_sampler = distributed.DistributedSampler(self.train_dataset)
            self.test_sampler = distributed.DistributedSampler(self.test_dataset)

    def get_train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size_train, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,
                          sampler=self.train_sampler if self.distributed_training else None)

    def get_test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_test, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          sampler=self.train_sampler if self.distributed_training else None)
