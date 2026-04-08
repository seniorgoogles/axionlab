import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, distributed

class Cifar10(object):

    def __init__(self, train_path, test_path, batch_size, distributed_training, num_workers):
        self.batch_size_train = batch_size[0]
        self.batch_size_test = batch_size[1]

        self.distributed_training = distributed_training
        self.num_workers = num_workers

        # Do preprocessing
        self.train_dataset, self.test_dataset = self.__do_preprocessing__()

        # If distributed, use distributed sampler for multiple GPUs
        if self.distributed_training:
            self.train_sampler = distributed.DistributedSampler(self.train_dataset)
            self.test_sampler = distributed.DistributedSampler(self.test_dataset)


    def __do_preprocessing__(self):

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
        train_dataset = datasets.Cifar10('../tmp/dataset/cifar10', train=True, download=True, transform=transform)
        test_dataset = datasets.Cifar10('../tmp/dataset/cifar10', train=False, transform=transform)

        return train_dataset, test_dataset

    def get_train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size_train, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,
                          sampler=self.train_sampler if self.distributed_training else None)

    def get_test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_test, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          sampler=self.test_sampler if self.distributed_training else None)