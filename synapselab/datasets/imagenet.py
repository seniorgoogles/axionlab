import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, distributed

class ImageNet(object):

    def __init__(self, train_path, test_path, batch_size, distributed_training, num_workers):
        self.batch_size_train = batch_size[0]
        self.batch_size_test = batch_size[1]
        self.train_path = train_path
        self.test_path = test_path
        self.distributed_training = distributed_training
        self.num_workers = num_workers

        # Do preprocessing
        self.train_dataset, self.test_dataset = self.__do_preprocessing__()

        # If distributed, use distributed sampler for multiple GPUs
        if self.distributed_training:
            self.train_sampler = distributed.DistributedSampler(self.train_dataset)
            self.test_sampler = distributed.DistributedSampler(self.test_dataset)


    def __do_preprocessing__(self):

        # === data transformation === #
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_T = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_T = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize, ])

        train_dataset = datasets.ImageFolder(root=self.train_path,transform=train_T)
        test_dataset = datasets.ImageFolder(root=self.test_path,transform=test_T)

        return train_dataset, test_dataset

    def get_train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size_train, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,
                          sampler=self.train_sampler if self.distributed_training else None)

    def get_test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_test, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          sampler=self.train_sampler if self.distributed_training else None)