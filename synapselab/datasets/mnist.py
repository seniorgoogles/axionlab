import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, distributed
import torchvision.transforms.functional as F

class BorderCrop:
    def __init__(self, border):
        self.border = border

    def __call__(self, img):
        width, height = img.size
        return F.crop(img, self.border, self.border, height - 2 * self.border, width - 2 * self.border)


class Mnist(object):

    def __init__(self, train_path, test_path, batch_size, distributed_training, num_workers, crop_border_pixels=0):
        self.batch_size_train = batch_size[0]
        self.batch_size_test = batch_size[1]

        self.distributed_training = distributed_training
        self.num_workers = num_workers

        # Do preprocessing
        self.train_dataset, self.test_dataset = self.__do_preprocessing__(crop_border_pixels)

        # If distributed, use distributed sampler for multiple GPUs
        if self.distributed_training:
            self.train_sampler = distributed.DistributedSampler(self.train_dataset)
            self.test_sampler = distributed.DistributedSampler(self.test_dataset)


    def __do_preprocessing__(self, crop_border_pixels):
        transform = None 
        
        if crop_border_pixels > 0:
            transform = T.Compose([
                BorderCrop(border=2),
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
            ])
        else:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),                
            ])
        
        
        train_dataset = datasets.MNIST('../tmp/dataset/mnist', train=True, download=True,transform=transform)
        test_dataset = datasets.MNIST('../tmp/dataset/mnist', train=False,transform=transform)
        
        # Print shape of the dataset
        print(f"{train_dataset.data.shape=}")
        
        return train_dataset, test_dataset

    def get_train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size_train, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,
                          sampler=self.train_sampler if self.distributed_training else None)

    def get_test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_test, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          sampler=self.train_sampler if self.distributed_training else None)