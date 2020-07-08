#The main purpose of the custom data loader is to feed custom data
#eg. random data to the model.
#This will be useful in the following cases
#1. feeding some data and be able to run the model when actual data set
#is not available.
#2. the actual data loading has a lot of overhead, say due to disk or
#network access/preproc, affecting overall execution time.
#
#This can be mainly used for perf profiling of model execution,
#stability tests and limited convergence tests (needs some enhancements
#for determinism) for feature additions that do not change the numerical
#aspects.
#
#Currently data loading with random data for models like resnet/MNIST that
#use image data is supported.
#
#Use the following env vars to configure the dataset attributes:
#
#CDL_IMG_NUM_CHANNELS
#	 --> number of channels in the image(default :3)
#CDL_IMG_HEIGHT
#	 --> image height( default : 224)
#CDL_IMG_WIDTH
#	  --> image width (default(: 224)
#
#CDL_NUM_CLASSES
#	 --> number of classes in the data set( default : 1000)
#
#CDL_TRAIN_DATASET_SIZE
#	--> Train data set size (default: 10 times batch size)
#
#CDL_TEST_DATASET_SIZE
#	--> Test data set size (default: 2 times batch size)


import os
import torch
# For topologies like resnet, mnist etc running on image data
#Default is imagenet data dims. Default data format is NCHW
#Data set size and number of classes are to be configured via env variables
class ImageRandomDataLoader():
    def __init__(self, batch_size, train=True, drop_last=False):

        def get_val(env_var, default_val):
            val = default_val
            val_str = os.environ.get(env_var)
            if val_str is not None:
                val = int(val_str)
            return val

        self.batch_size = batch_size
        self.channels   = C = get_val('CDL_IMG_NUM_CHANNELS', 3)
        self.height     = H = get_val('CDL_IMG_HEIGHT', 224)
        self.width      = W = get_val('CDL_IMG_WIDTH', 224)
        self.num_classes = get_val('CDL_NUM_CLASSES', 1000)
        if train:
            phase = 'train'
            self.dataset_size = get_val('CDL_TRAIN_DATASET_SIZE', batch_size*10) #num images; default 10 iterations
        else:
            phase = 'test'
            self.dataset_size = get_val('CDL_TEST_DATASET_SIZE', batch_size*2) #num images; default 2 iterations

        self.train = train
        self.num_batches = self.dataset_size // self.batch_size
        self.curr_batch_no = 0
        self.partial_batch_size = 0
        if not drop_last:
            self.partial_batch_size = self.dataset_size % self.batch_size

        if self.partial_batch_size:
            self.num_batches = self.num_batches + 1
        print("ImageRandomDataLoader: ", "phase=", phase, " N=", batch_size, " C=", C, " H=", H," W=", W,
                " num classes=", self.num_classes, " dataset size=", self.dataset_size,
                "num batches(iterations) per epoch=",self.num_batches, " drop_last =", drop_last)

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        self.curr_batch_no = self.curr_batch_no + 1
        if self.curr_batch_no <= self.num_batches :
            batch_size = self.batch_size
            if self.curr_batch_no == self.num_batches and self.partial_batch_size:
                batch_size = self.partial_batch_size

            target = torch.randint(0, self.num_classes, (batch_size,))
            image = torch.randn(batch_size, self.channels, self.height, self.width)
            return image, target
        else:
            self.curr_batch_no = 0
            raise StopIteration




