from easydict import EasyDict as edict

__C = edict()
cfg = __C


##################################
# general parameters
##################################

__C.general = {}

__C.general.csv_path = "train_occ.csv"

# the output of training models and logs
__C.general.save_dir = "model/fine" 

# continue training from certain epoch, -1 to train from scratch
__C.general.resume_epoch = -1

# when finetune from certain model, can choose clear start epoch idx
__C.general.clear_start_epoch = False

# the number of GPUs used in training
__C.general.num_gpus = 3

# random seed used in training (debugging purpose)
__C.general.seed = 0

##################################
# data set parameters
##################################

__C.dataset = {}

__C.dataset.num_modality = 2

# the number of classes
__C.dataset.num_classes = 2

# the resolution on which segmentation is performed
__C.dataset.spacing = [0.5,0.5,0.5]

# the sampling crop size, e.g., determine the context information
__C.dataset.crop_size = [128, 128, 128]

__C.dataset.window_center = 100

__C.dataset.window_width = 300


#####################################
# net
#####################################

__C.net = {}

# the network name
__C.net.name = 'unet'

# whether synchronizes bn layer between GPUs
# The batch size of BN layers is __C.train.batchsize if it is True, else __C.train.batchsize // num_gpus which is consistent with DP
# This parameter works only when using distributed training, but it can lead to slower training
__C.net.bn_sync = False


######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
__C.train.epochs = 50

# the number of samples in a batch
__C.train.batchsize = 12

# the number of threads for IO
__C.train.num_threads = 20

# the learning rate
__C.train.lr = 1e-4

##### lr_scheduler
__C.train.lr_scheduler = {}
__C.train.lr_scheduler.name = "MultiStep"
__C.train.lr_scheduler.params = {"milestones": [1000,2000,3000], "gamma": 0.1, "last_epoch": -1}

##### optimizer
__C.train.optimizer = {}
__C.train.optimizer.name = "Adam"
__C.train.optimizer.params = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0, "amsgrad": False}

# the number of batches to update loss curve
__C.train.plot_snapshot = 100

# the number of batches to save model
__C.train.save_epochs = 50

__C.train.weight_target = 10
__C.train.weight_background = 1

########################################
# debug parameters
########################################

__C.debug = {}

# whether to save input crops
__C.debug.save_inputs = False

