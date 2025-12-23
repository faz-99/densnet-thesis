# Configuration for DenseNet vs Swin Transformer comparison
# GPU parameters
is_cuda = False
device = 'cpu'
is_parallel = False
gpu_id = "0"
gpu_ids = [0]

# Dataset parameters
dataset_mean = (0.5613, 0.5778, 0.6032)
dataset_std = (0.2114, 0.1957, 0.1590)

# Training parameters
net_name = "iaff40"
batch_size = 32
num_workers = 0
max_epoch = 5
warmup_epochs = 5
warmup_steps = -1

lr = 0.003
min_lr = 1e-6
weight_decay = 0.05

milestones = [20, 40, 60, 80]
gamma = 0.5

img_s = 224

# Dataset configuration
class_num = 2
train = "datasets/BreaKHis 400X/train"
valid = "datasets/BreaKHis 400X/test"

# Model parameters
drop_path = 0.8