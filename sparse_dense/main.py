
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from data import nyu_v2
from models import tiramisu
import utils.training as train_utils

writer = SummaryWriter()

# CONSTANTS
NYU_V2_TRAINING_PATH = './data/NYU_V2/training'
NYU_V2_VALIDATING_PATH = './data/NYU_V2/validating'
NYU_V2_TESTING_PATH = './data/NYU_V2/testing'
RESULTS_PATH = Path('.results/')
WEIGHTS_PATH = Path('.weights/')
RESULTS_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH.mkdir(exist_ok=True)
batch_size = 2

if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
if not os.path.exists(WEIGHTS_PATH):
    os.makedirs(WEIGHTS_PATH)

# dataset
#transforms2apply = transforms.Compose((transforms.Resize(200), transforms.ToTensor()))
transforms2apply = transforms.ToTensor()

train_dset = nyu_v2.NYU_V2(NYU_V2_TRAINING_PATH, 0, transform=transforms2apply )
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=False)

val_dset = nyu_v2.NYU_V2(NYU_V2_VALIDATING_PATH, 0, transform=transforms2apply)
val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, shuffle=False)

for idx, data in enumerate(train_loader):
    inputs = data[0]
    targets = data[1]

    images = inputs[:,0:3,:,:]
    images_grid = vutils.make_grid(images, normalize=True, scale_each=True)
    writer.add_image('Image', images_grid, idx)

    s1 = inputs[:,3:4,:,:]
    s1_grid = vutils.make_grid(s1, normalize=True, scale_each=True)
    writer.add_image('Depths', s1_grid, idx)

    s2 = inputs[:,4:5,:,:]
    s2_grid = vutils.make_grid(s2, normalize=True, scale_each=True)
    writer.add_image('Dist', s2_grid, idx)

    depths_grid = vutils.make_grid(targets, normalize=True, scale_each=True)
    writer.add_image('GT_Depths', depths_grid, idx)

    dummy_s1 = torch.rand(1)
    writer.add_scalar('data/scalar1', dummy_s1[0], idx)

    if idx > 5:
        break
writer.close()
exit()

# training
LR = 1e-4
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1
N_EPOCHS = 2

use_cuda = False #torch.cuda.available()

if use_cuda :
    torch.cuda.manual_seed(0)
else:
    torch.manual_seed(0)

model = tiramisu.FCDenseNet(in_channels=5,
                            down_blocks=(4, 4, 4, 4, 4),
                            up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
                            growth_rate=12, out_chans_first_conv=48,
                            n_classes=1)

model.apply(train_utils.weights_init)
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
if use_cuda :
    criterion = nn.MSELoss().cuda()
else:
    criterion = nn.MSELoss()

for epoch in range(1, N_EPOCHS + 1):
    since = time.time()

    ### Train ###
    trn_loss, trn_err = train_utils.train(model, train_loader, optimizer, criterion, epoch)
    print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(epoch, trn_loss, 1 - trn_err))
    time_elapsed = time.time() - since
    print('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    ### Test ###
    val_loss, val_err = train_utils.test(model, val_loader, criterion, epoch)
    print('Val - Loss: {:.4f} | Acc: {:.4f}'.format(val_loss, 1 - val_err))
    time_elapsed = time.time() - since
    print('Total Time {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))

    ### Checkpoint ###
    train_utils.save_weights(model, epoch, val_loss, val_err)

    ### Adjust Lr ###
    train_utils.adjust_learning_rate(LR, LR_DECAY, optimizer, epoch, DECAY_EVERY_N_EPOCHS)