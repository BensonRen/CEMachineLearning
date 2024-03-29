import parameters
from makeModel import LeNet5
import argparse
import os, sys
import time
import datetime
import numpy as np
# Import pytorch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm
import flag_reader

# You cannot change this line.
from tools.dataloader import CIFAR10
def recoverfromckpt(flags,net):
    # Code for loading checkpoint and recover epoch id.
    CKPT_PATH = "./saved_model/model.h5"

    def get_checkpoint(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path)
        except Exception as e:
            print(e)
            return None
        return ckpt

    ckpt = get_checkpoint(CKPT_PATH)
    if ckpt is None or flags.train_from_scratch:
        if not flags.train_from_scratch:
            print("Checkpoint not found.")
        print("Training from scratch ...")
        start_epoch = 0
        current_learning_rate = flags.initial_lr
    else:
        print("Successfully loaded checkpoint: %s" % CKPT_PATH)
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch'] + 1
        current_learning_rate = ckpt['lr']
        print("Starting from epoch %d " % start_epoch)

    return start_epoch,current_learning_rate


def gettraintest(flags):
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip()])
    # transforms.RandomChoice([
    # transforms.Grayscale(num_output_channels = 3),
    # transforms.RandomCrop(224),
    # transforms.RandomVerticalFlip(0.5),
    # transforms.RandomHorizontalFlip(0.5)])])

    transform_val = transform_train = transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                               (0.2023, 0.1994, 0.2010)),
                                                          ])

    # Call the dataset Loader
    trainset = CIFAR10(root=flags.dataroot, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=flags.train_batch_size, shuffle=True, num_workers=1)
    valset = CIFAR10(root=flags.dataroot, train=False, download=True, transform=transform_val)
    valloader = torch.utils.data.DataLoader(valset, batch_size=flags.val_batch_size, shuffle=False, num_workers=1)

    return trainloader,valloader

def train_one_epoch(i,global_step,net,criterion, optimizer):
    print(datetime.datetime.now())
    # Switch to train mode
    net.train()
    print("Epoch %d:" % i)

    total_examples = 0
    correct_examples = 0

    train_loss = 0
    train_acc = 0
    device = net.device
    trainloader = net.trainloader
    valloader = net.valloader
    # Train the training dataset for 1 epoch.
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # Copy inputs to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        # Zero the gradient
        optimizer.zero_grad()
        # Generate output
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # Now backward loss
        loss.backward()
        # Apply gradient
        optimizer.step()
        # Calculate predicted labels
        _, predicted = outputs.max(1)
        # Calculate accuracy
        # print((predicted == targets).sum().item())
        total_examples += list(inputs.size())[0]
        correct_examples += (predicted == targets).sum().item()

        train_loss += loss

        global_step += 1
        if global_step % 100 == 0:
            train_avg_loss = train_loss / (batch_idx + 1)
        pass
    train_avg_acc = correct_examples / total_examples
    print("Training loss: %.4f, Training accuracy: %.4f" % (train_avg_loss, train_avg_acc))
    print(datetime.datetime.now())
    # Validate on the validation dataset
    print("Validation...")
    total_examples = 0
    correct_examples = 0

    net.eval()

    val_loss = 0
    val_acc = 0
    # Disable gradient during validation
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # Copy inputs to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Zero the gradient
            optimizer.zero_grad()
            # Generate output from the DNN.
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # Calculate predicted labels
            _, predicted = outputs.max(1)
            # Calculate accuracy
            total_examples += list(inputs.size())[0]
            correct_examples += (predicted == targets).sum().item()
            val_loss += loss

    val_avg_loss = val_loss / len(valloader)
    val_avg_acc = correct_examples / total_examples
    print("Validation loss: %.4f, Validation accuracy: %.4f" % (val_avg_loss, val_avg_acc))


    return train_avg_loss, train_avg_acc, val_avg_loss, val_avg_acc,global_step

def lr_decay(i, flags, current_learning_rate, optimizer):
    """
    function that makes the decay of the learning rate, every X epoch would times decay rate
    :param i: epoch number now
    :param flags: the input flags which contains the decay_epoch information and the decay rate from Hyperparams
    :param current_learning_rate: The current decay rate
    :param optimizer: Optimizer to decay
    :return:
    """
    if i % flags.decay_epochs == 0 and i != 0:
        current_learning_rate =current_learning_rate * flags.decay_rate
        for param_group in optimizer.param_groups:
        # Assign the learning rate parameter
            param_group['lr'] = current_learning_rate
        print("Current learning rate has decayed to %f" % current_learning_rate)
    return current_learning_rate

def saveifbest(val_avg_acc, best_val_acc,flags, i, current_learning_rate, net):
    if val_avg_acc > best_val_acc:
        best_val_acc = val_avg_acc
        if not os.path.exists(flags.checkpoint_path):
            os.makedirs(flags.checkpoint_path)
        print("Saving ...")
        state = {'net': net.state_dict(),
                 'epoch': i,
                 'lr': current_learning_rate}
        torch.save(state, os.path.join(flags.checkpoint_path, 'model.h5'))

def train_from_flags(flags):
    print("data acquiring and preprocessing...")
    trainloader, valloader = gettraintest(flags)
    print("Building the model and network")

    # Specify the device for computation
    device = flags.device if torch.cuda.is_available() else 'cpu'
    net = LeNet5(trainloader,valloader, device)
    net = net.to(device)
    if device == 'cuda':
        print("Train on GPU...")
    else:
        print("Train on CPU...")

    print("Getting the pre-trained model / train from scratch")
    start_epoch, current_learning_rate = recoverfromckpt(flags,net)

    print("Starting from learning rate %f:" % current_learning_rate)

    print('Setting up the loss function...')
    # Create loss function and specify regularization
    criterion = nn.CrossEntropyLoss()  # L2 norminat weight_decay for optimizer
    # Add optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=flags.initial_lr, momentum=flags.momentum, weight_decay=flags.reg)


    # Start the training/validation process
    # The process should take about 5 minutes on a GTX 1070-Ti
    # if the code is written efficiently.
    global_step = 0
    best_val_acc = 0
    # Initialize the lists for training and validations
    # train_avg_loss_list, train_avg_acc_list, val_avg_loss_list, val_avg_acc_list = [],[],[],[]
    train_loss_acc_val_loss_acc = [[], [], [], []]
    for i in range(start_epoch, flags.epochs):
        train_avg_loss, train_avg_acc, val_avg_loss, val_avg_acc, global_step = train_one_epoch(i,global_step,net,criterion, optimizer)
        current_learning_rate = lr_decay(i,flags,current_learning_rate,optimizer)       #Do the update the
        saveifbest(val_avg_acc, best_val_acc, flags, i, current_learning_rate, net)     #Save the model if it is best
        train_loss_acc_val_loss_acc[0].append(train_avg_loss)
        train_loss_acc_val_loss_acc[1].append(train_avg_acc)
        train_loss_acc_val_loss_acc[2].append(val_avg_loss)
        train_loss_acc_val_loss_acc[3].append(val_avg_acc)

    dirname = os.path.join(flags.result_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for i in range(4):
        np.savetxt(os.path.join(dirname, '{}.txt'.format(i)), train_loss_acc_val_loss_acc[i], delimiter=',')
    val_avg_acc_nparr = np.array(train_loss_acc_val_loss_acc[3])
    flag_reader.write_flags(flags, max(val_avg_acc_nparr), dirname)
    print("optimization finished")

if __name__ == '__main__':
    flags = flag_reader.read_flag()
    train_from_flags(flags)
