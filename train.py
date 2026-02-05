import argparse
import os
import torch.multiprocessing as mp

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.nn import CrossEntropyLoss, Conv2d
from torchvision.models import resnet34
from torchvision import transforms
from torchvision import models
import pandas as pd

from src.utils import calculate_weights
from src.dataset import DistributedWeightedSampler, ImageDataset
from src.models import ResnetABMIL

import wandb

from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score

import numpy as np
import random


def train_step(train_loader, model, criterion, optimizer):
    model.train()
    training_epoch_loss = 0
    acc = 0
    tests = 0
    targets = []
    outputs = []
    for i, batch in enumerate(train_loader):
        img_tensor, target, filename = batch[0].cuda(), batch[1].cuda(), batch[2]
        targets.append(target)
        #print(target)
        output = model(img_tensor)
        #print(output)
        if isinstance(output, (tuple, list)):
            output = output[0]
        # print(output)
        optimizer.zero_grad()
        loss = criterion(output, target)
        #print(loss)
        loss.backward()
        optimizer.step()
        training_epoch_loss += loss.item()
        predicted_classes = torch.max(output, dim=1)[1]
        #print(predicted_classes)
        outputs.append(predicted_classes)
        acc += (predicted_classes == target).sum()
        tests += len(predicted_classes)
        #raise RuntimeError('debug')
        # if "/projects/ag-bozek/sugliano/dlbcl/data/interim/resnet_imgs/004_for_resnet.npy" in filename:
        #     print(filename)
        #     print(target)
        #     print(output)
        #     print(loss)
        #     print(predicted_classes)

    true_vals = torch.tensor([k for t in targets for k in t]).cpu().numpy()#torch.tensor([t.cpu().numpy()[k] for t in targets for k in t])
    predicts = torch.tensor([k for t in outputs for k in t]).cpu().numpy()#torch.tensor([t.cpu().numpy()[k] for t in outputs for k in t])
    ccmm = confusion_matrix(true_vals, predicts)

    training_phase_results = {
        'Loss': training_epoch_loss / ( (i+1) ),
        'Accuracy': acc.item() / tests,
        'Balanced_acc': balanced_accuracy_score(true_vals, predicts),
        'F1': f1_score(true_vals, predicts),
        'confusion_matrix': ccmm,
        'Learning rate': optimizer.param_groups[0]['lr']}

    return training_phase_results


def validate_step(val_loader, model, criterion):
    model.eval()
    val_epoch_loss = 0
    acc = 0
    tests = 0
    targets = []
    outputs = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            img_tensor, target, filename = batch[0].cuda(), batch[1].cuda(), batch[2]
            # print("filename: ", filename)
            targets.append(target)
            # print("target: ", target)
            output = model(img_tensor)
            # print("output: ", output)
            if isinstance(output, (tuple, list)):
                output = output[0]
            loss = criterion(output, target)
            # print("loss: ", loss)
            val_epoch_loss += loss.item()
            predicted_classes = torch.max(output, dim = 1)[1]
            # print("predicted_class: ", predicted_classes)
            outputs.append(predicted_classes)
            acc += (predicted_classes == target).sum()
            tests += len(predicted_classes)
            # raise RuntimeError('debug')
            # if "/projects/ag-bozek/sugliano/dlbcl/data/interim/resnet_imgs/004_for_resnet.npy" in filename:
            #     print(filename)
            #     print(target)
            #     print(output)
            #     print(loss)
            #     print(predicted_classes)

    true_vals = torch.tensor([k for t in targets for k in t]).cpu().numpy()#torch.tensor([t.cpu().numpy()[k] for t in targets for k in t])
    predicts = torch.tensor([k for t in outputs for k in t]).cpu().numpy()#torch.tensor([t.cpu().numpy()[k] for t in outputs for k in t])
    ccmm = confusion_matrix(true_vals, predicts)


    val_phase_results = {
        'Loss': val_epoch_loss / ((i+1)),
        'Accuracy' : acc.item() / tests,
        'Balanced_acc': balanced_accuracy_score(true_vals, predicts),
        'F1' : f1_score(true_vals, predicts),
        'confusion_matrix' : ccmm
    }
    return val_phase_results

def main_worker(args):

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="bozek-lab",
        # Set the wandb project where this run will be logged.
        project="DLBCL_project",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": args.learning_rate, #todo change learning rate
            "architecture": args.model,
            "dataset": "DLBCL",
            "nb_channel": 14,
            "epochs": args.epochs,
            "task": args.task,
            "batch_size": args.batch_size, #todo larger batch?
            "image_size": args.img_size,
            "num_workers": args.num_workers,
            "weight_decay": 1e-8,
            "scheduler_factor": args.scheduler_factor,
            "scheduler_patience": args.scheduler_patience,
            "loss": "Cross_entropy", #todo change loss
            "optimizer": "ADAM",
            "scheduler": "ReduceLROnPlateau", #todo change for balanced accuracy?
            "shuffle": True,
            "train_set" : args.train_csv,
            "val_set" : args.val_csv,
            "preprocessing?" : "resizing + small crop + normalized"
        },
    )

    os.makedirs(os.path.join(args.checkpoints_dir,run.id))

    # if torch.cuda.is_available():
    #     torch.cuda.set_device(proc_index)
    # else:
    #     raise RuntimeError('CUDA not available!')

    # if dist.is_nccl_available():
    #     dist.init_process_group(
    #         backend = 'nccl',
    #         world_size = args.gpus,
    #         rank = proc_index
    #     )
    # else:
    #     raise RuntimeError('NCCL backend not available!')

    if args.task == 'ihc-score':
        args.num_classes = 4
    elif args.task == 'relapse':
        args.num_classes = 2
    elif args.task == 'hans_binary':
        args.num_classes = 2
    else:
        raise ValueError('Task should be ihc-score or her-status')

    if args.model == 'resnet34':
        model = resnet34(weights = models.ResNet34_Weights.IMAGENET1K_V1)#, num_classes = args.num_classes)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
        #for more channel we would need to change the first conv ->
        # model.conv1 = Conv2d(14, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model = model.cuda()
    elif args.model == 'abmil':
        model = ResnetABMIL(num_classes = args.num_classes).cuda()
    else:
        raise ValueError('Model should be resnet34 or abmil')

    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    # model = DistributedDataParallel(model, device_ids=[proc_index], output_device=proc_index)

    #remove this part later
    # checkpoint = torch.load("checkpoints/ibmv8i45/checkpoint_99.pth.tar")
    # model.load_state_dict(checkpoint['model'])
    # mean = [ 60.1976, 120.2014, 103.0581,  76.7479, 145.5347, 112.5264, 144.9628,
    #      127.3409,  53.1630, 144.0218, 111.1871, 125.4579, 134.6648, 133.0389]
    # std = [56.2254, 54.2254, 47.3853, 53.8022, 54.0022, 48.1101, 53.4882, 51.5245,
    #      38.4545, 60.6355, 47.2878, 47.2716, 52.5021, 46.8046]
    mean = [ 60.1976, 120.2014, 144.9628]
    std = [56.2254, 54.2254, 53.4882]

    train_transform = transforms.Compose([
        transforms.CenterCrop(5000),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Normalize(mean, std),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.ToTensor(),
    ])

    train_df = pd.read_csv(args.train_csv)
    train_dataset = ImageDataset(train_df, fn_col = 'filename', lbl_col = args.task, transform = train_transform, return_filename=True, which_channels = [[0, 1, 6]])
    if args.weighted_sampler_label == 'None':
        args.weighted_sampler_label = args.task
    # weights = calculate_weights(torch.tensor(train_df[args.weighted_sampler_label].values))
    # train_sampler = DistributedWeightedSampler(weights, num_replicas=args.gpus, rank=proc_index, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

    if args.val_csv != 'None':
        val_transform = transforms.Compose([
            transforms.CenterCrop(5000),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.Normalize(mean, std),
            #transforms.ToTensor(),
        ])
        val_df = pd.read_csv(args.val_csv)
        val_dataset = ImageDataset(val_df, fn_col = 'filename', lbl_col = args.task, transform = val_transform, return_filename=True, which_channels = [[0, 1, 6]])
        # val_sampler = DistributedSampler(val_dataset, num_replicas=args.gpus, rank=proc_index, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=1e-15)

    criterion = CrossEntropyLoss()

    epoch0 = 0
    epoch = epoch0
    while epoch < epoch0 + args.epochs:
        train_phase_results = {'Loss': '', 'Accuracy': '', "Balanced_acc": ""}
        train_phase_results = train_step(train_loader, model, criterion, optimizer)
        val_phase_results = {'Loss': '', 'Accuracy' : '', "Balanced_acc": ""}
        if args.val_csv != 'None':
            val_phase_results = validate_step(val_loader, model, criterion)
            # val_phase_results = train_step(val_loader, model, criterion, optimizer)
            acc = val_phase_results['Accuracy']
            scheduler.step(acc)

        if True:#(proc_index == 0):
            run.log({"loss_train": train_phase_results["Loss"],
                     "acc_train": train_phase_results["Accuracy"],
                     "bal_acc_train": train_phase_results["Balanced_acc"],
                     "loss_val": val_phase_results["Loss"],
                     "acc_val": val_phase_results["Accuracy"],
                     "bal_acc_val": val_phase_results["Balanced_acc"]})
            print('Epoch {} finished.'.format(epoch))
            print('Train phase: ', train_phase_results)
            print('Val phase: ', val_phase_results)
            print('\n')

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'accuracy': val_phase_results['Accuracy']

            }, os.path.join(args.checkpoints_dir,run.id,'checkpoint_{}.pth.tar'.format(epoch)))
        epoch += 1

    run.finish()



def get_args():
    parser = argparse.ArgumentParser(description='Train HER2 overexpression classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', dest='model', type=str, default='resnet34', help='resnet34 or abmil')
    parser.add_argument('--task', dest='task', type=str, default='hans_binary', help='ihc-score or her2-status')
    parser.add_argument('--weighted_sampler_label', dest='weighted_sampler_label', type=str, default='None', help='Additional label in the train .csv to weight the sampling')
    parser.add_argument('--gpus', dest='gpus', type=int, default=4, help='Number of GPUs')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs', dest='epochs')
    parser.add_argument('--learning_rate', dest="learning_rate", type=float, nargs='?', default=0.0001, help='Learning rate')
    parser.add_argument('--scheduler_factor', dest="scheduler_factor", type=float, nargs='?', default=0.1, help='Scheduler factor for decreasing learning rate')
    parser.add_argument('--scheduler_patience', dest="scheduler_patience", type=int, nargs='?', default=10, help='Scheduler patience for decreasing learning rate')
    parser.add_argument('--batch_size', type=int, nargs='?', default=4, help='Batch size', dest='batch_size')
    parser.add_argument('--train_csv', dest='train_csv', type=str, default='train_hans_test.csv', help='.csv file containing the training examples')
    parser.add_argument('--val_csv', dest='val_csv', type=str, default='train_hans_test.csv', help='.csv file containing the val examples')
    parser.add_argument('--checkpoints_dir', dest='checkpoints_dir', type=str, default='./checkpoints', help='Path to save model checkpoints')
    parser.add_argument('--ip_address', dest='master_addr', type=str, default='localhost', help='IP address of rank 0 node')
    parser.add_argument('--port', dest='master_port', type=str, default='8888', help='Free port on rank 0 node')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0, help='Number of workers for loading data')
    parser.add_argument('--img_size', dest='img_size', type=int, default=1024, help='Input image size for the model')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.checkpoints_dir, exist_ok = True)
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    main_worker(args=args)
    # mp.spawn(main_worker, nprocs=args.gpus, args=(args,))
