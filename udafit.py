import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from modifiedNet.transPCD import PcdmViT
import loss
import itertools
from torch.utils.data import DataLoader
from CTXdataset import testLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (11 + gamma * iter_num / max_iter) ** (-power)
    # decay = (1 + gamma) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-4
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def cal_cd_acc(loader, model, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            batchData = iter_test.next()
            data1 = batchData['I1'].float().to(device)
            data2 = batchData['I2'].float().to(device)
            labels = batchData['label'].float().to(device)
            outputs = model(data1, data2)

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item() / np.log(all_label.size()[0])

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int), :]
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc, predict, mean_ent
    else:
        return accuracy * 100, mean_ent, predict, mean_ent


def train_target(args):
    best_acc=0
    model = PcdmViT()
    path_checkpoint = args.resumepath
    checkpoint = torch.load(path_checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    CDNet = model.to(device, dtype=torch.float)

    #optimizer = optim.SGD(itertools.chain(CDNet.parameters()), lr=args.lr)
    optimizer = optim.Adam(itertools.chain(CDNet.parameters()), lr=args.lr, betas=(0.9, 0.999))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(testLoader)
    interval_iter = 240 #max_iter // 10
    iter_num = 0

    model.eval()
    acc_s_te, _, pry, mean_ent = cal_cd_acc(testLoader, model, False)
    log_str = 'Task: CTX, Iter:{}/{}; Accuracy={:.2f}%, Ent={:.3f}'.format(iter_num, max_iter, acc_s_te,mean_ent)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')
    model.train()


    old_pry = 0
    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            batchData = iter_test.next()
            data1 = batchData['I1'].float().to(device)
            data2 = batchData['I2'].float().to(device)
            labels = batchData['label'].float().to(device)
            #inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(testLoader)
            batchData = iter_test.next()
            data1 = batchData['I1'].float().to(device)
            data2 = batchData['I2'].float().to(device)
            labels = batchData['label'].float().to(device)

        
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter, power=0.75)

        # features_test = netB(netF(inputs_test))
        # outputs_test = netC(features_test)
        outputs_test = model(data1,data2)

        softmax_out = nn.Softmax(dim=1)(outputs_test)
        avalue = loss.Entropy(softmax_out)  
        entropy_loss = torch.mean(avalue)  # Okay, it's just a definition.

        msoftmax = softmax_out.mean(dim=0)
        bvalue = msoftmax * torch.log(msoftmax + 1e-5)
        gentropy_loss = -torch.sum(bvalue)
        entropy_loss -= gentropy_loss
        
        entropy_loss.backward()
        optimizer.step()  # all the same
        scheduler.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()
            acc_s_te, _, pry, mean_ent = cal_cd_acc(testLoader, model, False)
            log_str = 'Task: CTX, Iter:{}/{}; Accuracy={:.2f}%, Ent={:.3f}'.format(iter_num, max_iter, acc_s_te,
                                                                                   mean_ent)
            
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            if acc_s_te > best_acc or iter_num==1 or iter_num == max_iter:
                best_acc = acc_s_te
                torch.save(CDNet.state_dict(),  args.model_dir+'netCD_iter_%d.pth' % (iter_num))
                print("Best Already saved: acc=",best_acc)
            else:
                print("Model not saved: history best_acc=",best_acc)
            model.train()

            if torch.abs(pry - old_pry).sum() == 0:
                break
            else:
                old_pry = pry.clone()

    return model


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DINE')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--max_epoch', type=int, default=1000, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=2, help="batch_size")
    parser.add_argument('--worker', type=int, default=24, help="number of workers")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet18, resnet50, resnext50")
    parser.add_argument('--net_src', type=str, default='resnet50',
                        help="alexnet, vgg16, resnet18, resnet34, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='./ckps/tar')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--weight-decay', type = float, default = 1e-5, metavar = 'WD',
                help = 'Weight decay (default: 1e-5)')
    # network saving
    parser.add_argument('--model_dir', default='./finetune_mvit_seed2022/', type=str, help='model save path')
    parser.add_argument('--resumepath', default='./trainwell/netCD_epoch_170_097244.pth', type=str, help='histry saved path')
    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    #folder = './data/'
    #args.output_dir = './finetune_ctx/'
    if not osp.exists(args.model_dir):
        os.system('mkdir -p ' + args.model_dir)
    if not osp.exists(args.model_dir):
        os.mkdir(args.model_dir)
        

    args.out_file = open(osp.join(args.model_dir, 'log_finetune.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

    train_target(args)