import argparse

#training options
parser = argparse.ArgumentParser(description='Training Change Detection Network')

# training parameters
parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--val_batchsize', default=16, type=int, help='batchsize for validation')
parser.add_argument('--num_workers', default=24, type=int, help='num of workers')
parser.add_argument('--n_class', default=2, type=int, help='number of class')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
parser.add_argument('--suffix', default=['.png','.jpg','.tif'], type=list, help='the suffix of the image files.')

parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')


# network saving
parser.add_argument('--model_dir', default='str', type=str, help='model save path')
parser.add_argument('--resume_dir', default='str', type = str, help = 'histort best path')
