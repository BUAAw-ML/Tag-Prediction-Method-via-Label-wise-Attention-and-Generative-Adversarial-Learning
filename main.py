import argparse
from engine import *
from models import *
from util import *
from dataLoader import *
from transformers import BertModel

import datetime

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Training Super-parameters')

parser.add_argument('-seed', default=0, type=int, metavar='N',
                    help='random seed')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[40], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--G-lr', '--Generator-learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--D-lr', '--Discriminator-learning-rate', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--B-lr', '--Bert-learning-rate', default=0.001, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save_model_path', default='./checkpoint', type=str,
                    help='path to save checkpoint (default: none)')
parser.add_argument('--log_dir', default='./logs', type=str,
                    help='path to save log (default: none)')
parser.add_argument('--data_type', default='All', type=str,
                    help='The type of data')
parser.add_argument('--data_path', default='../datasets/AAPD/aapd2.csv', type=str,
                    help='path of data')
parser.add_argument('--bert_trainable', default=True, type=bool,
                    help='bert_trainable')
# parser.add_argument('--utilize_unlabeled_data', default=1, type=int,
#                     help='utilize_unlabeled_data')
parser.add_argument('--use_previousData', default=0, type=int,
                    help='use_previousData')
parser.add_argument('--method', default='MultiLabelMAP', type=str,
                    help='Method')
parser.add_argument('--overlength_handle', default='truncation', type=str,
                    help='overlength_handle')

global args, use_gpu
args = parser.parse_args()

use_gpu = torch.cuda.is_available()

result_path = os.path.join('result', datetime.date.today().strftime('%Y%m%d'))
if not os.path.exists(result_path):
    os.makedirs(result_path)
method_str = args.data_path.split("/")[-1] + '_' + args.method
fo = open(os.path.join(result_path, method_str + '.txt'), "a+")
fo.write('#' * 50)
setting_str = 'Setting: \t batch-size: {} \t epoch_step: {} \t G_LR: {} \t D_LR: {} \t B_LR: {}'\
              '\ndevice_ids: {} \t data_path: {} \t bert_trainable: {}' \
              '\nuse_previousData: {} \t method: {} \t overlength_handle: {}'.format(
                args.batch_size, args.epoch_step, args.G_lr, args.D_lr, args.B_lr,
                args.device_ids, args.data_path, args.bert_trainable,
                args.use_previousData, args.method, args.overlength_handle)

print(setting_str)
fo.write(setting_str)

dataset, encoded_tag, tag_mask = load_data(args.data_path, args.data_type, args.use_previousData, args.overlength_handle)

bert = BertModel.from_pretrained('bert-base-uncased')

model = {}
model['Discriminator'] = Discriminator(num_classes=len(dataset.tag2id))
model['Encoder'] = Bert_Encoder(bert, bert_trainable=args.bert_trainable)
model['Generator'] = Generator()
model['MABert'] = MABert(bert, num_classes=len(dataset.tag2id), bert_trainable=args.bert_trainable, device=args.device_ids[0])

# define loss function (criterion)
criterion = nn.BCELoss()

# define optimizer
optimizer = {}
optimizer['Generator'] = torch.optim.SGD([{'params': model['Generator'].parameters(), 'lr': args.G_lr}],
                                         momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer['enc'] = torch.optim.SGD([{'params': model['MABert'].parameters(), 'lr': 0.01}], lr=0.1,
#                                    momentum=args.momentum, weight_decay=args.weight_decay)

optimizer['enc'] = torch.optim.SGD(model['MABert'].get_config_optim(args.D_lr, args.B_lr),
                            momentum=args.momentum, weight_decay=args.weight_decay)

# optimizer['Generator'] = torch.optim.Adam([{'params': model['Generator'].parameters(), 'lr': 5e-3}], lr=5e-3)

state = {'batch_size': args.batch_size, 'max_epochs': args.epochs, 'evaluate': args.evaluate,
         'resume': args.resume, 'num_classes': dataset.get_tags_num(), 'difficult_examples': False,
         'save_model_path': args.save_model_path, 'log_dir': args.log_dir, 'workers': args.workers,
         'epoch_step': args.epoch_step, 'lr': args.D_lr, 'encoded_tag': encoded_tag, 'tag_mask': tag_mask,
         'device_ids': args.device_ids, 'print_freq': args.print_freq, 'id2tag': dataset.id2tag,
         'result_file': fo, 'method': args.method}

if args.evaluate:
    state['evaluate'] = True

if args.method == 'MultiLabelMAP':
    engine = MultiLabelMAPEngine(state)
elif args.method == 'semiGAN_MultiLabelMAP':
    engine = semiGAN_MultiLabelMAPEngine(state)

engine.learning(model, criterion, dataset, optimizer)

fo.close()