import os
import shutil
import time
import torch.optim
import torch.utils.data
import torchnet as tnt
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from util import *
import json

tqdm.monitor_interval = 0

class Engine(object):
    def __init__(self, state={}):
        self.writer = SummaryWriter(state['log_dir'])
        os.makedirs(state['log_dir'], exist_ok=True)
        self.result_file = state['result_file']

        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('batch_size') is None:
            self.state['batch_size'] = 16

        if self._state('workers') is None:
            self.state['workers'] = 25

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = False
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0
        # best score
        self.state['best_score'] = {'map': 0., 'OP': 0., 'OR': 0., 'OF1': 0., 'CP': 0., 'CR': 0., 'CF1': 0.}
        self.state['train_iters'] = 0
        self.state['eval_iters'] = 0

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].value()[0]
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}'.format(self.state['epoch'], loss=loss))
            else:
                print('Test: \t Loss {loss:.4f}'.format(loss=loss))
        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        input_var = self.state['input']
        target_var = self.state['target']

        # compute output
        self.state['output'] = model(input_var)
        self.state['loss'] = criterion(self.state['output'], target_var)

        if training:
            self.state['train_iters'] += 1
        else:
            self.state['eval_iters'] += 1

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

    def learning(self, model, criterion, dataset):
        ### data loading code ###
        train_loader = torch.utils.data.DataLoader(dataset.train_data,
                                                   batch_size=self.state['batch_size'], shuffle=False,
                                                   num_workers=self.state['workers'],
                                                   collate_fn=dataset.collate_fn)

        #for generative adversarial learning, which use train data without labels
        unlabeled_train_loader = torch.utils.data.DataLoader(dataset.unlabeled_train_data,
                                                   batch_size=self.state['batch_size'], shuffle=False,
                                                   num_workers=self.state['workers'],
                                                   collate_fn=dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(dataset.test_data,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'], collate_fn=dataset.collate_fn)

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model['Generator'].load_state_dict(checkpoint['state_dict-Generator'])
                model['Classifier'].load_state_dict(checkpoint['state_dict-Classifier'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))

        if self.state['use_gpu']:

            model['Generator'] = model['Generator'].cuda(self.state['device_ids'][0])
            model['Classifier'] = model['Classifier'].cuda(self.state['device_ids'][0])

            # model = torch.nn.DataParallel(model, device_ids=self.state['device_ids'])
            if 'encoded_tag' in self.state:
                self.state['encoded_tag'] = self.state['encoded_tag'].cuda(self.state['device_ids'][0])
            if 'tag_mask' in self.state:
                self.state['tag_mask'] = self.state['tag_mask'].cuda(self.state['device_ids'][0])
            criterion = criterion.cuda(self.state['device_ids'][0])

        if self.state['evaluate']:
            self.state['epoch'] = self.state['start_epoch']
            self.validate(val_loader, model, criterion, self.state['epoch'])
            return

        optimizer = {}

        # According to the relevant literature, generative adversarial learning requires a smaller learning rate
        if self.state['method'] == 'GAN_MultiLabelMAP':
            self.state['B_lr'] *= 0.1

        optimizer['Generator'] = torch.optim.SGD(
            [{'params': model['Generator'].parameters(), 'lr': self.state['G_lr']}],
            momentum=0.9, weight_decay=1e-4)
        print("B_lr {}".format(self.state['B_lr']))

        optimizer['Classifier'] = torch.optim.SGD(
            model['Classifier'].get_config_optim(self.state['D_lr'], self.state['B_lr']),
            momentum=0.9, weight_decay=1e-4)

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:', lr)

            # train for one epoch
            print("Train with labeled data:")
            self.train(train_loader, model, criterion, optimizer, epoch, False)
            #
            if self.state['method'] == 'GAN_MultiLabelMAP':
                # train for one epoch
                self.train(unlabeled_train_loader, model, criterion, optimizer, epoch, True)

            # evaluate on validation set
            prec1 = self.validate(val_loader, model, criterion, epoch)

            # remember best OF1 and save checkpoint
            # is_best = prec1['OF1'] > self.state['best_score']['OF1']
            # self.save_checkpoint({
            #     'epoch': epoch + 1,
            #     # 'arch': self._state('arch'),
            #     'state_dict-Generator': model['Generator'].state_dict() if self.state['use_gpu'] else model['Generator'].state_dict(),
            #     'state_dict-Classifier': model['Classifier'].state_dict() if self.state['use_gpu'] else model['Classifier'].state_dict(),
            #     'best_score': self.state['best_score'],
            # }, is_best)

            if prec1['OF1'] >= self.state['best_score']['OF1']:
                self.state['best_score']['OF1'] = prec1['OF1']
                self.state['best_score']['OP'] = prec1['OP']
                self.state['best_score']['OR'] = prec1['OR']
                self.state['best_score']['CF1'] = prec1['CF1']
                self.state['best_score']['CP'] = prec1['CP']
                self.state['best_score']['CR'] = prec1['CR']
                self.state['best_score']['map'] = prec1['map']

            best_str = '**best** map={map:.3f} OP={OP:.3f} OR={OR:.3f} OF1={OF1:.3f} CP={CP:.3f} CR={CR:.3f} CF1={CF1:.3f}'.format(
                map=self.state['best_score']['map'], OP=self.state['best_score']['OP'],
                OR=self.state['best_score']['OR'],
                OF1=self.state['best_score']['OF1'], CP=self.state['best_score']['CP'],
                CR=self.state['best_score']['CR'], CF1=self.state['best_score']['CF1'])

            print(best_str)
            self.result_file.write(best_str + '\n')

        return self.state['best_score']

    def train(self, data_loader, model, criterion, optimizer, epoch, GAN_training):

        # switch to train mode
        model['Generator'].train()
        model['Classifier'].train()

        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()
        for i, (input, target, _) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(True, model, criterion, data_loader, optimizer)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(self.state['device_ids'][0])

            self.on_forward(True, model, criterion, data_loader, optimizer, GAN_training=GAN_training)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)

        self.on_end_epoch(True, model, criterion, data_loader, optimizer)
    
    @torch.no_grad()
    def validate(self, data_loader, model, criterion, epoch):
        # switch to evaluate mode
        model['Generator'].eval()
        model['Classifier'].eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        end = time.time()
        for i, (input, target, self.state['dscp']) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(self.state['device_ids'][0])

            output = self.on_forward(False, model, criterion, data_loader)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)

        score = self.on_end_epoch(False, model, criterion, data_loader)

        return score

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):

        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)

        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'],
                                             'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']['OF1']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_list = []
        decay = 0.1 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
        for opt in optimizer.values():
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['lr'] * decay
                lr_list.append(param_group['lr'])
        return np.unique(lr_list)


class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True, GAN_training=False):
        target_var = self.state['target']
        ids, token_type_ids, attention_mask, dscp_tokens = self.state['input']
        ids = ids.cuda(self.state['device_ids'][0])
        token_type_ids = token_type_ids.cuda(self.state['device_ids'][0])
        attention_mask = attention_mask.cuda(self.state['device_ids'][0])

        if training:
            self.state['train_iters'] += 1
        else:
            self.state['eval_iters'] += 1

        z = torch.rand(ids.shape[0], 1, 768).type(torch.FloatTensor).cuda(self.state['device_ids'][0])
        x_g = model['Generator'](z)

        _, logits, _ = model['Classifier'](ids, token_type_ids, attention_mask,
                                                                      self.state['encoded_tag'],
                                                                      self.state['tag_mask'], x_g.detach())

        self.state['output'] = logits

        self.state['loss'] = criterion(logits, target_var)

        if training:
            optimizer['Classifier'].zero_grad()
            self.state['loss'].backward()
            nn.utils.clip_grad_norm_(optimizer['Classifier'].param_groups[0]["params"], max_norm=10.0)
            optimizer['Classifier'].step()
        else:
            return self.state['output']

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        map = 100 * self.state['ap_meter'].value().mean()
        loss = self.state['meter_loss'].value()[0]
        OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
        if display:
            if training:
                reselt_str = 'Epoch: [{0}]\t Loss {loss:.4f}\t mAP {map:.3f} \n ' \
                'OP: {OP:.4f}\t OR: {OR:.4f}\t OF1: {OF1:.4f}\t CP: {CP:.4f}\t CR: {CR:.4f}\t CF1: {CF1:.4f}'.format(
                self.state['epoch'], loss=loss, map=map, OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1)

            else:
                reselt_str = 'Test: \t Loss {loss:.4f}\t mAP {map:.3f} \n' \
                'OP: {OP:.4f}\t OR: {OR:.4f}\t OF1: {OF1:.4f}\t CP: {CP:.4f}\t CR: {CR:.4f}\t CF1: {CF1:.4f} \n' \
                'OP_3: {OP_3:.4f}\t OR_3: {OR_3:.4f}\t OF1_3: {OF1_3:.4f}\t CP_3: {CP_3:.4f}\t CR_3: {CR_3:.4f}\t CF1_3: {CF1_3:.4f}'.format(
                    loss=loss, map=map, OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1,
                    OP_3=OP_k, OR_3=OR_k, OF1_3=OF1_k, CP_3=CP_k, CR_3=CR_k, CF1_3=CF1_k)

            print(reselt_str)
            self.result_file.write(reselt_str + '\n')

            result = {'map': map, 'OP': OP, 'OR': OR, 'OF1': OF1, 'CP': CP, 'CR': CR, 'CF1': CF1}

        if training:
            self.writer.add_scalar('loss/train_epoch_loss', loss, self.state['epoch'])
            self.writer.add_scalar('mAP/train_mAP', map, self.state['epoch'])
            self.writer.add_scalar('OF1/train_OF1', OF1, self.state['epoch'])
            self.writer.add_scalar('CF1/train_CF1', CF1, self.state['epoch'])
        else:
            self.writer.add_scalar('loss/eval_epoch_loss', loss, self.state['epoch'])
            self.writer.add_scalar('mAP/eval_mAP', map, self.state['epoch'])
            self.writer.add_scalar('OF1/eval_OF1', OF1, self.state['epoch'])
            self.writer.add_scalar('CF1/eval_CF1', CF1, self.state['epoch'])

        return result

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        # record loss
        self.state['loss_batch'] = self.state['loss'].item()
        self.state['meter_loss'].add(self.state['loss_batch'])
        if training:
            self.writer.add_scalar('loss/train_batch_loss', self.state['loss_batch'], self.state['train_iters'] - 1)
        else:
            self.writer.add_scalar('loss/eval_batch_loss', self.state['loss_batch'], self.state['eval_iters'] - 1)

        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data.cpu(), self.state['target_gt'].cpu())

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))


class semiGAN_MultiLabelMAPEngine(MultiLabelMAPEngine):

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True,
                   GAN_training=False):
        target_var = self.state['target']
        ids, token_type_ids, attention_mask, dscp_tokens = self.state['input']
        ids = ids.cuda(self.state['device_ids'][0])
        token_type_ids = token_type_ids.cuda(self.state['device_ids'][0])
        attention_mask = attention_mask.cuda(self.state['device_ids'][0])

        if training:
            self.state['train_iters'] += 1
        else:
            self.state['eval_iters'] += 1

        epsilon = 1e-8

        z = torch.Tensor(ids.shape[0], 1, 768).uniform_(0, 1).cuda(self.state['device_ids'][0])

        x_g = model['Generator'](z)

        # -----------train enc-----------
        flatten, logits, prob = model['Classifier'](ids, token_type_ids, attention_mask,
                                                self.state['encoded_tag'],
                                                self.state['tag_mask'], x_g.detach())  #

        self.state['output'] = logits

        D_L_unsupervised = -1 * torch.mean(torch.log(1 - prob + epsilon))
        D_L_unsupervised2 = -1 * torch.mean(torch.log(flatten + epsilon))

        if GAN_training == False:  # train with labeled data
            d_loss = criterion(self.state['output'], target_var)

            if training:
                optimizer['Classifier'].zero_grad()
                d_loss.backward()
                nn.utils.clip_grad_norm_(optimizer['Classifier'].param_groups[0]["params"], max_norm=10.0)
                optimizer['Classifier'].step()

            self.state['loss'] = [d_loss, d_loss]

        else:
            # -----------generative adversarial learning-----------
            d_loss = D_L_unsupervised + D_L_unsupervised2
            if training:
                optimizer['Classifier'].zero_grad()
                d_loss.backward()
                nn.utils.clip_grad_norm_(optimizer['Classifier'].param_groups[0]["params"], max_norm=10.0)
                optimizer['Classifier'].step()

            flatten, _, prob = model['Classifier'](ids, token_type_ids, attention_mask,
                                               self.state['encoded_tag'],
                                               self.state['tag_mask'], x_g)
            g_loss = -1 * torch.mean(torch.log(prob + epsilon))
            if training:
                optimizer['Generator'].zero_grad()
                g_loss.backward()
                nn.utils.clip_grad_norm_(model['Generator'].parameters(), max_norm=10.0)
                optimizer['Generator'].step()

            self.state['loss'] = [d_loss, g_loss]

        if not training:
            return self.state['output']

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        # record loss
        self.state['loss_batch'] = self.state['loss'][0].item() + self.state['loss'][1].item()
        self.state['meter_loss'].add(self.state['loss_batch'])
        if training:
            self.writer.add_scalar('loss/train_batch_loss', self.state['loss_batch'], self.state['train_iters'] - 1)
        else:
            self.writer.add_scalar('loss/eval_batch_loss', self.state['loss_batch'], self.state['eval_iters'] - 1)

        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data.cpu(), self.state['target_gt'].cpu())

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'd_Loss {d_loss_current:.4f}\t'
                      'g_Loss {g_loss_current:.4f}'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time,
                    d_loss_current=self.state['loss'][0].item(),
                    g_loss_current=self.state['loss'][1].item()
                ))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'd_Loss {d_loss_current:.4f}\t'
                      'g_Loss {g_loss_current:.4f}'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time,
                    d_loss_current=self.state['loss'][0].item(),
                    g_loss_current=self.state['loss'][1].item()
                ))




