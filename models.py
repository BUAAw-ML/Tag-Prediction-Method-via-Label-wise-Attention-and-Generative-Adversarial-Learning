from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
from transformers import BertModel
from torch.autograd import Variable
import torch.nn.functional as F


class LABert(nn.Module):
    def __init__(self, bert, num_classes, bert_trainable=True, device=0):
        super(LABert, self).__init__()

        self.add_module('bert', bert)
        if not bert_trainable:
            for m in self.bert.parameters():
                m.requires_grad = False

        self.num_classes = num_classes

        self.class_weight = Parameter(torch.Tensor(num_classes, 768).uniform_(0, 1), requires_grad=False).cuda(device)
        self.class_weight.requires_grad = True

        self.GAN_output = nn.Softmax(dim=-1)

    def forward(self, ids, token_type_ids, attention_mask, encoded_tag, tag_mask, feat):
        token_feat = self.bert(ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)[0] #N, L, hidden_size

        embed = self.bert.get_input_embeddings()
        tag_embedding = embed(encoded_tag)
        tag_embedding = torch.sum(tag_embedding * tag_mask.unsqueeze(-1), dim=1) \
                        / torch.sum(tag_mask, dim=1, keepdim=True)

        masks = torch.unsqueeze(attention_mask, 1)  # N, 1, L  .bool() .byte()
        attention = (torch.matmul(token_feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill((~masks.bool()), torch.tensor(-np.inf))

        attention = F.softmax(attention, -1) #N, labels_num, L

        attention_out = attention @ token_feat   # N, labels_num, hidden_size
        attention_out = attention_out * self.class_weight
        attention_out = torch.sum(attention_out, -1)

        logit = torch.sigmoid(attention_out)

        realData_prob = torch.sum(attention_out, -1, keepdim=True)

        feat = feat * self.class_weight
        prob = torch.sum(feat, -1)
        prob = torch.sum(prob, -1, keepdim=True)

        prob = torch.cat((prob,realData_prob),-1)
        prob = self.GAN_output(prob)

        return prob[:,1], logit, prob[:,0]

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.class_weight, 'lr': lr},
            {'params': self.bert.parameters(), 'lr': lrp},
        ]


class Generator(nn.Module):
    def __init__(self, hidden_dim=768, input_dim=768, num_hidden_generator=2, hidden_dim_generator=2000):
        super(Generator, self).__init__()

        self.dropout = nn.Dropout(p=0.5)
        self.act = nn.LeakyReLU(0.2)

        self.num_hidden_generator = num_hidden_generator
        self.hidden_list_generator = nn.ModuleList()
        for i in range(num_hidden_generator):
            dim = input_dim if i == 0 else hidden_dim_generator
            self.hidden_list_generator.append(nn.Linear(dim, hidden_dim_generator))

        self.output = nn.Linear(hidden_dim_generator, hidden_dim)

    def forward(self, feat):

        x = feat

        for i in range(self.num_hidden_generator):
            x = self.hidden_list_generator[i](x)
            x = self.act(x)

        y = self.output(x)
        return y

    def get_config_optim(self, lr):
        return [
            {'params': self.hidden_list_generator.parameters(), 'lr': lr},
            {'params': self.output.parameters(), 'lr': lr},
        ]


class MLPBert(nn.Module):
    def __init__(self, bert, num_classes, hidden_dim, hidden_layer_num, bert_trainable=True):
        super(MLPBert, self).__init__()

        self.add_module('bert', bert)
        if not bert_trainable:
            for m in self.bert.parameters():
                m.requires_grad = False

        self.num_classes = num_classes
        self.hidden_layer_num = hidden_layer_num
        self.hidden_list = nn.ModuleList()
        for i in range(hidden_layer_num):
            if i == 0:
                self.hidden_list.append(nn.Linear(768, hidden_dim))
            else:
                self.hidden_list.append(nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU()

    def forward(self, ids, token_type_ids, attention_mask, encoded_tag, tag_mask, feat):

        token_feat = self.bert(ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)[0]
        sentence_feat = torch.sum(token_feat * attention_mask.unsqueeze(-1), dim=1) \
                        / torch.sum(attention_mask, dim=1, keepdim=True)

        x = sentence_feat
        for i in range(self.hidden_layer_num):
            x = self.hidden_list[i](x)
            x = self.act(x)
        y = torch.sigmoid(self.output(x))
        return y, y, y

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.bert.parameters(), 'lr': lrp},
            {'params': self.hidden_list.parameters(), 'lr': lr},
            {'params': self.output.parameters(), 'lr': lr},
        ]