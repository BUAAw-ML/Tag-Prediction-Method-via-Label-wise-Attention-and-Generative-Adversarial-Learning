import csv
import copy
import os
import sys
from random import shuffle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# from word_embedding import *
import pickle
import json

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# token_table = {'ecommerce': 'electronic commerce'}

def load_data(data_config, data_path=None, data_type='allData', use_previousData=False):
    cache_file_head = data_path.split("/")[-1]

    if use_previousData:
        print("load dataset from cache")
        dataset = dataEngine.from_dict(torch.load(os.path.join('cache', cache_file_head + '.dataset')))
        encoded_tag, tag_mask = torch.load(os.path.join('cache', cache_file_head + '.encoded_tag')), \
                                torch.load(os.path.join('cache', cache_file_head + '.tag_mask'))

    else:
        print("build dataset")
        if not os.path.exists('cache'):
            os.makedirs('cache')

        dataset = dataEngine(data_config=data_config)

        if data_type == 'TrainTest_pkl':
            file = os.path.join(data_path, 'train.pkl')
            dataset.filter_pkl(file)
            data = dataset.load_pkl(file)
            dataset.train_data, dataset.unlabeled_train_data = dataset.data_preprocess(data)

            file = os.path.join(data_path, 'test.pkl')
            dataset.test_data = dataset.load_pkl(file)

        elif data_type == 'TrainTest_text':

            file1 = os.path.join(data_path, 'train_texts.txt')
            file2 = os.path.join(data_path, 'train_labels.txt')
            dataset.filterTags_text(file2)
            data = dataset.load_EurLex_RCV2_SO(file1, file2)

            dataset.train_data, dataset.unlabeled_train_data = dataset.data_preprocess(data)

            file1 = os.path.join(data_path, 'test_texts.txt')
            file2 = os.path.join(data_path, 'test_labels.txt')
            dataset.test_data = dataset.load_EurLex_RCV2_SO(file1, file2)

        torch.save(dataset.to_dict(), os.path.join('cache', cache_file_head + '.dataset'))
        encoded_tag, tag_mask = dataset.encode_tag()
        torch.save(encoded_tag, os.path.join('cache', cache_file_head + '.encoded_tag'))
        torch.save(tag_mask, os.path.join('cache', cache_file_head + '.tag_mask'))

    return dataset, encoded_tag, tag_mask


class dataEngine(Dataset):
    def __init__(self, train_data=None, unlabeled_train_data=None, test_data=None,
                    tag2id={}, id2tag={}, co_occur_mat=None, tfidf_dict=None, data_config={}):
        self.train_data = train_data
        self.unlabeled_train_data = unlabeled_train_data
        self.test_data = test_data

        self.tag2id = tag2id
        self.id2tag = id2tag

        self.use_tags = {}

        self.co_occur_mat = co_occur_mat
        self.tfidf_dict = tfidf_dict

        self.data_config = data_config

    def random_permutation(self, data):
        data = np.array(data)
        ind = np.random.RandomState(seed=10).permutation(len(data))
        data = data[ind]
        return data

    @classmethod
    def from_dict(cls, data_dict):
        return dataEngine(data_dict.get('train_data'),
                       data_dict.get('unlabeled_train_data'),
                       data_dict.get('test_data'),
                       data_dict.get('tag2id'),
                       data_dict.get('id2tag'),
                       data_dict.get('co_occur_mat'),
                       data_dict.get('tfidf_dict'))

    def to_dict(self):
        data_dict = {
            'train_data': self.train_data,
            'unlabeled_train_data': self.unlabeled_train_data,
            'test_data': self.test_data,
            'tag2id': self.tag2id,
            'id2tag': self.id2tag,
            'co_occur_mat': self.co_occur_mat,
            'tfidf_dict': self.tfidf_dict
        }
        return data_dict

    def get_tags_num(self):
        return len(self.tag2id)

    def encode_tag(self):
        tag_ids = []
        tag_token_num = []
        for i in range(self.get_tags_num()):
            tag = self.id2tag[i]
            tokens = tokenizer.tokenize(tag)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            tag_ids.append(token_ids)
            tag_token_num.append(len(tokens))
        max_num = max(tag_token_num)
        padded_tag_ids = torch.zeros((self.get_tags_num(), max_num), dtype=torch.long)
        mask = torch.zeros((self.get_tags_num(), max_num))
        for i in range(self.get_tags_num()):
            mask[i, :len(tag_ids[i])] = 1.
            padded_tag_ids[i, :len(tag_ids[i])] = torch.tensor(tag_ids[i])
        return padded_tag_ids, mask

    def collate_fn(self, batch):
        # construct input
        inputs = [e['dscp_ids'] for e in batch]  #e['title_ids'] +
        dscp_tokens = [e['dscp_tokens'] for e in batch]


        lengths = np.array([len(e) for e in inputs])
        max_len = np.max(lengths)
        inputs = [tokenizer.prepare_for_model(e, max_length=max_len + 2, pad_to_max_length=True, truncation=True) for e in inputs]

        ids = torch.LongTensor([e['input_ids'] for e in inputs])
        token_type_ids = torch.LongTensor([e['token_type_ids'] for e in inputs])
        attention_mask = torch.FloatTensor([e['attention_mask'] for e in inputs])
        # construct tag
        tags = torch.zeros(size=(len(batch), self.get_tags_num()))
        for i in range(len(batch)):
            tags[i, batch[i]['tag_ids']] = 1.

        dscp = [e['dscp'] for e in batch]

        return (ids, token_type_ids, attention_mask, dscp_tokens), tags, dscp

    @classmethod
    def get_tfidf_dict(cls, document):
        tfidf_dict = {}
        tfidf_model = TfidfVectorizer(sublinear_tf=True,
                                        strip_accents='unicode',
                                        analyzer='word',
                                        token_pattern=r'\w{1,}',
                                        stop_words='english',
                                        ngram_range=(1, 1),
                                        max_features=10000).fit(document)
        for item in tfidf_model.vocabulary_:
            tfidf_dict[item] = tfidf_model.idf_[tfidf_model.vocabulary_[item]]

        return tfidf_dict


    def filter_pkl(self, file):
        tag_occurance = {}
        ignored_tags = set()

        with open(file, 'rb') as pklfile:
            reader = pickle.load(pklfile)
            for row in reader:

                tag = row["tags"]
                tag = list(set(tag))

                for t in tag:
                    if t in ignored_tags:
                        continue
                    elif t not in tag_occurance:
                        tag_occurance[t] = 1
                    else:
                        tag_occurance[t] += 1


        print('Total number of tags: {}'.format(len(tag_occurance)))
        tags = sorted(tag_occurance.items(), key=lambda x: x[1], reverse=True)

        print(tags)

        for item in tags[self.data_config['min_tagFrequence']:self.data_config['max_tagFrequence']]:
            self.use_tags[item[0]] = item[1]

    def load_pkl(self, file):
        data = []

        with open(file, 'rb') as pklfile:

            reader = pickle.load(pklfile)

            for row in reader:

                if len(row) != 4:
                    continue

                id = row["id"]
                title = row["name"]
                dscp = row["descr"]
                tag = row["tags"]

                title_tokens = tokenizer.tokenize(title.strip())
                dscp_tokens = title_tokens + tokenizer.tokenize(dscp.strip())

                if len(dscp_tokens) > 510:
                    if self.data_config['overlength_handle'] == 'truncation':
                        dscp_tokens = dscp_tokens[:510]
                    else:
                        continue

                dscp_ids = tokenizer.convert_tokens_to_ids(dscp_tokens)

                if self.use_tags is not None:
                    tag = [t for t in tag if t in self.use_tags]

                if len(tag) == 0:
                    continue

                for t in tag:
                    if t not in self.tag2id:
                        tag_id = len(self.tag2id)
                        self.tag2id[t] = tag_id
                        self.id2tag[tag_id] = t

                tag_ids = [self.tag2id[t] for t in tag]

                data.append({
                    'id': int(id),
                    'dscp_ids': dscp_ids,
                    'dscp_tokens': dscp_tokens,
                    'tag_ids': tag_ids,
                    'dscp': dscp
                })

        print("The number of tags for training: {}".format(len(self.tag2id)))

        return data

    def filterTags_text(self, file):
        tag_occurance = {}
        ignored_tags = set()
        with open(file, 'r') as f_tag:
            tags = f_tag.readlines()
            for tag in tags:
                tag = tag.strip().split()
                tag = [t.strip('#') for t in tag if t != '']  #

                for t in tag:
                    if t in ignored_tags:
                        continue
                    elif t not in tag_occurance:
                        tag_occurance[t] = 1
                    else:
                        tag_occurance[t] += 1

        print('Total number of tags: {}'.format(len(tag_occurance)))
        tags = sorted(tag_occurance.items(), key=lambda x: x[1], reverse=True)

        print(tags)

        for item in tags[self.data_config['min_tagFrequence']:self.data_config['max_tagFrequence']]:
            self.use_tags[item[0]] = item[1]

    def load_EurLex_RCV2_SO(self, file1, file2):
        data = []

        f_text = open(file1, 'r')
        texts = f_text.readlines()
        f_tag = open(file2, 'r')
        tags = f_tag.readlines()

        instanceCount = 0
        for text, tag in zip(texts, tags):

            dscp_tokens = tokenizer.tokenize(text.strip())
            if len(dscp_tokens) > 510:
                if self.data_config['overlength_handle'] == 'truncation':
                    dscp_tokens = dscp_tokens[:510]
                else:
                    continue

            dscp_ids = tokenizer.convert_tokens_to_ids(dscp_tokens)

            tag = tag.strip().split()
            tag = [t.strip('#') for t in tag if t != '']

            if self.use_tags is not None:
                tag = [t for t in tag if t in self.use_tags]

            if len(tag) == 0:
                continue

            if instanceCount > self.data_config['intanceNum_limit']:
                break
            instanceCount += 1

            for t in tag:
                if t not in self.tag2id:
                    tag_id = len(self.tag2id)
                    self.tag2id[t] = tag_id
                    self.id2tag[tag_id] = t

            tag_ids = [self.tag2id[t] for t in tag]

            data.append({
                'id': 0,
                'dscp_ids': dscp_ids,
                'dscp_tokens': dscp_tokens,
                'tag_ids': tag_ids,
                'dscp': text
            })

        print("The number of tags for training: {}".format(len(self.tag2id)))
        print(self.tag2id)

        f_text.close()
        f_tag.close()

        return data

    def data_preprocess(self, data):
        train_data = []
        
        data = np.array(data)
        ind = np.random.RandomState(seed=10).permutation(len(data))
        data = data[ind]

        #### If we use only partial training data, this ensures that the training set contains samples of each tag###
        for tag in self.use_tags.keys():
            self.use_tags[tag] *= self.data_config['data_split'] / len(data) if self.data_config['data_split'] < len(data) else 1

        tag_count = copy.deepcopy(self.use_tags)
        
        candidate = []
        rest = []

        print('The size of all train data: {}'.format(len(data)))

        for item in data:
            for tag_id in item['tag_ids']:
                if tag_count[self.id2tag[tag_id]] == self.use_tags[self.id2tag[tag_id]]:
                    for tag_id in item['tag_ids']:
                        tag_count[self.id2tag[tag_id]] -= 1
                    train_data.append(item)
                    break

                elif tag_count[self.id2tag[tag_id]] >= 1:
                    for tag_id in item['tag_ids']:
                        tag_count[self.id2tag[tag_id]] -= 1
                    candidate.append(item)
                    break
                else:
                    rest.append(item)
                    break

            if len(train_data) >= self.data_config['data_split']:
                print("len(train_data):{}".format(len(train_data)))
                break

        print(tag_count)

        assert len(data) == len(train_data) + len(candidate) + len(rest)

        if len(candidate) >= self.data_config['data_split'] - len(train_data):
            train_data.extend(candidate[:int(self.data_config['data_split'] - len(train_data))])

        else:
            train_data.extend(candidate)
            train_data.extend(rest[:int(self.data_config['data_split'] - len(train_data))])

        ###unlabeled_train_data is used for generative adversarial learning, which use train data without labels###
        unlabeled_train_data = copy.deepcopy(train_data)
        unlabeled_data_num = 1600

        if len(unlabeled_train_data) >= unlabeled_data_num:
            unlabeled_train_data = train_data[:unlabeled_data_num]

        while len(unlabeled_train_data) < unlabeled_data_num:
            unlabeled_train_data.extend(train_data)

        train_data = np.array(train_data)
        ind = np.random.RandomState(seed=10).permutation(len(train_data))
        train_data = train_data[ind]

        unlabeled_train_data = np.array(unlabeled_train_data)
        ind = np.random.RandomState(seed=10).permutation(len(unlabeled_train_data))
        unlabeled_train_data = unlabeled_train_data[ind]
        
        return train_data, unlabeled_train_data