import os
import torch
import collections
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader

from utils.process import getData
from utils.tools import getVocab


class CustomDataset(Dataset):
    def __init__(self, data,word2id,id2word, device='cpu'):
        self.word2id = word2id
        self.id2word = id2word
        self.pad = 'pad'
        self.data = data
        self.device = device

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return [self.data[0][idx], self.data[1][idx]]

    def collate_fn(self, batch):
        # used in sequence labeling task
        tensor_in = []
        tensor_decoder = []
        tensor_label = []

        lengths = []
        max_length_in = -1
        max_length_label = -1
        max_length_decoder = -1

        for data in batch:
            data_in, data_out = data
            # char level ids only
            tensor_in.append(data_in)
            tensor_decoder.append(data_out[:-1])
            tensor_label.append(data_out[1:])

            lengths.append(len(data_in))
            max_length_in = max(max_length_in, len(data_in))
            max_length_decoder = max(max_length_decoder, len(data_out[:-1]))
            max_length_label = max(max_length_label, len(data_out[1:]))

            assert max_length_decoder == max_length_label
        # 对齐
        tensor_in = [arr[:max_length_in] if len(arr) >= max_length_in else arr + [0] * (max_length_in - len(arr)) for
                     arr in tensor_in]
        tensor_mask = [[1 if v != 0 else 0 for v in arr] for arr in tensor_in]
        tensor_decoder = [
            arr[:max_length_decoder] if len(arr) >= max_length_decoder else arr + [0] * (max_length_decoder - len(arr))
            for arr in tensor_decoder]
        tensor_label = [arr[:max_length_label] if len(arr) >= max_length_label else arr + [-100] * (max_length_label - len(arr))
                        for arr in tensor_label]

        # to tensor
        tensor_in = torch.Tensor(np.array(tensor_in)).long().to(self.device)
        tensor_mask = torch.Tensor(np.array(tensor_mask)).long().to(self.device)
        tensor_decoder = torch.Tensor(np.array(tensor_decoder)).long().to(self.device)
        tensor_label = torch.Tensor(np.array(tensor_label)).long().to(self.device)

        return tensor_in, tensor_decoder, tensor_label

        # return (tensor_in, tensor_mask, lengths), tensor_decoder, tensor_label


if __name__ == '__main__':
    datasets = CustomDataset()
    dataloader = DataLoader(datasets, batch_size=2, shuffle=True,
                            collate_fn=datasets.collate_fn)
    for data in dataloader:
        print(data)
        break
