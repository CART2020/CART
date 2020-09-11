# -*- coding: utf-8 -*-
from collections import Counter
import torch
from typing import Dict, Tuple

Mapping = Dict[str, int]

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_path: str, entity2id: Mapping):
        self.entity2id = entity2id
        with open(data_path, "r") as f:
            for line in f:
                self.data = [line[:-1].split(" ") for line in f]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        head, relation, tail = self.data[index]

        relation_list = relation.split(",")

        return int(head), int(relation_list[0]),int(relation_list[1]), int(relation_list[2]), int(relation_list[3]), int(tail)

    @staticmethod
    def _to_idx(key: str, mapping: Mapping) -> int:
        try:
            return mapping[key]
        except KeyError:
            return len(mapping)
