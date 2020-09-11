# -*- coding: utf-8 -*
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class Attention_item_level(nn.Module):
    def __init__(self, input_dim, dim1, output_dim):
        super(Attention_item_level, self).__init__()
        self.fc1 = nn.Linear(input_dim, dim1)
        self.adding = nn.Linear(dim1, dim1)
        self.fc2 = nn.Linear(dim1, output_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.init_weight()
        
    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)       
        torch.nn.init.xavier_uniform_(self.fc2.weight)       
        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()
        
    def forward(self, input_tensor):
        x = self.fc1(input_tensor)
        x = self.dropout(x)
        x = self.sigmoid(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.sigmoid(x)

        return x
    
class Attention_relation_level(nn.Module):
    def __init__(self, input_dim, dim1, output_dim):
        super(Attention_relation_level, self).__init__()
        self.fc1 = nn.Linear(input_dim, dim1)
        self.adding = nn.Linear(dim1, dim1)
        self.fc2 = nn.Linear(dim1, output_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.init_weight()
        
    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)       
        torch.nn.init.xavier_uniform_(self.fc2.weight)       
        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()
        
    def forward(self, input_tensor):
        x = self.fc1(input_tensor)
        x = self.dropout(x)
        x = self.sigmoid(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.sigmoid(x)

        return x


class TransE(nn.Module):

    def __init__(self,  relation_count_list,entity_count_list, device, item_att_model, relation_att_model, norm=2, dim=100, margin=1.0):
        super(TransE, self).__init__()
        self.entity_count_list = entity_count_list
        self.relation_count_list = relation_count_list
        self.device = device
        self.norm = norm
        self.dim = dim
        self.entities_emb_head,self.entities_emb_tail = self._init_enitity_emb()
        self.relations_emb_time,self.relations_emb_category,self.relations_emb_clusters,self.relations_emb_poi_type = self._init_relation_emb()
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')
        self.item_att_model = item_att_model
        self.relation_att_model = relation_att_model

    def _init_enitity_emb(self):
        entities_emb_head = nn.Embedding(num_embeddings=self.entity_count_list[0] + 1,
                                    embedding_dim=self.dim,
                                    padding_idx=self.entity_count_list[0])
        uniform_range = 6 / np.sqrt(self.dim)
        entities_emb_head.weight.data.uniform_(-uniform_range, uniform_range)
        
        entities_emb_tail = nn.Embedding(num_embeddings=self.entity_count_list[1] + 1,
                                    embedding_dim=self.dim,
                                    padding_idx=self.entity_count_list[1])
        uniform_range = 6 / np.sqrt(self.dim)
        entities_emb_tail.weight.data.uniform_(-uniform_range, uniform_range)
        return entities_emb_head, entities_emb_tail

    def _init_relation_emb(self):
        relations_emb_time = nn.Embedding(num_embeddings=self.relation_count_list[0]+1,
                                     embedding_dim=self.dim,
                                     padding_idx=self.relation_count_list[0])
        uniform_range = 6 / np.sqrt(self.dim)
        relations_emb_time.weight.data.uniform_(-uniform_range, uniform_range)
        # -1 to avoid nan for OOV vector
        relations_emb_time.weight.data[:-1, :].div_(relations_emb_time.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        
        
        relations_emb_category = nn.Embedding(num_embeddings=self.relation_count_list[1]+1,
                                     embedding_dim=self.dim,
                                     padding_idx=self.relation_count_list[1])
        uniform_range = 6 / np.sqrt(self.dim)
        relations_emb_category.weight.data.uniform_(-uniform_range, uniform_range)
        # -1 to avoid nan for OOV vector
        relations_emb_category.weight.data[:-1, :].div_(relations_emb_category.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        
        
        relations_emb_clusters = nn.Embedding(num_embeddings=self.relation_count_list[2]+1,
                                     embedding_dim=self.dim,
                                     padding_idx=self.relation_count_list[2])
        uniform_range = 6 / np.sqrt(self.dim)
        relations_emb_clusters.weight.data.uniform_(-uniform_range, uniform_range)
        # -1 to avoid nan for OOV vector
        relations_emb_clusters.weight.data[:-1, :].div_(relations_emb_clusters.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        
        
        relations_emb_poi_type = nn.Embedding(num_embeddings=self.relation_count_list[3]+1,
                                     embedding_dim=self.dim,
                                     padding_idx=self.relation_count_list[3])
        uniform_range = 6 / np.sqrt(self.dim)
        relations_emb_poi_type.weight.data.uniform_(-uniform_range, uniform_range)
        # -1 to avoid nan for OOV vector
        relations_emb_poi_type.weight.data[:-1, :].div_(relations_emb_poi_type.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        
        return relations_emb_time,relations_emb_category,relations_emb_clusters,relations_emb_poi_type

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):

        self.entities_emb_head.weight.data[:-1, :].div_(self.entities_emb_head.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))
        self.entities_emb_tail.weight.data[:-1, :].div_(self.entities_emb_tail.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))

        assert positive_triplets.size()[1] == 6
        positive_distances = self._distance(positive_triplets)

        assert negative_triplets.size()[1] == 6
        negative_distances = self._distance(negative_triplets)

        return self.loss(positive_distances, negative_distances), positive_distances, negative_distances

    def predict(self, triplets: torch.LongTensor):
        return self._distance(triplets)

    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)
    
    def cal_item_context_embedding(self, triplets: torch.LongTensor):
        heads = triplets[:, 0]
        relation_time = triplets[:, 1]
        relation_category = triplets[:, 2]
        relation_cluster = triplets[:, 3]
        relation_type = triplets[:, 4]
        tails = triplets[:, 5]
        
        relation_emb_value = self.entities_emb_tail(tails) + self.relations_emb_time(relation_time) + self.relations_emb_category(relation_category) + \
            self.relations_emb_clusters(relation_cluster) + self.relations_emb_poi_type(relation_type)
            
        return relation_emb_value, torch.cat((self.entities_emb_head(heads),relation_emb_value), 1)
    
    def cal_context_embedding(self, triplets: torch.LongTensor):
        relation_time = triplets[:, 1]
        relation_category = triplets[:, 2]
        relation_cluster = triplets[:, 3]
        relation_type = triplets[:, 4]
        relation_emb_value = self.relations_emb_time(relation_time) + self.relations_emb_category(relation_category) + \
            self.relations_emb_clusters(relation_cluster) + self.relations_emb_poi_type(relation_type)
        
        return (relation_emb_value)
    
    def _distance(self, triplets):
        assert triplets.size()[1] == 6
        heads = triplets[:, 0]
        relation_time = triplets[:, 1]
        relation_category = triplets[:, 2]
        relation_cluster = triplets[:, 3]
        relation_type = triplets[:, 4]
        tails = triplets[:, 5]
        relation_emb_value = self.relations_emb_time(relation_time) + self.relations_emb_category(relation_category) + \
            self.relations_emb_clusters(relation_cluster) + self.relations_emb_poi_type(relation_type)

        return (self.entities_emb_head(heads) + relation_emb_value - self.entities_emb_tail(tails)).norm(p=self.norm,dim=1)
    
    def forward2(self, positive_triplets, negative_triplets):

        self.entities_emb_head.weight.data[:-1, :].div_(self.entities_emb_head.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))
        self.entities_emb_tail.weight.data[:-1, :].div_(self.entities_emb_tail.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))

        assert len(positive_triplets) == 3
        positive_distances = self._distance2(positive_triplets)

        assert len(negative_triplets) == 3
        negative_distances = self._distance2(negative_triplets)

        return self.loss(positive_distances, negative_distances), positive_distances, negative_distances
    
    def _distance2(self, triplets):
        heads = triplets[0]
        relation = triplets[1]
        tails = triplets[2]
        return (self.entities_emb_head(heads) + relation - self.entities_emb_tail(tails)).norm(p=self.norm,dim=1)