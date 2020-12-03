import random
import torch
import torch.nn as nn
import json
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import transE_model as model_definition

from collections import defaultdict

import argparse
import sys
from heapq import nlargest, nsmallest

from config import global_config as cfg
import operator


def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var


def evaluate_change(static_score, another_score, TopK):
    assert len(static_score) == len(another_score)

    static_dict = dict()
    another_dict = dict()

    for index in range(len(static_score)):
        static_dict[static_score[index]] = index
        another_dict[another_score[index]] = index

    s = nlargest(TopK, static_score)
    a = nlargest(TopK, another_score)

    s_index = [static_dict[item] for item in s]
    a_index = [another_dict[item] for item in a]

    intersection = set(s_index).intersection(set(a_index))

    return len(intersection)


def rank_items(given_preference, user_sequence, feature_sequence,  transE_model, candidate_list, rej_list):
    device = torch.device('cuda') 
    transE_model.eval()
    cluster_list = []
    type_list = []
    category_list = []
    
    np_array = np.zeros([1,50], dtype=np.int)
    asked_cluster = None
    asked_poi_type = None
    
    
    item_features_list = feature_sequence.strip().split(' ')

    target_time, target_category, target_cluster, target_poi_type = item_features_list[-1].split(',') 
    for i in range(len(given_preference)): 
        if i == 0:
            if len(given_preference[i]) > 0:
                for item in given_preference[i]:
                    cluster_list.append(item)
        if i == 1:
            if len(given_preference[i]) > 0:
                for item in given_preference[i]:
                    type_list.append(item)
        if i == 2:
            if len(given_preference[i]) > 0:
                for item in given_preference[i]:
                    category_list.append(item)
            
    if len(cluster_list) > 0:
        asked_cluster = cluster_list[0]
    if asked_cluster != None:
        asked_cluster = torch.LongTensor([asked_cluster])
        asked_cluster = asked_cluster.to(device)
        asked_cluster_embedding = transE_model.relations_emb_clusters(asked_cluster)
    else:
        asked_cluster_embedding = torch.LongTensor(np_array).to(device)
    
    if len(type_list) > 0:
        asked_poi_type = type_list[0]    
    if asked_poi_type != None:
        asked_poi_type = torch.LongTensor([asked_poi_type])
        asked_poi_type = asked_poi_type.to(device)
        asked_poi_type_embedding = transE_model.relations_emb_poi_type(asked_poi_type)
    else:
        asked_poi_type_embedding = torch.LongTensor(np_array).to(device)
        
    if len(category_list)>0:
        category_list = torch.LongTensor(category_list)
        category_list = category_list.to(device)
        asked_feature_embedding = transE_model.relations_emb_category(category_list)
    else:
        asked_feature_embedding = torch.LongTensor(np_array).to(device)
    
    target_time = torch.LongTensor([int(target_time)])
    target_time = target_time.to(device)
    time_embedding = transE_model.relations_emb_time(target_time)

    relation_1 =  asked_cluster_embedding + asked_poi_type_embedding + torch.sum(asked_feature_embedding, dim=0) + time_embedding
    
    user_item_list = user_sequence.strip().split(' ')
    user_id = user_item_list[0]
    item_id = user_item_list[-1]
    user_id = torch.LongTensor([int(user_id)])
    user_id = user_id.to(device)
    item_id = torch.LongTensor([int(item_id)])
    item_id = item_id.to(device)
    
    
    history_item_features_list=item_features_list[:-1]
    item_context_embedding_list=[] 
    Attention_item_list=[]
    embedding_list = []
    weighted_list = []
    softmax_list = []
    
    for i in range(len(history_item_features_list)):
        time, category, cluster, poi_type = item_features_list[i].split(',')
        time = torch.LongTensor([int(time)])
        time = time.to(device)
        category = torch.LongTensor([int(category)])
        category = category.to(device)
        cluster = torch.LongTensor([int(cluster)])
        cluster = cluster.to(device)
        poi_type = torch.LongTensor([int(poi_type)])
        poi_type = poi_type.to(device)
        
        item_triples = torch.stack((user_id, time, category, cluster, poi_type, item_id), dim=1)
        #    print (positive_triples)
        item_context_embedding, concat_embedding = transE_model.cal_item_context_embedding(item_triples)
        item_context_embedding_list.append(item_context_embedding)
        output = transE_model.item_att_model(concat_embedding)
        Attention_item_list.append(output)
        embedding_list.append(concat_embedding)

    softmax = torch.nn.Softmax(dim = 1)
    softmax_value = softmax(torch.Tensor([Attention_item_list]))
    softmax_list = softmax_value.detach().cpu().numpy()[0].tolist()

    
    for i in softmax_list:
        weighted_item_context_embedding = item_context_embedding_list[softmax_list.index(i)] * i
        weighted_list.append(weighted_item_context_embedding)

    relation_2_embedding = sum(weighted_list) #relation 2

    
    concat_relation_embedding_1 = torch.cat((transE_model.entities_emb_head(user_id),relation_1), 1) 
    concat_relation_embedding_2 = torch.cat((transE_model.entities_emb_head(user_id),relation_2_embedding),1) 

    
    relation_weight_list = []
    relation_weight_1 = transE_model.relation_att_model(concat_relation_embedding_1)
    relation_weight_list.append(relation_weight_1)
    relation_weight_2 = transE_model.relation_att_model(concat_relation_embedding_2)
    relation_weight_list.append(relation_weight_2)
    
    relation_softmax_value = softmax(torch.Tensor([relation_weight_list]))
    relation_softmax_list = relation_softmax_value.detach().cpu().numpy()[0].tolist()
    relation = relation_softmax_list[0]*relation_1 + relation_softmax_list[1] * relation_2_embedding #final relation
    
    ideal_item_embedding = transE_model.entities_emb_head(user_id) + relation
    distance_dict = dict()
    for i in candidate_list:
        key = str(i)
        i = torch.LongTensor([int(i)])
        i = i.to(device)        
        candidate_embedding = transE_model.entities_emb_tail(i)
        value = (ideal_item_embedding - candidate_embedding).norm(p=2, dim =1)
        distance_dict[key] = value.detach().data.cpu().numpy()[0]
    sorted_distance_dict = sorted(distance_dict.items(), key=lambda kv: kv[1])   
    ranked_item = []
    for key, value in sorted_distance_dict:
        ranked_item.append(key)
        
    return ranked_item
    