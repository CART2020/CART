import random
import torch
import torch.nn as nn
import json
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from config import global_config as cfg
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import sys
def feature_distance(given_preference, userID, TopKTaxo, features):        
    device = torch.device('cuda')
    cluster_list = []
    type_list = []
    category_list = []    

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
                    
    user_id = userID
    user_id = torch.LongTensor([user_id])
    user_id = user_id.to(device)
    

    given_preference = category_list
    given_preference = torch.LongTensor([given_preference])
    given_preference = given_preference.to(device)
    head_embeding = cfg.entities_emb_head(user_id)
    
    result_dict = dict()
    
    if len(given_preference) > 0:
        category_matrix = cfg.relations_emb_category(given_preference)
        target_item_features_list = features.strip().split(' ')
        time, category, cluster, poi_type = target_item_features_list[-1].split(',') 

        time = torch.LongTensor([int(time)])
        time = time.to(device)
        category_matrix = torch.sum(category_matrix, dim=1)
        user_context_embed = head_embeding + cfg.relations_emb_time(time) + category_matrix
        
    for index, big_feature in enumerate(cfg.FACET_POOL[: 2]):
        
        if big_feature == 'clusters':
            cluster_list = []
            for i in range(cfg.clusters_count):
                cluster_list.append(i)
            cluster_input = torch.LongTensor(cluster_list)
            cluster_input = cluster_input.to(device)
            big_feature_clusters_matrix = cfg.relations_emb_clusters(cluster_input)
            new_user_context_embed = user_context_embed.expand(len(big_feature_clusters_matrix), -1)
            
            cluster_distance = (new_user_context_embed - big_feature_clusters_matrix).norm(p=2,dim=1)
            cluster_distance = cluster_distance.detach().cpu().numpy()            
            cluster_distance = np.sort(cluster_distance)
            result_dict[big_feature] = np.sum(cluster_distance[: TopKTaxo])  / (len(given_preference) + len(cluster_list) + len(type_list) + 1)
   
        if big_feature == 'POI_Type':
            type_list = []
            for i in range(cfg.type_count):
                type_list.append(i)
            type_input = torch.LongTensor(type_list)
            type_input = type_input.to(device)
            big_feature_type_matrix = cfg.relations_emb_poi_type(type_input)
            new_user_context_embed = user_context_embed.expand(len(big_feature_type_matrix), -1)
            
            type_distance = (new_user_context_embed - big_feature_type_matrix).norm(p=2,dim=1)
            type_distance = type_distance.detach().cpu().numpy()
            type_distance = np.sort(type_distance)
            result_dict[big_feature] = np.sum(type_distance[:])  / (len(given_preference) + len(cluster_list) + len(type_list) + 1)
    

    for big_feature in cfg.FACET_POOL[2: ]:
        feature_index = [item for item in cfg.taxo_dict[big_feature]] 
        feature_index = torch.LongTensor([feature_index])
        feature_index = feature_index.to(device)
        feature_index_embedding = cfg.relations_emb_category(feature_index)
        new_user_context_embed = user_context_embed.expand(len(feature_index_embedding), -1)
        small_feature_distance = (new_user_context_embed - feature_index_embedding).norm(p=2,dim=1)
        small_feature_distance = small_feature_distance.detach().cpu().numpy()            
        small_feature_distance = np.sort(small_feature_distance)        
        result_dict[big_feature] = np.sum(small_feature_distance[: TopKTaxo])  / (len(given_preference) + len(cluster_list) + len(type_list) + 1)

    return result_dict 


        
    
    