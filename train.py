# -*- coding: utf-8 -*-
import pandas as pd
from collections import Counter
from torch.utils import data
from typing import Dict, Tuple
import transE_model as model_definition
import torch
import torch.nn as nn
import torch.optim as optim
import dataset as ds
import numpy as np
import pickle
import pandas as pd
import random
import json
from torch.utils import data as torch_data
import generate_data
import inside_category

train_dict,valid_dict,test_dict = generate_data.get_data()

df = pd.read_csv("../data/new_transE_3.csv") 
user_length = df['User_id'].max()+1
item_length = df['Item_id'].max()+1

time_length = df['new_time'].max()+1
category_length = df['L2_Category_name'].max()+1
cluster_length = df['clusters'].max()+1
poi_type_length = df['POI_Type'].max()+1

relation_count_list = []
relation_count_list.append(time_length)
relation_count_list.append(category_length)
relation_count_list.append(cluster_length)
relation_count_list.append(poi_type_length)

entity_count_list = []
entity_count_list.append(user_length)
entity_count_list.append(item_length)


vector_length = 50
margin = 1.0
device = torch.device('cuda') 
norm = 2
learning_rate = 0.001
model_item = model_definition.Attention_item_level(input_dim= 2 * vector_length, dim1=64, output_dim=1)
model_relation = model_definition.Attention_relation_level(input_dim= 2 * vector_length, dim1=64, output_dim=1)


model = model_definition.TransE( relation_count_list, entity_count_list, device=device, dim=vector_length,
                                margin=margin, norm=norm, item_att_model = model_item, relation_att_model=model_relation)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
best_score = 0.0
step = 0
sequence_list = []

f = open("new_train.txt", "w")
for key in train_dict.keys():
    num_total_sequences = len(train_dict[key][0])
    for i in range(num_total_sequences):
        sequence_length = len(train_dict[key][0][i])
        sequence_list.append(sequence_length)
        for j in range(sequence_length):
            row = ''
            row += str(train_dict[key][0][i][j]) + ' '+ str(train_dict[key][8][i][j])+',' +  str(train_dict[key][3][i][j]) +',' + \
            str(train_dict[key][4][i][j]) +',' + str(train_dict[key][5][i][j]) + ' ' + str(train_dict[key][1][i][j]) + '\n'
            f.write(row)
f.close()


train_set = ds.Dataset('new_train.txt',None)

path = "../weight/new_transE.pt"

sequence_index = 0
current_sequence = 0
embedding_list = []
Attention_item_list = []
item_context_embedding_list = []
softmax_list = []
weighted_list = []

model.train()

for epoch in range(20):
    print ('epoch: ', epoch)
    for i in range(len(train_set)): 
        if (sequence_index == sequence_list[current_sequence]-1):
            local_heads, local_relation_time,local_relation_category, local_relation_cluster, local_relation_type, local_tails = train_set[i]
            local_heads = torch.LongTensor([local_heads])
            local_relation_time = torch.LongTensor([local_relation_time])
            local_relation_category = torch.LongTensor([local_relation_category])
            local_relation_cluster = torch.LongTensor([local_relation_cluster])
            local_relation_type = torch.LongTensor([local_relation_type])
            local_tails = torch.LongTensor([local_tails])
            local_heads, local_relation_time, local_relation_category, local_relation_cluster, local_relation_type, local_tails = \
                (torch.LongTensor([local_heads]).to(device), torch.LongTensor([local_relation_time]).to(device),torch.LongTensor([local_relation_category]).to(device), 
                 torch.LongTensor([local_relation_cluster]).to(device), torch.LongTensor([local_relation_type]).to(device),torch.LongTensor([local_tails]).to(device))    
            
            item_triples = torch.stack((local_heads,local_relation_time, local_relation_category, local_relation_cluster, local_relation_type,local_tails), dim=1)
            context_embedding = model.cal_context_embedding(item_triples) # cal relation_1
            
            softmax = torch.nn.Softmax(dim = 1)
            softmax_value = softmax(torch.Tensor([Attention_item_list]))
            softmax_list = softmax_value.detach().cpu().numpy()[0].tolist()

            for i in softmax_list:
                weighted_item_context_embedding = item_context_embedding_list[softmax_list.index(i)] * i
                weighted_list.append(weighted_item_context_embedding)
                
            relation_2_embedding = sum(weighted_list) #relation 2
            
            concat_relation_embedding_1 = torch.cat((model.entities_emb_head(local_heads),context_embedding), 1) 
            concat_relation_embedding_2 = torch.cat((model.entities_emb_head(local_heads),relation_2_embedding),1) 

            relation_weight_1 = model.relation_att_model(concat_relation_embedding_1)
            relation_weight_2 = model.relation_att_model(concat_relation_embedding_2)
            relation = relation_weight_1*context_embedding + relation_weight_2 * relation_2_embedding
            
            optimizer.zero_grad()

            broken_tails = torch.randint(high=item_length, size=(1,), device=device)
            
            positive_item_list = [local_heads, relation, local_tails]
            negative_item_list = [local_heads, relation, broken_tails]
            
            lsigmoid = nn.LogSigmoid()
            diff,_,_ = model.forward2(positive_item_list,negative_item_list)
            loss = - lsigmoid(diff).sum(dim=0)
            # print(loss.data.cpu())
            loss.backward()
            
            optimizer.step()
    #        step += 1        

            embedding_list.clear()
            Attention_item_list.clear()
            item_context_embedding_list.clear()
            softmax_list.clear()
            weighted_list.clear()
            sequence_index = 0
            current_sequence += 1
#            j+=1
        else:
            local_heads, local_relation_time,local_relation_category, local_relation_cluster, local_relation_type, local_tails = train_set[i]
            local_heads = torch.LongTensor([local_heads])
            local_relation_time = torch.LongTensor([local_relation_time])
            local_relation_category = torch.LongTensor([local_relation_category])
            local_relation_cluster = torch.LongTensor([local_relation_cluster])
            local_relation_type = torch.LongTensor([local_relation_type])
            local_tails = torch.LongTensor([local_tails])    
            
            if int(local_relation_type.item()) == 0: 
                local_heads, local_relation_time, local_relation_category, local_relation_cluster, local_relation_type, local_tails = \
                    (torch.LongTensor([local_heads]).to(device), torch.LongTensor([local_relation_time]).to(device),torch.LongTensor([local_relation_category]).to(device), 
                     torch.LongTensor([local_relation_cluster]).to(device), torch.LongTensor([local_relation_type]).to(device),torch.LongTensor([local_tails]).to(device))    
    
                item_triples = torch.stack((local_heads, local_relation_time, local_relation_category, local_relation_cluster, local_relation_type, local_tails), dim=1)
    
                item_context_embedding, concat_embedding = model.cal_item_context_embedding(item_triples)
                item_context_embedding_list.append(item_context_embedding)
                
                concat_embedding = concat_embedding.to(device)
                
                output = model.item_att_model(concat_embedding)
                Attention_item_list.append(output)
                
                embedding_list.append(concat_embedding) 
    
                sequence_index += 1
            else:
                categorys, ratios = inside_category.getvalues()
                local_relation_category = torch.zeros([1,50]).to(device)
                for i in range(len(categorys)):
                    cat = categorys[i]
                    cat = torch.LongTensor([cat]).to(device)
                    rat = torch.Tensor([ratios[i]]).to(device)
                    rat = rat.expand(1,vector_length)
                    local_relation_category += model.relations_emb_category(cat) * rat
                
                local_heads, local_relation_time, local_relation_cluster, local_relation_type, local_tails = \
                    (torch.LongTensor([local_heads]).to(device), torch.LongTensor([local_relation_time]).to(device),
                     torch.LongTensor([local_relation_cluster]).to(device), torch.LongTensor([local_relation_type]).to(device),torch.LongTensor([local_tails]).to(device))    

                
                item_context_embedding = model.entities_emb_tail(local_tails) + model.relations_emb_time(local_relation_time) + local_relation_category  + \
                                            model.relations_emb_clusters(local_relation_cluster) + model.relations_emb_poi_type(local_relation_type)
                
                concat_embedding = torch.cat((model.entities_emb_head(local_heads),item_context_embedding), 1)
                item_context_embedding_list.append(item_context_embedding)
                
                concat_embedding = concat_embedding.to(device)
                
                output = model.item_att_model(concat_embedding)
                Attention_item_list.append(output)
                
                embedding_list.append(concat_embedding) 
    
                sequence_index += 1             
                
    current_sequence = 0

torch.save(model.state_dict(),path)

 