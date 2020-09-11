import sys
import json
import pickle
import time
import torch
import pandas as pd
import transE_model as model_file
from collections import Counter
import generate_data


def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var

def get_unique_column_values(column):
    # get unique column items
    column_list = []
    for i in column:
        for j in i:
            if j not in column_list:
                column_list.append(j)
    return column_list

class _Config():
    def __init__(self):
        self.init_data()
        self.init_basic()
        self.init_type()

        self.init_misc()
        self.init_test()
        self.init_FM_related()
        
        
    def init_data(self):
        train_dict,valid_dict,test_dict = generate_data.get_data()
        sequence_list =[]
        f1 = open("train_list_item.txt", "w") #user_id i1 i2 i3
        f2 = open("train_list_features.txt", "w") #3,3,3,4    2,2,2,2    9,9,9,9
        f7 = open("train_list_location.txt", "w") #Item's location_id
        for key in train_dict.keys():
            num_total_sequences = len(train_dict[key][0]) #eg. user '0' sequence number
            for i in range(num_total_sequences): #check each sequence
                sequence_length = len(train_dict[key][0][i])
                sequence_list.append(sequence_length)
                f1_row = str(key) + ' '
                f2_row = ''
                f7_row = '' 
                for j in range(sequence_length):
                    
                    f1_row +=  str(train_dict[key][1][i][j]) + ' '
                    f2_row +=  str(train_dict[key][8][i][j])+',' +  str(train_dict[key][3][i][j]) +',' + \
                    str(train_dict[key][4][i][j]) +',' + str(train_dict[key][5][i][j]) + ' '
                f7_row = str(train_dict[key][6][i][sequence_length-1])
                
                f1_row += '\n'
                f2_row += '\n'
                f7_row += '\n'
                f1.write(f1_row)
                f2.write(f2_row)
                f7.write(f7_row)
        f1.close()
        f2.close()
        f7.close()
        

        sequence_list =[]
        f3 = open("valid_list_item.txt", "w") #user_id i1 i2 i3
        f4 = open("valid_list_features.txt", "w") #3,3,3,4    2,2,2,2    9,9,9,9
        f8 = open("valid_list_location.txt", "w") #Item's location_id

        for key in valid_dict.keys():
            num_total_sequences = len(valid_dict[key][0]) #eg. user '0' sequence number
            for i in range(num_total_sequences): #check each sequence
                sequence_length = len(valid_dict[key][0][i])
                sequence_list.append(sequence_length)
                f3_row = str(key) + ' '
                f4_row = ''
                f8_row = ''
                for j in range(sequence_length):
                    
                    f3_row +=  str(valid_dict[key][1][i][j]) + ' '
                    f4_row +=  str(valid_dict[key][8][i][j])+',' +  str(valid_dict[key][3][i][j]) +',' + \
                    str(valid_dict[key][4][i][j]) +',' + str(valid_dict[key][5][i][j]) + ' '
                f8_row = str(valid_dict[key][6][i][sequence_length-1])
                
                f3_row += '\n'
                f4_row += '\n'
                f8_row += '\n'
                f3.write(f3_row)
                f4.write(f4_row)
                f8.write(f8_row)
        f3.close()
        f4.close()
        f8.close()
        

        sequence_list =[]
        f5 = open("test_list_item.txt", "w") #user_id i1 i2 i3
        f6 = open("test_list_features.txt", "w") #3,3,3,4    2,2,2,2    9,9,9,9
        f9 = open("test_list_location.txt", "w") #Item's location_id
        for key in test_dict.keys():
            num_total_sequences = len(test_dict[key][0]) #eg. user '0' sequence number
            for i in range(num_total_sequences): #check each sequence
                sequence_length = len(test_dict[key][0][i])
                sequence_list.append(sequence_length)
                f5_row = str(key) + ' '
                f6_row = ''
                f9_row = ''
                for j in range(sequence_length):
                    
                    f5_row +=  str(test_dict[key][1][i][j]) + ' '
                    f6_row +=  str(test_dict[key][8][i][j])+',' +  str(test_dict[key][3][i][j]) +',' + \
                    str(test_dict[key][4][i][j]) +',' + str(test_dict[key][5][i][j]) + ' '
                f9_row = str(test_dict[key][6][i][sequence_length-1])                
                
                f5_row += '\n'
                f6_row += '\n'
                f9_row += '\n'
                f5.write(f5_row)
                f6.write(f6_row)
                f9.write(f9_row)
        f5.close()
        f6.close()
        f9.close()
        
        train_list_item = []
        f = open("train_list_item.txt", "r")
        for x in f:
            train_list_item.append(x)
        f.close()
        
        train_list_features = []
        f = open("train_list_features.txt", "r")
        for x in f:
            train_list_features.append(x)
        f.close()

        train_list_location = []
        f = open("train_list_location.txt", "r")
        for x in f:
            train_list_location.append(x)
        f.close()
        
        valid_list_item = []
        f = open("valid_list_item.txt", "r")
        for x in f:
            valid_list_item.append(x)
        f.close()
        
        valid_list_features = []
        f = open("valid_list_features.txt", "r")
        for x in f:
            valid_list_features.append(x)
        f.close()

        valid_list_location = []
        f = open("valid_list_location.txt", "r")
        for x in f:
            valid_list_location.append(x)
        f.close()
        
        test_list_item = []
        f = open("test_list_item.txt", "r")
        for x in f:
            test_list_item.append(x)
        f.close()
        
        test_list_features = []
        f = open("test_list_features.txt", "r")
        for x in f:
            test_list_features.append(x)
        f.close()

        test_list_location = []
        f = open("test_list_location.txt", "r")
        for x in f:
            test_list_location.append(x)
        f.close()        

        self.train_list_item = train_list_item
        self.train_list_features = train_list_features
        self.train_list_location = train_list_location
        
        self.valid_list_item = valid_list_item
        self.valid_list_features = valid_list_features
        self.valid_list_location = valid_list_location
        
        self.test_list_item = test_list_item
        self.test_list_features = test_list_features
        self.test_list_location = test_list_location
        
        
    def init_basic(self):
        
        
        train_dict,valid_dict,test_dict = generate_data.get_data()

        
        
        df = pd.read_csv("new_transE_3.csv")
        self.user_length = df['User_id'].max()+1
#        print (self.user_length)
        self.item_length = df['Item_id'].max()+1
        self.entity_count_list=[]
        self.entity_count_list.append(self.user_length)
        self.entity_count_list.append(self.item_length)
        
        time_length = df['new_time'].max()+1
        category_length = df['L2_Category_name'].max()+1
        cluster_length = df['clusters'].max()+1
        poi_type_length = df['POI_Type'].max()+1
        self.type_count = poi_type_length
        self.relation_count_list = []
        self.relation_count_list.append(time_length)
        self.relation_count_list.append(category_length)
        self.relation_count_list.append(cluster_length)
        self.relation_count_list.append(poi_type_length)
        
        self.vector_length = 50
        self.margin = 1.0
        self.device = torch.device('cuda') 
        self.norm = 1
        self.learning_rate = 0.01
        
#        df2 = pd.read_csv("ui.csv") #UNIQUE ui
        df3 = pd.read_csv("dict.csv") #Item_id	stars	clusters	new_L2_Category_name	new_POI_Type from ear.csv
        user_list = df[['User_id']].values.tolist()
        self.user_list = get_unique_column_values(user_list)

        
        busi_list = df[['Item_id']].values.tolist()
        self.busi_list = get_unique_column_values(busi_list)
        
        
        # _______ String to Int _______

        with open('L2.json', 'r') as f: # "Food": [6, 7, 9, 14, 15, 17, 19, 20, 2]
            self.taxo_dict = json.load(f)
        with open('poi.json', 'r') as f:
            self.poi_dict = json.load(f)
        
        df3 = pd.read_csv("dict.csv") #Item_id	stars	clusters	new_L2_Category_name	new_POI_Type from ear.csv
        item_dict=dict()
        star_list =[]
    #    max_int = 0
        for index,  row in df3.iterrows():
            if str(int(row['Item_id'])) not in item_dict: 
                star_list.clear()
                row_dict = dict()
                star_list.append(row['stars'])
                row_dict['stars'] = row['stars']
                row_dict['clusters'] = int(row['clusters'])
                row_dict['L2_Category_name'] = [int(row['L2_Category_name'])]
                row_dict['POI_Type'] = int(row['POI_Type'])
                row_dict['feature_index'] = [int(row['L2_Category_name'])]
                row_dict['feature_index'].append(int(row['clusters'])+category_length)
                row_dict['feature_index'].append(int(row['POI_Type'])+category_length+cluster_length)
                row_dict['feature_index'].append(2*int(row['stars'])-2+category_length+cluster_length+poi_type_length)
                item_dict[str(int(row['Item_id']))] = row_dict
                
            else:
                star_list.append(row['stars'])
                item_dict[str(int(row['Item_id']))]['stars'] = (sum(star_list))/len(star_list)
                item_dict[str(int(row['Item_id']))]['L2_Category_name'].append(int(row['L2_Category_name']))
                item_dict[str(int(row['Item_id']))]['feature_index'].append(int(row['L2_Category_name']))
        self.item_dict = item_dict        

    def init_type(self):
        self.INFORM_FACET = 'INFORM_FACET'
        self.ACCEPT_REC = 'ACCEPT_REC'
        self.REJECT_REC = 'REJECT_REC'

        # define agent behavior
        self.ASK_FACET = 'ASK_FACET'
        self.MAKE_REC = 'MAKE_REC'
        self.FINISH_REC_ACP = 'FINISH_REC_ACP'
        self.FINISH_REC_REJ = 'FINISH_REC_REJ'
        self.EPISODE_START = 'EPISODE_START'

        # define the sender type
        self.USER = 'USER'
        self.AGENT = 'AGENT'

    def init_misc(self):
#        self.FACET_POOL = ['city', 'stars', 'RestaurantsPriceRange2']
#        self.FACET_POOL = ['clusters', 'stars', 'POI_Type']
        self.FACET_POOL = ['clusters',  'POI_Type']
        self.FACET_POOL += self.taxo_dict.keys()
        print('Total feature length is: {}, Top 10 namely: {}'.format(len(self.FACET_POOL), self.FACET_POOL[: 10]))
        self.REC_NUM = 10
        self.MAX_TURN = 10
        self.play_by = None
        self.calculate_all = None


    def init_FM_related(self):
        clusters_max = 0
        category_max = 0
        feature_max = 0
        for k, v in self.item_dict.items():
            if v['clusters'] > clusters_max:
                clusters_max = v['clusters']
            if max(v['L2_Category_name']) > category_max:
                category_max = max(v['L2_Category_name'])
            if max(v['feature_index']) > feature_max:
                feature_max = max(v['feature_index'])

        stars_list = [3.5, 2.5, 1.5, 3.0, 4.5, 1.0, 4.0, 5.0, 2.0]
        poi_list = [0,1]
        self.star_count, self.poi_count = len(stars_list), len(poi_list)
        self.clusters_count, self.category_count, self.feature_count = clusters_max + 1, category_max + 1, feature_max + 1

        self.clusters_span = (0, self.clusters_count)
        self.poi_span = (self.clusters_count, self.clusters_count + self.poi_count)
        self.star_span = (self.clusters_count + self.star_count, self.clusters_count + self.star_count + self.poi_count)

        self.spans = [self.clusters_span, self.star_span, self.poi_span]

        print('clusters max: {}, category max: {}, feature max: {}'.format(self.clusters_count, self.category_count, self.feature_count))
        
        model_item = model_file.Attention_item_level(input_dim= 2 * self.vector_length, dim1=64, output_dim=1)
        model_relation = model_file.Attention_item_level(input_dim= 2 * self.vector_length, dim1=64, output_dim=1)
        
        device = torch.device('cuda')
        model = model_file.TransE(self.relation_count_list, self.entity_count_list, device, dim=self.vector_length,
                                margin=self.margin, norm=self.norm, item_att_model = model_item, relation_att_model=model_relation)
        model.load_state_dict(torch.load("C:/Users/kvn646/Desktop/MSAI-Project/upload/CAL/basic/weight/new_transE.pt"))
        self.entities_emb_head = model.entities_emb_head
        self.entities_emb_tail = model.entities_emb_tail
        self.relations_emb_time = model.relations_emb_time
        self.relations_emb_category = model.relations_emb_category
        self.relations_emb_clusters = model.relations_emb_clusters
        self.relations_emb_poi_type = model.relations_emb_poi_type
        self.transE_model = model.to(device)

    def init_test(self):
        pass

    def change_param(self, playby, eval, update_count, update_reg, purpose, mod, mask):
        self.play_by = playby
        self.eval = eval
        self.update_count = update_count
        self.update_reg = update_reg
        self.purpose = purpose
        self.mod = mod
        self.mask = mask



start = time.time()
global_config = _Config()
print('Config takes: {}'.format(time.time() - start))

print('___Config Done!!___')
