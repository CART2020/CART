# BB-8 and R2-D2 are best friends.

import sys
import time
from collections import defaultdict
import random
random.seed(0)

import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.distributions import Categorical


from message import message
from config import global_config as cfg
from utils_entropy import cal_ent
from heapq import nlargest, nsmallest
from utils_fea_sim import feature_distance
from utils_sense import rank_items
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import  math
import torch.optim as optim

def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var


class agent():
    def __init__(self, transE_model, user_id, busi_id, do_random, write_fp, strategy, TopKTaxo, numpy_list, PN_model, log_prob_list, action_tracker, candidate_length_tracker, mini, optimizer1_fm, optimizer2_fm, alwaysupdate, do_mask, sample_dict, choose_pool, features, items):
        #_______ input parameters_______
        self.user_id = user_id
        self.busi_id = busi_id
        self.transE_model = transE_model

        self.turn_count = 0
        self.F_dict = defaultdict(lambda: defaultdict())
        self.recent_candidate_list = [int(k) for k, v in cfg.item_dict.items()]
#        print ("agent init candidate list: ", len(self.recent_candidate_list))
        self.recent_candidate_list_ranked = self.recent_candidate_list

        self.asked_feature = list() #record asked facets
        self.do_random = do_random
        self.rejected_item_list_ = list()

        self.history_list = list()

        self.write_fp = write_fp
        self.strategy = strategy
        self.TopKTaxo = TopKTaxo
        self.entropy_dict_10 = None
        self.entropy_dict_50 = None
        self.entropy_dict = None
        self.distance_dict = None
        self.distance_dict2 = None
        self.PN_model = PN_model

        self.known_feature = list() # category id list
        self.known_facet = list() 

        self.residual_feature_big = None
        self.change = None
        self.skip_big_feature = list()
        self.numpy_list = numpy_list

        self.log_prob_list = log_prob_list
        self.action_tracker = action_tracker
        self.candidate_length_tracker = candidate_length_tracker
        self.mini_update_already = False
        self.mini = mini
        self.optimizer1_fm = optimizer1_fm
        self.optimizer2_fm = optimizer2_fm
        self.alwaysupdate = alwaysupdate
        self.previous_dict = None
        self.rejected_time = 0
        self.do_mask = do_mask
        self.big_feature_length = 11 # 2+9 no starts
        self.feature_length = 289 #L2 + star + cluster +type in CRM
        self.sample_dict = sample_dict
        self.choose_pool = choose_pool #['clusters', 'POI_Type'] + big_feature_list 
        
        self.features = features # feature sequence
        self.items = items # item sequence
        self.known_feature_category = []
        self.known_feature_cluster =[]
        self.known_feature_type =[]
        self.known_feature_total =[]

    def get_batch_data(self, pos_neg_pairs, bs, iter_):
        # this function is used for Reflection Stage.
        PAD_IDX1 = len(cfg.user_list) + len(cfg.item_dict)
        PAD_IDX2 = cfg.feature_count

        left = iter_ * bs
        right = min((iter_ + 1) * bs, len(pos_neg_pairs))
        pos_list, pos_list2, neg_list, neg_list2 = list(), list(), list(), list()
        for instance in pos_neg_pairs[left: right]:
            # instance[0]: pos item, instance[1] neg item
            pos_list.append(torch.LongTensor([self.user_id, instance[0] + len(cfg.user_list)]))
            neg_list.append(torch.LongTensor([self.user_id, instance[1] + len(cfg.user_list)]))
        # end for
        preference_list = torch.LongTensor(self.known_feature).expand(len(pos_list), len(self.known_feature))

        pos_list = pad_sequence(pos_list, batch_first=True, padding_value=PAD_IDX1)
        pos_list2 = preference_list

        neg_list = pad_sequence(neg_list, batch_first=True, padding_value=PAD_IDX1)
        neg_list2 = preference_list

        return cuda_(pos_list), cuda_(pos_list2), cuda_(neg_list), cuda_(neg_list2)
    # end def

    def mini_update_transE(self):
        # This function is used to update the pretrained transE model.
        device = torch.device('cuda')
        self.transE_model.to(device)
        self.transE_model.train()
        optimizer = optim.SGD(self.transE_model.parameters(), lr=0.001)
        
        
        items = self.items
        features = self.features
#        asked_features = self.asked_feature
        item_features_list = features.strip().split(' ')
    #    print (item_features_list)
    #    print (item_features_list[-1])
        target_time, target_category, target_cluster, target_poi_type = item_features_list[-1].split(',') # target item features        
        
        userID = torch.LongTensor([self.user_id])
        userID = userID.to(device)
#        print (target_time)
        target_time = torch.LongTensor([int(target_time)])
        target_time = target_time.to(device)
        
        np_array = np.zeros([1,50], dtype=np.int)
        asked_cluster = None
        asked_poi_type = None
        feature_list = []
        if len(self.known_feature_cluster) > 0:
            asked_cluster = self.known_feature_cluster[0]
        if len(self.known_feature_type) > 0:
            asked_poi_type = self.known_feature_type[0]       
        if len(self.known_feature_category) > 0:
            feature_list = self.known_feature_category
        
    # check if value is None
        if asked_cluster != None:
            asked_cluster = torch.LongTensor([asked_cluster])
            asked_cluster = asked_cluster.to(device)
            asked_cluster_embedding = self.transE_model.relations_emb_clusters(asked_cluster)
        else:
            asked_cluster = torch.LongTensor([0])
            asked_cluster = asked_cluster.to(device)
            asked_cluster_embedding = torch.LongTensor(np_array).to(device)
            
        if asked_poi_type != None:
            asked_poi_type = torch.LongTensor([asked_poi_type])
            asked_poi_type = asked_poi_type.to(device)
            asked_poi_type_embedding = self.transE_model.relations_emb_poi_type(asked_poi_type)
        else:
            asked_poi_type = torch.LongTensor([0])
            asked_poi_type = asked_poi_type.to(device)            
            asked_poi_type_embedding = torch.LongTensor(np_array).to(device)
        
        item_features_list = features.strip().split(' ')
        target_time, target_category, target_cluster, target_poi_type = item_features_list[-1].split(',') # target item features
        target_time = torch.LongTensor([int(target_time)])
        target_time = target_time.to(device)   
        
        target_category = torch.LongTensor([int(target_category)])
        target_category = target_category.to(device) 

        
        user_item_list = items.strip().split(' ')
        target_item_id = user_item_list[-1]
        target_item_id = torch.LongTensor([int(target_item_id)])
        target_item_id = target_item_id.to(device)            
        positive_item_triples = torch.stack((userID, target_time, target_category, asked_cluster,  asked_poi_type, target_item_id), dim=1)
        
        for reject_item in self.rejected_item_list_:
            reject_item = torch.LongTensor([int(reject_item)])
            reject_item = reject_item.to(device)             
            negative_item_triples = torch.stack((userID, target_time, target_category, asked_cluster,  asked_poi_type, reject_item), dim=1)
            
            optimizer.zero_grad()
            lsigmoid = nn.LogSigmoid()
            diff,_,_ = self.transE_model.forward(positive_item_triples,negative_item_triples)
            loss = - lsigmoid(diff).sum(dim=0)
#            print(loss)
            loss.backward()    
            optimizer.step()
    # end def

    def vectorize(self):
        # This function is for the vectorization of our state vector.

        list4 = [v for k, v in self.distance_dict2.items()]

        list5 = self.history_list + [0] * (10 - len(self.history_list))
        
        list6 = [0] * 8 
        if len(self.recent_candidate_list) <= 5:
            list6[0] = 1
        if len(self.recent_candidate_list) > 5 and len(self.recent_candidate_list) <= 10:
            list6[1] = 1
        if len(self.recent_candidate_list) > 10 and len(self.recent_candidate_list) <= 15:
            list6[2] = 1
        if len(self.recent_candidate_list) > 15 and len(self.recent_candidate_list) <= 20:
            list6[3] = 1
        if len(self.recent_candidate_list) > 20 and len(self.recent_candidate_list) <= 25:
            list6[4] = 1
        if len(self.recent_candidate_list) > 25 and len(self.recent_candidate_list) <= 30:
            list6[5] = 1
        if len(self.recent_candidate_list) > 30 and len(self.recent_candidate_list) <= 35:
            list6[6] = 1
        if len(self.recent_candidate_list) > 35:
            list6[7] = 1


        list4 = [float(i)/sum(list4) for i in list4] #normalization
        list_cat = list4 + list5 + list6
        list_cat = np.array(list_cat)

        assert len(list_cat) == 29
        return list_cat
    # end def




    def update_upon_feature_inform(self, input_message):
        assert input_message.message_type == cfg.INFORM_FACET
        
        #_______ update F_dict________
        facet = input_message.data['facet'] # asked facet in last turn
        if facet is None:
            print('?')
        self.asked_feature.append(facet)
        value = input_message.data['value']

#        if facet in ['stars', 'RestaurantsPriceRange2', 'city']:
        if facet in ['clusters', 'POI_Type']:
            
            '''
            value is a list
            for Prices: [3.0]
            '''
            if value is not None and value[0] is not None: # value is in list.
                # for k in recent_candidate_list, if cfg.item_dict[str(k)][facet] in value, keep k(candidates)
                self.recent_candidate_list = [k for k in self.recent_candidate_list if cfg.item_dict[str(k)][facet] in value]
#                print ("recent_candidate_list length ", self.recent_candidate_list)
                # avoid busi_id(ground truth) ranking high in recent_candidate_list
                '''check the type of busi_id, int and str work different'''
                self.recent_candidate_list = list(set(self.recent_candidate_list) - set([self.busi_id])) + [self.busi_id]
                # add facet to know_facet
                self.known_facet.append(facet)
                fresh = True
                if facet == 'clusters':
                    if int(value[0]) not in self.known_feature_cluster:
                        # append new value which responding to facet, to known_feature
#                        self.known_feature.append(int(value[0]))# add region 1 to known feature
                        self.known_feature_cluster.append(int(value[0]))
                    else:
                        fresh = False
                if facet == 'POI_Type':
                    if int(value[0]) not in self.known_feature_type:
#                        self.known_feature.append(int(value[0] + cfg.clusters_count - 1))
                        self.known_feature_type.append(int(value[0]))
                    else:
                        fresh = False

                self.known_feature = list(set(self.known_feature)) # feature = values
#                print ('feature1 : ',self.known_feature[-1])

                
                # if we dont ask  
                if cfg.play_by != 'AOO' and cfg.play_by != 'AOO_valid':
                    # features = prefered values, known features similarity to preference features
                    self.known_feature_total.clear()
                    self.known_feature_total.append(self.known_feature_cluster)
                    self.known_feature_total.append(self.known_feature_type)
                    self.known_feature_total.append(self.known_feature_category)
                    
                    self.distance_dict = feature_distance(self.known_feature_total, self.user_id, self.TopKTaxo, self.features)
                    self.distance_dict2 = self.distance_dict.copy()
                    #print('Similarity ranking dict: {}'.format(self.distance_dict))

                    self.recent_candidate_list_ranked = rank_items(self.known_feature_total, self.items, self.features, self.transE_model, self.recent_candidate_list, self.rejected_item_list_)
        # end if
        else:  # means the feature is in the small tags
            if value is not None:
                self.recent_candidate_list = [k for k in self.recent_candidate_list if set(value).issubset(set(cfg.item_dict[str(k)]['L2_Category_name']))]
#                print ("recent_candidate_list length ", self.recent_candidate_list[-1])
                self.recent_candidate_list = list(set(self.recent_candidate_list) - set([self.busi_id])) + [self.busi_id]

                self.known_feature_category += [int(i) for i in value]
#                print ('feature2 : ',self.known_feature[-1])
                self.known_feature_category = list(set(self.known_feature_category))
                self.known_facet.append(facet)

                # dictionary
                l = list(set(self.recent_candidate_list) - set([self.busi_id]))
                random.shuffle(l)


                if cfg.play_by != 'AOO' and cfg.play_by != 'AOO_valid':
                    self.known_feature_total.clear()
                    self.known_feature_total.append(self.known_feature_cluster)
                    self.known_feature_total.append(self.known_feature_type)
                    self.known_feature_total.append(self.known_feature_category)
                    self.distance_dict = feature_distance(self.known_feature_total, self.user_id, self.TopKTaxo, self.features)
                    self.distance_dict2 = self.distance_dict.copy()
                    self.recent_candidate_list_ranked = rank_items(self.known_feature_total, self.items, self.features, self.transE_model, self.recent_candidate_list, self.rejected_item_list_)
        # end else

        start = time.time()
        if value is not None and value[0] is not None:

            c = cal_ent(self.recent_candidate_list)
#            print (c)
            d = c.do_job()
            self.entropy_dict = d

        # _______ set those asked feature's entropy to 0 _______
        for f in self.asked_feature:
            self.entropy_dict[f] = 0

        for f in self.asked_feature:
            if self.distance_dict is not None and f in self.distance_dict:
                self.distance_dict[f] = 10000
                if self.entropy_dict[f] == 0:
                    self.distance_dict[f] = 10000

        for f in self.asked_feature:
            if self.distance_dict2 is not None and f in self.distance_dict:
                self.distance_dict2[f] = 10000
                if self.entropy_dict[f] == 0:
                    self.distance_dict[f] = 10000

        self.residual_feature_big = list(set(self.choose_pool) - set(self.known_facet))
        ent_position, sim_position = None, None
        if self.entropy_dict is not None:
            ent_value = sorted([v for k, v in self.entropy_dict.items()], reverse=True)
            ent_position = [ent_value.index(self.entropy_dict[big_f]) for big_f in self.residual_feature_big]

        if self.distance_dict is not None:
            sim_value = sorted([v for k, v in self.distance_dict.items()], reverse=True)
            sim_position = [sim_value.index(self.distance_dict[str(big_f)]) for big_f in self.residual_feature_big]

        if len(self.residual_feature_big) > 0:
            with open(self.write_fp, 'a') as f:
                f.write('Turn Count: {} residual feature: {}***ent position: {}*** sim position: {}***\n'.format(self.turn_count, self.residual_feature_big, ent_position, sim_position))

    def prepare_next_question(self):
        if self.strategy == 'maxent':
            facet = max(self.entropy_dict, key=self.entropy_dict.get)
            data = dict()
            data['facet'] = facet
            new_message = message(cfg.AGENT, cfg.USER, cfg.ASK_FACET, data)
            self.asked_feature.append(facet)
            return new_message
        elif self.strategy == 'maxsim':
            # ask following max similarity between attributes
            for f in self.asked_feature: # asked factes
                if self.distance_dict is not None and f in self.distance_dict:
                    self.distance_dict[f] = 10000
            # if we dont have any feature, we use entropy dict.
            if len(self.known_feature) == 0 or self.distance_dict is None:
                # key is used to compare
               facet = max(self.entropy_dict, key=self.entropy_dict.get)
            else:
               facet = max(self.distance_dict, key=self.distance_dict.get)
            data = dict()
            data['facet'] = facet
            new_message = message(cfg.AGENT, cfg.USER, cfg.ASK_FACET, data)
            self.asked_feature.append(facet)
            return new_message
        else:
            pool = [item for item in cfg.FACET_POOL if item not in self.asked_feature]
            facet = np.random.choice(np.array(pool), 1)[0]
            data = dict()
            if facet in [item.name for item in cfg.cat_tree.children]:
                data['facet'] = facet
            else:
                data['facet'] = facet
            new_message = message(cfg.AGENT, cfg.USER, cfg.ASK_FACET, data)
            return new_message

    def prepare_rec_message(self):
        '''recent_candidate_list_ranked = ['3', '10']'''
        self.recent_candidate_list_ranked = [item for item in self.recent_candidate_list_ranked if item not in self.rejected_item_list_]  # Delete those has been rejected
        rec_list = self.recent_candidate_list_ranked[: 10]
        data = dict()
        data['rec_list'] = rec_list
        print ('recommend_list: ', rec_list)
        new_message = message(cfg.AGENT, cfg.USER, cfg.MAKE_REC, data)
        return new_message

    def response(self, input_message):
        '''
        The agent moves a step forward, upon receiving a message from the user.
        '''
        assert input_message.sender == cfg.USER
        assert input_message.receiver == cfg.AGENT
        #_______ update the agent self_______
        if input_message.message_type == cfg.INFORM_FACET:
            #update entropy and similarity
            self.update_upon_feature_inform(input_message)
        if input_message.message_type == cfg.REJECT_REC:
            self.rejected_item_list_ += input_message.data['rejected_item_list']
            self.rejected_time += 1
            if self.mini == 1:
                if self.alwaysupdate == 1:
                    for i in range(cfg.update_count):
                        self.mini_update_transE()
                    self.mini_update_already = True
                    self.recent_candidate_list = list(set(self.recent_candidate_list) - set(self.rejected_item_list_))
                    self.recent_candidate_list = list(set(self.recent_candidate_list) - set([self.busi_id])) + [self.busi_id]
                    self.recent_candidate_list_ranked = rank_items(self.known_feature_total, self.items, self.features, self.transE_model, self.recent_candidate_list, self.rejected_item_list_)

        #_______ Adding into history _______
        if input_message.message_type == cfg.INFORM_FACET:
            if self.turn_count > 0:  # means first doesn't# count
                if input_message.data['value'] is None:
                    self.history_list.append(0)  # ask attribute, fail
                else:
                    self.history_list.append(1)  # ask attribute, successful

        if input_message.message_type == cfg.REJECT_REC:
            self.history_list.append(-1)  # try recommendation, user doesn't want.
            self.recent_candidate_list = list(set(self.recent_candidate_list) - set(self.rejected_item_list_)) # don't consider

        if cfg.play_by != 'AOO' and cfg.play_by != 'AOO_valid':
            # Add control point here
            if cfg.mod == 'ours':
                state_vector = self.vectorize()

        action = None
        SoftMax = nn.Softmax(dim=-1)


        if cfg.play_by == 'policy':
            # means using Policy Network to determine action
            s = torch.from_numpy(state_vector).float()
            s = Variable(s, requires_grad=True)
            self.PN_model.eval()
            pred = self.PN_model(s)
            prob = SoftMax(pred)
            c = Categorical(prob)
#            print ("pred is: ", pred)
            # different way to choose action
            if cfg.eval == 1: # action selection based on max softmax value 
                # for evaluation of Action stage
                pred_data = pred.data.tolist()
                sorted_index = sorted(range(len(pred_data)), key=lambda k: pred_data[k], reverse=True)

                # The following line are for avoid asking same question
                # It is a fair evaluation, because all models have such operation.
                unasked_max = None
                for item in sorted_index:
                    if item < self.big_feature_length:
                        if cfg.FACET_POOL[item] not in self.asked_feature:
                            unasked_max = item
                            break
                    else:
                        unasked_max = self.big_feature_length
                        break
                action = Variable(torch.IntTensor([unasked_max]))  # make it compatible with torch
                print('action is: {}'.format(action))
            else: # for RL
                # for training of Action stage
                i = 0
                action_ = self.big_feature_length
                while(i < 10000):
                   action_ = c.sample()
#                   print ('action_ is: ', action_)
                   i += 1
                   if action_ <= self.big_feature_length:
                       if action_ == self.big_feature_length:
                           break
                       elif cfg.FACET_POOL[action_] not in self.asked_feature:
                           break
                action = action_
                print('action is: {}'.format(action))
#                print('facet is: {}'.format(cfg.FACET_POOL[action_]))

            log_prob = c.log_prob(action)
            if self.turn_count != 0:
                self.log_prob_list = torch.cat([self.log_prob_list, log_prob.reshape(1)])
            else:
                self.log_prob_list = log_prob.reshape(1)

            # translate into message
            if action < len(cfg.FACET_POOL):
                data = dict()
                data['facet'] = cfg.FACET_POOL[action]
                new_message = message(cfg.AGENT, cfg.USER, cfg.ASK_FACET, data)
            else:
                new_message = self.prepare_rec_message()
#                print ('*********** prepare_rec_message')
            self.action_tracker.append(action.data.numpy().tolist())
            self.candidate_length_tracker.append(len(self.recent_candidate_list))


        # following are for writing to numpy array
        action = None
        if new_message.message_type == cfg.ASK_FACET:
            action = cfg.FACET_POOL.index(new_message.data['facet'])

        if new_message.message_type == cfg.MAKE_REC:
            action = len(cfg.FACET_POOL)

        if cfg.purpose == 'pretrain':
            self.numpy_list.append((action, state_vector))
        # end following

        with open(self.write_fp, 'a') as f:
            f.write('Turn count: {}, candidate length: {}\n'.format(self.turn_count, len(self.recent_candidate_list)))
        return new_message
    # end def response
# end def class agent
