# -*- coding: utf-8 -*-
CUDA_LAUNCH_BLOCKING="1"
import sys

import pickle
import torch
import argparse

import time
import numpy as np
import json

from config import global_config as cfg
from epi import run_one_episode, update_PN_model
from pn import PolicyNetwork
import copy

from collections import defaultdict

import random
import os.path
import json
import math
import regex as re
import generate_data


random.seed(1)

the_max = 0
for k, v in cfg.item_dict.items():
    if the_max < max(v['feature_index']):
        the_max = max(v['feature_index'])
print(the_max)
FEATURE_COUNT = the_max + 1

success_at_turn_list_location_random = [0] * 10
success_at_turn_list_item = [0] * 10
success_at_turn_list_location_rate = [0] * 10




def main():
    parser = argparse.ArgumentParser(description="Run conversational recommendation.")
    parser.add_argument('-mt', type=int, dest='mt', help='MAX_TURN', default = 10)
    parser.add_argument('-playby', type=str, dest='playby', help='playby', default ='policy' )
    parser.add_argument('-optim', type=str, dest='optim', help='optimizer', default ='SGD')
    parser.add_argument('-lr', type=float, dest='lr', help='lr', default =0.001)
    parser.add_argument('-decay', type=float, dest='decay', help='decay', default =0)
    parser.add_argument('-TopKTaxo', type=int, dest='TopKTaxo', help='TopKTaxo', default =3)
    parser.add_argument('-gamma', type=float, dest='gamma', help='gamma', default =0)
    parser.add_argument('-trick', type=int, dest='trick', help='trick', default =0)
    parser.add_argument('-startFrom', type=int, dest='startFrom', help='startFrom', default =0)
    parser.add_argument('-endAt', type=int, dest='endAt', help='endAt', default =171) #test 171 
    parser.add_argument('-strategy', type=str, dest='strategy', help='strategy', default ='maxsim')
    parser.add_argument('-eval', type=int, dest='eval', help='eval', default =1)
    parser.add_argument('-mini', type=int, dest='mini', help='mini', default =1)
    parser.add_argument('-alwaysupdate', type=int, dest='alwaysupdate', help='alwaysupdate', default =1)
    parser.add_argument('-initeval', type=int, dest='initeval', help='initeval', default =0)
    parser.add_argument('-upoptim', type=str, dest='upoptim', help='upoptim', default ='SGD')
    parser.add_argument('-upcount', type=int, dest='upcount', help='upcount', default =4)
    parser.add_argument('-upreg', type=float, dest='upreg', help='upreg', default =0.001)
    parser.add_argument('-code', type=str, dest='code', help='code', default ='stable')
    parser.add_argument('-purpose', type=str, dest='purpose', help='purpose', default ='train' )
    parser.add_argument('-mod', type=str, dest='mod', help='mod', default ='ours')
    parser.add_argument('-mask', type=int, dest='mask', help='mask', default =0)
    # use for ablation study

    A = parser.parse_args()


    cfg.change_param(playby=A.playby, eval=A.eval, update_count=A.upcount, update_reg=A.upreg,
                     purpose=A.purpose, mod=A.mod, mask=A.mask)
    device = torch.device('cuda')
    random.seed(1)
#    random.shuffle(cfg.valid_list)
#    random.shuffle(cfg.test_list)
    the_valid_list_item = copy.copy(cfg.valid_list_item)
    the_valid_list_features = copy.copy(cfg.valid_list_features)
    the_valid_list_location = copy.copy(cfg.valid_list_location)
    
    the_test_list_item = copy.copy(cfg.test_list_item)
    the_test_list_features = copy.copy(cfg.test_list_features)
    the_test_list_location = copy.copy(cfg.test_list_location)
    
    the_train_list_item = copy.copy(cfg.train_list_item)
    the_train_list_features = copy.copy(cfg.train_list_features)
    the_train_list_location = copy.copy(cfg.train_list_location)    
#    random.shuffle(the_valid_list)
#    random.shuffle(the_test_list)


    gamma = A.gamma
    transE_model = cfg.transE_model
    if A.eval == 1:

        if A.mod == 'ours':
            fp = '../data/PN-model-ours/PN-model-ours.txt'
        if A.initeval == 1:

            if A.mod == 'ours':
                fp = '../data/PN-model-ours/pretrain-model.pt'
    else:
        # means training
        if A.mod == 'ours':
            fp = '../data/PN-model-ours/pretrain-model.pt'
            
    INPUT_DIM = 0

    if A.mod == 'ours':
        INPUT_DIM = 29 #11+10+8
    PN_model = PolicyNetwork(input_dim=INPUT_DIM, dim1=64, output_dim=12)
    start = time.time()

    try:
        PN_model.load_state_dict(torch.load(fp))
        print('Now Load PN pretrain from {}, takes {} seconds.'.format(fp, time.time() - start))
    except:
        print('Cannot load the model!!!!!!!!!\n fp is: {}'.format(fp))
        if cfg.play_by == 'policy':
            sys.exit()

    if A.optim == 'Adam':
        optimizer = torch.optim.Adam(PN_model.parameters(), lr=A.lr, weight_decay=A.decay)
    if A.optim == 'SGD':
        optimizer = torch.optim.SGD(PN_model.parameters(), lr=A.lr, weight_decay=A.decay)
    if A.optim == 'RMS':
        optimizer = torch.optim.RMSprop(PN_model.parameters(), lr=A.lr, weight_decay=A.decay)

    numpy_list = list()
    NUMPY_COUNT = 0

    sample_dict = defaultdict(list)
    conversation_length_list = list()
    
    combined_num = 0 
    total_turn = 0
    # start episode
    for epi_count in range(A.startFrom, A.endAt):
        if epi_count % 1 == 0:
            print('-----\nIt has processed {} episodes'.format(epi_count))

        start = time.time()

        current_transE_model = copy.deepcopy(transE_model)
        current_transE_model.to(device)
        
        param1, param2 = list(), list()
        i = 0
        for name, param in current_transE_model.named_parameters():
            if i == 0 or i==1:
                param1.append(param)
                # param1: head, tail
            else:
                param2.append(param)
                # param2: time, category, cluster, type
            i += 1

        '''change to transE embedding'''
        optimizer1_transE, optimizer2_transE = None, None
        if A.purpose != 'fmdata':
            optimizer1_transE = torch.optim.Adagrad(param1, lr=0.01, weight_decay=A.decay)
            if A.upoptim == 'Ada':
                optimizer2_transE = torch.optim.Adagrad(param2, lr=0.01, weight_decay=A.decay)
            if A.upoptim == 'SGD':
                optimizer2_transE = torch.optim.SGD(param2, lr=0.001, weight_decay=A.decay)

        if A.purpose != 'pretrain':
            items = the_valid_list_item[epi_count]  #0 18 10 3 
            features = the_valid_list_features[epi_count] #3,21,2,1    21,12,2,1   22,7,2,1 
            location = the_valid_list_location[epi_count]
            item_list = items.strip().split(' ') 
            u = item_list[0] 
            item = item_list[-1]
            if A.eval == 1:
#                u, item, l = the_test_list_item[epi_count]
                items = the_test_list_item[epi_count]  #0 18 10 3 
                features = the_test_list_features[epi_count] #3,21,2,1    21,12,2,1   22,7,2,1 
                location = the_test_list_location[epi_count]
                item_list = items.strip().split(' ') 
                u = item_list[0] 
                item = item_list[-1]          
                
            user_id = int(u)
            item_id = int(item)
            location_id = int(location)
        else:
            user_id = 0
            item_id = epi_count

        if A.purpose == 'pretrain':
            items = the_train_list_item[epi_count]  #0 18 10 3 
            features = the_train_list_features[epi_count] #3,21,2,1    21,12,2,1   22,7,2,1 
            location = the_train_list_location[epi_count]
            item_list = items.strip().split(' ') 
            u = item_list[0] 
            item = item_list[-1]
            user_id = int(u)
            item_id = int(item)
            location_id = int(location)
        print ("----target item: ", item_id)
        big_feature_list = list()
        
        '''update L2.json'''
        for k, v in cfg.taxo_dict.items():
#            print (k,v)
            if len(set(v).intersection(set(cfg.item_dict[str(item_id)]['L2_Category_name']))) > 0:
#                print(user_id, item_id) #433,122
#                print (k)
                big_feature_list.append(k)

        write_fp = '../data/interaction-log/{}/v4-code-{}-s-{}-e-{}-lr-{}-gamma-{}-playby-{}-stra-{}-topK-{}-trick-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-upreg-{}-m-{}.txt'.format(
            A.mod.lower(), A.code, A.startFrom, A.endAt, A.lr, A.gamma, A.playby, A.strategy, A.TopKTaxo, A.trick,
            A.eval, A.initeval,
            A.mini, A.alwaysupdate, A.upcount, A.upreg, A.mask)
        
        '''care the sequence of facet pool items'''
        if cfg.item_dict[str(item_id)]['POI_Type'] is not None:
            choose_pool = ['clusters', 'POI_Type'] + big_feature_list 

        choose_pool_original = choose_pool


        if A.purpose not in ['pretrain', 'fmdata']:
            choose_pool = [random.choice(choose_pool)]

        # run the episode
        for c in choose_pool:
            start_facet = c
            with open(write_fp, 'a') as f:
                f.write(
                    'Starting new\nuser ID: {}, item ID: {} episode count: {}\n'.format(user_id, item_id, epi_count))
            if A.purpose != 'pretrain':
                log_prob_list, rewards, success, turn_count, known_feature_category = run_one_episode(current_transE_model, user_id, item_id, A.mt, False, write_fp,
                                                         A.strategy, A.TopKTaxo,
                                                         PN_model, gamma, A.trick, A.mini,
                                                         optimizer1_transE, optimizer2_transE, A.alwaysupdate, start_facet, A.mask, sample_dict, choose_pool_original,features, items)
            else:
                current_np = run_one_episode(current_transE_model, user_id, item_id, A.mt, False, write_fp,
                                                         A.strategy, A.TopKTaxo,
                                                         PN_model, gamma, A.trick, A.mini,
                                                         optimizer1_transE, optimizer2_transE, A.alwaysupdate, start_facet, A.mask, sample_dict, choose_pool_original,features, items)
                numpy_list += current_np
        # end run
        
#         check POI type, recommend location id by star, check the success. 
        if A.purpose != 'pretrain': 
            total_turn += turn_count
            
        if A.purpose != 'pretrain':
            if success == True:
                if cfg.poi_dict[str(item_id)]['POI'] == 'Combined':
                    combined_num += 1
                    L2_Category_name_list = cfg.poi_dict[str(item_id)]['L2_Category_name']
                    category_list = known_feature_category
#                    print (category_list)
                    possible_L2_list = []
                    for i in range(len(L2_Category_name_list)):
                        for j in range(len(category_list)):
                            if L2_Category_name_list[i] == category_list[j]:
                                temp_location = cfg.poi_dict[str(item_id)]['Location_id'][i]
                                if temp_location not in possible_L2_list:
                                    possible_L2_list.append(temp_location)
                    location_list =[]
                    location_list = possible_L2_list
                    random_location_list = location_list.copy()
                    star_location_list = location_list.copy()
                    if location_id not in location_list:
#                        random_location_list += random.sample(cfg.poi_dict[str(item_id)]['Location_id'], int(len(cfg.poi_dict[str(item_id)]['Location_id'])/2))
#Can test random selection.
                        random_location_list = np.random.choice(category_list, int(len(category_list)/2), replace=False)
                        star_list = cfg.poi_dict[str(item_id)]['stars']
                        location_list = cfg.poi_dict[str(item_id)]['Location_id']
                        prob_star_list = [float(i)/sum(star_list) for i in star_list]               
                        if len(location_list) > 5:
                            random_location_list = np.random.choice(location_list, len(location_list)/2, p=prob_star_list, replace = False)
                        else:
                            random_location_list = location_list                        
                        star_list = cfg.poi_dict[str(item_id)]['stars']

                        star_list = [b[0] for b in sorted(enumerate(star_list),key=lambda i:i[1])]
                        print(star_list)
                        location_stars = star_list[:3]
                        for index in location_stars:
                            l = cfg.poi_dict[str(item_id)]['Location_id'][index]  
                            star_location_list.append(l)
                    print ('star_location_list: ', star_location_list)
                    print ('random_location_list', random_location_list)
                    if location_id in random_location_list:
                        print('Random Combined Rec Success! in episode: {}.'.format(epi_count))
                        success_at_turn_list_location_random[turn_count] += 1
                    else:
                        print('Random Combined Rec failed! in episode: {}.'.format(epi_count))
                        
                    if location_id in star_location_list:
                        print('Random Combined Rec Success! in episode: {}.'.format(epi_count))
                        success_at_turn_list_location_rate[turn_count] += 1
                    else:
                        print('Random Combined Rec failed! in episode: {}.'.format(epi_count))

                else:
                    print('Independent Rec Success! in episode: {}.'.format(epi_count))
                    success_at_turn_list_location_random[turn_count] += 1
                    success_at_turn_list_location_rate[turn_count] += 1
                
        if A.purpose != 'pretrain':
            if success == True:
                print('Rec Success! in episode: {}.'.format(epi_count))
                success_at_turn_list_item[turn_count] += 1
#        
        print ('total combined: ', combined_num)
        # update PN model
        if A.playby == 'policy' and A.eval != 1 and A.purpose != 'pretrain':
            update_PN_model(PN_model, log_prob_list, rewards, optimizer)
            print('updated PN model')
            current_length = len(log_prob_list)
            conversation_length_list.append(current_length)
        # end update


        check_span = 10
        if epi_count % check_span == 0 and epi_count >= 3 * check_span and cfg.eval != 1 and A.purpose != 'pretrain':
            PATH = '../data/PN-model-{}/v4-code-{}-s-{}-e-{}-lr-{}-gamma-{}-playby-{}-stra-{}-topK-{}-trick-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-upreg-{}-m-{}-epi-{}.txt'.format(
                A.mod.lower(), A.code, A.startFrom, A.endAt, A.lr, A.gamma, A.playby, A.strategy, A.TopKTaxo, A.trick,
                A.eval, A.initeval,
                A.mini, A.alwaysupdate, A.upcount, A.upreg, A.mask, epi_count)
            torch.save(PN_model.state_dict(), PATH)
            print('Model saved at {}'.format(PATH))

            a1 = conversation_length_list[epi_count - 3 * check_span: epi_count - 2 * check_span]
            a2 = conversation_length_list[epi_count - 2 * check_span: epi_count - 1 * check_span]
            a3 = conversation_length_list[epi_count - 1 * check_span: ]
            a1 = np.mean(np.array(a1))
            a2 = np.mean(np.array(a2))
            a3 = np.mean(np.array(a3))

            with open(write_fp, 'a') as f:
                f.write('$$$current turn: {}, a3: {}, a2: {}, a1: {}\n'.format(epi_count, a3, a2, a1))
            print('current turn: {}, a3: {}, a2: {}, a1: {}'.format(epi_count, a3, a2, a1))

            num_interval = int(epi_count / check_span)
            for i in range(num_interval):
                ave = np.mean(np.array(conversation_length_list[i * check_span: (i + 1) * check_span]))
                print('start: {}, end: {}, average: {}'.format(i * check_span, (i + 1) * check_span, ave))
                PATH = '../data/PN-model-{}/v4-code-{}-s-{}-e-{}-lr-{}-gamma-{}-playby-{}-stra-{}-topK-{}-trick-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-upreg-{}-m-{}-epi-{}.txt'.format(
                    A.mod.lower(), A.code, A.startFrom, A.endAt, A.lr, A.gamma, A.playby, A.strategy, A.TopKTaxo,
                    A.trick,
                    A.eval, A.initeval,
                    A.mini, A.alwaysupdate, A.upcount, A.upreg, A.mask, (i + 1) * check_span)
                print('Model saved at: {}'.format(PATH))

            if a3 > a1 and a3 > a2:
                print('Early stop of RL!')
                exit()

        # write control information
        if A.purpose != 'pretrain':
            with open(write_fp, 'a') as f:
                f.write('Big features are: {}\n'.format(choose_pool))
                if rewards is not None:
                    f.write('reward is: {}\n'.format(rewards.data.numpy().tolist()))
                f.write('WHOLE PROCESS TAKES: {} SECONDS\n'.format(time.time() - start))
        # end write

        # Write to pretrain numpy which is the pretrain data.
        if A.purpose == 'pretrain':
            if len(numpy_list) > 5000:
                with open('../data/pretrain-numpy-data-{}/segment-{}-start-{}-end-{}.pk'.format(
                        A.mod, NUMPY_COUNT, A.startFrom, A.endAt), 'wb') as f:
                    pickle.dump(numpy_list, f)
                    print('Have written 5000 numpy arrays!')
                NUMPY_COUNT += 1
                numpy_list = list()
        # end write
    
    print ('\nLocation result:')
    sum_sr = 0.0
    for i in range(len(success_at_turn_list_location_rate)):
        success_rate = success_at_turn_list_location_rate [i]/A.endAt
        sum_sr += success_rate
        print ('success rate is {} at turn {}, accumulated sum is {}'.format(success_rate, i+1, sum_sr))   
    print ('Average turn is: ', total_turn/A.endAt +1)   

if __name__ == '__main__':
    main()
