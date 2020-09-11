import env
import agent
from config import global_config as cfg
from message import message
import random
import torch
from torch.autograd import Variable
import numpy as np

def auxiliary_reward():
    import pandas as pd
    auxiliary_reward_dict = dict()
    df = pd.read_csv("reward.csv")
    max_index = df.shape[0] 
    for index,  row in df.iterrows():
        if str(int(row['Item_id'])) not in auxiliary_reward_dict: 
            auxiliary_reward_dict[str(int(row['Item_id']))] = 1
            
        else:
            auxiliary_reward_dict[str(int(row['Item_id']))] += 1
        
        
    for key in auxiliary_reward_dict.keys():
        auxiliary_reward_dict[key] = round(auxiliary_reward_dict[key] / max_index,4)
    return auxiliary_reward_dict

def choose_start_facet(busi_id):
    choose_pool = list()
    if cfg.item_dict[str(busi_id)]['stars'] is not None:
        choose_pool.append('stars')
    if cfg.item_dict[str(busi_id)]['clusters'] is not None:
        choose_pool.append('clusters')
    if cfg.item_dict[str(busi_id)]['POI_Type'] is not None:
        choose_pool.append('POI_Type')
    print('choose_pool is: {}'.format(choose_pool))

    THE_FEATURE = random.choice(choose_pool)

    return THE_FEATURE


def get_reward(history_list, gamma, trick):
    prev_reward = - 0.01

    # -2: reach maximum turn, end.
    # -1: recommend unsuccessful
    # 0: ask attribute, unsuccessful
    # 1: ask attribute, successful
    # 2: recommend successful!

    r_dict = {
        2: 1 + prev_reward,
        1: 0.1 + prev_reward,
        0: 0 + prev_reward,
        -1: 0 - 0.1,
        -2: -0.3
    }

    reward_list = [r_dict[item] for item in history_list]

    print('gamma: {}'.format(gamma))

    rewards = []
    R = 0
    for r in reward_list[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    # turn rewards to pytorch tensor and standardize
    rewards = torch.Tensor(rewards)
    print('history list: {}'.format(history_list))
    print('reward: {}'.format(rewards))

    # It is a trick for optimization of policy gradient, we can consider use it or not
    # We didn't use it. But the follower of our work can consider use it.
    if trick == 1:
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    return rewards


''' recommend procedure '''
def run_one_episode(transE_model, user_id, busi_id, MAX_TURN, do_random, write_fp, strategy, TopKTaxo,
                    PN_model, gamma, trick, mini, optimizer1_transE, optimizer2_transE, alwaysupdate, start_facet, mask, sample_dict, choose_pool,features, items):
    # _______ initialize user and agent _______
    
    success = None
    
    the_user = env.user(user_id, busi_id)

    numpy_list = list()
    log_prob_list, reward_list = Variable(torch.Tensor()) , list()
    action_tracker, candidate_length_tracker = list(), list()

    the_agent = agent.agent(transE_model, user_id, busi_id, do_random, write_fp, strategy, TopKTaxo, numpy_list, PN_model, log_prob_list, action_tracker, candidate_length_tracker, mini, optimizer1_transE, optimizer2_transE, alwaysupdate, mask, sample_dict, choose_pool, features, items)

    # _______ initialize start message _______
    data = dict()
    data['facet'] = start_facet
    start_signal = message(cfg.AGENT, cfg.USER, cfg.EPISODE_START, data)
 
    agent_utterance = None
    while(the_agent.turn_count < MAX_TURN):
        if the_agent.turn_count == 0:
            user_utterance = the_user.response(start_signal) # user responses start_signal
        else: 
            user_utterance = the_user.response(agent_utterance) 
        with open(write_fp, 'a') as f:
            f.write('The user {} utterance in #{} turn, type: {}, data: {}\n'.format(user_id, the_agent.turn_count, user_utterance.message_type, user_utterance.data))
                
        if user_utterance.message_type == cfg.ACCEPT_REC:
            success = True
            the_agent.history_list.append(2) #success recommend get reward 2， look for auxiliary_reward
            rewards = get_reward(the_agent.history_list, gamma, trick)
            if cfg.purpose == 'pretrain':
                return numpy_list
            else:
                return (the_agent.log_prob_list, rewards, success, int(the_agent.turn_count), the_agent.known_feature_category)

        agent_utterance = the_agent.response(user_utterance)

        the_agent.turn_count += 1

        if the_agent.turn_count == MAX_TURN:
            success = False
            the_agent.history_list.append(-2)#failed recommend get reward -2
            print('Max turn quit...')
            rewards = get_reward(the_agent.history_list, gamma, trick)
            if cfg.purpose == 'pretrain':
                return numpy_list
            else:
                return (the_agent.log_prob_list, rewards, success, int(the_agent.turn_count), the_agent.known_feature_category)


def update_PN_model(model, log_prob_list, rewards, optimizer):
    model.train()

    loss = torch.sum(torch.mul(log_prob_list, Variable(rewards)).mul(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
