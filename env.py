# -*- coding: utf-8 -*-
import sys

from collections import Counter
import numpy as np
from random import randint
import json
import random

from message import message
from config import global_config as cfg
import time


class user():
    def __init__(self, user_id, busi_id):
        self.user_id = user_id
        self.busi_id = busi_id
        self.recent_candidate_list = [int(k) for k, v in cfg.item_dict.items()]
        self.asked_feature = list()

    def find_brother(self, node):
        return [child.name for child in node.parent.children if child.name != node.name]

    def find_children(self, node):
        return [child.name for child in node.children if child.name != node.name]

    def inform_facet(self, facet):
        data = dict()
        data['facet'] = facet
        
        #check if facet asked
        if facet in self.asked_feature:
            data['value'] = None
            return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)

        self.asked_feature.append(facet)

        
        if facet in ['stars']:
            data['value'] = [cfg.item_dict[str(self.busi_id)][facet]]
            return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)

        elif facet in ['POI_Type']:
            data['value'] = [cfg.item_dict[str(self.busi_id)][facet]]
            if cfg.item_dict[str(self.busi_id)][facet] is None:
                data['value'] = None
            return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)

        elif facet in ['clusters']:
            data['value'] = [cfg.item_dict[str(self.busi_id)][facet]]
            return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)

        else:  # means now deal with other features
            
            #get L2 category from L1
            candidate_feature = cfg.taxo_dict[facet]
            ground_truth_feature = cfg.item_dict[str(self.busi_id)]['L2_Category_name']
            intersection_between = list(set(candidate_feature).intersection(set(ground_truth_feature)))
            
            
            if len(intersection_between) == 0:
                data['value'] = None
                return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)

            if len(intersection_between) >1:
                data['value'] = intersection_between[:2]
                return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)
            else:
                data['value'] = intersection_between
                return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)

    def response(self, input_message):
        assert input_message.sender == cfg.AGENT
        assert input_message.receiver == cfg.USER


        new_message = None
        if input_message.message_type == cfg.EPISODE_START or input_message.message_type == cfg.ASK_FACET:
            facet = input_message.data['facet']
            new_message = self.inform_facet(facet)

        if input_message.message_type == cfg.MAKE_REC:
            if str(self.busi_id) in input_message.data['rec_list']:
                data = dict()
                data['ranking'] = input_message.data['rec_list'].index(str(self.busi_id)) + 1
                data['total'] = len(input_message.data['rec_list'])
                new_message = message(cfg.USER, cfg.AGENT, cfg.ACCEPT_REC, data)
            else:
                data = dict()
                data['rejected_item_list'] = input_message.data['rec_list']
                new_message = message(cfg.USER, cfg.AGENT, cfg.REJECT_REC, data)
        return new_message
