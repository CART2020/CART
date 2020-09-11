from collections import Counter
import numpy as np
from config import global_config as cfg
import time


class cal_ent():
    '''
    given the current candidate list, calculate the entropy of every big feature(denote by f)
    '''

    def __init__(self, recent_candidate_list):
        self.recent_candidate_list = recent_candidate_list

    def calculate_entropy_for_one_tag(self, tagID, _counter):
        '''
        Args:
        tagID: int
        '''
        v = _counter[tagID]
#        print ("v: ", v)
        if v > 1:
            v=1
        p1 = float(v) / len(self.recent_candidate_list)
#        print ("p1: ", p1)
        p2 = 1.0 - p1

        if p1 == 0 or p1 == 1:
            return 0
        return (- p1 * np.log2(p1) - p2 * np.log2(p2))

    def do_job(self):
        entropy_dict_small_feature = dict()
        cat_list_all = list()
        for k in self.recent_candidate_list:
            cat_list_all += cfg.item_dict[str(k)]['L2_Category_name']
            

        c = Counter(cat_list_all)
        for k, v in c.items():

            node_entropy_self = self.calculate_entropy_for_one_tag(k, c)
            entropy_dict_small_feature[k] = node_entropy_self
 
        entropy_dict = dict()
        for f in (['clusters', 'POI_Type']):
            value_list = [cfg.item_dict[str(bid)][f] for bid in self.recent_candidate_list]
            c = Counter(value_list)
            v = [v for k, v in c.items()]
            entropy_dict[f] = np.sum([- (p / float(len(value_list))) * np.log2(p / float(len(value_list))) for p in v])

        for big_feature in cfg.FACET_POOL[2: ]:  # for those features, not in {city stars price}
#            print ("big_feature: ", big_feature)
            remained_small = [f for f in cfg.taxo_dict[big_feature] if f in entropy_dict_small_feature.keys()]
#            print ('remained_small',remained_small)
            if len(remained_small) == 0:
                entropy_dict[big_feature] = 0
                continue
            entropy_dict[big_feature] = sum(entropy_dict_small_feature[f] for f in remained_small)

        return entropy_dict
