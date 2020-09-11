# -*- coding: utf-8 -*-
import pandas as pd
import copy
def get_data():
    df = pd.read_csv("new_transE_3.csv")
    #get count list
    new = df.groupby(['User_id','Date']).size()
    df2 = pd.DataFrame(new)
    df2.rename(columns={0: 'counts'}, inplace=True)
    count_list = df2['counts'].tolist()
    #print (count_list)
    
    # frequency count for (user date)
    #data = []
    
    current_user = df.loc[0]['User_id']
    
    total_dict = dict()
    
    count_dict = dict()
    count_dict['user_list'] = []
    count_dict['item_list'] = []
    count_dict['L1_category_list'] = []
    count_dict['L2_category_list'] = []
    count_dict['cluster_list'] = []
    count_dict['type_list'] = []
    count_dict['location_list'] = []
    count_dict['star_list'] = []
    count_dict['time_list'] = []
    
    begin_row_index = 0
    #sub_count_list = count_list[0:48]
    previous_user = 0
    
    for count in count_list:
        current_user = df.loc[begin_row_index]['User_id']  
        
        if previous_user != current_user:
            total_dict[str(previous_user)] = copy.deepcopy(count_dict)
            count_dict.clear()
            count_dict['user_list'] = []
            count_dict['item_list'] = []
            count_dict['L1_category_list'] = []
            count_dict['L2_category_list'] = []
            count_dict['cluster_list'] = []
            count_dict['type_list'] = []
            count_dict['location_list'] = []
            count_dict['star_list'] = []
            count_dict['time_list'] = []
            previous_user = current_user
        else:
            total_dict[str(current_user)] = copy.deepcopy(count_dict)  
            
        for end_row_index in range(2,count+1):
            df3 = df[begin_row_index: begin_row_index + end_row_index]
            count_dict['user_list'].append(df3['User_id'].tolist())
            count_dict['item_list'].append(df3['Item_id'].tolist())
            count_dict['L1_category_list'].append(df3['L1_Category_name'].tolist())
            count_dict['L2_category_list'].append(df3['L2_Category_name'].tolist())
            count_dict['cluster_list'].append(df3['clusters'].tolist())
            count_dict['type_list'].append(df3['POI_Type'].tolist())
            count_dict['location_list'].append(df3['Location_id'].tolist())
            count_dict['star_list'].append(df3['stars'].tolist())
            count_dict['time_list'].append(df3['new_time'].tolist())
        begin_row_index += count 
    
    train_dict = dict()
    valid_dict = dict()
    test_dict = dict()
    sub_dict = dict()
    sub_dict['user_list'] = []
    sub_dict['item_list'] = []
    sub_dict['L1_category_list'] = []
    sub_dict['L2_category_list'] = []
    sub_dict['cluster_list'] = []
    sub_dict['type_list'] = []
    sub_dict['location_list'] = []
    sub_dict['star_list'] = []
    sub_dict['time_list'] = []        
    for key in total_dict.keys():
        length = len(total_dict[key]['user_list'])
    #    print (key, type(key))
        train_list_group = []
        valid_list_group = []
        test_list_group = []
        
        test_list_group.append( total_dict[key]['user_list'][: int(0.1*(length)+1)])
        valid_list_group.append( total_dict[key]['user_list'][int(0.1*(length)+1) : int(0.3*(length)+1)])
        train_list_group.append( total_dict[key]['user_list'][int(0.3*(length)+1) : ])
    #    total_dict[key]['user_list']
        
        test_list_group.append( total_dict[key]['item_list'][: int(0.1*(length)+1)])
        valid_list_group.append( total_dict[key]['item_list'][int(0.1*(length)+1) : int(0.3*(length)+1)])
        train_list_group.append( total_dict[key]['item_list'][int(0.3*(length)+1) : ])
    #    total_dict[key]['item_list']
        
        test_list_group.append( total_dict[key]['L1_category_list'][: int(0.1*(length)+1)])
        valid_list_group.append( total_dict[key]['L1_category_list'][int(0.1*(length)+1) : int(0.3*(length)+1)])
        train_list_group.append( total_dict[key]['L1_category_list'][int(0.3*(length)+1) : ])
    #    total_dict[key]['L1_category_list']
        
        test_list_group.append( total_dict[key]['L2_category_list'][: int(0.1*(length)+1)])
        valid_list_group.append( total_dict[key]['L2_category_list'][int(0.1*(length)+1) : int(0.3*(length)+1)])
        train_list_group.append( total_dict[key]['L2_category_list'][int(0.3*(length)+1) : ])
    #    total_dict[key]['L2_category_list']
        
        test_list_group.append( total_dict[key]['cluster_list'][: int(0.1*(length)+1)])
        valid_list_group.append( total_dict[key]['cluster_list'][int(0.1*(length)+1) : int(0.3*(length)+1)])
        train_list_group.append( total_dict[key]['cluster_list'][int(0.3*(length)+1) : ])
    #    total_dict[key]['cluster_list']
        
        test_list_group.append( total_dict[key]['type_list'][: int(0.1*(length)+1)])
        valid_list_group.append( total_dict[key]['type_list'][int(0.1*(length)+1) : int(0.3*(length)+1)])
        train_list_group.append( total_dict[key]['type_list'][int(0.3*(length)+1) : ])
    #    total_dict[key]['type_list']
        
        test_list_group.append( total_dict[key]['location_list'][: int(0.1*(length)+1)])
        valid_list_group.append( total_dict[key]['location_list'][int(0.1*(length)+1) : int(0.3*(length)+1)])
        train_list_group.append( total_dict[key]['location_list'][int(0.3*(length)+1) : ])
    #    total_dict[key]['location_list']
        
        test_list_group.append( total_dict[key]['star_list'][: int(0.1*(length)+1)])
        valid_list_group.append( total_dict[key]['star_list'][int(0.1*(length)+1) : int(0.3*(length)+1)])
        train_list_group.append( total_dict[key]['star_list'][int(0.3*(length)+1) : ])
    #    total_dict[key]['star_list']
        
        test_list_group.append( total_dict[key]['time_list'][: int(0.1*(length)+1)])
        valid_list_group.append( total_dict[key]['time_list'][int(0.1*(length)+1) : int(0.3*(length)+1)])
        train_list_group.append( total_dict[key]['time_list'][int(0.3*(length)+1) : ])
    #    total_dict[key]['time_list']
        test_dict[key] = test_list_group
        valid_dict[key] = valid_list_group
        train_dict[key] = train_list_group
        
    return train_dict,valid_dict,test_dict
        