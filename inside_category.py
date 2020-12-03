# -*- coding: utf-8 -*-
def getvalues():
    import pandas as pd
    
    df3 = pd.read_csv("inside_category.csv")
    inside_category_dict=dict()
    for index,  row in df3.iterrows():
        if str(int(row['Item_id'])) not in inside_category_dict: 
            l2_dict = dict()
            l2_dict[row['L2_Category_name']] = 1
            inside_category_dict[str(int(row['Item_id']))] = l2_dict
            
        else:
            if row['L2_Category_name'] not in inside_category_dict[str(int(row['Item_id']))]:
                inside_category_dict[str(int(row['Item_id']))][row['L2_Category_name']] = 1
            else:
                inside_category_dict[str(int(row['Item_id']))][row['L2_Category_name']] += 1
    
    total_dict= dict()
    
    for key in inside_category_dict.keys():
        total = 0
        for k in inside_category_dict[key].keys():
            total += inside_category_dict[key][k]
        total_dict[key] = total 
        
    for key in inside_category_dict.keys():
        for k in inside_category_dict[key].keys():
            temp = inside_category_dict[key][k] / total_dict[key]
            inside_category_dict[key][k] = round(temp, 4)
            
    categorys = list (k for k,v in inside_category_dict[str(12)].items())
    ratios = list (v for k,v in inside_category_dict[str(12)].items())
    return categorys, ratios