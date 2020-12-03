# -*- coding: utf-8 -*-
import pandas as pd
import json

df = pd.read_csv("transE.csv")
new = df.groupby(['User_id','Date']).size()
df2 = pd.DataFrame(new)
merged = df.merge(df2, on=['User_id','Date'],how='left')
merged.rename(columns={0: 'counts'}, inplace=True)
merged.to_csv("new_transE.csv",index=False)


df3 = pd.read_csv("new_transE.csv")
df3.sort_values(['User_id','Date'])
indexNames = df3[ df3['counts'] <= 1 ].index
df3.drop(indexNames , inplace=True)
df3.to_csv("new_transE.csv", index=False)

df4 = pd.read_csv("new_transE.csv", encoding= 'unicode_escape')
time_list = df4['Time'].to_list()
new_time=[]
for i in time_list:
    time=''
    for j in i:
        if j ==':':
            break
        else:
            time += j
    new_time.append(int(time))
df4['new_time'] = new_time
df4.to_csv("new_transE.csv", index=False)


def get_column_values(column):
    column_list = []
    for i in column:
        for j in i:
            if j not in column_list:
                column_list.append(j)
    return column_list

columns_to_change = ["User_id","Location_id","Item_id","L2_Category_name", "L1_Category_name", "POI_Type", "clusters"]
old_df = pd.read_csv("new_transE.csv", encoding= 'unicode_escape')

for column in columns_to_change:
    df1 = old_df[[column]].values.tolist()
    new_df1 = get_column_values(df1)
    
    POI_Type_list = []
    for item in new_df1:
        if item not in POI_Type_list:
            POI_Type_list.append(item)
    
    new_column_list = []
    for item in df1:
        for i in item:
            new_column_list.append(POI_Type_list.index(i))

    old_df.drop([column], axis=1)
    old_df[column] = new_column_list
    old_df.to_csv("new_transE_3.csv", index=False)
    
    
def get_unique_column_values(column):
    column_list = []
    for i in column:
        for j in i:
            if j not in column_list:
                column_list.append(j)
    return column_list
    
df_12 = pd.read_csv("new_transE_3.csv", usecols=['Item_id','POI_Type','stars','L2_Category_name','clusters'], encoding= 'unicode_escape')
df_12 = df_12.drop_duplicates()
df_12.to_csv("dict.csv", index=False)

df = pd.read_csv("new_transE_3.csv")
df3 = pd.read_csv("dict.csv")

L2_Category_name_list = df[['L2_Category_name']].values.tolist()
L2_Category_name_list = get_unique_column_values(L2_Category_name_list)
clusters_list = df[['clusters']].values.tolist()
clusters_list = get_unique_column_values(clusters_list)
POI_Type_list = df[['POI_Type']].values.tolist()
POI_Type_list = get_unique_column_values(POI_Type_list)

category_length = max(L2_Category_name_list) +1
cluster_length = max(clusters_list) +1
POI_Type_length = max(POI_Type_list) +1
item_dict=dict()
star_list =[]
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
        row_dict['feature_index'].append(2*int(row['stars'])-2+category_length+cluster_length+POI_Type_length)
        item_dict[str(int(row['Item_id']))] = row_dict
        
    else:
        star_list.append(row['stars'])
        item_dict[str(int(row['Item_id']))]['stars'] = round((sum(star_list))/len(star_list),1)
        item_dict[str(int(row['Item_id']))]['L2_Category_name'].append(int(row['L2_Category_name']))
        item_dict[str(int(row['Item_id']))]['feature_index'].append(int(row['L2_Category_name']))
L2_dict = dict()
df = pd.read_csv('new_transE_3.csv', usecols=['L1_Category','L2_Category_name']) 
for index, row in df.iterrows():
    if row['L1_Category'] not in L2_dict:
        row_list = []
        row_list.append(row['L2_Category_name'])
        L2_dict[str(row['L1_Category'])] = row_list
    else:
        if row['L2_Category_name'] not in L2_dict[str(row['L1_Category'])]:
            L2_dict[str(row['L1_Category'])].append(row['L2_Category_name'])


with open('L2.json', 'w') as json_file:
    json.dump(L2_dict, json_file)

poi_dict = dict()
star_list = dict()
df = pd.read_csv('new_transE_3.csv', usecols=['Item_id','Location_id','POI_Type','POI','stars','L2_Category_name']) 
for index, row in df.iterrows():
    if str(int(row['Item_id'])) not in poi_dict:
        row_dict = dict()
        row_dict['stars'] = [row['stars']]
        row_dict['Location_id'] = [int(row['Location_id'])]
        row_dict['POI'] = row['POI']
        row_dict['POI_Type'] = int(row['POI_Type'])
        row_dict['L2_Category_name'] = [int(row['L2_Category_name'])]
        poi_dict[str(int(row['Item_id']))] = row_dict
    else:
        if int(row['Location_id']) not in poi_dict[str(int(row['Item_id']))]['Location_id']:
            poi_dict[str(row['Item_id'])]['Location_id'].append(row['Location_id'])
            poi_dict[str(row['Item_id'])]['stars'].append(row['stars'])
            poi_dict[str(row['Item_id'])]['L2_Category_name'].append(row['L2_Category_name'])


with open('poi.json', 'w') as json_file:
    json.dump(poi_dict, json_file)