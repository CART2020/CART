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
