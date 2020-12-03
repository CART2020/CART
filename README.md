# CART
This is the code of a conversation-based adaptive relational translation framework (CART) for next POI recommendation with uncertain check-ins. The CART consists of two modules: 
1. The recommender built upon the adaptive relational translation method performs location prediction; 
2. The conversation manager aims to achieve successful recommendations with the fewest conversation turns. 

## Pre-requisits
* ### Running environment
  - Python 3.7.4
  - Pytorch 1.4.0
  - pandas 0.25.1
  
* ### Datasets
Three datasets which are generated from Foursquare in three cities, i.e., Calgary (CAL), Charlotte (CHA) and Phoenix (PHO).
```bash
https://developer.foursquare.com/docs/build-with-foursquare/categories/
```
```
https://sites.google.com/site/yangdingqi/home/foursquare-dataset
```


* ### Modules of CART
  - #### Recommender
      generate_data.py (input generation)
      
      train.py (train translation model)
      
  - #### Conversation Manager
      pn.py (action generator)
      
      agent.py (state tracker, online update)
      
      env.py (user response)
      
## How to run
You can run Train_TransE.py directly. Change Parsers in need. 

## Reference
EAR System -- https://ear-conv-rec.github.io/manual.html#1-system-overview                
Lei, Wenqiang and He, Xiangnan and Miao, Yisong and Wu, Qingyun and Hong, Richang and Kan, Min-Yen and Chua, Tat-Seng, Estimation-Action-Reflection: Towards Deep Interaction Between Conversational and Recommender Systems, Proceedings of the 13th International Conference on Web Search and Data Mining

