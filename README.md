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

## Printed Result 
success rate is 0.0 at turn 1, accumulated sum is 0.0\\
success rate is 0.10238095238095238 at turn 2, accumulated sum is 0.10238095238095238
success rate is 0.49166666666666664 at turn 3, accumulated sum is 0.594047619047619
success rate is 0.27976190476190477 at turn 4, accumulated sum is 0.8738095238095238
success rate is 0.09761904761904762 at turn 5, accumulated sum is 0.9714285714285714
success rate is 0.002380952380952381 at turn 6, accumulated sum is 0.9738095238095238
success rate is 0.0011904761904761906 at turn 7, accumulated sum is 0.975
success rate is 0.0 at turn 8, accumulated sum is 0.975
success rate is 0.0 at turn 9, accumulated sum is 0.975
success rate is 0.0 at turn 10, accumulated sum is 0.975
Average turn is:  3.5845238095238097

## Reference
EAR System -- https://ear-conv-rec.github.io/manual.html#1-system-overview                
Lei, Wenqiang and He, Xiangnan and Miao, Yisong and Wu, Qingyun and Hong, Richang and Kan, Min-Yen and Chua, Tat-Seng, Estimation-Action-Reflection: Towards Deep Interaction Between Conversational and Recommender Systems, Proceedings of the 13th International Conference on Web Search and Data Mining

