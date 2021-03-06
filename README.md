# Coupled Variational Recurrent Collaborative Filtering
Code of KDD 2019 paper [Coupled Variational Recurrent Collaborative Filtering](https://arxiv.org/pdf/1906.04386.pdf).


## First Step:
Download Data from https://drive.google.com/drive/folders/1HeudBdACXQHSYTb04fNQNk7mVxiBGq7C?usp=sharing and put them in the 'data' folder. Each txt file has six columns as follows:
```
36955 21  3 789652009 0 0
36955 47  5 789652009 789652009 0
36955 1058  3 789652009 789652009 0
35139 1 4 822873600 0 0
35139 10  4 822873600 822873600 0
35435 10  3 822873600 0 822873600
35435 11  4 822873600 822873600 0
35139 18  4 822873600 822873600 0
35139 19  4 822873600 822873600 0
35621 19  1 822873600 0 822873600
    ...
```
The six columns correspond to:
```
user_id  item_id  rating  timestamp user_last_rating_timestamp  item_last_rated_timestamp
```
Each file is sorted based on the timestamp in the fourth column. The '0' elements in the fifth or the sixth column represent the new users or new items. 

## Second Step:
In command line run:
```
sh ml_10M.sh
```

## Cite this work

Qingquan Song, Shiyu Chang, and Xia Hu. "Coupled Variational Recurrent Collaborative Filtering." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2019. ([Download](https://www.kdd.org/kdd2019/accepted-papers/view/coupled-variational-recurrent-collaborative-filtering))

Biblatex entry:

    @inproceedings{song2019coupled,
      title={Coupled Variational Recurrent Collaborative Filtering},
      author=Song, Qingquan and Chang, Shiyu and Hu, Xia},
      booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
      pages={335--343},
      year={2019},
      organization={ACM}
    }
    
