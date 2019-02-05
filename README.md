# airbnb-booking-predict

## Score
- Single XGB on whole set of features score **ndcg_5 = 0.8766** which correspondes to **top 25%** in private LB
Alas, the competition is closed and this score isn't in LB.

## Steps to run
- Unpack *data/train_users_2_norm.csv.zip* and *data/test_users_norm.csv.zip*
- Run **train_and_predict_singleXGB.py** (<10 min at 4 threads) to generate *predict/xgb.csv* submission data  
**OR**  
- Run **train_and_predict_stacking.py** to generate *predict/ensemble.csv* submission data  

## Other runnables
- **action_count.py** (4-6 hrs at 15-20 sec per 10k entries) requires *data/sessions.csv* from kaggle and generates *data/user_actions_count.csv*
- **process_input.py** (~6 min) to generates *data/train_users_2_norm.csv* and *data/test_users_norm.csv*
- **xgb_gridsearch.py** finds best params (from described) for each classifier in ensemble
