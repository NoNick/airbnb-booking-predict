# airbnb-booking-predict

## Score
- Single XGB  on whole set of features score **0.87 <= ndcg_5 <= 0.876** which correspondes to **top 25%** in private LB (score in the last commit isn't highest since single xgb params wasn't calibrated on the last feature set)

## Steps to run
- Unpack *data/train_users_2_norm.csv.zip* and *data/test_users_norm.csv.zip*
- Run **train_and_predict_ensemble.py** to generate *predict/ensemble.csv* submission data  
**OR**  
- Run **train_and_predict_singleXGB.py** (<10 min at 4 threads) to generate *predict/xgb.csv* submission data

## Other runnables
- **action_count.py** (4-6 hrs at 15-20 sec per 10k entries) requires *data/sessions.csv* from kaggle and generates *data/user_actions_count.csv*
- **process_input.py** (~6 min) to generates *data/train_users_2_norm.csv* and *data/test_users_norm.csv*
- **xgb_gridsearch.py** finds best params (from described) for each classifier in ensemble
