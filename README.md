# airbnb-booking-predict

## Steps to run
- Unpack *data/sessions.csv* from kaggle
- Run **action_count.py** (4-6 hrs at 15-20 sec per 10k entries) to generate *data/user_actions_count.csv*
- Run **process_input.py** (~6 min) to generate *data/train_users_2_norm.csv* and *data/test_users_norm.csv*
- Run **train_and_predict_ensemble.py** to generate *predict/enA.csv* and *predict/enB.csv* submission data  
**OR**  
- Run **train_and_predict_singleXGB.py** (<10 min at 4 threads) to generate *predict/xgb.csv* submission data

## Score
- Single XGB  on whole set of features score **0.87 <= ndcg_5 <= 0.876** which correspondes to **top 25%** in private LB (max score not in the last commit since it has too many features)
