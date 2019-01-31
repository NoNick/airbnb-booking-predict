# airbnb-booking-predict

## Steps to run
- Unpack *data/sessions.csv* from kaggle
- Run **action_count.py** (4-6 hrs at 15-20 sec per 10k entries) to generate *data/user_actions_count.csv*
- Run **process_input.py** (couple minutes) to generate *data/train_users_2_norm.csv* and *data/test_users_norm.csv*
- Run **3level.py** to generate *predict/enA.csv* and *predict/enB.csv* submission data
**OR**
- Run **gradient_boost.py** to generate *predict/xgb.csv* submission data

## Score
- Two classifiers on whole set of features (commit 09b4fea) take less than hour to learn in 4 threads and score **ndcg_5 = 0.20633**
- Two classifiers (LinReg & XGB) per each of 13 features (branch 13features) take 7 hrs in 4 threads and score **ndcg_5 = 0.23625**
- XGB on whole set of features take 5 min and score **ndcg_5 = 0.87562** (correspondes to **top 25%** in private LB)
