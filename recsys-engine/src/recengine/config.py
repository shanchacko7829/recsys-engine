SEED = 42
# time runs from 0 to 1; the data is split by time, never randomly
T_TRAIN, T_VAL = 0.80, 0.90          # train: t<0.8, validation: 0.8-0.9, test: 0.9-1.0
K_EVAL = (10, 20)
CAND_ALS, CAND_KNN, CAND_CONTENT, CAND_POP, CAND_COLD = 100, 50, 50, 30, 30
COLD_SHARE = 0.2
