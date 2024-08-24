import streamlit as st
import catboost as ctb
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
st.title('Damage state prediction for RCSWs----<Hamidia et al.>')
algorithm= st.radio('ML algorithm:', ['Extreme Gradient Boost','CatBoost'])
df = pd.read_csv('RCSW-Input.csv', index_col=[])
data=df.head(285)
y = data["DS"]

def predictionCB (Aspect_Ratio, TB, LR, RL, BT, Lacunarity, D_1, D0, D1):
 X1 = data[['h/l','T2B', 'L2R', 'R2L', 'B2T', 'LC', '-1', '0', '1']]
 X_trainset, X_testset, y_trainset, y_testset = train_test_split(X1, y, test_size=0.3, random_state=1, stratify= y)
 model_CBC = ctb.CatBoostClassifier(depth = 10, iterations = 100, learning_rate = 0.1, border_count = 10, l2_leaf_reg = 3)
 model_CBC.fit(X_trainset, y_trainset)
 predicted_y = model_CBC.predict([Aspect_Ratio, TB, LR, RL, BT, Lacunarity, D_1, D0, D1])
 return predicted_y

def predictionXGB (A):
 X = data[['h/l', 'fc', 'rhov', 'rhoh', 'fyv', 'fyh', 'AL', 'BE', 'T2B', 'L2R', 'R2L', 'B2T', 'LC','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','10','9','8','7','6','5','4','3','2','1','0']]
 X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=1, stratify= y)
 xgbModel = xgb.XGBClassifier(max_depth = 12, n_estimators = 150, colsample_bytree = 1,learning_rate = 0.1, min_child_weight = 1, reg_alpha = 0, reg_lambda = 1, subsample = 0.9)
 xgbModel.fit(X_trainset,y_trainset)
 predicted_y = xgbModel.predict(A)
 return predicted_y

if algorithm=='CatBoost':
      Aspect_Ratio =st.number_input("Aspect Ratio__H/L")
      TB =st.number_input("Succolarity__T2B")
      LR = st.number_input("Succolarity_L2R")
      RL = st.number_input("Succolarity_R2L")
      BT = st.number_input("Succolarity_B2T")
      Lacunarity = st.number_input("Lacunarity")
      D_1 = st.number_input("D_1")
      D0 = st.number_input("Monofractal")
      D1 = st.number_input("D+1")
      yy=predictionCB(Aspect_Ratio, TB, LR, RL, BT, Lacunarity, D_1, D0, D1)
else:
      Aspect_Ratio =st.number_input("Aspect Ratio__H/L")
      fc =st.number_input("Compressive strength of concrete (MPa)")
      fyh =st.number_input("Vertical rebar strength (MPa)")
      fyv =st.number_input("Horizontal rebar strength (MPa)")
      rhov =st.number_input("Horizontal reinforcement ratio (%)")
      rhoh =st.number_input("Vertical reinforcement ratio (%)")
      AL =st.number_input("Axial load capacity")
      BE =st.number_input("Boundry element: 0 for No 1 for Yes")
      TB =st.number_input("Succolarity__T2B")
      LR = st.number_input("Succolarity_L2R")
      RL = st.number_input("Succolarity_R2L")
      BT = st.number_input("Succolarity_B2T")
      Lacunarity = st.number_input("Lacunarity")
      D_10 = st.number_input("D_10")
      D_9 = st.number_input("D_9")
      D_8 = st.number_input("D_8")
      D_7 = st.number_input("D_7")
      D_6 = st.number_input("D_6")
      D_5 = st.number_input("D_5")
      D_4 = st.number_input("D_4")
      D_3 = st.number_input("D_3")
      D_2 = st.number_input("D_2")
      D_1 = st.number_input("D_1")
      D0 = st.number_input("Monofractal")
      D1 = st.number_input("D+1")
      D2 = st.number_input("D+2")
      D3 = st.number_input("D+3")
      D4 = st.number_input("D+4")
      D5 = st.number_input("D+5")
      D6 = st.number_input("D+6")
      D7 = st.number_input("D+7")
      D8 = st.number_input("D+8")
      D9 = st.number_input("D+9")
      D10 = st.number_input("D+10")
      a=np.asarray([[Aspect_Ratio, fc, rhov, rhoh, fyv, fyh, AL, BE, TB, LR, RL, BT, Lacunarity,D_10,D_9,D_8,D_7,D_6,D_5,D_4,D_3,D_2,D_1,D10,D9,D8,D7,D6,D5,D4,D3,D2,D1,D0]])
      yy=predictionXGB(a)

st.button("DS prediction")
st.write(yy)
