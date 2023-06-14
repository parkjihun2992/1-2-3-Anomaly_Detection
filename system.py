import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import os

from collections import deque
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, RepeatVector, LSTM, Dense, Lambda, TimeDistributed
from tensorflow.keras import losses, metrics, activations
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
import seaborn as sns

split = pickle.load(open('./pred_model/final_data_ST_120_2019.pkl', 'rb'))['ST4000DM000']
file_list = os.listdir('./ST4000DM000')
train_list = split['train']
val_list = split['val']
test_list = split['test']
col = ['smart_7_raw', 'smart_9_raw', 'smart_240_raw', 'smart_241_raw', 'smart_242_raw', 'until_fail_days']
timestep_list = [5, 10, 20, 30]

# Anomaly Detection Scaler
scaler_path = './std_scaler.pkl'
scaler_zip = {f'{file_name[:-4]}': {} for file_name in file_list}
if not os.path.isfile(scaler_path):
    for file_name in file_list:
        db = pickle.load(open(f'./ST4000DM000/{file_name}', 'rb'))
        for db_type in [30, 60, 120]:
            temp = db[db['until_fail_days'] > db_type][col[:-1]]
            if len(temp) != 0:
                scaler = StandardScaler()
                scaler.fit(temp)
                scaler_zip[f'{file_name[:-4]}'][f'{db_type}'] = scaler

    pickle.dump(scaler_zip, open(scaler_path, 'wb'))
else:
    scaler_zip = pickle.load(open(scaler_path, 'rb'))

# Prediction Scaler
pred_scaler_path = './pred_model/final_data_ST_120_2019.pkl'

prediction_model = tf.keras.models.load_model('./pred_model/model_120d.h5')
diagnostic_model = tf.keras.models.load_model('./Model_STD/db_type_120_time_step_20_batch_128_node_32/db_type_120_time_step_20_batch_128_node_32.h5')


def flatten(X):
    '''
    Flatten a 3D array.

    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.

    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1] - 1), :]
    return (flattened_X)


file_number = f'{test_list[6]}'

test_file = pickle.load(open(f'./ST4000DM000/{file_number}.pkl', 'rb'))
pred_scaler = pickle.load(open(pred_scaler_path, 'rb'))['ST4000DM000']['std_scaler_(120d)'][f'{file_number}']

real_y = test_file[col[-1]]
test_db = test_file[col[:-1]]

# Applied Scaler
diagnosis_db = scaler_zip[f'{file_number}']['120'].transform(test_db)
prediction_db = pred_scaler.transform(test_db)

threshold = 2.79082426643083

db_d = deque(maxlen=20)
db_p = deque(maxlen=20)

time = []
yy = []
dy_list, py_list = [], []

my_labels = {"real": "Real value", "pred": "Prediction value", "Failure start point": "Failure start point", "Prediction start point": "Prediction start point", "Threshold": "Threshold", "Normal": "Normal", "Failure":"Failure"}

print(len(diagnosis_db)-120)

for i in range(len(diagnosis_db)):
    db_d.append(diagnosis_db[i])
    db_p.append(prediction_db[i])

    if len(db_d) == 20:
        time.append(i)

        pred_dy = diagnostic_model.predict(np.array([db_d]))
        mse = np.mean(np.power(flatten(np.array([db_d])) - flatten(pred_dy), 2), axis=1)
        dy_list.append(mse)

        pred_py = prediction_model.predict(np.array([db_p]))
        py_list.append(pred_py[0])
        yy.append(real_y[i])

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 10
        plt.rcParams["figure.figsize"] = (12, 8)

        plt.subplot(2, 1, 1)
        if dy_list[-1] <= threshold:
            plt.plot(time[-1], dy_list[-1], 'g.', label=my_labels['Normal'])
            my_labels["Normal"] = "_nolegend_"
        else:
            plt.plot(time[-1], dy_list[-1], 'rx', label=my_labels['Failure'])
            my_labels["Failure"] = "_nolegend_"
        threshold_line = plt.axhline(threshold, label=my_labels['Threshold'], color='black')
        mal_line = plt.axvline(len(diagnosis_db) - 120, label=my_labels['Failure start point'], color='orange')
        pred_line = plt.axvline(len(diagnosis_db) - 100, label=my_labels['Prediction start point'], color='purple')
        # my_labels["Failure start point"] = "_nolegend_"
        # my_labels["Prediction start point"] = "_nolegend_"
        my_labels["Threshold"] = "_nolegend_"

        plt.xlabel('Days')
        plt.ylabel('Reconstruction Error')
        plt.title('Condition Diagnosis')
        plt.legend()
        plt.tight_layout()

        plt.subplot(2, 1, 2)

        plt.plot(time, py_list, 'r.', label=my_labels['pred'])
        plt.plot(time, yy, 'b.', label=my_labels['real'])
        my_labels["pred"] = "_nolegend_"
        my_labels["real"] = "_nolegend_"

        mal_line = plt.axvline(len(diagnosis_db)-120, label=my_labels['Failure start point'], color='orange')
        pred_line = plt.axvline(len(diagnosis_db)-100, label=my_labels['Prediction start point'], color='purple')
        my_labels["Failure start point"] = "_nolegend_"
        my_labels["Prediction start point"] = "_nolegend_"

        plt.xlabel('Days')
        plt.ylabel('RUL (days)')
        plt.title('RUL Prediction')

        plt.legend()
        plt.tight_layout()

        plt.pause(0.01)
