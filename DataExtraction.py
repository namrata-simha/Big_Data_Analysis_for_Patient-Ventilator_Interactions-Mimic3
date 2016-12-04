import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.style.use('ggplot')
#matplotlib inline
from sklearn.linear_model.logistic import LogisticRegression
def get_admission():
        admission_df = pd.read_csv("C:/Users/namra/Desktop/Study Material/Master Of Science (M.S.) Computer Science/Fall 2016/TIM 209 Statistical learning, data science and business analytics/Project/mimic3/demo/ADMISSIONS.csv", sep=',',header=0)
        return admission_df \
            .drop('ROW_ID', 1) \
            .drop('SUBJECT_ID', 1) \
            .drop('ADMISSION_LOCATION', 1) \
            .drop('DISCHARGE_LOCATION', 1) \
            .drop('INSURANCE', 1) \
            .drop('LANGUAGE', 1) \
            .drop('RELIGION', 1) \
            .drop('MARITAL_STATUS', 1) \
            .drop('ETHNICITY', 1) \
            .drop('EDREGTIME', 1) \
            .drop('EDOUTTIME', 1) \
            .drop('HAS_CHARTEVENTS_DATA', 1)

def get_chartevents():
        chartevents_df = pd.read_csv("C:/Users/namra/Desktop/Study Material/Master Of Science (M.S.) Computer Science/Fall 2016/TIM 209 Statistical learning, data science and business analytics/Project/mimic3/demo/CHARTEVENTS.csv", sep=',',header=0)
        return chartevents_df \
            .drop('ROW_ID', 1) \
            .drop('SUBJECT_ID', 1) \
            .drop('ADMISSION_LOCATION', 1) \
            .drop('DISCHARGE_LOCATION', 1) \
            .drop('INSURANCE', 1) \
            .drop('LANGUAGE', 1) \
            .drop('RELIGION', 1) \
            .drop('MARITAL_STATUS', 1) \
            .drop('ETHNICITY', 1) \
            .drop('EDREGTIME', 1) \
            .drop('EDOUTTIME', 1) \
            .drop('HAS_CHARTEVENTS_DATA', 1)
data_path = os.path.abspath(os.path.dirname("C:/Users/namra/Desktop/Study Material/Master Of Science (M.S.) Computer Science/Fall 2016/TIM 209 Statistical learning, data science and business analytics/Project/mimic3/demo/"))
#admission_train = pd.read_csv("C:/Users/namra/Desktop/Study Material/Master Of Science (M.S.) Computer Science/Fall 2016/TIM 209 Statistical learning, data science and business analytics/Project/mimic3/demo/ADMISSIONS.csv")
admission_df=get_admission()
#vitals_labs = pd.merge(vitals_train,labs_train, left_index=True,right_index=True, how='outer')
vitals_labs_icu = admission_df[admission_df['HOSPITAL_EXPIRE_FLAG'] == 1 ]
vitals_labs_icu = vitals_labs_icu[vitals_labs_icu['DIAGNOSIS'] == 'SEPSIS']
print(vitals_labs_icu)

hospital_ids = vitals_labs_icu['HADM_ID']
print(hospital_ids)
