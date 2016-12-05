import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
from itertools import groupby
#matplotlib.style.use('ggplot')
#matplotlib inline
from sklearn.linear_model.logistic import LogisticRegression
def get_admission():
        admission_df = pd.read_csv("/home/madhu/Desktop/TIM project/mimic3/demo/ADMISSIONS.csv", sep=',',header=0)
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
        chartevents_df = pd.read_csv("/home/madhu/Desktop/TIM project/mimic3/demo/CHARTEVENTS.csv", sep=',',header=0)
        return chartevents_df \
            .drop('ROW_ID', 1) \
            .drop('SUBJECT_ID', 1) \
            .drop('ICUSTAY_ID', 1) \
            .drop('CHARTTIME', 1) \
            .drop('STORETIME', 1) \
            .drop('CGID', 1) \
            .drop('VALUE', 1) \
            .drop('VALUEUOM', 1) \
            .drop('WARNING', 1) \
            .drop('ERROR', 1) \
            .drop('RESULTSTATUS', 1) \
            .drop('STOPPED', 1)

admission_df=get_admission()
chartevents_df=get_chartevents()
#vitals_labs = pd.merge(vitals_train,labs_train, left_index=True,right_index=True, how='outer')
vitals_labs_icu = admission_df[admission_df['HOSPITAL_EXPIRE_FLAG'] == 1 ]
vitals_labs_icu = vitals_labs_icu[vitals_labs_icu['DIAGNOSIS'] == 'SEPSIS']
print vitals_labs_icu

hospital_ids = vitals_labs_icu['HADM_ID']
print hospital_ids

chartevents_df=chartevents_df[chartevents_df['HADM_ID'].isin(hospital_ids)]

print chartevents_df

item_list=[]
for i in chartevents_df['ITEMID']:
    item_list.append(i)
value_list=[]
for i in chartevents_df['VALUENUM']:
    value_list.append(i)


item_values=[[],[[]]]
i_found = 0
for i in range(0,len(item_list)-1):
    item_values_length=len(item_values[0])
    print item_values_length
    for j in range(0,item_values_length-1):
        if item_list[i]==item_values[0][j]:
            item_values[1][j].append(value_list[i])
            i_found = 1
            break
    if i_found == 0:
        item_values[0].append(item_list[i])
        print item_values_length
        item_values[1][item_values_length].append(value_list[i])
        print "Item values after iteration:",i," is ",item_values

mean_list=[]
for i in range(0,len(item_values[0])-1):
    valuenum_list=np.array(item_values[1][i])
    mean_list.append(valuenum_list.mean)
print mean_list
