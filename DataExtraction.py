import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.style.use('ggplot')
#matplotlib inline
from sklearn.linear_model.logistic import LogisticRegression
def get_admission():
        admission_df = pd.read_csv("mimic3/demo/ADMISSIONS.csv", sep=',',header=0)
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
        chartevents_df = pd.read_csv("mimic3/demo/CHARTEVENTS.csv", sep=',',header=0)
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

admission_df = get_admission()
chartevents_df = get_chartevents()
# vitals_labs = pd.merge(vitals_train,labs_train, left_index=True,right_index=True, how='outer')
vitals_labs_icu = admission_df[admission_df['HOSPITAL_EXPIRE_FLAG'] == 1]
vitals_labs_icu = vitals_labs_icu[vitals_labs_icu['DIAGNOSIS'] == 'SEPSIS']
#print(vitals_labs_icu)

hospital_ids = vitals_labs_icu['HADM_ID']
#print(hospital_ids)

chartevents_df = chartevents_df[chartevents_df['HADM_ID'].isin(hospital_ids)]

#print(chartevents_df)
item_list = []
for i in chartevents_df['ITEMID']:
    item_list.append(i)
value_list = []
for i in chartevents_df['VALUENUM']:
    value_list.append(i)

#print("Value_list",value_list)

##find unique list of itemids for length
item_list_unique = []
item_list_unique = set(item_list)
n = len(item_list_unique)
#print ("Unique values count:", n)

item_values = [[], [[]for i in range(n)]]
for i in range(0, len(item_list) - 1):
    i_found = 0
    item_values_length = len(item_values[0])
    for j in range(0, item_values_length):
        if item_list[i] == item_values[0][j]:
            if not math.isnan(value_list[i]):
                item_values[1][j].append(value_list[i])
                i_found = 1
                break
    if i_found == 0:
        if not math.isnan(value_list[i]):
            item_values[0].append(item_list[i])
            #print("Length: ",item_values_length)
            item_values[1][item_values_length].append(value_list[i])
            #print("Item values after iteration:", i, " is ", item_values)

"""item_list_unique_after_picking = []
item_list_unique_after_picking = set(item_values[0])
n1 = len(item_list_unique_after_picking)
print ("Unique values count after picking:", n1, "and length of item_values[0] is:",len(item_values[0]))
"""

mean_list = []
std_dev_list = []
for i in range(0, len(item_values[0])):
    ##convert item_values[1][i] into a numpy array
    valuenum_list = np.array(item_values[1][i])
    #print("Valuenum_list:",i,":",valuenum_list)
    #masked_valuenum = ma.masked_equal(valuenum_list, 'nan')
    mean_list.append(np.mean(valuenum_list))
    std_dev_list.append(np.std(valuenum_list))

print("Mean list: ",mean_list)
print("Std Deviation list: ",std_dev_list)

##for logit, SVM and random forests
final_items_list = []

for i in range(0, len(item_values[0])):
    temp = []
    temp.append(item_values[0][i])
    temp.append(mean_list[i])
    temp.append(std_dev_list[i])
    final_items_list.append(temp)

print (final_items_list)