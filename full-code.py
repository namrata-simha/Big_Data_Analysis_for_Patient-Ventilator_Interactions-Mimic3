import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import matplotlib
from itertools import groupby
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


def create_datatable():
    allitems_df = pd.read_csv("mimic3/demo/CHARTEVENTS.csv", sep=',', header=0)
    return allitems_df \
        .drop('ROW_ID', 1) \
        .drop('VALUENUM', 1) \
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


def item_value_series(hadmid, chareventsdf, admissiondf):
    hospital_ids = hadmid
    chartevents_df = chareventsdf
    # print(hospital_ids)

    chartevents_df = chartevents_df[chartevents_df['HADM_ID'] == hospital_ids]

    # print(chartevents_df)
    item_list = []
    for i in chartevents_df['ITEMID']:
        item_list.append(i)
    value_list = []
    for i in chartevents_df['VALUENUM']:
        value_list.append(i)

    # print("Value_list",value_list)

    ##find unique list of itemids for length
    item_list_unique = []
    item_list_unique = set(item_list)
    n = len(item_list_unique)
    # print ("Unique values count:", n)

    item_values = [[], [[] for i in range(n)]]
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
                # print("Length: ",item_values_length)
                item_values[1][item_values_length].append(value_list[i])
                # print("Item values after iteration:", i, " is ", item_values)

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
        # print("Valuenum_list:",i,":",valuenum_list)
        # masked_valuenum = ma.masked_equal(valuenum_list, 'nan')
        mean_list.append(np.mean(valuenum_list))
        std_dev_list.append(np.std(valuenum_list))

    # print("Mean list: ",mean_list)
    # print("Std Deviation list: ",std_dev_list)

    ##for logit, SVM and random forests
    final_items_list = []

    for i in range(0, len(item_values[0])):
        temp = []
        temp.append(item_values[0][i])
        temp.append(mean_list[i])
        temp.append(std_dev_list[i])
        final_items_list.append(temp)

    # print final_items_list
    return final_items_list


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
allitems_df = create_datatable()
allitems_df = np.array(allitems_df)
allitems_df = np.sort(allitems_df, axis=None)
# print data
x = ([list(j) for i, j in groupby(allitems_df)])
numcols = []
newlist = []
for i in range(len(x)):
    newlist.append(x[i][0])
allitems_df = pd.DataFrame(columns=newlist)

admission_df = get_admission()
chartevents_df = get_chartevents()

hospital_ids = admission_df['HADM_ID']
for i in range(len(hospital_ids)):
    itemslist = item_value_series(hospital_ids[i], chartevents_df, admission_df)
    # print itemslist
    for j in range(len(itemslist)):
        val = itemslist[j][0]
        allitems_df.set_value(hospital_ids[i], val, itemslist[j][1])

#print (allitems_df)

# print len(allitems_df)
# hospital_expire_flag=hospital_expire_flag
# print hospital_expire_flag

hospital_expire_flag = (admission_df['HOSPITAL_EXPIRE_FLAG'])[0:len(allitems_df)]
#print(hospital_expire_flag)

##Take all Hospital_expire_flag values and append it to the dataframe
series = pd.Series(hospital_expire_flag)
allitems_df['HOSPITAL_EXPIRE_FLAG'] = series.values

print(allitems_df)

# allitems_df=allitems_df.dropna(axis=1,how='all')
# allitems_df=allitems_df.dropna(thresh=100)
# print allitems_df

"""allitems_length = len(allitems_df)
allitems_df_training, allitems_df_test = allitems_df[:int(0.8*allitems_length), :], allitems_df[int(allitems_length*0.8):, :] if len(allitems_df) > 10 else allitems_df, None
"""
msk = np.random.rand(len(allitems_df)) < 0.8

allitems_df_training = allitems_df[msk]

allitems_df_test = allitems_df[~msk]

print("Training length",len(allitems_df_training))
print("Test length: ",len(allitems_df_test))