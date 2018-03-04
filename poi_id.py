# -*- coding: utf-8 -*-
#!/usr/bin/python
#!/usr/bin/python2.7 
from __future__ import division # need this to get decimal results for division and this must occur at the beginning of the file
# Superscrip #30 in my references: get rid of EOF errors in Spyder: https://stackoverflow.com/questions/20073639/python-syntaxerror-invalid-syntax-end
from __future__ import print_function # Required to get rid of Spyder EOF syntax errors. Must use python 3 print format
# DAND ML Final Project Aysan
# Numbers in front of links refers to the same number in the references file

import sys # Original
#print("sys version", sys.version)

import pickle # Original
sys.path.append("../tools/") # Original

# Superscript #25 in my references document to learn which program is running as __main__
import os # Good practice to always put this at the start of my code
assert os.path.basename(__file__) != '__main__.py'  # Good practice to always put this at the start of my code
#print("at start of program poi_id print __name__: {}".format(__name__)) # Returns __main__

from tester import dump_classifier_and_data # Original 
#print("after from tester import dump_classifier_and_data print __name__: {}".format(__name__)) # Returns __main__)

from sklearn.naive_bayes import GaussianNB # Original for task #4
from sklearn.cross_validation import train_test_split # Original for task #5

# More of my imports 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd # Need for standard scaling
import sys
import random # Needed to create training data
import seaborn as sns # Need for standard scaling
import warnings
warnings.filterwarnings("ignore")

# External programs required - copy to GitHub
# From feature_format import featureFormat, targetFeatureSplit # For simplicity, I copied these definitions to this code and 
# Gave credit via reference #28 to Udacity
from feature_format import featureFormat, targetFeatureSplit # Include in my list of files.
from tester import test_classifier # Include tester.py in my list of files so that Main does not crash

# imports from sklearn
from sklearn.cluster import KMeans 
from sklearn import cross_validation 
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.metrics import accuracy_score # Accuracy: no. of all data points labeled correctly divided by all data points. 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import preprocessing # Need for standard scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn import tree

from time import time

# I want a general dictionary of values that I created that I want to call, so that I can print the dictionary to see the variables and values
general_dict = {}

# Set up global features for use in functions that will change the features:
arrFeatures = []
arrnew_dataColor = []
data = []
data_dict = {}
dataColor = []
dataFinal = []
dataFinalFloat = []
dataFinalRatioPOI = []
dataRatio = []
dataRatioN = []
feature_testLen = []
features = {} # {} for an array, [] for a list
features_list =[]
features_test = []
features_train = []
labels = []
labels_test = []
labels_train = []
labelsFinal = []
my_dataset = {}
numFeatures = {}
varBegin = []


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
### Task 1: Select what features you'll use. # Original
### features_list is a list of strings, each of which is a feature name. # Original
### The first feature must be "poi". # Original
# features_list = ['poi','salary'] # You will need to use more features # Original

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file: # Original
    data_dict = pickle.load(data_file) # Original

def myDataset():
    with open("final_project_dataset.pkl", "r") as data_file: # Original
        data_dict = pickle.load(data_file) # Original
    
    # Delete the "TOTAL" outlier key & features
    del data_dict["TOTAL"]
    my_full_dataset = data_dict
    
    # Define my_dataset
    my_dataset = data_dict
    #print("in myDataset - my_dataset before isinstance: ", len(my_dataset), my_dataset)
    """ Output:
    in myDataset - my_dataset before isinstance:  145 {'METTS MARK': {'salary': 365788, 'to_messages': 807, 
    'deferral_payments': 'NaN', ...
    """

    # Remove rows if both "deferral_payments" and "bonus" are NaN
    #print ("\nlen of my_dataset before remove rows: ", len(my_dataset)) # len of my_dataset before remove rows:  145 
    #Need to remove the rows where deferral_payments and bonus are both NaN
    #print ("\nmy_dataset before conditionally removing rows:", my_dataset) # my_dataset before conditionally removing rows: {'METTS MARK': {'salary': 365788, 'to_messages': 807, 'deferral_payments': 'NaN',

    if isinstance(my_dataset,dict):
        for key, value in my_dataset.items():
            if isinstance(value, dict) or isinstance(value, list):
                for k, v in value.items():
                    if (k == 'deferral_payments' and v == 'NaN') and (k == 'bonus' and v == 'NaN'):
                        del my_dataset[key]
    #print("in myDataset - my_dataset after if delete: ", len(my_dataset), my_dataset)
    """ Output:
    in myDataset - my_dataset after if delete:  145 {'METTS MARK': {'salary': 365788, 'to_messages': 807, 
    'deferral_payments': 'NaN',
    ...
    """

    general_dict.update({'my_dataset':my_dataset, 'my_full_dataset':my_full_dataset})
    
    return my_dataset
    
def defineFeatures():
    global features_list
    #print("\nAt start of defineFeatures - features_list: ", len(features_list), features_list) # At start of defineFeatures - features_list:  0 []
    #print("\nAt start of defineFeatures - features: ", len(features), features) # At start of defineFeatures - features:  0 []
    #print("\nAt start of defineFeatures - data_dict: ", len(data_dict), data_dict) # At start of defineFeatures - data_dict:  146 {'METTS MARK': {'salary': 365788, 'to_messages': 807, 'deferral_payments': 'NaN', ...
    features_list = ['poi','deferral_payments','bonus'] 
    general_dict.update({'features_list':features_list})
    #print("\nIn defineFeatures - features_list defined: ", len(features_list), features_list) # In defineFeatures - features_list defined:  3 ['poi', 'deferral_payments', 'bonus']
    # How many features are there?
    for key, value in data_dict.iteritems():
        numFeatures[key] = len(value)
    #print("\nIn defineFeatures after for loop - features_list: ", len(features_list), features_list) # In defineFeatures after for loop - features_list:  3 ['poi', 'deferral_payments', 'bonus']
    #print("\nIn defineFeatures after for loop - features: ", len(features), features) # In defineFeatures after for loop - features:  146 {'METTS MARK': 21, 'BAXTER JOHN C': 21, 'ELLIOTT STEVEN': 21, 'CORDES WILLIAM R': 21, ...
    lenNumFeatures = len(value)
    #print("\nIn defineFeatures after for loop - Number of Features: ", lenFeatures) # In defineFeatures after for loop - Number of Features:  21. There are 21 features and each person has 21 features.
    general_dict.update({'lenNumFeatures': lenNumFeatures})
    return features_list, numFeatures, lenNumFeatures, general_dict

def formatSplit(my_dataset):
    #my_dataset = data_dict # Original from Task 3
    #print("\n\nlen and my_dataset start of formatSplit: ", len(my_dataset), my_dataset) # len and my_dataset start of formatSplit:  145 {'METTS MARK': {'salary': 365788, 'to_messages': 807, 'deferral_payments': 'NaN', ...
    global features
    global labels
    data = featureFormat(my_dataset, features_list, sort_keys = True) # Original
    #print("data in formatSplit: ", len(data), data) # 98 [[       0.  2869717.  4175000.] [       0.   178980.        0.] [       0.  1295738.  1200000.] ..., 
    labels, features = targetFeatureSplit(data) # Original 
    #print("\nlen and features in formatSplit: ", len(features), features) # 98 [array([ 2869717.,  4175000.]), array([ 178980.,       ...
    #print("\nlen and labels in formatSplit: ", len(labels), labels) # len and labels in formatSplit:  98 [0.0, 0.0, 0.0, 0.0
    return my_dataset, data, labels, features


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
### Task 2: Remove outliers # Original

def removeOutlierTotal(data):
    # Explore the data first
    #print("\nlen data start removeOutlierTotal def: ", len(data)) # 146
    # Delete the "TOTAL" key & features
    del data["TOTAL"]
    #print("len and data after del TOTAL: ", len(data), data) # len and data after del TOTAL:  145 {'METTS MARK': {'salary': 365788, 'to_messages': 807, 'deferral_payments': 'NaN',  ...
    #print("\nlen data within removeOutlierTotal def: ", len(data)) # 145
    return data

#print("\nlen data_dict before Task 3 - after loading data_dict: ", len(data_dict)) # 146 before running removeOUtlierTotal)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
#my_dataset = data_dict # Original - this is copied to my def myDataset():
### Extract features and labels from dataset for local testing

def copyDataNewFeaturesRemoveOutlierEtc():
    global arrnew_dataColor
    global arrFeatures
    global dataFinal
    global dataFinalFloat
    # Make diff types of copies of data for graphing below, Add new features and Remove Outlier
    # Data no longer arrays within arrays
    #print("\nin copyDataNewFeaturesRemoveOutlierEtc len and features: ", len(features), features) # in copyDataNewFeaturesRemoveOutlierEtc len and features:  98 [array([ 2869717.,  4175000.]), array([ 178980.,       0.]),...
    new_features = np.vstack(features)
    #print("new_features: ", new_features) # [[2869717.0, 4175000.0], [178980.0, 0.0], and so on. no longer arrays within arrays
    
    # The new_features dataset shows the tuples with commas between all of the values:
    # I need the array without the commas:
    # Superscrip 5 in references help from here https://stackoverflow.com/questions/27516849/how-to-convert-list-of-numpy-arrays-into-single-numpy-array
    arrFeatures = np.array(features)
    #print("\narrFeatures: ", arrFeatures) # They are no longer tuples [[ 2869717.  4175000.] and so on. That's better

    ################################# Add a float feature for "ratio of deferral_payments to bonus" feature 
    # Superscript 14 in my references document: Start by creating an array of zeros found here: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.zeros.html#numpy.zeros
    ratio_dp_b = np.zeros((len(arrFeatures), 1))
    #print("\nlen and ratio_dp_b: ", len(ratio_dp_b), ratio_dp_b) # len and ratio_dp_b:  98 [[ 0.] [ 0.] [ 0.] ...,  - Good. This is what I want.
    
    dataRatio = np.c_[ arrFeatures, ratio_dp_b ] # Add the ratio_dp_b column to arrFeatures and create a new dataset called dataRatio
    #print("\nlen and dataRatio: ", len(dataRatio), dataRatio) #  len and dataRatio:  98 [[ 2869717.  4175000.        0.] [  178980.        0.        0.]... This is what I want. The ration is item[2] - the third one

    # Convert the new dataset to floats
    dataRatioN = dataRatio.astype(np.float) 
    #print("\nlen and dataRatioN before for loop: ", len(dataRatioN), dataRatioN) # Yep Floats. len and dataRatioN before for loop:  98 [[ 2869717.  4175000.        0.] [  178980.        0.        0.] ...

    # Calculate the ratio and update the new dataRatioN dataset with the values
    for item in dataRatioN:
        if item[1] <> 0.:
            item[2]=item[0] / item[1] # Item[0] is deferral_payments, item[1] is bonus
        #print("\nitem[2] after divide in the loop: ", item[2]) # Looks good
    #print("\nlen and dataRatioN after for loop: ", len(dataRatioN), dataRatioN) # Yes, the ratio is there. len and dataRatioN after for loop:  98 [[  2.86971700e+06   4.17500000e+06   6.87357365e-01] [  1.78980000e+05  ...

    #Convert the file back to strings so that I can add a character feature for color
    dataNoColor = dataRatioN.astype(np.str)
    #print("\nlen and arrFeatures after conversion back to char: ", len(arrFeatures), arrFeatures) # Yes, that's good  len and arrFeatures after conversion back to char:  98 [[ 2869717.  4175000.] [  178980.        0.] ...

    ################################# Add a character feature for color
    # Superscript 8 found here: https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-an-numpy-array
    charar = np.chararray((len(dataNoColor),1))
    charar[:] = 'a' 
    #print("\nlen(charar), len(arrFeatures), len(labels), charar: ", len(charar), len(arrFeatures), len(labels), charar) # 98 Good... the datasets are the same length len(charar), len(arrFeatures), len(labels), charar:  98 98 98 [['a'] ['a'] ...
    
    # Add the labels and charar column to data and create a new dataset
    dataColor = np.c_[ labels,dataNoColor,charar ]  
    #print("\nlen and dataColor: ", len(dataColor), dataColor) # The labels and char are there. len and dataColor:  98 [['0.0' '2869717.0' '4175000.0' '0.687357365269' 'a'] ['0.0' ...

    # Update the new character feature with values for red and green color
    # Superscript 9 in reference file: https://stackoverflow.com/questions/45709743/how-to-replace-a-value-in-array-in-numpy-python
    for item in dataColor:
        if item[0]=="1.0": # I see in the values that 'poi' is either True or False. True == 1
            item[4]='r' # Red for POIs
        else:
            item[4]='g' # Green for non-POIs
         
    #print("\nlen and dataColor after for loop: ", len(dataColor), dataColor) # Good the color column has changed. Only shows some records at the top and at the bottom and they just all happen to be "g" because there are more non-POIs
    # len and dataColor after for loop:  98 [['0.0' '2869717.0' '4175000.0' '0.687357365269' 'g'] ['0.0' '178980.0' '0.0' '0.0' 'g'] ...
    
    # Convert the dataColor dataset to tupple dataset
    new_dataColor = np.array(dataColor)

    # Convert the new_dataColor to numpy array
    arrnew_dataColor = np.array(new_dataColor)
    #print("\nlen and arrnew_dataColor: ", len(arrnew_dataColor), arrnew_dataColor) # They are no longer tuples. len and arrnew_dataColor:  98 [['0.0' '2869717.0' '4175000.0' '0.687357365269' 'g'] ['0.0' '178980.0' '0.0' '0.0' 'g'] ...
    
    # See if my fields have names:
    # Superscript 12 in my references document from here: https://stackoverflow.com/questions/7561017/get-the-column-names-of-a-python-numpy-ndarray
    #print("\nlen and arrnew_dataColor.dtype.names: ", len(arrnew_dataColor), arrnew_dataColor.dtype.names) # My data has no field names, so I could not use the field names in the legend. len and arrnew_dataColor.dtype.names:  98 None

    """ First review response by reviewer:
    "The report mentioned that "LAY KENNETH L was removed. But because of the low number of POIs in the dataset, an outlier like "LAY KENNETH L should be 
    retained. The only exception to this is if it's removed from the training set."
    So, this code is not required. I already removed the TOTAL outlier, and I'm removing 'THE TRAVEL AGENCY IN THE PARK' in task 6.
    ########################################### Remove the outlier 
    # Find the outlier:
    # Superscript 16 in my list of references. https://stackoverflow.com/questions/18079029/index-of-element-in-numpy-array
    outIndexRow, outIndexCol = np.where(dataRatioN > 3500000)
    #print("\noutIndexRow, outIndexCol: ", outIndexRow, outIndexCol) # The zero in outIndexCol refers back to the 36 in outIndexRow. outIndexRow, outIndexCol:  [ 0  6 36 58 59 85] [1 1 0 1 1 1]
    # 36 means that it is the 37th row, because the rows start at number 0
    
    # Superscript 15 in my list of references. Figure out which row is the outlier to use later: https://stackoverflow.com/questions/27350048/python-return-index-of-an-element-of-an-array-given-specific-conditions
    outIndexRowCol0 = (next(i for i,v in enumerate(outIndexCol) if v == 0))
    outReferenceRow = outIndexRow[outIndexRowCol0]
    #print("Outlier Reference Row ", outReferenceRow) # it's row [36] which is the 37th row. Outlier Reference Row  36
    
    outReferenceRowData = dataRatioN[outIndexRow[outIndexRowCol0]] # Keep this, it is referred to in the next print stmt which is commented out
    #print("Outlier Reference Row data to check: ", outReferenceRowData) # Yes, that's the row with the oulier of deferral payment. Outlier Reference Row data to check:  [  6.42699000e+06   2.00000000e+06   3.21349500e+00]
    
    # Subscript 17 in my reference document: https://stackoverflow.com/questions/3877491/deleting-rows-in-numpy-array
    # The outlier is on row outReferenceRow = 36 - but the reference could change, so I'll use the variable name
    #print("\nlen and dataFinal row outReferenceRow before delete: ", len(arrnew_dataColor), arrnew_dataColor[outReferenceRow,]) # Yes, that's the right row and the number of records is 98
    # from print stmt above: len and dataFinal row outReferenceRow before delete:  98 ['0.0' '6426990.0' '2000000.0' '3.213495' 'g']    
    #print("dataFinal row after the outlier before delete: ", len(arrnew_dataColor), arrnew_dataColor[37,]) # dataFinal row after the outlier before delete:  98 ['0.0' '50591.0' '0.0' '0.0' 'g']
    
    # Remove the outlier and create my dataFinal dataset
    dataFinal = np.delete(arrnew_dataColor, (outReferenceRow), axis=0) # Remove the outlier based on the row reference #
    
    # I know now that I could have simplified the remove outlier by just putting: del data_dict["LAY, KENNETH L"]
    
    #print("\n len and dataFinal: ", len(dataFinal), dataFinal) # len and dataFinal:  97 [['0.0' '2869717.0' '4175000.0' '0.687357365269' 'g'] ['0.0' '178980.0' '0.0' '0.0' 'g'] ...
    """
    dataFinal = arrnew_dataColor # added this after review since I am no longer creating dataFinal in the previous commented out section
    # Remove the string field (color) and convert dataFinal to float:
    dataFinalFloat = dataFinal[:, :4]
    dataFinalFloat = dataFinalFloat.astype(np.float) 
    #print("\nin copyDataNewFeaturesRemoveOutlierEtc len and dataFinalFloat: ", len(dataFinalFloat), dataFinalFloat) # in copyDataNewFeaturesRemoveOutlierEtc len and dataFinalFloat:  98 [[  0.00000000e+00   2.86971700e+06   4.17500000e+06   ...
    
    return new_features, arrFeatures, arrnew_dataColor, dataRatioN, dataNoColor, dataFinal, dataFinalFloat


def defineVariables1():
 
    # I have already created the nomenclature of the variable names.
    # Superscript 23 in my reference document on adding to dictionaries: https://stackoverflow.com/questions/1024847/add-new-keys-to-a-dictionary
    
    # How many people are there?
    totalPeople = len(data_dict) # 145 now that I removed TOTAL
    general_dict.update({'totalPeople': totalPeople})
    #print("\ngeneral_dict top defineVariables1:, ", general_dict) # general_dict top defineVariables1:,  {'lenNumFeatures': 21, 'totalPeople': 145}
    #print("general_dict totalPeople: ", general_dict['totalPeople']) # general_dict totalPeople:  145
    
    # How many in the dataset 
    totalPOIs = 0
    countDeferralPayments = 0
    countLoanAdvances = 0
    countBonus = 0
    for key in data_dict.keys():
        if (data_dict[key]["poi"]==1): # I see in the values that 'poi' is either True or False. True == 1
            totalPOIs += 1
        if (data_dict[key]["deferral_payments"]<> 'NaN'): 
            countDeferralPayments += 1
        if (data_dict[key]["loan_advances"]<> 'NaN'): 
            countLoanAdvances += 1
        if (data_dict[key]["bonus"]<> 'NaN'): 
            countBonus += 1
            #print("\nkey: ", key)
    general_dict.update({'totalPOIs': totalPOIs, 'countDeferralPayments':countDeferralPayments, 'countLoanAdvances':countLoanAdvances, 'countBonus':countBonus})

    # Set this to True to see all the Keys, features, values.
    if False:
        for key, value in data_dict.items():
            print("\nKEY: ", key) # The peoples' names in caps are the keys. The others are the values
            
            print("VALUE: ", value) # I may refer back to these values in my analysis
    
    totalRatioPOIs = totalPOIs / totalPeople
    general_dict.update({'totalRatioPOIs':totalRatioPOIs})
    #print("\ngeneral_dict middle of defineVariables1:, ", general_dict) # general_dict middle of defineVariables1:,  {'lenNumFeatures': 21, 'totalRatioPOIs': 0.12413793103448276, 'totalPOIs': 18, 'totalPeople': 145, 'countBonus': 81, 'countDeferralPayments': 38, 'countLoanAdvances': 3}
    
    # How many POIs are in the names file:
    poi_qty_all_names = 0
    with open("poi_names.txt") as poiNamesFile:
        #print("\npoiNamesFile: ", poiNamesFile)
        
        poiNamesFile.next() #skip url line
        poiNamesFile.next() #skip blank line
        for line in poiNamesFile:
    		poi_qty_all_names +=1
    
    # Superscript #24 in my references document for len of file: https://stackoverflow.com/questions/38622975/typeerror-object-of-type-file-has-no-len
    readNames = open("poi_names.txt").read()
    readNamesLen = len(readNames)-2 # -2 for the url line and blank line at the top
    readNamesPOIRatio = poi_qty_all_names / readNamesLen
    
    # Superscript 4: From https://classroom.udacity.com/nanodegrees/nd002/parts/0021345409/modules/317428862475460/lessons/2254358555/concepts/24196685390923
    general_dict.update({'poi_qty_all_names':poi_qty_all_names, 'readNamesLen':readNamesLen, 'readNamesPOIRatio':readNamesPOIRatio})
    
    # How many people have quantified deferral_payments and loan_advances:
    #print("\ngeneral_dict bottom of defineVariables1:, ", general_dict) # general_dict bottom of defineVariables1:,  {'lenNumFeatures': 21, 'readNamesPOIRatio': 0.04691689008042895, 'totalRatioPOIs': 0.12413793103448276, 'totalPOIs': 18, 'poi_qty_all_names': 35, 'totalPeople': 145, 'countBonus': 81, 'readNamesLen': 746, 'countDeferralPayments': 38, 'countLoanAdvances': 3}

    return general_dict

def reportFeaturesOutliersEtc():
    print ("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SELECT WHAT FEATURES I'LL USE AND REMOVE THE OUTLIER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("\n################################################## Stats on the Basic Data #################################################")
    print("\nLength of the data_dict file after removing the 'Total' record: ", general_dict['totalPeople']) # There are 146 people - 145 now that I've deleted TOTAL
    print("\nNumber of Features: ", general_dict['lenNumFeatures']) # There are 21 features and each person has 21 features. 
    #print("\nList of persons & # of features each: ", features) # I copied this to a word document for my reference. I no longer need it here. I am not submitting the word document.
    print("\nNumber of POIs in the dictionary:", general_dict['totalPOIs']) # There are 18 POIs in the dictionary
    #print("\nNumber of POIs in the dictionary:", general_dict['totalPOIs']) # There are 18 POIs in the dictionary # I dont need "general_dict['totalPOIs]". It automatically picks up the variable from the general_dict.

    # Set this to True to see the output.
    if False:
        print("\ngeneral_dict: ", general_dict)
        
        print("\nNumber of people in the names txt file readNamesLen: ", general_dict['readNamesLen'])
        
        print("Number of POIs in the names txt file poi_qty_all_names: ", general_dict['poi_qty_all_names'])
        
        print("Ratio of POIs to number of names: ", general_dict['readNamesPOIRatio'])
        
    # There are 35 POIs in the names file, but only 18 POIs in the data_dict.  
    # "More data is always better--only having 18 data points doesn't give you that many examples to learn from." Copied from this link:
    # Superscript #3 in my references file: https://classroom.udacity.com/nanodegrees/nd002/parts/0021345409/modules/317428862475460/lessons/2291728537/concepts/30100186330923 
    
    print("\n################ Results for my initially chosen features: deferral_payments and loan_advances in that order ###############")
    print("\nSelect a POI and a non-POI and examine their deferral_payments and loan_advances in that order.")
    print("POI HIRKO JOSEPH: ", data_dict["HIRKO JOSEPH"]["deferral_payments"], ", ",data_dict["HIRKO JOSEPH"]["loan_advances"])
    
    print("Interesting: as a POI, Joseph Hirko had $10,259 in deferral_payments and no loan_advances.")
    print("\nnon-POI TAYLOR MITCHELL S: ", data_dict["TAYLOR MITCHELL S"]["deferral_payments"], ", ",data_dict["TAYLOR MITCHELL S"]["loan_advances"])
    
    print("As a non-POI, Mitchell Taylor had much higher deferral_payments than POI Joseph Hirko, but once again no loan_advances.")
    
    # I may want to pay particular attention to these people during my analysis.
    # Enron's CEO during most of the fraud time: Jeffrey Skilling
    # Enron's Chairman of the board: Kenneth Lay
    # Enron's CFO: Andrew Fastow
    # So, what are the values for my 2 features for each of them:
    print("\nValues of deferral_payments and loan_advances features for the Enron Executives:") 
    print("\nPOI Jeffrey K Skilling, Enron's CEO: ", data_dict["SKILLING JEFFREY K"]["deferral_payments"], ", ",data_dict["SKILLING JEFFREY K"]["loan_advances"])
    
    print("POI Kenneth L Lay, Enron's Chairman of the Board: ", data_dict["LAY KENNETH L"]["deferral_payments"], ", ",data_dict["LAY KENNETH L"]["loan_advances"])
    
    print("POI Andrew S Faston, Enron's CFO: ", data_dict["FASTOW ANDREW S"]["deferral_payments"], ", ",data_dict["FASTOW ANDREW S"]["loan_advances"])
    
    print("\nSkilling and Fastow had no deferral_payments or loan_advances, but Lay had huge deferral_payment and loan_advances:")
    print("Since Lay is a POI, this is not an outlier.")
    # POI SKILLING JEFFREY K deferral_payments and loan_advances:  NaN ,  NaN
    # POI LAY KENNETH L deferral_payments and loan_advances:  202911 ,  81525000 
    # POI FASTOW ANDREW S deferral_payments and loan_advances:  NaN ,  NaN
    # Of course NaN is not zero, it just means that it is an unfilled feature

    print("\nNumber of people with deferral_payments:", general_dict['countDeferralPayments']) # 38 now that I deleted TOTAL
    
    print("Number of people with loan_advances:", general_dict['countLoanAdvances']) # only 3 people, and one of them... hmmmm this may end up not being such a great feature to use. There is not enough data.
    
    # So I added a print statement to print the key
    # I see that "Total" is a key. I will have to remove that from the data base. I removed TOTAL. 
    # Now there are 3 people with loan_advances
    print("\nHow much loan advance did they have, and are they marked in the data_dict as a POI: ")
    
    print("\nKenneth L Lay loan_advances and POI? ", data_dict["LAY KENNETH L"]["loan_advances"], data_dict["LAY KENNETH L"]["poi"])
    
    print("Mark R Pickering loan_advances and POI? ", data_dict["PICKERING MARK R"]["loan_advances"], data_dict["PICKERING MARK R"]["poi"])
    
    # Of course Total is not a POI. 
    #print("TOTAL loan_advances and POI: ", data_dict["TOTAL"]["loan_advances"], data_dict["TOTAL"]["poi"]) # I deleted the TOTAL key & featues
    
    print("Mark A Frevert loan_advances and POI? ", data_dict["FREVERT MARK A"]["loan_advances"], data_dict["FREVERT MARK A"]["poi"])
    
    print ("\nThere are only 3 people with loan_advances, two of whom are not POIs.")
    print ("So, I will change the feature that I want to select with deferral_payments.")
    print ("Let's look at Bonus instead.")

    print("\n############################### Results for my feature 'bonus' which replaces loan_advances ##############################\n")
    
    print("Number of people with bonus:", general_dict['countBonus']) # There are 81 now that I deleted TOTAL. 
    print("That will be a good chunk of the database and big enough for it to have POIs and non-POIs.")
    
    # What ratio of people have deferral_payments and bonus non NaN
    print("People with deferral_payments as ratio of total people: ", (general_dict['countDeferralPayments']/general_dict['totalPeople'])) # Now that I've deleted TOTAL it is: 0.262068965517
    
    print("People with bonus as ratio of total people: ", (general_dict['countBonus']/general_dict['totalPeople'])) # Now that I've deleted TOTAL it is: 0.558620689655
    if False: # Set this to True to see what the data looks like
        print("\n############################################## What does the data look like ##############################################\n")
        
        print("len(data) and data: ", len(data), data) # 98 [[       0.  2869717.  4175000.] and so on... not a tuple
        
        print("A np array of labels which are the poi values: ", labels) # These are the POI values for each person [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, and so on
        
        print("Np arrays of 2 values per person for the Features: ", features) # A 2 value np array for each person showing deferral_payments and bonus [array([ 2869717.,  4175000.]), array([ 178980.,       0.]),and so on
        
        print("Number of records in labels - len(labels): ", len(labels)) # 98
        
        print("Number of records in features - len(features): ", len(features)) # 98
    
    print("\n####################### Add 'ratio of deferral_payments to bonus' and 'color' features to my dataset #######################")
    print("\nThe ratio feature is out of interest only, to see if I can use it in my analysis. I decided not to.")
    print("\nThe color feature will color code POIs and non-POIs in my scatter plots.")
    print("In the printout of the first few and last few rows of the database below, the new color feature populated with g for green, and other records, that have r for red, are in the rows that are not shown.")    
    print("\nI have also added the POI feature as the first feature.")    

    print("\nThe length of my dataset after removing total and all of the records that have NaN for both deferral_payments and bonus is: ", len(arrnew_dataColor))
    print("\nThe features are as follows. The first 3 are existing features from the original dataset. The last two are new features.")
    print("'poi', 'deferral_payments', 'bonus', 'ratio of deferral_payments to bonus' and 'color' for the graph data points.")
    print("")
    print(arrnew_dataColor) # They are tuples. len and new_dataColor:  98 [['0.0', '2869717.0', '4175000.0', '0.687357365269', 'g'], ['0.0', ...

    
    print("\n######################################## Scatter plots to help me analyse my data ########################################\n")
    print("In these graphs, the deferral_payment on the far right is indentified in the enron database as Mr. Lay who is a POI, so it is not an outlier.")
    print("However in the second graph which has color coding for POIs and non-POIs, the Enron database shows Mr. Lay as a non-POI, which is not correct in the database.")
    print("")

def graphData():
    # I want to use some graphing to see if there are any outlier(s).

    # Scatter plot of features without POI and Non-POI colors:
    x = arrFeatures[:,0]
    y = arrFeatures[:,1]
    plt.scatter(x,y, color='b')
    plt.xlabel("deferral_payments")
    plt.ylabel("bonus")
    
    # Superscript 13 in my references document: https://stackoverflow.com/questions/8598163/split-title-of-a-figure-in-matplotlib
    plt.title("Deferral Payments and Bonus - markers all one color")
    plt.show() # Checked
    
    # Scatter Plot of Deferral_payments and Bonus with colors for POI and non-POI
    c1 = arrnew_dataColor[:,4]
    #print("\nc2: ", c2) # Yes, list of colors
    
    # Superscript 11 in reference file: https://stackoverflow.com/questions/39500265/manually-add-legend-items-python-matplotlib
    x1 = arrnew_dataColor[:,1]
    y1 = arrnew_dataColor[:,2]
    red_patch = mpatches.Patch(color='red', label='POIs')
    green_patch = mpatches.Patch(color='g', label='non-POIs')
    
    plt.scatter(x1,y1, color=c1, label = "POI, non-POI")
    plt.xlabel("deferral_payments")
    plt.ylabel("bonus")
    plt.title("Deferral Payments and Bonus\n with colors for POI and non-POI")
    plt.legend(handles=[red_patch, green_patch])
    plt.show() # Checked

    """ These last 2 graphs are no longer required because the first reviewer said that Mr. Lay is NOT an outlier and should be retained.
    # And now without x outlier by using xlim
    
    plt.scatter(x1,y1, color=c1, label = "POI, non-POI")
    plt.xlabel("deferral_payments")
    plt.ylabel("bonus")
    plt.legend(handles=[red_patch, green_patch])
    plt.title("Deferral Payments and Bonus\n without Outlier using xlim")
    plt.xlim(-250000,3500000)
    
    plt.show() # Checked
    

    print("\n################################################### Removing the outlier ###################################################")
    print("\nRemove the outlier, based on it's row number, so that I can do some training and testing.")
    print("\nAfter removing the outlier and the records where both deferral_payments and bonus are both NaN, the length of the data is: ", len(dataFinal))
    #print("\nSample of dataFinal after removing outlier. POI, ratio of deferral to bonus and color are still in the data: ", dataFinal)

    # print(the scatter plot without xlim)
    
    x2 = dataFinal[:,1]
    y2 = dataFinal[:,2]
    c2 = dataFinal[:,4]
    
    plt.scatter(x2,y2, color=c2, label = "POI, non-POI")
    plt.xlabel("deferral_payments")
    plt.ylabel("bonus")
    plt.legend(handles=[red_patch, green_patch])
    plt.title("Deferral Payments and Bonus\n after removing Outlier from the database")
    
    plt.show() # Checked
    """
    
    # It's interesting to note that most of the POIs do not have deferred payments
    
    # Draw a boundary line from trial and error on the slope
    # Subscript 20 in my reference document. Use of ax and how to set ticks:  https://stackoverflow.com/questions/20335290/matplotlib-plot-set-x-ticks
    x2 = dataFinal[:,1]
    y2 = dataFinal[:,2]
    c2 = dataFinal[:,4]

    fig, ax = plt.subplots()
    plt.scatter(x2,y2, color=c2, label = "POI, non-POI")
    plt.xlabel("deferral_payments")
    plt.ylabel("bonus")
    plt.legend(handles=[red_patch, green_patch])
    plt.title("Deferral_payments and Bonus\n with estimated line dividing POI and non-POI")
    
    # Superscript 18 in my references document: https://stackoverflow.com/questions/43152529/python-matplotlib-adding-regression-line-to-a-plot
    f = lambda x: 1.7*x - 0 # I played around with the slope of this line until I got something that I liked 
    # x values of line to plot
    x = np.array([0,3500000]) # I wanted the maximum to be the max of bonus
    # Superscript 19 in my references document: adding dashes and line style: https://stackoverflow.com/questions/35099130/change-spacing-of-dashes-in-dashed-line-in-matplotlib
    plt.plot(x,f(x),lw=1.5, c="b", linestyle = '--', dashes= [5,5])
    
    ax.set_xticks(ax.get_xticks()[::2]) # Set every 2nd tick so that they dont run together
    
    plt.show() # Checked
    
    print("\nIt is interesting to note by looking at the plot with the estimated line dividing POIs and non-POIs,")
    print("that most persons identified as POIs in the Enron database did not take deferred_payments, but took bonuses.")
    print("Perhaps they knew something that the rest of the non-POIs did not know. Perhaps they knew that the deferred payments would never materialize.")
    print("Of course this is just conjecture which cannot be proven since we cannot get into the minds of the POIs.")
    print("That type of comment would not be put into a formal Data Analytics report.")
    print("")

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
### Task 4: Try a varity of classifiers # Original
### Please name your classifier clf for easy export below. # Original
### Note that if you want to do PCA or other multi-stage operations, # Original
### you'll need to use Pipelines. For more info: # Original
### http://scikit-learn.org/stable/modules/pipeline.html # Original

def defineVariables2():
    global labelsFinal
    # Superscript 21 in my reference document: extracting fields: https://stackoverflow.com/questions/4455076/how-to-access-the-ith-column-of-a-numpy-multidimensional-array
    labelsFinal = dataFinal[:,[0]] # First definition of labelsFinal
    #print("\nUnder #4: len(labelsFinal), labelsFinal: ", len(labelsFinal), labelsFinal) # Under #4: len(labelsFinal), labelsFinal:  98 [['0.0'] ['0.0'] ...
    
    featuresFinal12 = dataFinal[:,[1,2]] ##### these are my final features that I had before testing my new feature
    featuresFinal13 = dataFinal[:,[1,3]] ##### these are my features using deferral_payments and ratio

    general_dict.update({'labelsFinal':labelsFinal, 'featuresFinal12':featuresFinal12, 'featuresFinal13':featuresFinal13})

    return general_dict, labelsFinal, featuresFinal12, featuresFinal13

def defineVariables3():
    global dataFinalRatioPOI
    # Some "total" variables that I will need to compare classifiers
    #print("in defineVariables3 labelsFinal: ", len(labelsFinal), labelsFinal) # in defineVariables3 labelsFinal:  98 [['0.0'] ['0.0'] ...

    labelsFinalN = general_dict['labelsFinal'].astype(np.float)
    dataFinalLen = len(labelsFinalN)
    dataFinalSumPOI = sum(labelsFinalN)[0]
    dataFinalRatioPOI = dataFinalSumPOI / dataFinalLen
    
    general_dict.update({'labelsFinalN':labelsFinalN, 'dataFinalLen':dataFinalLen, 'dataFinalSumPOI':dataFinalSumPOI, 'dataFinalRatioPOI':dataFinalRatioPOI})
    #print("\nin defineVariables3 general_dict: ", general_dict) # in defineVariables3 general_dict:  {'readNamesPOIRatio': 0.04691689008042895, 'totalRatioPOIs': 0.12413793103448276, 'poi_qty_all_names': 35, 'dataFinalLen': 97, 'countDeferralPayments': 38, 'dataFinalRatioPOI': 0.17525773195876287, 'dataFinalSumPOI': 17.0, 'countLoanAdvances': 3, 'lenNumFeatures': 21, 'featuresFinal13': array([['2869717.0', '0.687357365269'], ...
    
    #general_dict['mynewkey'] = 'mynewvalue' Another way to add a key / value to my dictionary...
    
    #print("\nfeatures_train: ", features_train, len(features_train)) # [[0.6394267984578837, 0.529114345099137], [0.025010755222666936, 0.9710783776136181], ... Train is 72 length
    testClassifiers = [GaussianNB(), 
                       tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=42, splitter='best'), 
            RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False),     
            KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                   n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
                   random_state=42, tol=0.0001, verbose=0), 
            SVC(kernel="linear") ]
            # To use in a loop when I am testing various classifiers, rather than repeating code.
            # Note that I know that KMeans is not a classifier. It is in this variable to store the features.

    testClassifiersAbbr = ['NB', 'Tree', 'Forest', 'KMeans', 'SVM'] # To use to create variable names. Note that I know that KMeans is not a classifier. It is stored here for convenience.
    varBegin = ["timeToFit", "timeToPredict", "clfPredictSumPOI", "clfPredictRatioPOI", "ratioPredictedPOIToWhole", "accuracy", 
                "score", "clfPredict", "truePositives", "falsePositives","trueNegatives", "falseNegatives", 
                "totalAllTrueFalsePosNeg", "totalTest", "equalTest", "pBase", "pBinary", "pMacro", "pMicro", "pWeighted",
                "rBase", "rBinary", "rMacro", "rMicro", "rWeighted", "f1Base", "f1Binary", "f1Macro", "f1Micro", "f1Weighted",
                ]
    varEnd = ['NBBasic', 'TreeBasic', 'ForestBasic', 'KMeansBasic', 'NBComplex', 'TreeComplex', 'ForestComplex', 'KMeansComplex', 'SVMComplex'] # Note that I know that KMeans is not a classifier. It is stored here for convenience.

    myCode = ["Basic", "Complex"]
    general_dict.update({'testClassifiers':testClassifiers, 
                         'testClassifiersAbbr':testClassifiersAbbr,
                         'varBegin':varBegin,
                         'varEnd':varEnd,
                         'myCode':myCode})
    #print("\ngeneral_dict testClassifiers: ", general_dict['testClassifiers', general_dict['varBegin']])
    
    return varBegin, general_dict

def myCodeFitPredict(Type, elemStart, elemEnd): # Where Type is "Basic" or "Complex" and elemStart / elemEnd allows me to exclude some classifiers if I want to
    feature_testLen = len(features_test)
    #print("\nin myCodeFitPredict elemStart, elemEnd: ", elemStart, elemEnd, general_dict['testClassifiers'][elemStart:elemEnd], general_dict['testClassifiersAbbr'][elemStart:elemEnd])
    for elemA, elemB in zip((general_dict['testClassifiers'][elemStart:elemEnd]), (general_dict['testClassifiersAbbr'][elemStart:elemEnd])): 
        #print("\nin myCodeFitPredict elemA, elemB, Type, varBegin[0]: ", elemA, elemB, Type, general_dict['varBegin'][0]) # in myCodeFitPredict elemA, elemB, Type, varBegin[0]:  GaussianNB(priors=None) NB Basic timeToFit ... same idea for Tree
        clf = elemA
        t0 = time()
        clf.fit(features_train, labels_train)
        timeToFit = round(time()-t0, 4)
        #print("\nTraining time to fit ", i, " classifier:", timeToFit, "s")
        # Superscript #26 in my references document: https://stackoverflow.com/questions/40801789/using-return-value-inside-another-function
        varFullFitTimeS = timeToFit
        varFullFitTime = general_dict['varBegin'][0]+elemB+Type
        t1 = time()
        clfPredict = clf.predict(features_test)
        #print("\nin myCodeFitPredict clfPredict: ", clfPredict) # Needs to be float for calculation below
        varFullPredictTimeS = round(time()-t1, 4)
        clfPredictSumPOI = sum(clfPredict.astype(np.float))
        #print("\nin myCodeFitPredict clfPredictSumPOI: ", clfPredictSumPOI) # OK now sums because it is a float
        #print("\nin myCodeFitPredict feature_testLen and dataFinalRatioPOI: ", feature_testLen, dataFinalRatioPOI) # in myCodeFitPredict feature_testLen and dataFinalRatioPOI:  30 0.175257731959
        clfPredictRatioPOI = clfPredictSumPOI / feature_testLen
        ratioPredictedPOIToWhole = clfPredictRatioPOI / dataFinalRatioPOI
        acc = accuracy_score(clfPredict, labels_test)
        clfScore = clf.score(features_test, labels_test) 
        
        varFullPredictTime = general_dict['varBegin'][1]+elemB+Type
        varFullClfPredict = general_dict['varBegin'][7]+elemB+Type
        varFullPredictSumPOI = general_dict['varBegin'][2]+elemB+Type
        varFullPredictRatioPOI = general_dict['varBegin'][3]+elemB+Type
        varFullPredictedPOIToWhole = general_dict['varBegin'][4]+elemB+Type
        varFullAccuracy = general_dict['varBegin'][5]+elemB+Type
        varFullScore = general_dict['varBegin'][6]+elemB+Type

        general_dict.update({varFullFitTime:varFullFitTimeS, 
                             varFullPredictTime:varFullPredictTimeS, 
                             varFullClfPredict:clfPredict,
                             varFullPredictSumPOI:clfPredictSumPOI,
                             varFullPredictRatioPOI:clfPredictRatioPOI,
                             varFullPredictedPOIToWhole:ratioPredictedPOIToWhole,
                             varFullAccuracy:acc,
                             varFullScore:clfScore})
        #print("\nin myCodeFitPredict - elemA, elemB: ", elemA, elemB) # in myCodeFitPredict - elemA, elemB, general_dict:  DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, ...
        #print("\nin myCodeFitPredict general_dict: ", general_dict) # Good. e.g. 'clfPredictNBBasic', 'scoreNBBasic': 0.73333333333333328 'scoreTreeBasic': 1.0

    return general_dict

def basicTrainTestData(data, labels, testPct, randomSeed, svcKernel, svcC):
    global features_train
    global features_test
    global labels_train
    global labels_test
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(data, labels, test_size = testPct, random_state = randomSeed)
    general_dict.update({"features_train_basic":features_train, 
                         "features_test_basic":features_test, 
                         "labels_train_basic":labels_train, 
                         "labels_test_basic":labels_test}) 
    #print("\nin basicTrainTestData at top general_dict: ", general_dict) # e.g. 'features_test_basic': array([[  0.00000000e+00,   0.00000000e+00,   9.00000000e+05, ...

    #print("in basicTrainTestData, len and features_train, features_test, labels_train, labels_test: ", len(features_train), len(features_test), len(labels_train), len(labels_test), features_train, features_test, labels_train, labels_test)
    """ Output, with these parameters basicTrainTestData(dataFinalFloat, labelsFinal, 0.3, 42, "linear", 1.), is:
    in copyDataNewFeaturesRemoveOutlierEtc len and dataFinalFloat:  97 [[  0.00000000e+00   2.86971700e+06   4.17500000e+06   6.87357365e-01]
    [  0.00000000e+00   1.78980000e+05   0.00000000e+00   0.00000000e+00]
    ...
    """
    #clf = SVC(kernel='linear', C=1.) # Original from course
    #clf = SVC(kernel=svcKernel, C=svcC) # This one took too long to run
    # So I decided to eliminate the SVC from my basicTrainTestData definition but I will use it in my complexTrainTestData definition
    myCodeFitPredict(general_dict['myCode'][0], 0, -1) # -1 eliminates the SVC from the list of classifiers that I loop through

    return general_dict

# Superscript #27 in my reference file. https://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points
def reportBasicClf():
    print ("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BASIC CLASSIFIER RESULTS - using sklearn cross_validation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    #print("\ndeferral_payments and bonus: len(featuresFinal12), featuresFinal12: ", len(general_dict['featuresFinal12']), general_dict['featuresFinal12']) 
    # When Final features are deferral payments and bonus: 98 good  [['2869717.0' '4175000.0'] ['178980.0' '0.0'] ...

    #print("\ndeferral_payments (dp) and ratio of dp to bonus: len(featuresFinal13), featuresFinal13: ", len(general_dict['featuresFinal13']), general_dict['featuresFinal13']) 
    # When features are deferral_payments and ratio: len(featuresFinal), featuresFinal:  98 good  [['2869717.0' '0.687357365269'] ['178980.0' '0.0'] ...

    print("\n############### Table of Statistics from the Basic Classifiers (without Complex Parameters) and Some Notes ###############\n")
    
    print("Description                               Names File  Enron Data   My Data     GaussianNB  Tree        Forest   ")
    
    print("----------------------------------------- ----------  ----------  ----------  ----------  ----------  ----------")
    
    print("Number of people:                         ", general_dict['readNamesLen'], "       ", general_dict['totalPeople'], "       ", general_dict['dataFinalLen'], "        ", general_dict['dataFinalLen'], "        ", general_dict['dataFinalLen'], "        ", general_dict['dataFinalLen'])
    
    print("Number of POIs:                            ", general_dict['poi_qty_all_names'], "        ", general_dict['totalPOIs'], "       ", ("{0:.0f}".format(general_dict['dataFinalSumPOI'])), "         ", ("{0:.0f}".format(general_dict['clfPredictSumPOINBBasic'])), "         ", ("{0:.0f}".format(general_dict['clfPredictSumPOITreeBasic'])), "         ", ("{0:.0f}".format(general_dict['clfPredictSumPOIForestBasic'])))
    
    print("Ratio of POIs to number of people (note 1): ", ("{0:.3f}".format(general_dict['readNamesPOIRatio'])), "     ",("{0:.3f}".format(general_dict['totalRatioPOIs'])), "    ",("{0:.3f}".format(dataFinalRatioPOI)), "     ", ("{0:.3f}".format(general_dict['clfPredictRatioPOINBBasic'])), "     ", ("{0:.3f}".format(general_dict['clfPredictRatioPOITreeBasic'])), "     ", ("{0:.3f}".format(general_dict['clfPredictRatioPOIForestBasic'])))
    
    print("Ratio of pred POIs ratio to total ratio:      N/A         N/A        N/A       ", ("{0:.3f}".format(general_dict['ratioPredictedPOIToWholeNBBasic'])), "     ", ("{0:.3f}".format(general_dict['ratioPredictedPOIToWholeTreeBasic'])), "     ", ("{0:.3f}".format(general_dict['ratioPredictedPOIToWholeForestBasic'])))
    
    print("Accuracy (Note 2):                            N/A         N/A        N/A       ", ("{0:.3f}".format(general_dict['accuracyNBBasic'])), "     ", ("{0:.3f}".format(general_dict['accuracyTreeBasic'])), "     ", ("{0:.3f}".format(general_dict['accuracyForestBasic'])))
    
    print("Score:                                        N/A         N/A        N/A       ", ("{0:.3f}".format(general_dict['scoreNBBasic'])), "     ", ("{0:.3f}".format(general_dict['scoreTreeBasic'])), "     ", ("{0:.3f}".format(general_dict['scoreForestBasic'])))
    
    print("Training time to fit:                         N/A         N/A        N/A       ", ("{0:.3f}".format(general_dict['timeToFitNBBasic'])), "     ", ("{0:.3f}".format(general_dict['timeToFitTreeBasic'])), "     ", ("{0:.3f}".format(general_dict['timeToFitForestBasic'])))
    
    print("Training time to predict:                     N/A         N/A        N/A       ", ("{0:.3f}".format(general_dict['timeToPredictNBBasic'])), "     ", ("{0:.3f}".format(general_dict['timeToPredictTreeBasic'])), "     ", ("{0:.3f}".format(general_dict['timeToPredictForestBasic'])))

    print("\nNotes:")
    
    print("Note 1: In the case of the classifiers, the ratio is ratio of # of POIs to # of people in the test dataset.")

    print("Note 2: The accuracy score is perfect for the DecisionTree and Forest classifiers.")

    print("Note 3: The SVC classifier takes too long to run when I use sklearn.cross_validation, so it is not included in this report. It will be included in the Complex Report.")

    print("")

# The complexTrainTestData code is based on the Machine Learning course code, My reference 33 in my list of references. But I've modified it quite a bit
# Lesson 14, Quiz 8. https://classroom.udacity.com/nanodegrees/nd002/parts/0021345409/modules/317428862475460/lessons/2960698751/concepts/29872685780923
def complexTrainTestData(data, randomSeed, testPct, errorPct, maxDeferral, maxBonus, splitPct):
    global features_train
    global features_test
    global labels_train
    global labels_test

    #print("\nrandomSeed:", randomSeed) # randomSeed: 42 good
    n_points=len(data)
    #print("\nn_points, len(data):", n_points, len(data)) # n_points, len(data): 98 98 good

    random.seed(randomSeed) # From Lesson 14: https://classroom.udacity.com/nanodegrees/nd002/parts/0021345409/modules/317428862475460/lessons/2960698751/concepts/30383385630923 
    # "Random_state controls which points go into the training set and which are used for testing; setting it to 42 means we know exactly which events are in which set, and can check the results you get."
    deferralPoints = [random.random() for ii in range(0,n_points)] # random.random() Return the next random floating point number in the range [0.0, 1.0).
    #print("\nlen(deferralPoints), deferralPoints: ", len(deferralPoints), deferralPoints)

    """ Output: the numbers are always the same regardless of how often I run it... due to the random_seed:
        len(deferralPoints), deferralPoints:  98 [0.6394267984578837, 0.025010755222666936, 0.27502931836911926, 0.22321073814882275, ...]
    """
    
    bonusPoints = [random.random() for ii in range(0,n_points)]
    #print("\nlen(bonusPoints), bonusPoints: ", len(bonusPoints), bonusPoints)

    """ Output: the numbers are always the same regardless of how often I run it... due to the random_seed:
        len(bonusPoints), bonusPoints:  98 [0.9710783776136181, 0.8607797022344981, 0.011481021942819636, 0.7207218193601946, ...]
    """

    # Include a margin of error:
    # By adding "+errorPct*error[ii]" to the value of y, sometimes y will round up to 1 with error because it will bump it up to 0.5 or more which, when rounded will make it 1
    # whereas without adding "+errorPct*error[ii]", it would stay below 0.5 and round to zero.
    error = [random.random() for ii in range(0,n_points)] # 
    #print("\nlen(error), error: ", len(error), error) 

    """ output is - the numbers are always the same regardless of how often I run it... due to the random_seed:
        len(error), error:  98 [0.5503253124498481, 0.05058832952488124, 0.9992824684127266, 0.8360275850799519, 0.9689962572847513, ...]
    """

    y = [round(deferralPoints[ii]*bonusPoints[ii]+testPct+errorPct*error[ii]) for ii in range(0,n_points)]    
    #print("\nlen(y), y before max: ", len(y), y) # len(y), y before max:  98 [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,

    """ output is - the numbers are always the same regardless of how often I run it... due to the random_seed:
        len(y), y before max:  97 [1.0, 0.0, 1.0, 0.0, ...]
    """

    # If the deferralPoints or the bonusPoints are in the top x% of the data (e.g. 20%: 1-.8), then make the person a POI.
    # The for statement only replaces the y if TRUE, otherwise, it keeps what is already there, which may already be a 1
    for ii in range(0, len(y)):
        #print("\ny, dp and bp at start of max for loop: ", y, deferralPoints, bonusPoints)
        if deferralPoints[ii]>maxDeferral or bonusPoints[ii]>maxBonus:
            #print("y, dp and bp start of if of max for loop: ", y, deferralPoints, bonusPoints)
            y[ii] = 1.0
        #print("y, dp and bp after if in max for loop: ", y, deferralPoints, bonusPoints)
    #print("\nlen(y), y after max: ", len(y), y) # len(y), y after max:  98 [1.0, 1.0, 0.0, 1.0, ...

    """ output is - the numbers are always the same regardless of how often I run it... due to the random_seed:
        len(y), y after max:  97 [1.0, 1.0, 1.0, 0.0, ...] # y values are different after max.
    """
    
    # split into train/test sets
    X = [[dp, bp] for dp, bp in zip(deferralPoints, bonusPoints)]
    #print("\nlen(X) and X in complexTrainTestData...: ", len(X), X) # len(X) and X in complexTrainTestData...:  98 [[0.6394267984578837, 0.9710783776136181],  ...

    split = int(splitPct*n_points)
    #print("\nsplit in complexTrainTestData...: ", split)  # split in complexTrainTestData...:  73

    features_train = X[0:split]
    features_test  = X[split:]
    labels_train = y[0:split]
    labels_test  = y[split:]

    general_dict.update({"features_train_complex":features_train, 
                         "features_test_complex":features_test, 
                         "labels_train_complex":labels_train, 
                         "labels_test_complex":labels_test,
                         "dataComplex":data,
                         "testPctComplex":testPct, 
                         "errorPctComplex":errorPct, 
                         "maxDeferralComplex":maxDeferral, 
                         "maxBonusComplex":maxBonus, 
                         "splitPctComplex":splitPct,
                         "randomSeedComplex":randomSeed}) 
    #print("\nin complexTrainTestData at top general_dict: ", general_dict) # e.g. 'labels_test_complex': [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0], 'features_test_complex': [[0.8763676264726689, 0.132311848725025], [0.3146778807984779, ...

    if False: # If set to True, this will print to check the values,

        print("\n########################## features_train, features_test, labels_train, labels_test on finalData ##########################\n")
        #  The numbers are always the same regardless of how often I run it... due to the random_seed
        print("len(features_train), features_train: ", len(features_train), features_train) # len(features_train), features_train:  73 ...
        #[[0.6394267984578837, 0.9710783776136181], [0.025010755222666936, 0.8607797022344981], ...]]
             
        print("len(features_test), features_test: ", len(features_test), features_test) #  len(features_test), features_test:  25 ...
        #[[0.3146778807984779, 0.5710430933252845], [0.65543866529488, 0.47267102631179414], ...]]
             
        print("len(labels_train), labels_train: ", len(labels_train), labels_train) # len(labels_train), labels_train:  73 [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,  ... ]
        
        print("len(labels_test), labels_test: ", len(labels_test), labels_test) # len(labels_test), labels_test:  25 [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, ... ]
    
    # Create the training data:
    deferred_nonPOI = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
    bonus_nonPOI =    [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
    deferred_POI =    [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
    bonus_POI =       [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]
    
    training_data = {"nonPOI":{"deferred":deferred_nonPOI, "bonus":bonus_nonPOI}
            , "POI":{"deferred":deferred_POI, "bonus":bonus_POI}}

    if False: # if set to True check and print training data

        print("\n################################ deferred and bonus, POI and non-POI creating TRAINING_data ################################")
        print("\nTraining data: Train Pct, len(deferred_nonPOI), len(bonus_nonPOI), len(deferred_POI), len(bonus_POI): ", splitPct, len(deferred_nonPOI), 
              len(bonus_nonPOI), len(deferred_POI), len(bonus_POI)) # Training data: Train Pct, len(deferred_nonPOI), len(bonus_nonPOI), len(deferred_POI), len(bonus_POI):  0.75 27 27 46 46
        print("\nTraining data: deferred_nonPOI, bonus_nonPOI, deferred_POI, bonus_POI: ", deferred_nonPOI, bonus_nonPOI, deferred_POI, bonus_POI) # 
        """ Output: the numbers are always the same regardless of how often I run it... due to the random_seed:
                
        Training data: deferred_nonPOI, bonus_nonPOI, deferred_POI, bonus_POI:  [0.27502931836911926, 0.08693883262941615, 
        0.4219218196852704, 0.029797219438070344, 0.21863797480360336, 0.1988376506866485, 0.5449414806032167, 
        0.006498759678061017, 0.6981393949882269, 0.15547949981178155, 0.09274584338014791, 0.09671637683346401, 
        0.5362280914547007, 0.045824383655662215, 0.22789827565154686,
        ...
        """
        print("\ntraining_data: ", training_data)
        """ Output: the numbers are always the same regardless of how often I run it... due to the random_seed:
        training_data:  {'nonPOI': {'bonus': [0.011481021942819636, 0.6409617985798081,  ... ],
            'deferred': [0.27502931836911926, 0.08693883262941615, ...]
            }, 
            'POI': {'bonus': [0.9710783776136181, 0.8607797022344981, ...], 
            'deferred': [0.6394267984578837, 0.025010755222666936, ...]
            }}
        """

    # Create the testing data:
    deferred_nonPOI = [features_test[ii][0] for ii in range(0, len(features_test)) if labels_test[ii]==0]
    bonus_nonPOI =    [features_test[ii][1] for ii in range(0, len(features_test)) if labels_test[ii]==0]
    deferred_POI =    [features_test[ii][0] for ii in range(0, len(features_test)) if labels_test[ii]==1]
    bonus_POI =       [features_test[ii][1] for ii in range(0, len(features_test)) if labels_test[ii]==1]
    
    test_data = {"nonPOI":{"deferred":deferred_nonPOI, "bonus":bonus_nonPOI}
            , "POI":{"deferred":deferred_POI, "bonus":bonus_POI}}

    if False: # if set to True check and print testing data

        print("\n################################## deferred and bonus, POI and non-POI creating TEST_data ##################################")
        print("\nTest data: Test Pct, len(deferred_nonPOI), len(bonus_nonPOI), len(deferred_POI), len(bonus_POI): ", (1-splitPct), len(deferred_nonPOI), 
              len(bonus_nonPOI), len(deferred_POI), len(bonus_POI)) # Test data: Test Pct, len(deferred_nonPOI), len(bonus_nonPOI), len(deferred_POI), len(bonus_POI):  0.25 11 11 14 14

        print("\nTest data: deferred_nonPOI, bonus_nonPOI, deferred_POI, bonus_POI: ", deferred_nonPOI, bonus_nonPOI, deferred_POI, bonus_POI)  
        """ Output: the numbers are always the same regardless of how often I run it... due to the random_seed:
                
        Test data: deferred_nonPOI, bonus_nonPOI, deferred_POI, bonus_POI:  [0.3146778807984779, 0.4588518525873988, 
        0.26488016649805246, 0.24662750769398345, 0.26274160852293527, 0.21932075915728333, 0.5095262936764645, 
        0.04711637542473457, 0.10964913035065915, 0.42215996679968404, 0.06352770615195713] [0.5710430933252845, 
        0.1904099143618777, 0.09693081422882333, 0.4310511824063775, 0.467024668036675, 0.09841787115195888, 
        0.33930260539496315, 0.24865633392028563, 0.1902089084408115, 0.27854514466694047, 0.2498064478821005] 
        [0.65543866529488, 0.39563190106066426, 0.9145475897405435, 0.5613681341631508, 0.5845859902235405, 
        0.897822883602477, 0.39940050514039727, 0.9975376064951103, 0.09090941217379389, 0.62744604170309, 
        0.7920793643629641, 0.38161928650653676, 0.9961213802400968, 0.529114345099137] [0.47267102631179414, 
        0.7846194242907534, 0.8074969977666434, 0.4235786230199208, 0.7290758494598506, 0.6733645472933015, 
        0.9841652113659661, 0.4026212821022688, 0.8616725363527911, 0.4486135478331319, 0.4218816398344042, 
        0.9232655992760128, 0.44313074505345695, 0.8613491047618306]
        """
        print("\ntest_data: ", test_data)
        """ Output: the numbers are always the same regardless of how often I run it... due to the random_seed:
        test_data:  {'nonPOI': {'bonus': [0.5710430933252845, 0.1904099143618777,  ...], # with comma
        'deferred': [0.3146778807984779, 0.4588518525873988, ...]
        },
        'POI': {'bonus': [0.47267102631179414, 0.7846194242907534, ...],
        'deferred': [0.65543866529488, 0.39563190106066426, ...]
        }}
        """    

    myCodeFitPredict(general_dict['myCode'][1], 0, 5)

    return training_data, test_data
    return features_train, labels_train, features_test, labels_test

def reportComplexTrainTestData():
    #print("\nin reportComplexTrainTestData - General Dictionary: ", general_dict)
    print("\n%%%%%% COMPLEX: TRY A VARIETY OF CLASSIFIERS AND TUNE MY CLASSIFIER TO TRY TO GET BETTER THAN 0.3 PRECISION AND RECALL %%%%%%")
    print("\n##################### Table of Statistics from the Classifiers on data created by Split and Some Notes #####################")

    print("\n********** Parameters used to create the training and test data: ")
    print("Random Seed = ", general_dict['randomSeedComplex'], 
          "Test Percent = ", general_dict['testPctComplex'], 
          "Error Percent = ", general_dict['errorPctComplex'], 
          "Max Deferral Payments = ", general_dict['maxDeferralComplex'], 
          "Max Bonus = ", general_dict['maxBonusComplex'], 
          "Split Percent = ", general_dict['splitPctComplex'], ".")
    
    print("\nDescription                                Names File  Enron Data   My Data     GaussianNB  Tree        Forest       SVC")
    
    print("------------------------------------------ ----------  ----------  ----------  ----------  ----------  ----------  ----------")
    
    print("Number of people:                          ", general_dict['readNamesLen'], "       ", general_dict['totalPeople'], "       ", general_dict['dataFinalLen'], "        ", general_dict['dataFinalLen'], "        ", general_dict['dataFinalLen'], "        ", general_dict['dataFinalLen'], "        ", general_dict['dataFinalLen'])

    print("Number of POIs:                             ", general_dict['poi_qty_all_names'], "        ", general_dict['totalPOIs'], "       ", ("{0:.0f}".format(general_dict['dataFinalSumPOI'])), "         ", ("{0:.0f}".format(general_dict['clfPredictSumPOINBBasic'])), "        ", ("{0:.0f}".format(general_dict['clfPredictSumPOITreeComplex'])), "        ", ("{0:.0f}".format(general_dict['clfPredictSumPOIForestComplex'])), "        ", ("{0:.0f}".format(general_dict['clfPredictSumPOISVMComplex'])))
    
    print("Ratio of POIs to number of people (note 1):  ", ("{0:.3f}".format(general_dict['readNamesPOIRatio'])), "     ",("{0:.3f}".format(general_dict['totalRatioPOIs'])), "    ",("{0:.3f}".format(dataFinalRatioPOI)), "     ", ("{0:.3f}".format(general_dict['clfPredictRatioPOINBComplex'])), "     ", ("{0:.3f}".format(general_dict['clfPredictRatioPOITreeComplex'])), "     ", ("{0:.3f}".format(general_dict['clfPredictRatioPOIForestComplex'])), "     ", ("{0:.3f}".format(general_dict['clfPredictRatioPOISVMComplex'])))
    
    print("Ratio of pred POIs ratio to total ratio:       N/A         N/A        N/A       ", ("{0:.3f}".format(general_dict['ratioPredictedPOIToWholeNBComplex'])), "     ", ("{0:.3f}".format(general_dict['ratioPredictedPOIToWholeTreeComplex'])), "     ", ("{0:.3f}".format(general_dict['ratioPredictedPOIToWholeForestComplex'])), "     ", ("{0:.3f}".format(general_dict['ratioPredictedPOIToWholeSVMComplex'])))
    
    print("Accuracy (Note 2):                             N/A         N/A        N/A       ", ("{0:.3f}".format(general_dict['accuracyNBComplex'])), "     ", ("{0:.3f}".format(general_dict['accuracyTreeComplex'])), "     ", ("{0:.3f}".format(general_dict['accuracyForestComplex'])), "     ", ("{0:.3f}".format(general_dict['accuracySVMComplex'])))
    
    print("Score:                                         N/A         N/A        N/A       ", ("{0:.3f}".format(general_dict['scoreNBComplex'])), "     ", ("{0:.3f}".format(general_dict['scoreTreeComplex'])), "     ", ("{0:.3f}".format(general_dict['scoreForestComplex'])), "     ", ("{0:.3f}".format(general_dict['scoreSVMComplex'])))
    
    print("Training time to fit:                          N/A         N/A        N/A       ", ("{0:.3f}".format(general_dict['timeToFitNBComplex'])), "     ", ("{0:.3f}".format(general_dict['timeToFitTreeComplex'])), "     ", ("{0:.3f}".format(general_dict['timeToFitForestComplex'])), "     ", ("{0:.3f}".format(general_dict['timeToFitSVMComplex'])))
    
    print("Training time to predict:                      N/A         N/A        N/A       ", ("{0:.3f}".format(general_dict['timeToPredictNBComplex'])), "     ", ("{0:.3f}".format(general_dict['timeToPredictTreeComplex'])), "     ", ("{0:.3f}".format(general_dict['timeToPredictForestComplex'])), "     ", ("{0:.3f}".format(general_dict['timeToPredictSVMComplex'])))
    
    print("\nNotes:")
    
    print("Note 1: In the case of the classifiers, the ratio is ratio of # of POIs to # of people in the test dataset.")

    print("Note 2: The accuracy scores are high for all classifiers.")
    
    print("Note 3: By using split instead of sklearn cross_validation; I have more control over the split of testing and training data.")
    print("The numbers seem more reliable, since there are no perfect 1.0 metrics.")

    print("")

def treeAccuracy(): # I did not run this in main. It is just here to test.
    # Try using min_samples_split on Tree classifier to see if I can raise the accuracy:
    # Learned in Lesson 4 Quiz 12
    if False: # Try Decision Tree min_samples_split
        clf = tree.DecisionTreeClassifier(min_samples_split = 20) # I also tried 30 which yielded the same result
        clf = clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        acc_min_samples_split = accuracy_score(pred, labels_test)
        print("\n########## Tree Classifier with other splits #########\n")
        
        print("acc_min_samples_split_20: ", acc_min_samples_split)
        
        clf = tree.DecisionTreeClassifier(min_samples_split = 2)
        clf = clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        acc_min_samples_split = accuracy_score(pred, labels_test)
        print("acc_min_samples_split_2: ", acc_min_samples_split)
        
        # Answer is still accuracy = 80% with split of 20 or 30 and bounced between 76% and 80% with split of 2 

def regressionDefBonus():
    # Regression: list the features I want to look at--first item in the list will be the "target" feature
    # training-testing split needed in regression, just like classification
    # Superscript #29 in my references file: based on Udacity lesson on how to calculate and plot regressions
    regDictionary = data_dict
    reg_features_list = ["bonus", "deferral_payments"]
    regData = featureFormat( regDictionary, reg_features_list, remove_any_zeroes=True)
    #print("\nregData: ", regData) # [[ 1200000.  1295738.] [  400000.  1130036.] Good, bonus and deferral_payments
    """ Output:
    regData:  [[ 1200000.  1295738.]
    [  400000.  1130036.]
    [ 1200000.    27610.]
    ..., 
    [  400000.   260455.]
    [  900000.   649584.]
    [  600000.   227449.]]
    """


    regTarget, regFeatures = targetFeatureSplit( regData )
    
    train_color = "b"
    test_color = "r" # To differentiate training points from test points.

    testSize = (0.8, 0.5, 0.3, 0.25)
    testSizeA = ("TestSize0.8", "TestSize0.5", "TestSize0.3", "TestSize0.25")
    general_dict.update({'testSizeA':testSizeA})
    #print("\nin regressionDefBonus: general_dict: ", general_dict) # ... 'testSizeA': ('TestSize0.8', 'TestSize0.5', 'TestSize0.3', 'TestSize0.25'), ...
    elemC = -1
    for elem in testSize: 
        elemC = elemC+1
        #print("\nin regressionDefBonus - elemC, elem and testSize: ", elemA, elem, testSize) # in regressionDefBonus - elemC, elem and testSize:  0 0.25 (0.8, 0.5, 0.3, 0.25)

        #reg_feature_train, reg_feature_test, reg_target_train, reg_target_test = train_test_split(regFeatures, regTarget, test_size=0.8, random_state=42)
        reg_feature_train, reg_feature_test, reg_target_train, reg_target_test = train_test_split(regFeatures, regTarget, test_size=elem, random_state=42)
    
        # Naming it reg, so that the plotting code below picks it up and plots it correctly. 
        reg = linear_model.LinearRegression() 

        reg.fit(reg_feature_test, reg_target_test) 

        varSlope = reg.coef_
        varFullSlope = "regSlope"+general_dict['testSizeA'][elemC]
        #print("\nin regressionDefBonus varFullSlope by elem: ", varFullSlope) 
        """ Output:
        in regressionDefBonus varFullSlope by elem:  regSlopeTestSize0.8
        in regressionDefBonus varFullSlope by elem:  regSlopeTestSize0.5
        in regressionDefBonus varFullSlope by elem:  regSlopeTestSize0.3
        in regressionDefBonus varFullSlope by elem:  regSlopeTestSize0.25
        """


        varIntercept = reg.intercept_
        varFullIntercept = "regIntercept"+general_dict['testSizeA'][elemC]
        #print("\nin regressionDefBonus varFullIntercept by elem: ", varFullIntercept) # in regressionDefBonus varFullIntercept by elem:  regInterceptTestSize0.8

        varRegScoreOnTrain = reg.score(reg_feature_train, reg_target_train)
        varFullRegScoreOnTrain = "regScoreOnTrain"+general_dict['testSizeA'][elemC]
        #print("\nin regressionDefBonus varFullRegScoreOnTrain by elem: ", varFullRegScoreOnTrain) # in regressionDefBonus varFullRegScoreOnTrain by elem:  regScoreOnTrainTestSize0.8

        varRegScoreOnTest = reg.score(reg_feature_test, reg_target_test)
        varFullRegScoreOnTest = "regScoreOnTest"+general_dict['testSizeA'][elemC]
        #print("\nin regressionDefBonus varFullRegScoreOnTest by elem: ", varFullRegScoreOnTest) # in regressionDefBonus varFullRegScoreOnTest by elem:  regScoreOnTestTestSize0.8

        general_dict.update({varFullSlope:varSlope, varFullIntercept:varIntercept, varFullRegScoreOnTrain:varRegScoreOnTrain, varFullRegScoreOnTest:varRegScoreOnTest}) 

    print("\n######################### Regression Stats on bonus (target) and deferral_payments - by Test Size ########################\n")
    print("Statistic                          Test Size =>      25%               30%               50%               80% ")

    print("----------------------------------------------- ----------------  ----------------  ----------------  ----------------")

    print("")

    print('Slope (Coefficient)                             ', (general_dict['regSlopeTestSize0.25']), 
          "   ", (general_dict['regSlopeTestSize0.3']), 
          "   ", (general_dict['regSlopeTestSize0.5']), 
          "   ", (general_dict['regSlopeTestSize0.8']))
    print('Intercept                                       ', ("{0:.0f}".format(general_dict['regInterceptTestSize0.25'])), 
          "         ", ("{0:.0f}".format(general_dict['regInterceptTestSize0.3'])), 
          "         ", ("{0:.0f}".format(general_dict['regInterceptTestSize0.5'])), 
          "         ", ("{0:.0f}".format(general_dict['regInterceptTestSize0.8'])))
    print('r-Square Score On Train Dataset                      ', ("{0:.3f}".format(general_dict['regScoreOnTrainTestSize0.25'])), 
          "          ", ("{0:.3f}".format(general_dict['regScoreOnTrainTestSize0.3'])), 
          "           ", ("{0:.3f}".format(general_dict['regScoreOnTrainTestSize0.5'])), 
          "           ", ("{0:.3f}".format(general_dict['regScoreOnTrainTestSize0.8'])))
    print('r-Square Score On Test Dataset                        ', ("{0:.3f}".format(general_dict['regScoreOnTestTestSize0.25'])), 
          "           ", ("{0:.3f}".format(general_dict['regScoreOnTestTestSize0.3'])), 
          "           ", ("{0:.3f}".format(general_dict['regScoreOnTestTestSize0.5'])), 
          "           ", ("{0:.3f}".format(general_dict['regScoreOnTestTestSize0.8'])))

    # Number 34 in my references file - meaning of regression: https://www.google.ca/search?dcr=0&ei=ufZQWrSJEOaP0gKdyIHYAQ&q=what+does+regression+mean+in+statistics&oq=what+does+regression+mean+in+statistics&gs_l=psy-ab.12...0.0.0.6725.0.0.0.0.0.0.0.0..0.0....0...1c..64.psy-ab..0.0.0....0.cCN20jYIRYI
    print("\nThe strength of the relationship between the dependent variable, bonus, (denoted by the Y axis) and the independent variable, deferral_payments, (on the x axis) is low. There is not a strong correlation between these 2 features.")

    # Number 35 in my references file - meaning of R-squared: https://www.google.ca/search?dcr=0&ei=CflQWsDRMsmN0wKE_oSYAg&q=what+does+r+squared+mean+in+statistics&oq=what+does+rmean+in+statistics&gs_l=psy-ab.1.1.0i7i30k1l10.5826.5826.0.9552.1.1.0.0.0.0.250.250.2-1.1.0....0...1c..64.psy-ab..0.1.249....0.EssQFqbfAxs
    print("\nR-squared is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression. ... 100% indicates that the model explains all the variability of the response data around its mean.")

    print("\nSince the r-squared score on test dataset is close to zero, the features are not correlated.")

    # labels for the legend
    plt.scatter(reg_feature_test[0], reg_target_test[0], color=test_color, label="test")
    plt.scatter(reg_feature_test[0], reg_target_test[0], color=train_color, label="train")

            
    # draw the scatterplot, with color-coded training and testing points
    for feature, target in zip(reg_feature_test, reg_target_test):
        plt.scatter( feature, target, color=test_color ) 
    for feature, target in zip(reg_feature_train, reg_target_train):
        plt.scatter( feature, target, color=train_color ) 

    # Draw the regression line, once it's coded
    try:
        plt.plot( feature_test, reg.predict(feature_test) )
    except NameError:
        pass
    
    plt.plot(reg_feature_train, reg.predict(reg_feature_train), color="b")  # this plots the regression line

    plt.title("Scatterplot of Test and Train with Regression Line for 0.25 test size")
    plt.xlabel(reg_features_list[1])
    plt.ylabel(reg_features_list[0])
    plt.legend()
    plt.show()
    return general_dict

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
### Task 5: Tune your classifier to achieve better than .3 precision and recall # Original
### using our testing script. Check the tester.py script in the final project # Original
### folder for details on the evaluation method, especially the test_classifier # Original
### function. Because of the small size of the dataset, the script uses # Original
### stratified shuffle split cross validation. For more info:  # Original
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html # Original

# Example starting point. Try investigating other evaluation techniques! # Original
#from sklearn.cross_validation import train_test_split # Original
#features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42) # Original 

def myValidationStats(Type, elemStart, elemEnd): # Where Type is "Basic" or "Complex" and elemStart / elemEnd allows me to exclude some classifiers if I want to
           
    labels_test = Type
    #print("\nin myValidationStats len and labels_test: ", len(labels_test), labels_test) # in myValidationStats len and labels_test:  30 [ 0.  1.  0. ...,  0.  1.  0.]
    # Ouput first time through: in myValidationStats len and labels_test:  30 [ 0.  0.  0. ...,  0.  1.  0.]
    # Output 2nd time through: in myValidationStats len and labels_test:  25 [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    for elem in (general_dict['listClfPredict'][elemStart:elemEnd]): 

        clfPredict = general_dict[elem]
        clfPredict = clfPredict.astype(np.float)
        #print("\nin myValidationStats len and clfPredict: ", len(clfPredict), clfPredict) 

        varIndex = (general_dict['listClfPredict'].index(elem))
        #print("\nin myValidationStats varIndex: ", varIndex) 
            
        truePositives = [1 for j in zip(labels_test, clfPredict) if j[0] == j[1] and j[1] == 1]
        #print("\nlen(truePositives) and truePositives: ", len(truePositives), truePositives) 
            
        falsePositives = [1 for j in zip(labels_test, clfPredict) if j[0] != j[1] and j[1] == 1]
        #print("\nlen(falsePositives) and falsePositives: ", len(falsePositives), falsePositives) 
            
        trueNegatives = [1 for j in zip(labels_test, clfPredict) if j[0] == j[1] and j[1] == 0] 
        #print("\nlen(trueNegatives) and trueNegatives: ", len(trueNegatives), trueNegatives)

        falseNegatives = [1 for j in zip(labels_test, clfPredict) if j[0] != j[1] and j[1] == 0] 
        #print("\nlen(falseNegatives) and falseNegatives: ", len(falseNegatives), falseNegatives)

        lenTruePositives = len(truePositives) 
        lenFalsePositives = len(falsePositives)  
        lenTrueNegatives = len(trueNegatives)  
        lenFalseNegatives = len(falseNegatives)  

        # To check that the sum of the parts adds up to the total
        totalAllTrueFalsePosNeg = len(truePositives) + len(trueNegatives) + len(falsePositives) + len(falseNegatives)
        totalTest = len(labels_test)
        equalTest = (totalTest == totalAllTrueFalsePosNeg)
        #print("\ntotalAllTrueFalsePosNeg, totalTest, equalTest: ", totalAllTrueFalsePosNeg, totalTest, equalTest) 
        
        #print("\ntotalAllTrueFalsePosNeg and does the total equal total tests? ", totalAllTrueFalsePosNeg, equalTest) 
            
        pBase = precision_score(labels_test, clfPredict) 
        #print("\npBase: ", pBase) 
            
        pBinary = precision_score(labels_test, clfPredict, average = 'binary') 
        #print("\npBinary: ", pBinary) 
            
        pMacro = precision_score(labels_test, clfPredict, average = 'macro') 
        #print("\npMacro: ", pMacro) 
            
        pMicro = precision_score(labels_test, clfPredict, average = 'micro')  
        #print("\npMicro: ", pMicro) 
            
        pWeighted = precision_score(labels_test, clfPredict, average = 'weighted') 
        #print("\npWeighted: ", pWeighted) 
            
        rBase = recall_score(labels_test, clfPredict) 
        #print("\nrBase: ", rBase) 
            
        rBinary = recall_score(labels_test, clfPredict, average = 'binary') 
        #print("\nrBinary: ", rBinary) 
            
        rMacro = recall_score(labels_test, clfPredict, average = 'macro') 
        #print("\nrMacro: ", rMacro) 
            
        rMicro = recall_score(labels_test, clfPredict, average = 'micro') 
        #print("\nrMicro: ", rMicro) 
            
        rWeighted = recall_score(labels_test, clfPredict, average = 'weighted') 
        #print("\nrWeighted: ", rWeighted) 
            
        f1Base = 2*((pBase*rBase)/(pBase+rBase)) 
        #print("\nf1Base: ", f1Base) 
        
        f1Binary = 2*((pBinary*rBinary)/(pBinary+rBinary)) 
        #print("\nf1Binary: ", f1Binary) 
            
        f1Macro = 2*((pMacro*rMacro)/(pMacro+rMacro)) 
        #print("\nf1Macro: ", f1Macro) 
            
        f1Micro = 2*((pMicro*rMicro)/(pMicro+rMicro)) 
        #print("\nf1Micro: ", f1Micro) 
            
        f1Weighted = 2*((pWeighted*rWeighted)/(pWeighted+rWeighted)) 
        #print("\nf1Weighted: ", f1Weighted) 
        
        # Save the output in the general_dict so that I can use it in the report
        varFulltruePositives = general_dict['varBegin'][8] + general_dict['varEnd'][varIndex] 
        varFullfalsePositives = general_dict['varBegin'][9] + general_dict['varEnd'][varIndex]
        varFulltrueNegatives = general_dict['varBegin'][10] + general_dict['varEnd'][varIndex] 
        varFullfalseNegatives = general_dict['varBegin'][11] + general_dict['varEnd'][varIndex] 
        varFulllentruePositives = 'len' + general_dict['varBegin'][8] + general_dict['varEnd'][varIndex] 
        varFulllenfalsePositives = 'len' + general_dict['varBegin'][9] + general_dict['varEnd'][varIndex]
        varFulllentrueNegatives = 'len' + general_dict['varBegin'][10] + general_dict['varEnd'][varIndex] 
        varFulllenfalseNegatives = 'len' + general_dict['varBegin'][11] + general_dict['varEnd'][varIndex] 
        varFulltotalAllTrueFalsePosNeg = general_dict['varBegin'][12] + general_dict['varEnd'][varIndex] 
        varFulltotalTest = general_dict['varBegin'][13] + general_dict['varEnd'][varIndex] 
        varFullequalTest = general_dict['varBegin'][14] + general_dict['varEnd'][varIndex] 
        varFullpBase = general_dict['varBegin'][15] + general_dict['varEnd'][varIndex] 
        varFullpBinary = general_dict['varBegin'][16] + general_dict['varEnd'][varIndex]
        varFullpMacro = general_dict['varBegin'][17] + general_dict['varEnd'][varIndex] 
        varFullpMicro = general_dict['varBegin'][18] + general_dict['varEnd'][varIndex] 
        varFullpWeighted = general_dict['varBegin'][19] + general_dict['varEnd'][varIndex]
        varFullrBase = general_dict['varBegin'][20] + general_dict['varEnd'][varIndex] 
        varFullrBinary = general_dict['varBegin'][21] + general_dict['varEnd'][varIndex] 
        varFullrMacro = general_dict['varBegin'][22] + general_dict['varEnd'][varIndex] 
        varFullrMicro = general_dict['varBegin'][23] + general_dict['varEnd'][varIndex] 
        varFullrWeighted = general_dict['varBegin'][24] + general_dict['varEnd'][varIndex] 
        varFullf1Base = general_dict['varBegin'][25] + general_dict['varEnd'][varIndex] 
        varFullf1Binary = general_dict['varBegin'][26] + general_dict['varEnd'][varIndex] 
        varFullf1Macro = general_dict['varBegin'][27] + general_dict['varEnd'][varIndex] 
        varFullf1Micro = general_dict['varBegin'][28] + general_dict['varEnd'][varIndex] 
        varFullf1Weighted = general_dict['varBegin'][29] + general_dict['varEnd'][varIndex]

        general_dict.update({varFulltruePositives:truePositives, 
                             varFullfalsePositives:falsePositives, 
                             varFulltrueNegatives:trueNegatives, 
                             varFullfalseNegatives:falseNegatives,
                             varFulllentruePositives:lenTruePositives, 
                             varFulllenfalsePositives:lenFalsePositives, 
                             varFulllentrueNegatives:lenTrueNegatives, 
                             varFulllenfalseNegatives:lenFalseNegatives,
                             varFulltotalAllTrueFalsePosNeg:totalAllTrueFalsePosNeg, 
                             varFulltotalTest:totalTest, 
                             varFullequalTest:equalTest, 
                             varFullpBase:pBase, 
                             varFullpBinary:pBinary, 
                             varFullpMacro:pMacro, 
                             varFullpMicro:pMicro, 
                             varFullpWeighted:pWeighted, 
                             varFullrBase:rBase, 
                             varFullrBinary:rBinary, 
                             varFullrMacro:rMacro, 
                             varFullrMicro:rMicro, 
                             varFullrWeighted:rWeighted, 
                             varFullf1Base:f1Base, 
                             varFullf1Binary:f1Binary, 
                             varFullf1Macro:f1Macro, 
                             varFullf1Micro:f1Micro, 
                             varFullf1Weighted:f1Weighted})
        #print("\n\n\n\nin myValidationStats - General Dictionary: ", general_dict)

def myValidation():

    listLabelsTest = ['labels_test_basic', 'labels_test_complex']
    listClfPredict = ['clfPredictNBBasic', 'clfPredictTreeBasic', 'clfPredictForestBasic', 'clfPredictKMeansBasic', 'clfPredictNBComplex', 'clfPredictTreeComplex', 'clfPredictForestComplex', 'clfPredictKMeansComplex', 'clfPredictSVMComplex']

    general_dict.update({'listLabelsTest':listLabelsTest, 
                         'listClfPredict':listClfPredict})
    for elemE in general_dict['listLabelsTest']:
        #print("\nelemE: ", elemE) # first time through the loop: elemE:  labels_test_basic
        # Second time through the loop: elemE:  labels_test_complex

        if elemE == 'labels_test_basic':
            elemStart = 0
            elemEnd = 4
            #print("\nin myValidation if basic - len and general_dict[elemE]: ", len(general_dict[elemE]), general_dict[elemE]) # in myValidation - len and general_dict[elemE]:  30 [['0.0'] ['1.0'] ['0.0'] ...,  ['0.0'] ['1.0'] ['0.0']]
            Type = np.concatenate(general_dict[elemE])
            #print("\nin myValidation if basic after concatenate - len and Type: ", len(Type), Type) # in myValidation - len and Type:  30 ['0.0' '1.0' '0.0' ..., '0.0' '1.0' '0.0']

            Type = Type.astype(np.float)
        else:
            elemStart = 4
            elemEnd = 9
            Type = general_dict[elemE]
            #print("\nin myValidation if complex - len and general_dict[elemE]: ", len(general_dict[elemE]), general_dict[elemE]) 
            # Output: in myValidation if complex - len and general_dict[elemE]:  25 [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        
        #print("\nin myValidation - len and Type: ", len(Type), Type) 
        # Ouput first time through - basic: in myValidation - len and Type:  30 [ 0.  0.  0. ...,  0.  1.  0.]
        # OUtput second time through - complex: in myValidation - len and Type:  in myValidation - len and Type:  25 [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]

        myValidationStats(Type, elemStart, elemEnd)

def myValidationReport():

    print("\n%%%%%%%%%%%%%%%%%%%%%%%%% Precision, Recall and F1 for various classifiers with various parameters %%%%%%%%%%%%%%%%%%%%%%%%%")
    print("\n############################ Basic: Training and Test Data Created by sklearn cross_validation #############################")
    
    print("\n********** Classifiers used in this section: ")

    print("\n", (general_dict['testClassifiers'][0]))
    print("\n", (general_dict['testClassifiers'][1]))
    print("\n", (general_dict['testClassifiers'][2]))

    print("\n                                   Test  Len        Len        Len        Len")
    print("                                   Case  True       False      True       False")
    print("Classifier / Average Parameter     Size  Positives  Positives  Negatives  Negatives  Precision  Recall  F1") 
    print("---------------------------------  ----  ---------  ---------  ---------  ---------  ---------  ------  -----")
    
    print("GaussianNB      Average = Binary:  ", general_dict['totalTestNBBasic'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesNBBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesNBBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesNBBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesNBBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['pBinaryNBBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['rBinaryNBBasic'])), 
          " ", ("{0:.3f}".format(general_dict['f1BinaryNBBasic'])),)

    print("Decision Tree   Average = Binary:  ", general_dict['totalTestTreeBasic'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesTreeBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesTreeBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesTreeBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesTreeBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['pBinaryTreeBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['rBinaryTreeBasic'])), 
          " ", ("{0:.3f}".format(general_dict['f1BinaryTreeBasic'])),)

    print("Forest          Average = Binary:  ", general_dict['totalTestForestBasic'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesForestBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesForestBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesForestBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesForestBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['pBinaryForestBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['rBinaryForestBasic'])), 
          " ", ("{0:.3f}".format(general_dict['f1BinaryForestBasic'])),)

    print(".................................  ....  .........  .........  .........  .........  .........  ......  .....")

    print("GaussianNB      Average = Macro:   ", general_dict['totalTestNBBasic'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesNBBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesNBBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesNBBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesNBBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['pMacroNBBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['rMacroNBBasic'])), 
          " ", ("{0:.3f}".format(general_dict['f1MacroNBBasic'])),)

    print("Decision Tree   Average = Macro:   ", general_dict['totalTestTreeBasic'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesTreeBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesTreeBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesTreeBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesTreeBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['pMacroTreeBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['rMacroTreeBasic'])), 
          " ", ("{0:.3f}".format(general_dict['f1MacroTreeBasic'])),)

    print("Forest          Average = Macro:   ", general_dict['totalTestForestBasic'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesForestBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesForestBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesForestBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesForestBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['pMacroForestBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['rMacroForestBasic'])), 
          " ", ("{0:.3f}".format(general_dict['f1MacroForestBasic'])),)

    print(".................................  ....  .........  .........  .........  .........  .........  ......  .....")

    print("GaussianNB      Average = Micro:   ", general_dict['totalTestNBBasic'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesNBBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesNBBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesNBBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesNBBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['pMicroNBBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['rMicroNBBasic'])), 
          " ", ("{0:.3f}".format(general_dict['f1MicroNBBasic'])),)

    print("Decision Tree   Average = Micro:   ", general_dict['totalTestTreeBasic'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesTreeBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesTreeBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesTreeBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesTreeBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['pMicroTreeBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['rMicroTreeBasic'])), 
          " ", ("{0:.3f}".format(general_dict['f1MicroTreeBasic'])),)

    print("Forest          Average = Micro:   ", general_dict['totalTestForestBasic'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesForestBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesForestBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesForestBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesForestBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['pMicroForestBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['rMicroForestBasic'])), 
          " ", ("{0:.3f}".format(general_dict['f1MicroForestBasic'])),)

    print(".................................  ....  .........  .........  .........  .........  .........  ......  .....")

    print("GaussianNB      Average = Weighted:", general_dict['totalTestNBBasic'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesNBBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesNBBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesNBBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesNBBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['pWeightedNBBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['rWeightedNBBasic'])), 
          " ", ("{0:.3f}".format(general_dict['f1WeightedNBBasic'])),)

    print("Decision Tree   Average = Weighted:", general_dict['totalTestTreeBasic'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesTreeBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesTreeBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesTreeBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesTreeBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['pWeightedTreeBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['rWeightedTreeBasic'])), 
          " ", ("{0:.3f}".format(general_dict['f1WeightedTreeBasic'])),)
    
    print("Forest          Average = Weighted:", general_dict['totalTestForestBasic'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesForestBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesForestBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesForestBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesForestBasic'])), 
          "   ", ("{0:.3f}".format(general_dict['pWeightedForestBasic'])), 
          "    ", ("{0:.3f}".format(general_dict['rWeightedForestBasic'])), 
          " ", ("{0:.3f}".format(general_dict['f1WeightedForestBasic'])),)
    
    print(".................................  ....  .........  .........  .........  .........  .........  ......  .....")
    
    print("\nNotes: ")
    print("When the training and test data were created by sklearn cross_validation,")
    print("the best Precision, Recall and F1 scores were when using the Decision Tree and Forest Classifiers, regardless of the type of average.")
    print("All the results are 1.0 which means that the data was perfectly predicted.")
    print("The best results for the GaussianNB classifier was when the Average = Micro.")
    print("Normally high scores would mean that I can have full confidence in recall of the suspects,")
    print("However, because some scores are perfect, the results are suspect.")

    print("\nThe number of POIs in the Enron dataset is very low, so the numbers are not perfectly reliable. ")

    print("\nIn the next section, we will add several parameters, to see how the metrics line up.")

    print("\nAnd in the last section, task 6 in this project, we will run the full dataset through multiple folds / repeats, to get more accurate data.")
    print("In task 6, we will also program SelectKBest to pick the best features, and allow cvGridSearch to pick the best classifier, to optimize the precision and recall scores.")

    print("\n##################################### Complex: Training and Test Data created with Split ####################################")
    print("\n********** Classifiers used in this section: ")

    print("\n", (general_dict['testClassifiers'][0]))
    print("\n", (general_dict['testClassifiers'][1]))
    print("\n", (general_dict['testClassifiers'][2]))
    print("\n", (general_dict['testClassifiers'][4]))

    print("\n********** Parameters used to create the training and test data: ")
    print("Random Seed = ", general_dict['randomSeedComplex'], 
          "Test Percent = ", general_dict['testPctComplex'], 
          "Error Percent = ", general_dict['errorPctComplex'], 
          "Max Deferral Payments = ", general_dict['maxDeferralComplex'], 
          "Max Bonus = ", general_dict['maxBonusComplex'], 
          "Split Percent = ", general_dict['splitPctComplex'], ".")

    print("\n                                   Test  Len        Len        Len        Len")
    print("                                   Case  True       False      True       False")
    print("Classifier / Average Parameter     Size  Positives  Positives  Negatives  Negatives  Precision  Recall  F1") 
    print("---------------------------------  ----  ---------  ---------  ---------  ---------  ---------  ------  -----")
    
    print("GaussianNB      Average = Binary:  ", general_dict['totalTestNBComplex'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesNBComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesNBComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesNBComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesNBComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['pBinaryNBComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['rBinaryNBComplex'])), 
          " ", ("{0:.3f}".format(general_dict['f1BinaryNBComplex'])),)

    print("Decision Tree   Average = Binary:  ", general_dict['totalTestTreeComplex'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesTreeComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesTreeComplex'])), 
          "  ", ("{0:.3f}".format(general_dict['lentrueNegativesTreeComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesTreeComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['pBinaryTreeComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['rBinaryTreeComplex'])), 
          " ", ("{0:.3f}".format(general_dict['f1BinaryTreeComplex'])),)

    print("Forest          Average = Binary:  ", general_dict['totalTestForestComplex'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesForestComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesForestComplex'])), 
          "  ", ("{0:.3f}".format(general_dict['lentrueNegativesForestComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesForestComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['pBinaryForestComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['rBinaryForestComplex'])), 
          " ", ("{0:.3f}".format(general_dict['f1BinaryForestComplex'])),)

    print("SVM             Average = Binary:  ", general_dict['totalTestSVMComplex'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesSVMComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesSVMComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesSVMComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesSVMComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['pBinarySVMComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['rBinarySVMComplex'])), 
          " ", ("{0:.3f}".format(general_dict['f1BinarySVMComplex'])),)

    print(".................................  ....  .........  .........  .........  .........  .........  ......  .....")

    print("GaussianNB      Average = Macro:   ", general_dict['totalTestNBComplex'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesNBComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesNBComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesNBComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesNBComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['pMacroNBComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['rMacroNBComplex'])), 
          " ", ("{0:.3f}".format(general_dict['f1MacroNBComplex'])),)

    print("Decision Tree   Average = Macro:   ", general_dict['totalTestTreeComplex'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesTreeComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesTreeComplex'])), 
          "  ", ("{0:.3f}".format(general_dict['lentrueNegativesTreeComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesTreeComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['pMacroTreeComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['rMacroTreeComplex'])), 
          " ", ("{0:.3f}".format(general_dict['f1MacroTreeComplex'])),)

    print("Forest          Average = Macro:   ", general_dict['totalTestForestComplex'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesForestComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesForestComplex'])), 
          "  ", ("{0:.3f}".format(general_dict['lentrueNegativesForestComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesForestComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['pMacroForestComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['rMacroForestComplex'])), 
          " ", ("{0:.3f}".format(general_dict['f1MacroForestComplex'])),)

    print("SVM             Average = Macro:   ", general_dict['totalTestSVMComplex'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesSVMComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesSVMComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesSVMComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesSVMComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['pMacroSVMComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['rMacroSVMComplex'])), 
          " ", ("{0:.3f}".format(general_dict['f1MacroSVMComplex'])),)

    print(".................................  ....  .........  .........  .........  .........  .........  ......  .....")

    print("GaussianNB      Average = Micro:   ", general_dict['totalTestNBComplex'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesNBComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesNBComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesNBComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesNBComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['pMicroNBComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['rMicroNBComplex'])), 
          " ", ("{0:.3f}".format(general_dict['f1MicroNBComplex'])),)

    print("Decision Tree   Average = Micro:   ", general_dict['totalTestTreeComplex'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesTreeComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesTreeComplex'])), 
          "  ", ("{0:.3f}".format(general_dict['lentrueNegativesTreeComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesTreeComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['pMicroTreeComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['rMicroTreeComplex'])), 
          " ", ("{0:.3f}".format(general_dict['f1MicroTreeComplex'])),)

    print("Forest          Average = Micro:   ", general_dict['totalTestForestComplex'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesForestComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesForestComplex'])), 
          "  ", ("{0:.3f}".format(general_dict['lentrueNegativesForestComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesForestComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['pMicroForestComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['rMicroForestComplex'])), 
          " ", ("{0:.3f}".format(general_dict['f1MicroForestComplex'])),)

    print("SVM             Average = Micro:   ", general_dict['totalTestSVMComplex'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesSVMComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesSVMComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesSVMComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesSVMComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['pMicroSVMComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['rMicroSVMComplex'])), 
          " ", ("{0:.3f}".format(general_dict['f1MicroSVMComplex'])),)

    print(".................................  ....  .........  .........  .........  .........  .........  ......  .....")

    print("GaussianNB      Average = Weighted:", general_dict['totalTestNBComplex'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesNBComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesNBComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesNBComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesNBComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['pWeightedNBComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['rWeightedNBComplex'])), 
          " ", ("{0:.3f}".format(general_dict['f1WeightedNBComplex'])),)

    print("Decision Tree   Average = Weighted:", general_dict['totalTestTreeComplex'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesTreeComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesTreeComplex'])), 
          "  ", ("{0:.3f}".format(general_dict['lentrueNegativesTreeComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesTreeComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['pWeightedTreeComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['rWeightedTreeComplex'])), 
          " ", ("{0:.3f}".format(general_dict['f1WeightedTreeComplex'])),)

    print("Forest          Average = Weighted:", general_dict['totalTestForestComplex'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesForestComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesForestComplex'])), 
          "  ", ("{0:.3f}".format(general_dict['lentrueNegativesForestComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesForestComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['pWeightedForestComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['rWeightedForestComplex'])), 
          " ", ("{0:.3f}".format(general_dict['f1WeightedForestComplex'])),)

    print("SVM             Average = Weighted:", general_dict['totalTestSVMComplex'], 
          "  ", ("{0:.3f}".format(general_dict['lentruePositivesSVMComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalsePositivesSVMComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['lentrueNegativesSVMComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['lenfalseNegativesSVMComplex'])), 
          "   ", ("{0:.3f}".format(general_dict['pWeightedSVMComplex'])), 
          "    ", ("{0:.3f}".format(general_dict['rWeightedSVMComplex'])), 
          " ", ("{0:.3f}".format(general_dict['f1WeightedSVMComplex'])),)

    print(".................................  ....  .........  .........  .........  .........  .........  ......  .....")

    print("\nNotes: ")
    print("When the training and test data were created using")
    print("          Random Seed = ", general_dict['randomSeedComplex']) 
    print("          Test Percent = ", general_dict['testPctComplex'])
    print("          Error Percent = ", general_dict['errorPctComplex'])
    print("          Max Deferral Payments = ", general_dict['maxDeferralComplex'])
    print("          Max Bonus = ", general_dict['maxBonusComplex'])
    print("          Split Percent = ", general_dict['splitPctComplex'], ",")
    print("the best F1 score was when using the Forest Classifier and the type of average set to binary (the default).")
    print("For that row, my false negative rate is zero.")
    print("For that row, the precision score was 0.933, and the recall was perfect at 1.0.")
    print("The high recall score would normally mean that whenever a POI gets flagged in my test set, ")
    print("   * I know with a lot of confidence that it’s very likely to be a real POI, ")
    print("   * not a false alarm; ")
    print("   * and that I can identify POI’s reliably and accurately.")
    print("   * If my identifier finds a POI then the person is almost certainly a POI, ")
    print("   * and if the identifier does not flag someone, then they are almost certainly not a POI.")
    print("   * I am likely not missing the real POIs, so I can put the cuffs on the edge cases and call them in for questioning.")

    print("\nHowever, there are so few POIs in the enron dataset that the results are not perfectly reliable.")
    print("\nLater in task 6, we will run the full dataset through multiple folds / repeats, to get more accurate data.")
    print("In task 6, we will also program SelectKBest to pick the best features, and allow cvGridSearch to pick the best overall solution, to optimize the precision and recall scores.")


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
### Task 6: Dump your classifier, dataset, and features_list so anyone can # Original
### check your results. You do not need to change anything below, but make sure # Original
### that the version of poi_id.py that you submit can be run on its own and # Original
### generates the necessary .pkl files for validating your results. # Original

# Try some scaling
def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):

    """ some plotting code designed to help visualize clusters """

    # Plot each cluster with a different color--add more colors for
    # Drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    # Place red stars over points that are POIs
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.title(name)
    plt.savefig(name)
    plt.show()

def my_kmeans():
    # Use kmeans to process features into groups
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Using KMeans to cluster my features into groups %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    feature_1 = "deferral_payments"
    feature_2 = "bonus"
    poi  = "poi"
    features_list_kmeans = [poi, feature_1, feature_2]
    data_kmeans_my_dataset = general_dict['my_dataset']
    data_kmeans = featureFormat(data_kmeans_my_dataset, features_list_kmeans )
    poi_kmeans, finance_features_kmeans = targetFeatureSplit( data_kmeans )
    #print("\nin my_kmeans plt.show() basic without Draw(): ")

    for f1, f2 in finance_features_kmeans:
        plt.scatter( f1, f2 )
    
    plt.xlabel("deferral_payments")
    plt.ylabel("bonus")
    plt.title("Deferral Payments and Bonus - with my features colored")

    plt.show()
    
    # Cluster here; create predictions of the cluster labels
    # For the data and store them to a list called pred
    # Rename the "name" parameter when the number of features is changed
    
    clfDataKmeans = KMeans(n_clusters=2, random_state=0) # I know that KMeans is not a classifier, that it is used for processing features into groups
    clfDataKmeans = clfDataKmeans.fit(data_kmeans) # I added the clfDataKmeans =
    predKmeans = clfDataKmeans.predict(data_kmeans)
    #print("\nin my_kmeans with predicted cluster_centers using Draw: ", clfDataKmeans.cluster_centers_)

    try:
        Draw(predKmeans, finance_features_kmeans, poi_kmeans, mark_poi=False, name="Clusters predicted by KMeans", f1_name=feature_1, f2_name=feature_2)
    except NameError:
        print("\nin my_kmeans no predictions object named pred found, no clusters to plot")
        
    # This is on finance_features 
    clfFFP_kmeans = KMeans(n_clusters=2).fit(finance_features_kmeans)
    pred_kmeans = clfFFP_kmeans.predict(finance_features_kmeans)
    #print("\nin my_kmeans with Finance Features - Labels", clfFFP_kmeans.labels_)
    #print("\nin my_kmeans with Finance Features and Cluster Centers using Draw with POI markers: ", clfFFP_kmeans.cluster_centers_)
    
    try:
        Draw(pred_kmeans, finance_features_kmeans, poi_kmeans, mark_poi=True, name="Clusters with red POI markers", f1_name=feature_1, f2_name=feature_2)
    except NameError:
        print("\nin my_kmeans with Pred - no predictions object named pred found, no clusters to plot")

    #print("\nin my_kmeans clf: ", clfFFP_kmeans)        

def standardScaling(df):

    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Scaling my Features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # From my ref # 37: http://benalexkeen.com/feature-scaling-with-scikit-learn/
    # My code: 
    #print("\nin standardScaling(df) - df before pd.DataFrame(df): ", len(df), df) # len = 145
    """ output from above print statement:
    in standardScaling(df) - df before pd.DataFrame(df):  145 {'METTS MARK': {'salary': 365788, 'to_messages': 807, 
    'deferral_payments': 'NaN', 'total_payments': 1061827, 'exercised_stock_options': 'NaN', 'bonus': 600000, 
    'restricted_stock': 585062, 'shared_receipt_with_poi': 702, 'restricted_stock_deferred': 'NaN', 
    'total_stock_value': 585062, 'expenses': 94299, 'loan_advances': 'NaN',   
    ...
    """
    df = pd.DataFrame(df)
    #print("\nin standardScaling(df) - df after pd.DataFrame(df): ", len(df), df) # len = 21. [21 rows x 144 columns] The features are in 21 rows and the people are in 144 columns:
    """ output from above print statement:
    in standardScaling(df) - df after pd.DataFrame(df):  21                                    
                                        ALLEN PHILLIP K BADUM JAMES P  \
    bonus                                      4175000           NaN   
    deferral_payments                          2869717        178980   
    deferred_income                           -3081055           NaN   
    director_fees                                  NaN           NaN   
    email_address              phillip.allen@enron.com           NaN   
    exercised_stock_options                    1729541        257817   
    expenses                                     13868          3486   
    from_messages                                 2195           NaN   
    from_poi_to_this_person                         47           NaN   
    from_this_person_to_poi                         65           NaN   
    loan_advances                                  NaN           NaN   
    long_term_incentive                         304805           NaN   
    other                                          152           NaN   
    poi                                          False         False   
    restricted_stock                            126027           NaN   
    restricted_stock_deferred                  -126027           NaN   
    salary                                      201955           NaN   
    shared_receipt_with_poi                       1407           NaN   
    to_messages                                   2902           NaN   
    total_payments                             4484442        182466   
    total_stock_value                          1729541        257817   
    
                                       BANNANTINE JAMES M BAXTER JOHN C  \
    bonus                                             NaN       1200000   
    deferral_payments                                 NaN       1295738 
    
    ...
    [21 rows x 144 columns]
    
    """
    # Transpose the dataframe so that the columns are the features:
    df = df.T
    
    #print("\nin standardScaling(df) - df after pd.DataFrame.T(df): ", len(df), df) # len = . [144 rows x 21 columns]

    """ output now:
    ...
    WROBEL BRUCE                                      NaN                     NaN   
    YEAGER F SCOTT                                    NaN                     NaN   
    YEAP SOON                                         NaN                     NaN   
           long_term_incentive    other  \
    ALLEN PHILLIP K                      ...                     304805      152   
    BADUM JAMES P                        ...                        NaN      NaN   
    BANNANTINE JAMES M                   ...                        NaN   864523   
    ...
                                 poi restricted_stock  \
    ALLEN PHILLIP K                False           126027   
    BADUM JAMES P                  False              NaN   
    BANNANTINE JAMES M             False          1757552       
    ...
                              restricted_stock_deferred   salary  \
    ALLEN PHILLIP K                                 -126027   201955   
    BADUM JAMES P                                       NaN      NaN   
    BANNANTINE JAMES M                              -560222      477   
    ...
                              shared_receipt_with_poi to_messages  \
    ALLEN PHILLIP K                                  1407        2902   
    BADUM JAMES P                                     NaN         NaN   
    BANNANTINE JAMES M                                465         566   
    ...
                              total_payments total_stock_value  
    ALLEN PHILLIP K                      4484442           1729541  
    BADUM JAMES P                         182466            257817  
    BANNANTINE JAMES M                    916197           5243487  
    ...
    
    [145 rows x 21 columns]
    
    """
    
    # Select only the features that I want
    df = df[['poi','deferral_payments', 'bonus']]
    #print("\nin standardScaling df", len(df), df)

    """ output is:
    in standardScaling df 145                                  poi deferral_payments    bonus
    ALLEN PHILLIP K                False           2869717  4175000
    BADUM JAMES P                  False            178980      NaN
    BANNANTINE JAMES M             False               NaN      NaN
    ...
    [145 rows x 3 columns]
    
    """
    
    # Delete rows where deferral_payments and bonus are both NaN
    df = df.drop(df[(df.deferral_payments == 'NaN') & (df.bonus == 'NaN')].index)
    #print("\nin standardScaling df after deletes: ", len(df), df)

    """ output is:
    in standardScaling df after deletes:  98                                poi deferral_payments    bonus
    ALLEN PHILLIP K              False           2869717  4175000
    BADUM JAMES P                False            178980      NaN
    BAXTER JOHN C                False           1295738  1200000    ...
    """   
    
    # Replace NaN with 0. Have to convert the columns from object to float first.
    df['deferral_payments'] = df['deferral_payments'].astype(str).astype(float)
    df['bonus'] = df['bonus'].astype(str).astype(float)
    df = df.fillna(0) 
    #print("\nin standardScaling df after replacing NaN: ",  len(df), df)

    """ output is:
    in standardScaling df after replacing NaN:  98                                poi  deferral_payments      bonus
    ALLEN PHILLIP K              False          2869717.0  4175000.0
    BADUM JAMES P                False           178980.0        0.0
    BAXTER JOHN C                False          1295738.0  1200000.0
    ...
    [97 rows x 3 columns]
    """
    
    np.random.seed(1)
    scaler = preprocessing.MinMaxScaler()
    scaled_df = df[['deferral_payments', 'bonus']] 
    scaled_df = scaler.fit_transform(scaled_df)
    #print("\nin standardScaling(df) - scaled_df after fit_transform: ", len(scaled_df), scaled_df)
    """ output is
    in standardScaling(df) - scaled_df after fit_transform:  98 [[ 0.45519895  0.521875  ]
    [ 0.04310903  0.        ]
    [ 0.21414199  0.15      ]
    ..., 
    [ 0.14301255  0.040625  ]
    [ 0.01569801  0.375     ]
    [ 0.01569801  0.05625   ]]
    """
    scaled_df = pd.DataFrame(scaled_df, columns=['deferral_payments', 'bonus'], index=df.index)
    scaled_df.columns = ['scaled_deferral_payments', 'scaled_bonus']
    #print("\nin standardScaling(df) - scaled_df after pd.DataFrame(scaled_df...: ", len(scaled_df), scaled_df)
    
    """ output is:
    in standardScaling(df) - scaled_df after pd.DataFrame(scaled_df...:  
    98                              scaled_deferral_payments  scaled_bonus
    ALLEN PHILLIP K                              0.455199      0.521875
    BADUM JAMES P                                0.043109      0.000000
    BAXTER JOHN C                                0.214142      0.150000
    BAY FRANKLIN R                               0.055587      0.050000
    ...
    """
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
    
    ax1.set_title('Before Scaling')
    sns.kdeplot(df['deferral_payments'], ax=ax1)
    sns.kdeplot(df['bonus'], ax=ax1)
    ax2.set_title('After Standard Scaler')
    sns.kdeplot(scaled_df['scaled_deferral_payments'], ax=ax2)
    sns.kdeplot(scaled_df['scaled_bonus'], ax=ax2)
    plt.show() # figures are a little different before and after scaling.
    
    dfAndScaled = pd.concat([df, scaled_df], axis=1) # Combine df and scaled_df
    #print("\nafter merging the scaled_df to df:" , len(dfAndScaled), dfAndScaled)

    """ output is:
    after merging the scaled_df to df: 98                                poi  deferral_payments      bonus  \
    ALLEN PHILLIP K              False          2869717.0  4175000.0   
    BADUM JAMES P                False           178980.0        0.0 
    ...
                             scaled_deferral_payments  scaled_bonus  
    ALLEN PHILLIP K                              0.455199      0.521875  
    BADUM JAMES P                                0.043109      0.000000  
    ...
    WHALLEY LAWRENCE G                           0.015698      0.375000  
    WHITE JR THOMAS E                            0.015698      0.056250  

    [98 rows x 5 columns]
    """
    
    # Convert the data back to a dictionary:
    # I need the column headings
    scaled_df = scaled_df.T
    """ output is:
    after making scaled_df a dict:  2                           ALLEN PHILLIP K  BADUM JAMES P  BAXTER JOHN C  \
    scaled_deferral_payments         0.455199       0.043109       0.214142   
    scaled_bonus                     0.521875       0.000000       0.150000   
    """
    scaled_df = scaled_df.to_dict('dict')
    #print("\nafter making scaled_df a dict: ", len(scaled_df), scaled_df)
    """ output is:
    after making scaled_df a dict:  98 {'METTS MARK': {'scaled_bonus': 0.074999999999999997, 
    'scaled_deferral_payments': 0.015698010104923971}, 'BAXTER JOHN C': ...
    """
    # That's good. But I also need 'poi', so I want to transform dfAndScaled to dict
    dfAndScaled = dfAndScaled.T
    dfAndScaled = dfAndScaled.to_dict('dict')
    #print("\nafter making dfAndScaled a dict: ", len(dfAndScaled), dfAndScaled)
    """ output is:
    after making dfAndScaled a dict:  98 {'METTS MARK': {'bonus': 600000.0, 'scaled_bonus': 0.075, 'deferral_payments': 0.0, 
    'scaled_deferral_payments': 0.01569801010492397, 'poi': False}, 'BAXTER JOHN C': ...    
    """

    # Add these databases to the dictionary
    general_dict.update({'scaled_df':scaled_df, 'dfAndScaled':dfAndScaled})

    print("\nStandard Scaling does not significantly improve the relationship between my features of Deferral Payments and Bonus.")
    print("")
    
# My ref # 38 for code assistance from my mentor, Myles: https://discussions.udacity.com/t/error-using-pipeline/171750/4
def task6FromScratch():
    
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SelectKBest and GridSearch %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("\n########################################################### Notes ##########################################################")
    print("\nThis is task 6 in the Data Analytics Nanodegree Machine Learning Final Project.")

    print("\nGoing through all of the above machinations in the prior sections of the course have helped me to understand: ")
    print("          * that I should not rely on my intuition or on guesswork to select the best features to use in analysing data,")
    print("          * that the SVM classifier is not the best classifier for the Enron dataset because of the few number of POIs in")
    print("            the dataset,")
    print("          * how the GaussianNB, Tree and Forest classifiers behave when sklearn cross_validation is used versus using a") 
    print("            more manual split option,")
    print("          * how regression analysis is useful to highlight how the features that I had selected were not correlated, and")
    print("            that I could have used that earlier in the project to eliminate that combination of features,")
    print("          * how scaling would be useful in some situations, but was not useful for the features that I selected, because")
    print("            both features were already aligned with the numbering that they used,")
    print("          * how to use loops to create variables in a general dictionary so that I can call them to put on reports, and more")
    print("            easily compare classifiers and statistics,")
    print("          * per the first reviewer, that I should name my definitions with underscores between the words,")
    print("            and not use capital letters to separate the words; I will do that in future code. I did not change this code.")
    print("          * That Spyder has a fundamental flaw where shadow code is kept at the EOF even after it is deleted. That the")
    print("            workaround is to put 'from __future__ import print_function' at the top of my code. However, it did not")
    print("            completely get rid of the problem, and I constantly had to save the file as a new version in order to run it")
    print("            again.")

    print("\nIn going through the final task 6, on which the course tester.py evaluates my results, I realized:")
    print("          * that SelectKBest in combination with GridSearch is much better at automatically picking the best features and the")
    print("            number of features in order to optimize the precision and recall scores,")
    print("          * that 'THE TRAVEL AGENCY IN THE PARK'is an outlier that needs to be excluded,")
    print("          * and that adding in my new features: 'scaled_deferral_payments', 'scaled_bonus', 'ratio of deferral_payments to bonus'")
    print("            did not affect the outcome of the best features that were selected by SelectKBest and GridSearch.")
    print("            However they negatively impacted the results of my DecisionTree classifier. When run without those 3 new features,")
    print("            the precision and recall scores were above 0.3, but when run WITH those features included in my dataset, ")
    print("            the recall score dropped BELOW 0.3. So I excluded those 3 features in my final dataset.")
    print("          * That the classifier GaussianNB does not have any parameters to be used for GridSearch, but that I can still use")
    print("            SelectKBest in the GridSearch pipeline,")
    print("          * and that in order for GaussianNB to have precision and recall > 0.3, I need to set the SelectKBest range to (1,10).")
    print("            When I do that, GaussianNB selects 5 features to get a precision and recall score > 0.3, whereas Decision Tree")
    print("            only required 3 features.")

    print("\nIn this part of the final project, I am selecting all of the features in the Enron dataset, except the email addresses, ")
    print("and allowing SelectKBest in combination with GridSearch to select the best fit.")

    print("\nIn the end, GridSearch selected 3 features as optimal, none of which were the 2 features that I had selected in the prior sections of the project.")
    
    print("\nIn task 6 I chose to use DecisionTree as my classifier, but I could have chosen GaussianNB and had precision and recall that would have met the tester.py threshhold required to pass the project.")

    print("\nI learned that GridSearch takes time to process through all of the combinations of features, and so I put in a print")
    print("statement to show that the machine is working, so that the user will not give up.")

    print("\nFor DecisionTree classifier, GridSearch selected 3 features, and I wondered if it was because I had set 'kbest__k': range(1,4), which")
    print("only allows up to 3 features to be selected. So, I temporarily changed it to 'kbest__k': range(1,10), to")
    print("allow up to 9 features, but GridSearch still select the same 3 features, and took a much longer time to go through all")
    print("of the possible combinations. So, I set 'kbest__k' back to range(1,4) to reduce the processing time.")
    
    with open("final_project_dataset.pkl", "r") as data_file: # Original
        data_dict = pickle.load(data_file) # Original
    
    # Delete the "TOTAL" and other main outlier
    data_dict.pop('TOTAL', 0)
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

    feature_list_task6 = ['poi','salary', 'to_messages', 'deferral_payments', 
    'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 
    'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 
    'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 
    'from_poi_to_this_person']

    if False: # Set to True to see how adding my extra features influences the results.
        print("\n################################################### Adding extra features ##################################################")
        # When I add these 3 additional features into my dataset, the precision and recall scores drop below 0.3, 
        # so I will exclude them though the if False statement.
        # Add scaled deferral_payments and scaled_bonus to data_dict
        # Convert the dictionary to a dataframe for scaling   
        df = pd.DataFrame(data_dict)
    
        # Transpose the dataframe so that the columns are the features:
        df = df.T
    
        # Replace NaN with 0. Have to convert the columns from object to float first.
        df['deferral_payments'] = df['deferral_payments'].astype(str).astype(float)
        df['bonus'] = df['bonus'].astype(str).astype(float)
        df = df.fillna(0)
    
        np.random.seed(1)
        scaler = preprocessing.MinMaxScaler()
        scaled_df = df[['deferral_payments', 'bonus']] 
        scaled_df = scaler.fit_transform(scaled_df)
    
        scaled_df = pd.DataFrame(scaled_df, columns=['deferral_payments', 'bonus'], index=df.index)
        scaled_df.columns = ['scaled_deferral_payments', 'scaled_bonus']
    
        # I don’t think I need this one:
        dfAndScaled = pd.concat([df, scaled_df], axis=1) # Combine df and scaled_df
    
        #scaled_df = scaled_df.T
        dfAndScaled = dfAndScaled.T
    
        #scaled_df = scaled_df.to_dict('dict')
        dfAndScaled = dfAndScaled.to_dict('dict')
    
        #data_dict = scaled_df
        data_dict = dfAndScaled
    
        # Add a new feature to data_dict for the ratio between deferral_payments and bonus
        # My ref #41 in my reference file: https://www.quora.com/In-Python-can-we-add-a-dictionary-inside-a-dictionary-If-yes-how-can-we-access-the-inner-dictionary-using-the-key-in-the-primary-dictionary
        for key,values in data_dict.items():
            print("")
            if data_dict[key]['bonus'] <> 0.:
                data_dict[key]['ratio_dp_b'] = data_dict[key]['deferral_payments'] / data_dict[key]['bonus']
            else:
                data_dict[key]['ratio_dp_b'] = 0
            #print(key, values)
            print(key, data_dict[key]['deferral_payments'], data_dict[key]['bonus'], data_dict[key]['scaled_deferral_payments'], 
                  data_dict[key]['scaled_bonus'], data_dict[key]['ratio_dp_b'])

        feature_list_task6 = ['poi','salary', 'to_messages', 'deferral_payments', 
                                  'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 
                                  'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 
                                  'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 
                                  'from_poi_to_this_person', 'scaled_deferral_payments', 'scaled_bonus', 'ratio_dp_b']

    my_dataset_task6 = data_dict

    # Add the new features to the list of all of the enron features. Exclude the email_address feature
    
    #print("\n BEFORE GridSearch")
    #test_classifier(tree.DecisionTreeClassifier( random_state = 1), my_dataset_task6, feature_list_task6, folds = 100)
    """ output is:
    BEFORE GridSearch
    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='best')
        Accuracy: 0.79600       Precision: 0.24519      Recall: 0.25500 F1: 0.25000     F2: 0.25298
        Total predictions: 1500 True positives:   51    False positives:  157   False negatives:  149   True negatives: 1143
    """

    general_dict.update({'my_dataset_task6':my_dataset_task6, 'feature_list_task6':feature_list_task6})
    
    data = featureFormat(my_dataset_task6, feature_list_task6, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    print("\n######################################## GaussianNB with SelectKBest and Grid Search #######################################")
    print("\nGridSearch is processing GaussianNB")

    # SelectKBest and GridSearch with GaussianNB

    clfNB = GaussianNB()
    # At this link, GaussianNB has no parameters: https://discussions.udacity.com/t/gaussiannb-parameters/22982
    parameters = {'kbest__k': range(1,10)}
    # use scaling in GridSearchCV
    Min_Max_scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    pipeline = Pipeline(steps=[('scaler', Min_Max_scaler), ('kbest', SelectKBest()), ('clfNB', clfNB)])
    cv = StratifiedShuffleSplit(labels, 100, random_state = 42) 
    gs = GridSearchCV(pipeline, parameters, cv=cv, scoring='f1')

    gs.fit(features, labels)
    clf = gs.best_estimator_
    general_dict.update({'clf_best_NB':clf})

    print("\nAfter GridSearch for GaussianNB, here is the Tester Classification report, which shows that" )
    print("it has selected 5 features and precision and recall are greater than 0.3.")
    print("")
    test_classifier(clf, my_dataset_task6, feature_list_task6, folds=1000)
    
    K_best = gs.best_estimator_.named_steps['kbest']

    # Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
    feature_scores = ['%.2f' % elem for elem in K_best.scores_ ]
    # Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
    feature_scores_pvalues = ['%.3f' % elem for elem in  K_best.pvalues_ ]
    # Get SelectKBest feature names, whose indices are stored in 'K_best.get_support',
    # create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"

    # With my feature_list_task6
    #print("\nfeature_list_task6: ", feature_list_task6) # Prints all 21 features, including poi as the first feature.
    features_selected_tuple=[(feature_list_task6[j+1], feature_scores[j], feature_scores_pvalues[j]) for j in K_best.get_support(indices=True)]
    
    # Sort the tuple by score, in reverse order
    features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)
    print("Here are the features selected for GaussianNB, and the score / pvalue for each: \n", features_selected_tuple)
    
    """ the output is: 
    After GridSearch for GaussianNB, here is the Tester Classification report, which shows that
    it has selected 5 features and precision and recall are greater than 0.3.

    Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('kbest', SelectKBest(k=5, 
    score_func=<function f_classif at 0x000000000ECF74A8>)), ('clfNB', GaussianNB(priors=None))])
    Accuracy: 0.84833       Precision: 0.41964      Recall: 0.35900 F1: 0.38696     F2: 0.36968
    Total predictions: 15000        True positives:  718    False positives:  993   False negatives: 1282   
    True negatives: 12007

    Here are the features selected for GaussianNB, and the score / pvalue for each: 
    [('total_stock_value', '14.69', '0.000'), ('exercised_stock_options', '13.71', '0.000'), 
    ('salary', '11.20', '0.001'), ('bonus', '11.13', '0.001'), ('restricted_stock', '6.58', '0.012')]

    """
    
    print("\n######################################### DecisionTree with SelectKBest and Grid Search ########################################")
    print("\nGridSearch is processing DecisionTree")

    # SelectKBest and GridSearch with DecisionTree
    clfTree = tree.DecisionTreeClassifier()

    parameters = {'clfTree__criterion': ['gini','entropy'],
              'clfTree__splitter':['best','random'],
              'clfTree__min_samples_split':[2, 10, 20],
                'clfTree__max_depth':[10,15,20,25,30],
                'clfTree__max_leaf_nodes':[5,10,30],
              'clfTree__random_state': [42],
              'kbest__k': range(1,4)
                }
    # use scaling in GridSearchCV
    Min_Max_scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    pipeline = Pipeline(steps=[('scaler', Min_Max_scaler), ('kbest', SelectKBest()), ('clfTree', clfTree)])
    cv = StratifiedShuffleSplit(labels, 100, random_state = 42) 
    gs = GridSearchCV(pipeline, parameters, cv=cv, scoring='f1')

    gs.fit(features, labels)
    clf = gs.best_estimator_
    general_dict.update({'clf_best':clf})


    print("\nAfter GridSearch with DecisionTree classifier, here is the Tester Classification report, which clearly shows that" )
    print("precision and recall are greater than 0.3 as required by the final project evaluation.")
    print("")
    test_classifier(clf, my_dataset_task6, feature_list_task6, folds = 1000)
  
    # Code assistance in my ref #40: https://discussions.udacity.com/t/selectpercentile-how-to-fill-the-parameters/174780/15
    
    K_best = gs.best_estimator_.named_steps['kbest']
    
    # Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
    feature_scores = ['%.2f' % elem for elem in K_best.scores_ ]
    # Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
    feature_scores_pvalues = ['%.3f' % elem for elem in  K_best.pvalues_ ]
    # Get SelectKBest feature names, whose indices are stored in 'K_best.get_support',
    # create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"

    # With my feature_list_task6
    #print("\nfeature_list_task6: ", feature_list_task6) # Prints all 21 features, including poi as the first feature.
    features_selected_tuple=[(feature_list_task6[j+1], feature_scores[j], feature_scores_pvalues[j]) for j in K_best.get_support(indices=True)]
    
    # Sort the tuple by score, in reverse order
    features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)
    print("Here are the features selected, and the score / pvalue for each: \n", features_selected_tuple)

    """ The output is:
        After GridSearch with DecisionTree classifier, here is the Tester Classification report, which clearly shows that
        precision and recall are greater than 0.3 as required by the final project evaluation.

        Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('kbest', SelectKBest(k=3, 
                        score_func=<function f_classif at 0x000000000ECF74A8>)), ('clfTree', 
            DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=10,
            max_features=None, max_leaf_nodes=30, min_impurity_split=1e-07,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='random'))])
        Accuracy: 0.83680       Precision: 0.38297      Recall: 0.36650 F1: 0.37455     F2: 0.36968
        Total predictions: 15000        True positives:  733    False positives: 1181   False negatives: 1267   
        True negatives: 11819

        Here are the features selected, and the score / pvalue for each: 
            [('total_stock_value', '14.69', '0.000'), ('exercised_stock_options', '13.71', '0.000'), ('salary', '11.20', '0.001')]
    """

# Choose my classifier
def task6Dump():
    # Create my_dataset
    
    my_dataset = general_dict['my_dataset_task6']
    features_list = general_dict['feature_list_task6'] 
    
    clf = general_dict['clf_best']
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Dump of my clf, feature_list and dataset for tester.py %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    print("\nMy clf as shown in the Pipeline:\n\n", clf)
    print("\nMy features list to input to the tester.py for evaluation and selection of best features:\n\n", features_list)
    print("\nThe length of my dataset, showing that the full enron dataset is being used, except for the TOTAL and ")
    print("the 'THE TRAVEL AGENCY IN THE PARK' outlier: ", len(my_dataset))
    #print("\nThe beginning and ending rows of my dataset:\n\n", my_dataset) # Use this to see the full 144 records

    print("\nThis is the end of my project code and report. I hope you enjoyed reading it.")

    """ output is:
    My clf as shown in the Pipeline:

    Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('kbest', SelectKBest(k=3, 
                    score_func=<function f_classif at 0x000000000ECF74A8>)), 
            ('clfTree', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=10,
            max_features=None, max_leaf_nodes=30, min_impurity_split=1e-07,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='random'))])

    My features list to input to the tester.py for evaluation and selection of best features:

        ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 
        'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 
        'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 
        'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

    The length of my dataset, showing that the full enron dataset is being used, except for the TOTAL and 
    the 'THE TRAVEL AGENCY IN THE PARK' outlier:  144    
    
    This is the end of my project code and report. I hope you enjoyed reading it.")
    """
    
    dump_classifier_and_data(clf, my_dataset, features_list) # Original 
    ##### I need the above line as part of my final code

def main():
    myDataset() # Checked
    defineFeatures() # Checked
    removeOutlierTotal(data_dict) # Checked 
    formatSplit(data_dict) # Formats and Splits the data. Checked
    copyDataNewFeaturesRemoveOutlierEtc() # Checked
    defineVariables1() # Checked
    reportFeaturesOutliersEtc() # Checked
    graphData() # Checked
    defineVariables2() # Checked
    defineVariables3() # Checked
    #def basicTrainTestData(data, labels, testPct, randomSeed, svcKernel, svcC): # I put this here so that I could see the order of the def parameters.
    basicTrainTestData(dataFinalFloat, labelsFinal, 0.3, 42, "linear", 1.) # Checked
    reportBasicClf() # Checked
    # def complexTrainTestData(data, randomSeed, testPct, errorPct, maxDeferral, maxBonus, splitPct): # I put this here so that I could see the order of the def parameters.
    complexTrainTestData(dataFinal, 42, 0.3, 0.1, 0.8, 0.8, 0.75) # Checked - See prior row for what the numbers in brackets represent.
    reportComplexTrainTestData() # Checked
    regressionDefBonus() # Checked
    myValidation() # Checked
    myValidationReport() # Checked
    my_kmeans() # Checked
    standardScaling(general_dict['my_dataset'])
    task6FromScratch() 
    """ I used this for my other classifiers, but before I did gsGridSearch, none of them returned precision and recall > 0.3
    x = [0, 1, 2, 3, 4]
    for x in x:
        task6Dump(x) # if I use this code, I have to add (x) back to the def task6Dump
        print("\n############ from Udacity tester.py #####################\n")
        #print("\nBefore test_classifier: classifier, len data and feature list: ", general_dict['testClassifiers'][x], len(general_dict['dfAndScaled']), general_dict['features_list']) 
        test_classifier(general_dict['testClassifiers'][x], general_dict['dfAndScaled'], general_dict['features_list'], folds = 1000) # from tester.py
    """
    task6Dump()
    
if __name__ == "__main__":  # Always put this in my programs
    main()

