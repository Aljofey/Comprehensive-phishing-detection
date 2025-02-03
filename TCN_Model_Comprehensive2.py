from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import SpatialDropout1D
from tensorflow import keras
#from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dropout, Add, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
#from keras.layers.merge import concatenate
from keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling2D,Concatenate
import seaborn as sns
import requests
import psycopg2
import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
#from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional
from tcn import TCN, tcn_full_summary
from matplotlib import pyplot
import matplotlib
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import GRU
from sklearn.feature_extraction.text import TfidfVectorizer
from keras import regularizers
from keras.preprocessing import sequence
from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import LSTM, Reshape, Bidirectional
from keras.layers import Activation, BatchNormalization
from keras.layers import AveragePooling1D, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import (EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout, GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense, TimeDistributed)
#from keras_self_attention import SeqSelfAttention
from tensorflow.keras.utils import to_categorical, custom_object_scope
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm, tqdm_pandas
import scipy
from scipy.stats import skew
##import librosa
##import librosa.display
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import seaborn as sns
import glob 
import os
import sys
import pickle
#import IPython.display as ipd  # To play sound in the notebook
import warnings
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.callbacks import EarlyStopping
#from keras_transformer import get_model, decode
#from transformers import BertTokenizer, TFBertModel,TFBertForSequenceClassification
#import transformers
import tensorflow_hub as hub
from keras.utils.np_utils import to_categorical

import itertools
#from tensorflow_model_optimization.sparsity import keras as sparsity
#import absl.flags
import csv
import torch
import tokenization
from bert import tokenization
import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, f1_score, recall_score, roc_curve
import xgboost as xgb
from tensorflow.keras import backend as k
from tensorflow.keras.layers import  Attention
from keras.layers import AlphaDropout
from tensorflow.keras.layers import  MultiHeadAttention
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
from absl import flags
sys.argv=['preserve_unused_tokens=False']
flags.FLAGS(sys.argv)
with tf.device('/gpu:0'):

        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['font.size'] = 14
        sns.set(font_scale=1.4)

        good_plain,good_noicy, good_hy, good_urls = [], [],[],[]
        bad_plain, bad_noicy, bad_hy, bad_urls = [], [],[],[]
        labels, texts=[],[]
        train_u, train_h=[],[]
        test_u, test_h=[],[]
        

##        #old reading
##        with open('C:\\Users\\ps\\Documents\\datasets\\dataset1.txt', 'r') as file:
##            for line in file:
##                columns = line.strip().split('|')
##                texts.append(columns[0])
##                if (columns[1]=='+1'):
##                    labels.append('1')
##                else:
##                    labels.append('0')


        import pandas as pd

##        #reading the second dataset
##        # Define paths
##        file_path ='C:\\Users\\ps\\Documents\\datasets\\the third paper\\archive\\phishing_site_urls.csv' # Update to the actual CSV file path
##        # Read the CSV file
##        df = pd.read_csv(file_path)
##        # Check the structure of the CSV
##        print(f"Number of rows: {df.shape[0]}")
##        print(df.head())  # Check the first few rows to ensure proper formatting
##        
##        if 'URL' in df.columns and 'Label' in df.columns:
##           # Process the CSV rows
##           for index, row in df.iterrows():
##               texts.append(row['URL'])
##       
##           # Convert 'bad' -> 1, 'good' -> 0
##               label = row['Label'].strip().lower()
##               if label == 'bad':
##                  labels.append(1)
##               elif label == 'good':
##                  labels.append(0)
##               else:
##                  print(f"Invalid label at row {index}: {label}")
##        else:
##              print("Error: CSV file does not contain 'URL' or 'Label' columns.")
##
##        # Print the size of the lists
##        print(f"Number of texts: {len(texts)}")
##        print(f"Number of labels: {len(labels)}")


        
###spilit the data
##        good_data = pd.read_csv('C:\\Users\\ps\\Documents\\datasets\\DatasetwithHTML\\D333_good.csv')
##        bad_data = pd.read_csv('C:\\Users\\ps\\Documents\\datasets\\DatasetwithHTML\\D333_phishing.csv')
##        good_data['label'] = 0
##        bad_data['label'] = 1
##        combined_data = pd.concat([good_data, bad_data], ignore_index=True)
####        for index, row in combined_data.iterrows():
####            match = re.match(r'(\d+)', row[0])
####            numeric_part = match.group(1)
####            combined_data.at[index, 'id'] = numeric_part
##            
##        combined_data = combined_data.values.astype(float)
##        scaler = StandardScaler()
##        X = scaler.fit_transform(combined_data)
##        X_train, X_test, y_train, y_test = train_test_split(combined_data[:, :-1], combined_data[:, -1], test_size=0.2, random_state=42)
##        encoder = preprocessing.LabelEncoder()
##        y_train = encoder.fit_transform(y_train)
##        y_test = encoder.fit_transform(y_test)
##        y_train1 = to_categorical(y_train)
##        y_test1 = to_categorical(y_test)
##        folder_path1 ="C:\\Users\ps\\Documents\\datasets\\D3333_good"
##        items1= os.listdir(folder_path1)
##        folder_path2 ="C:\\Users\\ps\\Documents\\datasets\\D333_bad"
##        items2= os.listdir(folder_path2)
##        j=0
##        for i in X_train:
##            h=0
##            u=0
##            if  y_train[j]==0:
##                  for item in items1:
##                      item_path1 = os.path.join(folder_path1, item)
##                      match = re.match(r'(\d+)',item)
##                      numeric_part = match.group(1)
##                      if int(i[0]) == int(numeric_part):
####                        if "h" not in str(item) and  "u" not in str(item)  :
####                            h=1
####                            with open(item_path1, 'rb') as file1:
####                              train_h.append(file1.read().decode('utf-8', errors='ignore'))
####                              #print(item)
##                        if "u" in str(item):
##                            u=1
##                            with open(item_path1, 'rb') as file2:
##                              train_u.append(file2.read().decode('utf-8', errors='ignore'))
##                              #print(item)
##                        #if h==1 and u==1:
##                        if u==1:
##                            break;
##               
##                    
##            if  y_train[j]==1:
##                      for item in items2:
##                          item_path2 = os.path.join(folder_path2, item)
##                          match = re.match(r'(\d+)',item)
##                          numeric_part = match.group(1)
##                          if int(i[0]) == int(numeric_part):
####                            if "h" not in str(item) and  "u" not in str(item)  :
####                                h=1
####                                with open(item_path2, 'rb') as file3:
####                                   train_h.append(file3.read().decode('utf-8', errors='ignore'))
####                                   #print(item)
##                            if "u" in str(item):
##                                u=1
##                                with open(item_path2, 'rb') as file4:
##                                  train_u.append(file4.read().decode('utf-8', errors='ignore'))
##                                  #print(item)
##                            if  u==1:
##                                break;
##            j=j+1
##
##        j=0
##        for i in X_test:
##            h=0
##            u=0
##            if  y_test[j]==0:
##                  for item in items1:
##                      item_path1 = os.path.join(folder_path1, item)
##                      match = re.match(r'(\d+)',item)
##                      numeric_part = match.group(1)
##                      if int(i[0]) == int(numeric_part):
####                        if "h" not in str(item) and  "u" not in str(item)  :
####                            h=1
####                            with open(item_path1, 'rb') as file1:
####                              test_h.append(file1.read().decode('utf-8', errors='ignore'))
####                              #print(item)
##                        if "u" in str(item):
##                            u=1
##                            with open(item_path1, 'rb') as file2:
##                              test_u.append(file2.read().decode('utf-8', errors='ignore'))
##                              #print(item)
##                        if u==1:
##                            break;
##               
##                    
##            if  y_test[j]==1:
##                      for item in items2:
##                          item_path2 = os.path.join(folder_path2, item)
##                          match = re.match(r'(\d+)',item)
##                          numeric_part = match.group(1)
##                          if int(i[0]) == int(numeric_part):
####                            if "h" not in str(item) and  "u" not in str(item)  :
####                                h=1
####                                with open(item_path2, 'rb') as file3:
####                                   test_h.append(file3.read().decode('utf-8', errors='ignore'))
####                                   #print(item)
##                            if "u" in str(item):
##                                u=1
##                                with open(item_path2, 'rb') as file4:
##                                  test_u.append(file4.read().decode('utf-8', errors='ignore'))
##                                  #print(item)
##                            if u==1:
##                                break;
##            j=j+1
##
##        train_u = pd.DataFrame({'train_u': train_u})
##        #train_h = pd.DataFrame({'train_h': train_h})
##        #test_h = pd.DataFrame({'test_h': test_h})
##        test_u = pd.DataFrame({'test_u': test_u})
##        X_train= pd.DataFrame(X_train, columns=["id","URLLength1","DigitAlphabetRatio1",  "SpecialcharAlphabetRatio1","UppercaseLowercaseRatio1", "DomainURLRatio1", "NumericCharCount1",
##                                                "EnglishLetterCount1",  "SpecialCharCount1",  "DotCount1",  "SemiColCount1" , "UnderscoreCount1","QuesMarkCount1",
##                                                "HashCharCount1", "EqualCount1",
##                                                "PercentCharCount1",  "AmpersandCount1",  "DashCharCount1", "DelimiterCount1",  "AtCharCount1","TildeCharCount1",
##                                                "DoubleSlashCount1", "HostNameLength1","QueryLength1", "HttpsInHostName1","TLDInSubdomain1",
##                                                "TLDInPath1", "HttpsInUrl1", "IsDomainEnglishWord1",
##                                                "Unigram1", "Bigram1",  "Trigram1", "count_subdomains1",  "measure_subdirectory_depth1", "detect_url_encoding1",  "keyword_analysis1",
##                                                "is_legitimate_tld1",
##                                                "analyze_hyphen_distribution1", "has_unique_identifiers1", "count_parameters1",  "contains_javascript1", "contains_brand_keywords1",
##                                                "contains_leetspeak1",  "measure_path_length1",  "resource_type_detection1", "ImgCount1",    "TotalLinks1",	"TitleCheck1",
##                                                "CheckIframeOrFrame1",  "CheckPopupCommands1",  "CountBodyTags1",   "CountMetaTags1", "CountDivTags1", "MeasureNodeDepth1",
##                                                "Sibling", "DetectSemanticElements1-headings", "DetectSemanticElements1-lists", "CountInternalAndExternalLinks1-internal_link_count",
##                                                "CountInternalAndExternalLinks1-external_link_count", "CountInternalAndExternalLinks1-link_ratio",  "count_embedded_content1-images",
##                                                "count_embedded_content1-iframes", "char_count", "word_count","sum_density", "broken", "empty",	"count_open_graph_tags1",
##                                                "form_count", "analyze_form_action_urls1",  "count_hidden_form_fields1",  "inline_scripts", "match",  "total_tags",   "raw_word_count",
##                                                "avg_length", "shortest_length",	"longest_length",	"std_deviation",	"adjacent_word_count",	"avg_length", "separated_count",
##                                                "count_random",
##                                                "lengths_Domain_Length",	"lengths_Subdomain_Length",   "lengths_Path_Length" ,"results_www_in_domain",
##                                                "results_com_in_domain",
##                                                "results_www_in_subdomain",   "has_consecutive_char_repeats1"])
##        X_test=pd.DataFrame(X_test, columns=["id","URLLength1","DigitAlphabetRatio1",  "SpecialcharAlphabetRatio1","UppercaseLowercaseRatio1", "DomainURLRatio1", "NumericCharCount1",
##                                                "EnglishLetterCount1",  "SpecialCharCount1",  "DotCount1",  "SemiColCount1" , "UnderscoreCount1","QuesMarkCount1",
##                                                "HashCharCount1", "EqualCount1",
##                                                "PercentCharCount1",  "AmpersandCount1",  "DashCharCount1", "DelimiterCount1",  "AtCharCount1","TildeCharCount1",
##                                                "DoubleSlashCount1", "HostNameLength1","QueryLength1", "HttpsInHostName1","TLDInSubdomain1",
##                                                "TLDInPath1", "HttpsInUrl1", "IsDomainEnglishWord1",
##                                                "Unigram1", "Bigram1",  "Trigram1", "count_subdomains1",  "measure_subdirectory_depth1", "detect_url_encoding1",  "keyword_analysis1",
##                                                "is_legitimate_tld1",
##                                                "analyze_hyphen_distribution1", "has_unique_identifiers1", "count_parameters1",  "contains_javascript1", "contains_brand_keywords1",
##                                                "contains_leetspeak1",  "measure_path_length1",  "resource_type_detection1", "ImgCount1",    "TotalLinks1",	"TitleCheck1",
##                                                "CheckIframeOrFrame1",  "CheckPopupCommands1",  "CountBodyTags1",   "CountMetaTags1", "CountDivTags1", "MeasureNodeDepth1",
##                                                "Sibling", "DetectSemanticElements1-headings", "DetectSemanticElements1-lists", "CountInternalAndExternalLinks1-internal_link_count",
##                                                "CountInternalAndExternalLinks1-external_link_count", "CountInternalAndExternalLinks1-link_ratio",  "count_embedded_content1-images",
##                                                "count_embedded_content1-iframes", "char_count", "word_count","sum_density", "broken", "empty",	"count_open_graph_tags1",
##                                                "form_count", "analyze_form_action_urls1",  "count_hidden_form_fields1",  "inline_scripts", "match",  "total_tags",   "raw_word_count",
##                                                "avg_length", "shortest_length",	"longest_length",	"std_deviation",	"adjacent_word_count",	"avg_length", "separated_count",
##                                                "count_random",
##                                                "lengths_Domain_Length",	"lengths_Subdomain_Length",   "lengths_Path_Length" ,"results_www_in_domain",
##                                                "results_com_in_domain",
##                                                "results_www_in_subdomain",   "has_consecutive_char_repeats1"])
##        y_train=pd.DataFrame(y_train, columns=["label"])
##        y_test=pd.DataFrame(y_test, columns=["label"])
##       
##        
##        
##        
##        train_u.to_csv("C:\\Users\\ps\\Documents\\datasets\\train_u2.csv", index=False)
##        #train_h.to_csv("C:\\Users\\ps\\Documents\\datasets\\train_h1.csv", index=False)
##        test_u.to_csv("C:\\Users\\ps\\Documents\\datasets\\test_u2.csv", index=False)
##        #test_h.to_csv("C:\\Users\\ps\\Documents\\datasets\\test_h2.csv", index=False)
##        X_train.to_csv("C:\\Users\\ps\\Documents\\datasets\\X_train2.csv", index=False)
##        X_test.to_csv("C:\\Users\\ps\\Documents\\datasets\\X_test2.csv", index=False)
##        y_train.to_csv("C:\\Users\\ps\\Documents\\datasets\\y_train2.csv", index=False)
##        y_test.to_csv("C:\\Users\\ps\\Documents\\datasets\\y_test2.csv", index=False)


##        test_h = pd.read_csv('C:\\Users\\ps\\Documents\\datasets\\test_h2.csv')
##        train_h = pd.read_csv('C:\\Users\\ps\\Documents\\datasets\\train_h2.csv')

        
####      # Load your data and spilit the data using 5-fold cross-validation
        from sklearn.model_selection import StratifiedKFold
##        # Load data
##        train_u = pd.read_csv('C:\\Users\\ps\\Documents\\datasets\\train_u2.csv')
##        test_u = pd.read_csv('C:\\Users\\ps\\Documents\\datasets\\test_u2.csv')
##        X_train = pd.read_csv('C:\\Users\\ps\\Documents\\datasets\\X_train2.csv')
##        X_test = pd.read_csv('C:\\Users\\ps\\Documents\\datasets\\X_test2.csv')
##        y_train = pd.read_csv('C:\\Users\\ps\\Documents\\datasets\\y_train2.csv')
##        y_test = pd.read_csv('C:\\Users\\ps\\Documents\\datasets\\y_test2.csv')

        #data3 = pd.read_csv('C:\\Users\\ps\\Documents\\datasets\\output.csv')
##
##        # Encode labels
##        encoder = preprocessing.LabelEncoder()
##        y_train_encoded = encoder.fit_transform(y_train.values.ravel())  # Flatten y_train to a 1D array
##        y_test_encoded = encoder.fit_transform(y_test.values.ravel())    # Flatten y_test to a 1D array
##
##        # Drop unnecessary columns if needed
##        X_train = X_train.drop(X_train.columns[0], axis=1)
##        X_test = X_test.drop(X_test.columns[0], axis=1)
##        data3 = data3.drop(data3.columns[0], axis=1)
##        print("data3", data3.shape)
##
##        # Ensure all data is in DataFrame format for concatenation
##        train_u = pd.DataFrame(train_u)
##        test_u = pd.DataFrame(test_u)
##        X_train = pd.DataFrame(X_train)
##        X_test = pd.DataFrame(X_test)
##        y_train_encoded = pd.DataFrame(y_train_encoded, columns=['label'])
##        y_test_encoded = pd.DataFrame(y_test_encoded, columns=['label'])
####
##        print(train_u.shape)
##        print(test_u.shape)
##        print(X_train.shape)
##        print(X_test.shape)
##        print(y_train_encoded.shape)
##        print(y_test_encoded.shape)
##
##        # Combine datasets
##        train_combined = pd.concat([train_u, X_train, y_train_encoded], axis=1)
##        test_combined = pd.concat([test_u, X_test, y_test_encoded], axis=1)
####
##             # Check and align column names
##        if not all(train_combined.columns == test_combined.columns):
##            test_combined.columns = train_combined.columns
##
##        # Concatenate train and test into one dataset
##        full_dataset = pd.concat([train_combined, test_combined], axis=0).reset_index(drop=True)
##
##        print("Full dataset shape:", full_dataset.shape)
##
##        # Extract features and labels
##        features = data3.iloc[:, :-1].values  # All columns except the last one
##        labels = data3.iloc[:, -1].values    # The last column is the label
####
##        features = np.array(features)
##        labels = np.array(labels)
####
####        #labels = np.array(labels)
####        # Initialize StratifiedKFold
##        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
######
######        # Ensure the directory for saving splits exists
##        output_dir = 'C:\\Users\\ps\\Documents\\datasets\\DS3_kfold_splits\\'
##        if not os.path.exists(output_dir):
##            os.makedirs(output_dir)
######
##        fold_no = 1
##        for train_index, val_index in skf.split(features, labels):
##            print(f'Processing Fold {fold_no}')
##            
###           # Split data into training and validation sets
##            X_train_input_ids, X_val_input_ids = features[train_index], features[val_index]
##            y_train, y_val = labels[train_index], labels[val_index]
####
####            # Save training data to CSV
##            train_df = pd.DataFrame(X_train_input_ids)
##            train_df['label'] = y_train
##            train_file = os.path.join(output_dir, f'train_fold_{fold_no}.csv')
##            train_df.to_csv(train_file, index=False)
####
####            # Save validation data to CSV
##            val_df = pd.DataFrame(X_val_input_ids)
##            val_df['label'] = y_val
##            val_file = os.path.join(output_dir, f'val_fold_{fold_no}.csv')
##            val_df.to_csv(val_file, index=False)
####
##            print(f"Fold {fold_no} saved: Train shape {train_df.shape}, Validation shape {val_df.shape}")
##            fold_no += 1
##
##     


##        scaler = StandardScaler()
##        X_train_normalized = scaler.fit_transform(X_train)  # X is your feature matrix
##        X_test_normalized = scaler.fit_transform(X_test)  # X is your feature matrix
##
### Apply PCA
##        pca = PCA(n_components=0.95)  # Retain 95% variance
##        Xtrain_pca = pca.fit_transform(X_train)  # Fit PCA on training data
##        Xtest_pca = pca.transform(X_test)  # Transform test data using same PCA
##
##        print("Xtrain_pca", Xtrain_pca.shape[1])
##        print("Xtest_pca", Xtest_pca.shape[1])
##        # Ensure consistency
##        if Xtrain_pca.shape[1] != Xtest_pca.shape[1]:
##                print("Warning: Number of components in train and test sets do not match!")


##        # Plot explained variance ratio
##        plt.figure(figsize=(10, 6))
##        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
##        plt.xlabel("Principal Components")
##        plt.ylabel("Explained Variance Ratio")
##        plt.title("Explained Variance by Principal Components")
##        plt.show()
        
##       #print("train_h.shape", train_h.shape)
##        print("train_u.shape", train_u.shape)
####        print("test_h.shape", test_h.shape)
##        print("test_u.shape", test_u.shape)
##
##        #test_h=test_h['test_h'].values
##        test_u=test_u['test_u'].values
##        #train_h=train_h['train_h'].values
##        train_u=train_u['train_u'].values

        
##        all_traintext=[]
##        all_testtext=[]
##        j=0
##        for i in train_h:
##           str1=""
##           str1= train_u[j]+i
##           all_traintext.append(str1)
##           j=j+1
##
##        j=0
##        for i in test_h:
##           str1=""
##           str1= test_u[j]+i
##           all_testtext.append(str1)
##           j=j+1
##
##        all_traintext = pd.DataFrame({'all_traintext': all_traintext})
##        all_testtext = pd.DataFrame({'all_testtext': all_testtext})
##        all_traintext=all_traintext['all_traintext'].values
##        all_testtext=all_testtext['all_testtext'].values

##        print("all_traintext: ",all_traintext.shape)
##        print("all_testtext: ",all_testtext.shape)

        #print(all_traintext[10]) 
 
##        print("test_h.shape", test_h.shape)
##        print("test_u.shape", test_u.shape)
        
    
##        folder_path1 ="C:\\Users\ps\\Documents\\datasets\\D3333_good_text"
##        items1= os.listdir(folder_path1)
##        for item in items1:
##          item_path1 = os.path.join(folder_path1, item)
##          #if "h" not in str(item) and  "u" not in str(item)  :
##          if "h" not in str(item):
##                with open(item_path1, 'rb') as file1:
##                  texts.append(file1.read().decode('utf-8', errors='ignore'))
##                  labels.append('0')
####          if "plain" in str(item):
####              with open(item_path1, 'rb') as file1:
####                  good_plain.append(file1.read().decode('utf-8', errors='ignore'))        
####
####          if "noicy" in str(item):
####              with open(item_path1, 'rb') as file1:
####                  good_noicy.append(file1.read().decode('utf-8', errors='ignore'))
####
####          if "url" in str(item):
####              with open(item_path1, 'rb') as file1:
####                  good_urls.append("http://"+file1.read().decode('utf-8', errors='ignore'))
##                  #labels.append('0')
        ##
##        
##        folder_path2 ="C:\\Users\\ps\\Documents\\datasets\\D333_bad_text"
##        items2= os.listdir(folder_path2)
##        for item in items2:
##          item_path2 = os.path.join(folder_path2, item)
##          #if "h" not in str(item) and  "u" not in str(item):
##          if "h" not in str(item):
##                with open(item_path2, 'rb') as file2:
##                  texts.append(file2.read().decode('utf-8', errors='ignore'))
##                  labels.append('1')
####          if "plain" in str(item):
####              with open(item_path2, 'rb') as file2:
####                  bad_plain.append(file2.read().decode('utf-8', errors='ignore'))        
####
####          if "noicy" in str(item):
####              with open(item_path2, 'rb') as file2:
####                  bad_noicy.append(file2.read().decode('utf-8', errors='ignore'))
####
####          if "url" in str(item):
####              with open(item_path2, 'rb') as file2:
####                  bad_urls.append(file2.read().decode('utf-8', errors='ignore'))


# Assuming you have loaded your data into the following variables: good_urls, good_pages, bad_urls, bad_pages

##        good_data = pd.DataFrame({'plain': good_noicy})
##        malicious_data = pd.DataFrame({'plain': bad_noicy})

### Add labels for good and malicious data
##        good_data['label'] = 0
##        malicious_data['label'] = 1

# Combine good and malicious data
##        combined_data = pd.concat([good_data, malicious_data], ignore_index=True)

##        trainDF = pd.DataFrame()
##        trainDF['text'] = texts
##        trainDF['label'] =labels
##          #for i in  trainDF['label']:
##           #     print(i)
##     
##        train_xh, valid_xh, train_yh, valid_yh = model_selection.train_test_split(trainDF['text'], trainDF['label'],test_size=0.2, random_state=0)

##        print(X_train.shape)
##        print(X_test.shape)
##        print(y_train.shape)
##        print(y_test.shape)

##        for i in y_test:
##          print(i)

        
        ###new reading
        ##
        ##file_path = "C:\\Users\\ps\\Documents\\datasets\\phishing_site_urls.csv"  # Update with your file path
        ##
        ##with open(file_path, "r",encoding='latin-1') as file:
        ##    reader = csv.reader(file, delimiter=",")  # Set the delimiter to "\t" for tab-separated values
        ##    for row in reader:
        ##        texts.append(row[0])
        ##        if (row[1]=='bad'):
        ##               labels.append('1')
        ##        else:
        ##               labels.append('0')

        ##with open('C:\\Users\\ps\\Documents\\datasets\\urlset.csv', 'r') as file:
        ##   reader = csv.reader(file)
        ##   headers=next(reader)  # Skip the header row
        ##   for row in reader:
        ##        print(len(row))
        ##        labels.append(row[13])
        ##        texts.append(row[0])

        ##with open('C:\\Users\\ps\\Documents\\datasets\\dataset9.txt', 'r') as file:
        ##    for line in file:
        ##
        ##        print(line)
        ##        columns = line.strip().split('|')
        ##        texts.append(columns[1])
        ##        if (columns[2]=='1'):
        ##            labels.append('1')
        ##        else:
        ##            labels.append('0')
        ##
        ##for i in rows:
        ##    labels.append(i[1])
        ##    texts.append(i[0])
        ##        
        ##for i in texts:
        ##    print (i)

##        # create a dataframe using texts and lables
##        trainDF = pd.DataFrame()
######
##        trainDF['text'] = texts
##        trainDF['label'] =labels

##        #for i in  trainDF['label']:
##         #     print(i)
##        X_train, X_test, y_train, y_test = model_selection.train_test_split(trainDF['text'], trainDF['label'],test_size=0.2, random_state=0)
##        print(X_train.shape)
##        print(X_test.shape)
##        print(y_train.shape)
##        print(y_test.shape)

        ##for i in train_x:
        ##   print(i)


        # load a clean dataset
        def load_dataset(filename):
                return load(open(filename, 'rb'))

        # fit a tokenizer
        def create_tokenizer(lines):
                tokenizer = Tokenizer()
                tokenizer.fit_on_texts(lines)
                return tokenizer

        # calculate the maximum document length
        def max_length(lines):
                return max([len(s.split()) for s in lines])

        # encode a list of lines
        def encode_text(tokenizer, lines, length):
                # integer encode
                encoded = tokenizer.texts_to_sequences(lines)
                # pad encoded sequences
                padded = pad_sequences(encoded, maxlen=length, padding='post')
                return padded
        def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
      
              sns.set(style="whitegrid", font_scale=1.2)
              plt.grid(False)
              plt.imshow(cm, interpolation='nearest', cmap=cmap)
              #plt.title(title)
              #plt.colorbar()
              # Change the border color and width of the outside borders
              border_color = 'black'
              border_width = 0.8
              
              # Change the border color and width of the inside borders
              inside_border_color = 'white'
              inside_border_width = 0.8
              tick_marks =np.arange(len(classes))
              plt.xticks(tick_marks, classes, rotation=0,fontsize=18, fontweight='bold')
              plt.yticks(tick_marks, classes,fontsize=18, fontweight='bold')
            
              if normalize:
                      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                  #print("Normalized confusion matrix")
              else:
                      1#print('Confusion matrix, without normalization')
          
                      #print(cm)
          
              thresh = cm.max() / 2.
              for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                    plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black",fontsize=20,     fontweight='bold')
              plt.tight_layout()
              plt.ylabel('True label',fontsize=16,fontweight='bold')
              plt.xlabel('Predicted label',fontsize=16,fontweight='bold')
              
                  # Set the border color and width of the outside borders
              plt.gca().spines['top'].set_color(border_color)
              plt.gca().spines['bottom'].set_color(border_color)
              plt.gca().spines['left'].set_color(border_color)
              plt.gca().spines['right'].set_color(border_color)
              
              plt.gca().spines['top'].set_linewidth(border_width)
              plt.gca().spines['bottom'].set_linewidth(border_width)
              plt.gca().spines['left'].set_linewidth(border_width)
              plt.gca().spines['right'].set_linewidth(border_width)
      

        class gMLPLayer(layers.Layer):
                def __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs):
                    super(gMLPLayer, self).__init__(*args, **kwargs)
                    self.num_patches = num_patches
                    self.embedding_dim=embedding_dim
                    self.dropout_rate=dropout_rate
                    
                    
                    self.channel_projection1 = keras.Sequential(
                          [
                              layers.Dense(units=embedding_dim * 2),
                              layers.ReLU(),
                              layers.Dropout(rate=dropout_rate),
                          ]
                      )

                    self.channel_projection2 = layers.Dense(units=embedding_dim)

                    self.spatial_projection = layers.Dense(
                          units=num_patches, bias_initializer="Ones"
                      )

                    self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
                    self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

                def spatial_gating_unit(self, x):
                  u, v = tf.split(x, num_or_size_splits=2, axis=2)
                  v = self.normalize2(v)
                  v_channels = tf.linalg.matrix_transpose(v)
                  v_projected = self.spatial_projection(v_channels)
                  v_projected = tf.linalg.matrix_transpose(v_projected)
                  return u * v_projected

                def get_config(self):
                    config = super(gMLPLayer, self).get_config()
                    config.update({
                        'num_patches': self.num_patches,
                        'embedding_dim': self.embedding_dim,
                        'dropout_rate': self.dropout_rate,
                    })
                    return config

                def call(self, inputs):
                  x = self.normalize1(inputs)
                  x_projected = self.channel_projection1(x)
                  x_spatial = self.spatial_gating_unit(x_projected)
                  x_projected = self.channel_projection2(x_spatial)
                  return x + x_projected

        def bert_encode(texts, tokenizer, max_len):
            all_tokens = []
            all_masks = []
            all_segments = []
            
            for text in texts:
                text = tokenizer.tokenize(text)
                
                text = text[:max_len-2]
                input_sequence = ["[CLS]"] + text + ["[SEP]"]
                pad_len = max_len-len(input_sequence)
                
                tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
                pad_masks = [1] * len(input_sequence) + [0] * pad_len
                segment_ids = [0] * max_len
                
                all_tokens.append(tokens)
                all_masks.append(pad_masks)
                all_segments.append(segment_ids)
                
            return np.array(all_tokens)
        #np.array(all_masks), np.array(all_segments)

        max_len = 200

        #this function is used for character embeding fearures of URLs
        def charcter_embedding(texts, max_len=200):
              tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
              tk.fit_on_texts(texts)
              ##
              alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
              ##
              char_dict = {}
              for i, char in enumerate(alphabet):
                  char_dict[char] = i + 1
              
                  tk.word_index = char_dict.copy()
              # Add 'UNK' to the vocabulary
              tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
              
              sequences = tk.texts_to_sequences(texts)
              #test_texts = tk.texts_to_sequences(valid_x)
              
              ### Padding
              data = pad_sequences(sequences, maxlen=200, padding='post')
              print('Load')
              return data

            
           #this extended function that used for character embeding fearures of URLs used by Zhang et al. [22] method    
        def charcter_embedding2(texts, max_len):
            # Initialize the tokenizer for character-level embedding
            tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK', lower=True)
            tk.fit_on_texts(texts)

            # Define character and custom token indices
            alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
            special_tokens = ["account", "admin", "administrator", "auth", "bank", "client", "confirm", "cmd", "email", 
                        "host", "login", "password", "pay", "private", "registed", "safe", "secure", "security", 
                        "sign", "service", "signin", "submit", "user", "update", "validation", "verification", "webscr"]

            char_dict = {char: i + 1 for i, char in enumerate(alphabet)}

            # Set up indices for <PAD> and <UNK>
            pad_index = len(char_dict) + len(special_tokens) + 1
            unk_index = pad_index + 1

            # Update tokenizer's word index to include custom characters and tokens
            tk.word_index = char_dict.copy()
            tk.word_index.update({token: idx + len(char_dict) for idx, token in enumerate(special_tokens)})
            tk.word_index[tk.oov_token] = unk_index
            tk.word_index['<PAD>'] = pad_index

            # Convert training and test sequences to integer sequences
            sequences = tk.texts_to_sequences(texts)
           

            #Pad the sequences to ensure consistent input size
            max_len = 200
            data = pad_sequences(sequences, maxlen=max_len, padding='post', value=pad_index)
            
            return data

               #this function is used to get weights matrix when we use character embeding fearures of URLs used by Zhang et al. [22] method  
        def get_embedding_weights2(text):
        
             # Initialize the tokenizer for character-level embedding
            tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK', lower=True)
            tk.fit_on_texts(texts)

            # Define character and custom token indices
            alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
            special_tokens = ["account", "admin", "administrator", "auth", "bank", "client", "confirm", "cmd", "email", 
                        "host", "login", "password", "pay", "private", "registed", "safe", "secure", "security", 
                        "sign", "service", "signin", "submit", "user", "update", "validation", "verification", "webscr"]

            char_dict = {char: i + 1 for i, char in enumerate(alphabet)}

            # Set up indices for <PAD> and <UNK>
            pad_index = len(char_dict) + len(special_tokens) + 1
            unk_index = pad_index + 1

            # Update tokenizer's word index to include custom characters and tokens
            tk.word_index = char_dict.copy()
            tk.word_index.update({token: idx + len(char_dict) for idx, token in enumerate(special_tokens)})
            tk.word_index[tk.oov_token] = unk_index
            tk.word_index['<PAD>'] = pad_index
            vocab_size = len(tk.word_index) + 1  # +1 for 0 index padding
            embedding_weights2 = []
            embedding_weights2 = np.zeros((vocab_size, vocab_size))

            for char, idx in tk.word_index.items():
                if idx < vocab_size:
                    embedding_weights2[idx, idx] = 1.0

            embedding_weights2 = np.array(embedding_weights2)
            print("Embedding weights2 shape:", embedding_weights2.shape)
            print("Vocab size2:", vocab_size)
            return embedding_weights2


         #this function is used to get weights matrix when we use character embeding fearures of URLs used by Aljofey et al. [8] method  
        def get_embedding_weights(text):
                    #character embeding fearures
               tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
               tk.fit_on_texts(text)
               ##
               alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
               ##
               char_dict = {}
               for i, char in enumerate(alphabet):
                   char_dict[char] = i + 1
               
                   tk.word_index = char_dict.copy()
               # Add 'UNK' to the vocabulary
               tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
               vocab_size = len(tk.word_index)+1
                             # Embedding weights
               embedding_weights = []  # (70, 69)
               embedding_weights.append(np.zeros(vocab_size))  # (0, 69)
             
               for char, i in tk.word_index.items():  # from index 1 to 69
                 onehot = np.zeros(vocab_size)
                 onehot[i - 1] = 1
                 embedding_weights.append(onehot)
             
               embedding_weights = np.array(embedding_weights)
               return embedding_weights


##        print(input_size)

        #word embeding
##        text = np.concatenate((train_u, test_u), axis=0)
##        text = np.array(text)
##        tokenizer2 = Tokenizer()
##        tokenizer2.fit_on_texts(text)
##        sequences1 = tokenizer2.texts_to_sequences(train_u)
##        sequences2 = tokenizer2.texts_to_sequences(test_u)
##        ##word_index = tokenizer.word_index
##        ##size_of_vocabulary=len(tokenizer.word_index) + 1
##        #####size_of_vocabulary=X.shape[0]
##        ##print(size_of_vocabulary)
##        X_seq_train = pad_sequences(sequences1, maxlen=500)
##        X_seq_test = pad_sequences(sequences2, maxlen=500)
##        X_seq_train=np.array(X_seq_train, dtype='float32')
##        X_seq_test=np.array(X_seq_test, dtype='float32')

##        glove_embeddings = {}
##        with open('C:\\Users\\ps\\Documents\\glove.6B\\glove.6B.300d.txt', encoding='utf-8') as f:
##          for line in f:
##                values = line.split()
##                word = values[0]
##                vector = np.asarray(values[1:], dtype='float32')
##                glove_embeddings[word] = vector
##
##                          # Create an embedding matrix
##        embedding_matrix2 = np.zeros((len(tokenizer2.word_index) + 1, 300))
##        for word, i in tokenizer2.word_index.items():
##              embedding_vector = glove_embeddings.get(word)
##              if embedding_vector is not None:
##                  embedding_matrix2[i] = embedding_vector

##        #fasttext embedding
##        import fasttext
##        fasttext_model_path = 'cc.en.300.bin'  # Replace with the path to your pre-trained model
##        model = fasttext.load_model(fasttext_model_path)
##
##        def text_to_embedding(text):
##                words = text.split()
##                embeddings = [model.get_word_vector(word) for word in words]
##                return np.mean(embeddings, axis=0)
##
##        X_train_embeddings = np.array([text_to_embedding(text) for text in train_u])
##        X_test_embeddings = np.array([text_to_embedding(text) for text in test_u])

        ###TF-IDf features
        ##tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3),max_features=7000)
        ##tfidf_vect_ngram_chars.fit(trainDF['text'])
        ##xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.fit_transform(train_x).toarray()  
        ##xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x).toarray()

        # label encode the target variable 
####        encoder = preprocessing.LabelEncoder()
####        y_train = encoder.fit_transform(y_train)
####        y_test = encoder.fit_transform(y_test)
##        ##for i in train_y:
##        ##    print(i)
##        y_train1 = to_categorical(y_train)
##        y_test1 = to_categorical(y_test)
        ##for i in valid_y1:
        ##    print(i)

##        #BERT word embeding features
##        m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
##        #m_url2='https://tfhub.dev/google/universal-sentence-encoder/4'
##        bert_layer = hub.KerasLayer(m_url, trainable=True)
##        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
##        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
##        tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
##        max_len = 200
##        embedding_matrix = bert_layer.get_weights()[0]
##        print("embedding_matrix:", embedding_matrix.shape)
##        train_input = bert_encode(X_train, tokenizer, max_len=max_len)
##        test_input = bert_encode(X_test, tokenizer, max_len=max_len)
##        train_input2 = bert_encode(X_train['hy'], tokenizer, max_len=max_len)
##        test_input2 = bert_encode(X_test['hy'], tokenizer, max_len=max_len)
##        train_xh = bert_encode(train_xh, tokenizer, max_len=200)
##        valid_xh = bert_encode(valid_xh, tokenizer, max_len=200)
##        test_h = bert_encode(test_h, tokenizer, max_len=500)
##        test_u1 = bert_encode(test_u, tokenizer, max_len=200)

        #encoder = preprocessing.LabelEncoder()
##        train_yh = encoder.fit_transform(train_yh)
##        valid_yh = encoder.fit_transform(valid_yh)
##        train_yh = tf.cast(train_yh, dtype=tf.float32)  # Cast to float32 if needed
##        valid_yh = tf.cast(valid_yh, dtype=tf.float32)  # Cast to float32 if needed
        #y_train = encoder.fit_transform(y_train)
        #y_test = encoder.fit_transform(y_test)
##        y_test = tf.cast(y_test, dtype=tf.float32)  # Cast to float32 if needed
##        y_train = tf.cast(y_train, dtype=tf.float32)  # Cast to float32 if needed

##       # print(train_h.shape)
##        print(train_u.shape)
##        print(test_u.shape)
##        #print(test_h.shape)
##        print(y_train.shape)
##        print(y_test.shape)

        composite_data_train = []
        composite_data_test = []
##
##        train_u = np.array(train_u)
##        X_train=np.array(X_train)
##        test_u = np.array(test_u)
##        X_test=np.array(X_test)

        
##        for i in test_u:
##          print(i)
##        for i in X_test:
##          print(i)

##        j=0
##        for i in range(len(train_u)):
##          composite_data_train.append(np.concatenate((train_u[i], X_train[j]), axis=None))
##          j=j+1
####
##        j=0 
##        for i in range(len(test_u)):
##          composite_data_test.append(np.concatenate((test_u[i], X_test[j]), axis=None))
##          j=j+1
######
######          # Convert the list of tuples to a numpy array if needed
##        composite_data_train = np.array(composite_data_train)
##        print("composite_data_train: ", composite_data_train.shape)
##        composite_data_test = np.array(composite_data_test)
##        print("composite_data_test: ", composite_data_test.shape)

          
##        for i in composite_data_test :
##              print(i)

        #concate_hyper_text_train=np.array(concate_hyper_text_train, dtype='float32')

        
        ##
        #x=np.array(test_input)


        ##print("test_input:", x.shape)
        ##print("test_input:", test_input.shape)
        ##for i in test_input:
        ##    print(i)

        #TFDistilBertModel
        ##from transformers import TFDistilBertModel
        ##from transformers import DistilBertTokenizer
        ##transformer_layer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        ##tokenizer_DistilBERT = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        ##train_input = bert_encode(train_x, tokenizer_DistilBERT, max_len=max_len)
        ##test_input = bert_encode(valid_x, tokenizer_DistilBERT, max_len=max_len)


        def DNN_model(n_features):
            inputs = layers.Input(name="input", shape=(n_features,))
            h1 = layers.Dense(name="h1", units=int(round((n_features+1)/2)), activation='relu')(inputs)
            h1 = layers.Dropout(name="drop1", rate=0.2)(h1)
           ### hidden layer 2
            h2 = layers.Dense(name="h2", units=int(round((n_features+1)/4)), activation='relu')(h1)
            h2 = layers.Dropout(name="drop2", rate=0.2)(h2)
            h3 = layers.Dense(name="h3", units=int(round((n_features+1)/8)), activation='relu')(h2)
            h3 = layers.Dropout(name="drop3", rate=0.2)(h3)
            h4 = layers.Dense(name="h4", units=int(round((n_features+1)/16)), activation='relu')(h3)
            h4 = layers.Dropout(name="drop4", rate=0.2)(h4)
            flat1 = Flatten()(h4)
            out = Dense(2, activation='softmax')(flat1)
            model = Model(inputs=inputs, outputs=out)
            model.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics=['accuracy'])
            print(model.summary())
            return model

        

        def aljofey_2020 ():
                inputs = tf.keras.layers.Input(shape = (200,))
                x = Embedding(vocab_size + 1, 95, weights=[embedding_weights])(inputs)              
                conv_layers = [[256, 7, 3],
                     [256, 7, 3],
                     [256, 3, -1],
                     [256, 3, -1],
                     [256, 3, -1],
                     [256, 3, -1],
                     [256, 3, 3]]

                fully_connected_layers = [2028, 2048]
                dropout_p = 0.5
                #optimizer = 'adam'
                for filter_num, filter_size, pooling_size in conv_layers:
                    x = Conv1D(filter_num, filter_size)(x)
                    x = Activation('relu')(x)
                    if pooling_size != -1:
                        x = MaxPooling1D(pool_size=pooling_size)(x)  # Final shape=(None, 34, 256)
                x = Flatten()(x)  # (None, 8704)

                for dense_size in fully_connected_layers:
                  x = Dense(dense_size, activation='relu')(x)  # dense_size == 1024
                  x = Dropout(dropout_p)(x)
# Output Layer
                out = Dense(1,activation='sigmoid')(x)
# Build model
                model = Model(inputs=inputs, outputs=out)
                model.compile(Adam(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])
                print(model.summary())
                return model

        def wei_wei (embedding_weights):
              inputs2 = tf.keras.layers.Input(shape=(200,))
              x4 = Embedding(*embedding_weights.shape, weights=[embedding_weights])(inputs2)
              x4 = Conv1D(64, kernel_size=8, activation='relu')(x4)
              x4 = MaxPooling1D(pool_size=2)(x4)

              x4 = Conv1D(16, kernel_size=16, activation='relu')(x4)
              x4 = MaxPooling1D(pool_size=2)(x4)

              x4 = Conv1D(8, kernel_size=32, activation='relu')(x4)
              x4 = MaxPooling1D(pool_size=2)(x4)

              x4 = Flatten()(x4)  # Add this line to flatten the output

              out = Dense(32, activation='relu')(x4)
              out = Dense(1, activation='sigmoid')(out)

              model = Model(inputs=inputs2, outputs=out)
              model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
              print(model.summary())
              return model


        def Mohammed_2022 ():
              inputs2 = tf.keras.layers.Input(shape = (200,))
              x4 = Embedding(vocab_size + 1, 95, weights=[embedding_weights])(inputs2)
              c1=Conv1D (128, kernel_size=4, activation='relu')(x4)
              c2=Conv1D (128, kernel_size=6, activation='relu')(x4)
              c3=Conv1D (128, kernel_size=10, activation='relu')(x4)
              c4=Conv1D (128, kernel_size=20, activation='relu')(x4)

              flat1 = Flatten()(c1)
              flat2 = Flatten()(c2)
              flat3 = Flatten()(c3)
              flat4 = Flatten()(c4)
              flat5 = Flatten()(x4)

              concatenated1 = Concatenate()([flat1,flat2,flat3,flat4,flat5])

              concatenated1=Dropout(0.5)(concatenated1)

              concatenated1 = Dense(64, activation='relu')(concatenated1)
              concatenated1 = Dense(64, activation='relu')(concatenated1)
              concatenated1 = Dense(64, activation='relu')(concatenated1)

              logits = layers.Dense(1, activation='sigmoid')(concatenated1)
              model=Model(inputs=inputs2, outputs=logits)
                        #model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
              model.compile(Adam(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])
              print(model.summary())
              return model
              
        def hybrid_DNN_LSTM (embedding_weights):
              inputs1 = tf.keras.layers.Input(shape = (60,))
              inputs2 = tf.keras.layers.Input(shape = (200,))
                       # Embedding layer Initialization
              
              x2 = Embedding(*embedding_weights.shape, weights=[embedding_weights])(inputs2)

              d1=Dense(40, activation='relu')(inputs1)
              d1=Dense(64, activation='relu')(d1)
              d1=Dense(32, activation='relu')(d1)
              d1=Dense(16, activation='relu')(d1)

              x=layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x2)
              x=layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
              x=Dropout(0.1)(x)
              x = layers.Dense(16, activation='relu')(x)
              flat1 = Flatten()(x)
              flat2 = Flatten()(d1)

              concatenated1 = Concatenate()([flat1, flat2])
              concatenated1=Dense(8, activation='relu')(concatenated1)
              logits = layers.Dense(1, activation='sigmoid')(concatenated1)

              model=Model(inputs=[inputs2,inputs1], outputs=logits)
                        #model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
              model.compile(Adam(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])
              print(model.summary())
              return model

        def biLSTM (embedding_matrix, shape=60):
             
              #inputs1 = tf.keras.layers.Input(shape = (200,))
              inputs1 = tf.keras.layers.Input(shape = (60,))
                       # Embedding layer Initialization
              
              #x1 = Embedding(*embedding_matrix.shape, weights=[embedding_matrix],trainable=False)(inputs1)
              x1 = Embedding(shape, 1024,trainable=True)(inputs1)

              x1=layers.Bidirectional(layers.LSTM(768, return_sequences=True))(x1)
              x1=layers.Bidirectional(layers.LSTM(768, return_sequences=True))(x1)
              x=Dropout(0.1)(x1)
              x = layers.Dense(128, activation='relu')(x)
              flat1 = Flatten()(x)
             
              logits = layers.Dense(1, activation='sigmoid')(flat1)

              model=Model(inputs=[inputs1], outputs=logits)
                        #model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
              model.compile(Adam(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])
              print(model.summary())
              return model



        def focal_loss(gamma=1.0, alpha=0.12):
              def focal_loss_fixed(y_true, y_pred):
                  # Calculate binary cross entropy
                  bce = k.binary_crossentropy(y_true, y_pred)
                  
                  # Calculate the modulating factor (focal term)
                  pt = tf.exp(-bce)
                  focal_term = (1 - pt) ** gamma
                  
                  # Calculate the final focal loss
                  loss = alpha * focal_term * bce
                  
                  return loss
              
              return focal_loss_fixed


        def build_rnn_model():
                    # Input layer for sequences of length 200
##                    inputs = tf.keras.layers.Input(shape=(200,))
##                    x = Embedding(vocab_size + 1, 95, weights=[embedding_weights])(inputs)
                    inputs = tf.keras.layers.Input(shape = (89,))
                       # Embedding layer Initialization
                      #x1 = Embedding(*embedding_matrix.shape, weights=[embedding_matrix],trainable=False)(inputs1)
                    x1 = Embedding(60, 1024,trainable=True)(inputs)
                    # Embedding layer using pretrained embeddings
                    #x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(inputs)
                    # First RNN layer, returning sequences (output shape: (batch_size, timesteps, 64))
                    x = SimpleRNN(64, return_sequences=True)(x1)
                    #x = LSTM(64, return_sequences=True)(x)
##                    x=Conv1D(64, 3, activation='relu')(x)
##                    x=Conv1D(128, 5, activation='relu')(x)
##                    x=Conv1D(128, 10, activation='relu')(x)
                    # Second RNN layer, returning sequences (output shape: (batch_size, timesteps, 64))
                    x = SimpleRNN(64, return_sequences=True)(x)
                    #x = LSTM(64, return_sequences=True)(x)
##                    # Third RNN layer, no sequences returned (output shape: (batch_size, 128))
##                    x = SimpleRNN(128, return_sequences=False)(x)
                    # Dense layer
                    x = Dense(64, activation='relu')(x)
                    # Dropout layer
                    x = Dropout(0.5)(x)
                    flat = Flatten()(x)    
                    # Output layer for binary classification
                    out = Dense(1, activation='sigmoid')(flat)
                    # Build the model
                    model = Model(inputs=[inputs], outputs=out)
                    # Compile the model
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    # Print the summary of the model
                    print(model.summary())
                    return model
          
          
        def DNN_model2(embedding_matrix):
                  #inputs = tf.keras.layers.Input(shape = (200,))
                  #x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix],trainable=False)(inputs)

                  inputs = layers.Input(shape=89,)
                  #x = Embedding(89, 1024,trainable=True)(inputs)

                  #x = Embedding(vocab_size + 1, 95, weights=[embedding_weights])(inputs)
                  x = layers.Dense(64, activation='relu')(inputs)
                   ### hidden layer 2
                  x = layers.Dense(32, activation='relu')(x)
                  x=Dropout(0.1)(x)
                  x=layers.Dense(16, activation='relu')(x)
                  x=layers.Dense(8, activation='relu')(x)
                  out = Dense(1,activation='sigmoid')(x)
                  model = Model(inputs=[inputs], outputs=out)
                  model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
                  print(model.summary())
                  return model

        def build_classifier(blocks):
                        inputs = layers.Input(shape=89,)
                        x = Embedding(89, 768,trainable=True)(inputs)
                        #x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix],trainable=False)(inputs)

                        #concatenated1 = Concatenate()([cnn6,cnn7,cnn8])
                        #inputs = tf.keras.layers.Input(shape = (200,))
                        #x = Embedding(89, 768,trainable=True)(inputs)
                        #x = Embedding(vocab_size + 1, 95, weights=[embedding_weights])(inputs)
                    
                        x = blocks(x)
##                        x = SpatialDropout1D(0.1)(x)
##                        t1 = TCN(128,dilations = [1,1,1], return_sequences=True, activation = 'relu')(x)
##            #max_pool1 = GlobalMaxPooling1D()(t1)
##            
##                        t3 = TCN(256,dilations = [1,1,1], return_sequences=True, activation = 'relu')(x)
##            #max_pool3 = GlobalMaxPooling1D()(t3)
##
##                        t4 = TCN(512,dilations = [1,1,1], return_sequences=True, activation = 'relu')(x)
##            #max_pool4 = GlobalMaxPooling1D()(t4)
####                        gvp = layers.GlobalAveragePooling1D()(x)
##                        concatenated1 = Concatenate()([t1, t3, t4])

                        
                        gmp = layers.GlobalMaxPooling1D()(x)
                        representation = layers.Dropout(rate=dropout_rate)(gmp)
                        conc = Dense(64, activation="relu")(representation)
                        conc = Dense(64, activation="relu")(conc)
                        conc = Dense(64, activation="relu")(conc)
                        out1 = Dense(1, activation="sigmoid")(conc)
                        
                        #logits = layers.Dense(1, activation='sigmoid')(representation)
                        model=Model(inputs=inputs, outputs=out1)
                        #model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
                        model.compile(Adam(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])
                        print(model.summary())
                        return model
            

        def TCN_model(embedding_matrix, shape=89):

            inputs1 = tf.keras.layers.Input(shape = (200,))
            inputs2 = tf.keras.layers.Input(shape = (shape,))

            x1 = Embedding(*embedding_matrix.shape, weights=[embedding_matrix],trainable=False)(inputs1)
            x2 = Embedding(shape, 1024,trainable=True)(inputs2)

          # TCN1
            x1 = SpatialDropout1D(0.1)(x1)
            t1 = TCN(128,dilations = [1,1,1,1],  return_sequences=True, activation = 'relu')(x1)
            t3 = TCN(256,dilations = [1,1,1,1],  return_sequences=True, activation = 'relu')(x1)
            t4 = TCN(512,dilations = [1,1,1,1], return_sequences=True, activation = 'relu')(x1)

           
          #TCN2
            x2 = SpatialDropout1D(0.1)(x2)
            t12 = TCN(128,dilations = [1,1,1,1], return_sequences=True, activation = 'relu')(x2)         
            t32 = TCN(256,dilations = [1,1,1,1], return_sequences=True, activation = 'relu')(x2)
            t42 = TCN(512,dilations = [1,1,1,1], return_sequences=True, activation = 'relu')(x2)
            concatenated1 = Concatenate()([t1, t3, t4])
            concatenated2 = Concatenate()([t12, t32, t42])
            max_pool = GlobalMaxPooling1D()(concatenated1)
            max_poo2 = GlobalMaxPooling1D()(concatenated2)
            concatenated3 = Concatenate()([ max_pool, max_poo2])
            conc = Dropout(0.1)(concatenated3)
            conc = Dense(64, activation="relu")(conc)
            conc = Dense(64, activation="relu")(conc)
            conc = Dense(64, activation="relu")(conc)
            out1 = Dense(1, activation="sigmoid")(conc)
            model3 = Model(inputs=[inputs1,inputs2], outputs=out1)
            model3.compile(Adam(lr=2e-5), loss='binary_crossentropy' ,metrics=['accuracy'])
            print(model3.summary())
            return model3

        
        #LSTM +CNN

        def CNN_LSTM(SEQ_LEN=60):

            inputs = tf.keras.layers.Input(shape = (SEQ_LEN,))
            x1 = Embedding(89, 1024,trainable=True)(inputs)

            #x1 = Embedding(*embedding_matrix.shape, weights=[embedding_matrix],trainable=False)(inputs)

            #x1 = Embedding(vocab_size + 1, 95, weights=[embedding_weights])(inputs)
            
            x1=Dropout(0.25)(x1)
##            x1=Conv1D(128, 8, activation='relu')(x1)
##            
##            x1=Conv1D(128, 10, activation='relu')(x1)
##            x1=MaxPooling1D(pool_size=2)(x1)
##            x1=Conv1D(256, 12, activation='relu')(x1)
##            x1=MaxPooling1D(pool_size=2)(x1)
            x1=layers.Bidirectional(LSTM(768, return_sequences=True, recurrent_dropout=0.2))(x1)
            x1=MaxPooling1D(pool_size=2)(x1)
            x1=layers.Bidirectional(LSTM(768, return_sequences=True, recurrent_dropout=0.2))(x1)
            output=Dense(1024,activation='relu')(x1)
            flat = Flatten()(output)    
            outputs=Dense(1,activation='sigmoid')(flat)
            model3 = Model(inputs=inputs, outputs=outputs)
            #model3.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
            model3.compile(Adam(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])
            print(model3.summary())
            return model3


        def CNN_Fusion():
  
               inputs = tf.keras.layers.Input(shape = (200,))
               #x1 = Embedding(*embedding_matrix.shape, weights=[embedding_matrix],trainable=False)(inputs)
               x1 = Embedding(vocab_size + 1, 95, weights=[embedding_weights])(inputs)
               
               cnn = Conv1D(filters=128, kernel_size=8, padding='same',activation='relu')(x1)
               cnn = SpatialDropout1D(0.4)(cnn)
               cnn = GlobalMaxPooling1D()(cnn) 
               
               cnn1 = Conv1D(filters=128, kernel_size=10, padding='same',activation='relu')(x1)
               cnn1 = SpatialDropout1D(0.4)(cnn1)
               cnn1 = GlobalMaxPooling1D()(cnn1) 
               
               cnn3 = Conv1D(filters=256, kernel_size=12, padding='same',activation='relu')(x1)
               cnn3 = SpatialDropout1D(0.4)(cnn3)
               cnn3 = GlobalMaxPooling1D()(cnn3)
               
               concatenated = Concatenate()([cnn, cnn1, cnn3])
               output = Dense(128, activation='relu')(concatenated)
             
               output=Dropout(0.4)(output)
               output = Dense(1, activation='sigmoid')(output)
               model3 = Model(inputs=inputs, outputs=output)
               model3.compile(loss='binary_crossentropy', optimizer=Adam(1e-5), metrics=['accuracy'])
           
               print(model3.summary())
               return model3


             # Zhang_2021 et al. [23] method     
        def CNN_biLSTM(embedding_weights1,vocab_size=123):  
               inputs = tf.keras.layers.Input(shape = (200,))
               x1 = Embedding(vocab_size + 1, 124, weights=[embedding_weights1])(inputs)
            
               x1=Dropout(0.25)(x1)
               x1=Conv1D(128, 8, activation='relu')(x1)  
               x1=Conv1D(128, 10, activation='relu')(x1)
               x1=MaxPooling1D(pool_size=2)(x1)
               x1=Conv1D(256, 12, activation='relu')(x1)
               x1=MaxPooling1D(pool_size=2)(x1)
               x1=LSTM(64, return_sequences=True, recurrent_dropout=0.2)(x1)
               x1=MaxPooling1D(pool_size=2)(x1)
               x1=LSTM(128, return_sequences=True, recurrent_dropout=0.2)(x1)
               output=Dense(1024,activation='relu')(x1)
               flat = Flatten()(output)    
               outputs=Dense(1,activation='sigmoid')(flat)
               model3 = Model(inputs=inputs, outputs=outputs)
               #model3.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
               model3.compile(Adam(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])
               print(model3.summary())
               return model3


        num_patches = 89
        dropout_rate = 0.1
        embedding_dim = 768  # Number of hidden units.
        num_blocks = 2  # Number of blocks.
        gmlp_blocks = keras.Sequential ([gMLPLayer(num_patches, embedding_dim, dropout_rate) for _ in range(num_blocks)])
        learning_rate = 0.003


        import matplotlib.pyplot as plt

        def plot_history(history):
                  acc = history.history['accuracy']
                  val_acc = history.history['val_accuracy']
                  loss = history.history['loss']
                  val_loss = history.history['val_loss']
                  epochs = range(1, len(acc) + 1)

                  # Plotting accuracy
                  plt.figure(figsize=(8, 8))
                  plt.plot(epochs, acc, 'b', label='Training Accuracy', linewidth=6)
                  plt.plot(epochs, val_acc, 'r', label='Validation Accuracy', linewidth=6)
                  plt.ylabel('Accuracy', fontsize=18, fontweight='bold')
                  plt.xlabel('Epochs', fontsize=18, fontweight='bold')
                  plt.legend(loc='lower right', fontsize=18, prop={'weight': 'bold'})  # Making legend labels bold
                  #plt.tick_params(axis='both', which='major', labelsize=16)
                  plt.yticks(fontsize=16, fontweight='bold')
                  plt.xticks(fontsize=16,fontweight='bold')

                  # Set border color for accuracy plot
                  for spine in plt.gca().spines.values():
                      spine.set_color('black')
                  
                  plt.grid(False)
                  plt.tight_layout()
                  plt.show()

                  # Plotting loss
                  plt.figure(figsize=(8, 8))
                  plt.plot(epochs, loss, 'b', label='Training Loss', linewidth=6)
                  plt.plot(epochs, val_loss, 'r', label='Validation Loss', linewidth=6)
                  plt.xlabel('Epochs', fontsize=18, fontweight='bold')
                  plt.ylabel('Loss', fontsize=18, fontweight='bold')
                  plt.legend(loc='upper right', fontsize=18, prop={'weight': 'bold'})  # Making legend labels bold
                  #plt.tick_params(axis='both', which='major', labelsize=16)
                  plt.yticks(fontsize=16, fontweight='bold')
                  plt.xticks(fontsize=16,fontweight='bold')

                  # Set border color for loss plot
                  for spine in plt.gca().spines.values():
                      spine.set_color('black')
                  
                  plt.grid(False)
                  plt.tight_layout()
                  plt.show()


        def training():
                fold_no = 5
                print(f'Running Fold {fold_no}')
              # Load training data
                train_file = f'C:\\Users\\ps\\Documents\\datasets\\DS2_kfold_splits\\train_fold_{fold_no}.csv'
                
                X_train = pd.read_csv(train_file)
                #X_train = train_df['X_train_input_ids'].values

##                X_train = pd.read_csv(train_file)
##                #X_train = train_df['X_train_input_ids'].values

                train_df = pd.read_csv(train_file)
                u_train = train_df['0'].values

                
                y_train = train_df['label'].values
              # Load validation data
                val_file = f'C:\\Users\\ps\\Documents\\datasets\\DS2_kfold_splits\\val_fold_{fold_no}.csv'
                X_test = pd.read_csv(val_file)
##                #X_test = val_df['X_val_input_ids'].values
#                y_test = X_test['label'].values

                test_df = pd.read_csv(val_file)
                u_test = test_df['0'].values
                y_test = test_df['label'].values

                  #here were extract charcter features of URL within each fold of the dataset to avoid the leak of features...
                train_data=charcter_embedding(u_train)
                test_data=charcter_embedding(u_test)
                embedding_weights1=get_embedding_weights(u_train)
                
                X_train = X_train.drop(X_train.columns[0], axis=1)
                X_test = X_test.drop(X_test.columns[0], axis=1)
                X_train = X_train.drop(X_train.columns[60], axis=1)
                X_test = X_test.drop(X_test.columns[60], axis=1)
                print("X_train:",X_train.shape)
                print("X_test:",X_test.shape)
                model=TCN_model(embedding_weights1)
                #model=biLSTM(embedding_weights1)
                #model=hybrid_DNN_LSTM(embedding_weights1)
                #model=CNN_LSTM()
                #model = build_classifier(gmlp_blocks)
                #model=build_rnn_model()
                #model=DNN_model2(embedding_weights1)
                checkpoint = ModelCheckpoint('ne3.h5', monitor='val_accuracy', save_best_only=True)
        #xvalid_tfidf_ngram_chars
                history=model.fit([train_data, X_train], y_train, epochs=50, batch_size=256, callbacks=[checkpoint], validation_data=([test_data,X_test], y_test), verbose=2)
                 # evaluate model on training dataset
                model.load_weights('ne3.h5')
                loss, acc = model.evaluate([test_data, X_train], array(y_train), verbose=0)
                print('Train Accuracy: %f' % (acc*100))
        # evaluate model on test dataset dataset
                loss, acc = model.evaluate([test_data, X_test], array(y_test), verbose=0)
                print('Test Accuracy: %f' % (acc*100))
                p= model.predict([test_data, X_test])
                print(p.shape)
                print(y_test.shape)
# Assuming y_true contains continuous values
                threshold = 0.5
                y_true = (p > threshold).astype(int)
                y_true=np.array(y_true)
                t=[]
                for i in p:
                      if (i >= 0.5).any():
                          t.append(1)
                      else:
                          t.append(0)
                t=np.array(t)
        #print(metrics.classification_report(y_test, t))
                print("\n f1_score(in %):", metrics.f1_score(y_test, y_true)*100)
                print("model accuracy(in %):", metrics.accuracy_score(y_test, y_true)*100)
                print("precision_score(in %):", metrics.precision_score(y_test,y_true)*100)
                print("roc_auc_score(in %):", metrics.roc_auc_score(y_test,y_true)*100)
                print("recall_score(in %):", metrics.recall_score(y_test,y_true)*100)
        #acc = accuracy_score(y_test, np.array(predicted.flatten() >= .5, dtype='int'))
##                fpr, tpr, thresholds = roc_curve(y_test, p)
##                auc = roc_auc_score(y_test, y_true)
##                print("auc:\n",auc)
        
                sns.set(style="whitegrid", font_scale=1.2)

                acc = history.history['accuracy']
                val_acc = history.history['val_accuracy']
                loss = history.history['loss']
                val_loss = history.history['val_loss']
                x = range(1, len(acc) + 1)
                print("acc list:\n")
                for i in acc:
                       print(",",i)
                       
                print("val_acc list:\n")
                for i in val_acc:
                       print(",",i)
                       
                print("loss list:\n")
                       
                for i in loss:
                       print(",",i)
                       
                print("val loss:\n")     
                  
                for i in val_loss:
                       print(",",i)
                            
                print("x range list:\n")
                  
                for i in x:
                       print(",",i) 

                from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
                cnf_matrix_tra = confusion_matrix(y_test, y_true)
                print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))
                class_names = [0,1]
                print("cnf_matrix_tra:\n")
                for i in cnf_matrix_tra:
                        print(",",i) 
                print("cnf_matrix_tra 2:\n") 
                print(cnf_matrix_tra)  
                pyplot.figure()
                plot_confusion_matrix(cnf_matrix_tra , classes=class_names)
                pyplot.show()
                plot_history(history)
                fpr, tpr, thresholds = roc_curve(y_test, y_true)
                roc_auc = auc(fpr,tpr)

                print("TPR list:\n")
##  
                for i in tpr:
                    print(",",i)
      
                print("FPR list:\n")
                for i in fpr:
                    print(",",i)



        training()


  
 








