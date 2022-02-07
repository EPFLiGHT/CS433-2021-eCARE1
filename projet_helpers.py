# File with usefull functions for the project eCare
#
# author Xavier Nal, Louis Gounot, Louis Bettens
#
#some functions come from the epoct_simplified code but 
#are here adapted to the eCare Project
#
# date 23.12.2021
#

import numpy as np
import pandas as pd
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import epoct_helpers

from epoct_helpers import* 

# for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def get_accuracy(qst, X_test, y_test, initial_state, encoders,
                       diseases_decoders):
    """
    Use disease and question decoders on unseen test set
    :param qst: questionnaire data set object
    :param X_cont_test: tensor with continuous features of test data
    :param X_cat_test: tensor with categorical features of test data
    :param y_test: tensor with targets of test data
    :param continuous_feature_encoders: dict of trained continuous_feature_encoders
    :param categorical_feature_encoders: dict of trained categorical feature encoders
    :param diseases_decoders: trained disease decoder
    :param encoders_scalars: parameters to weight contribution of encoders
    :return: print the accuracy of the prediction 
    """
 
    y_prediction = torch.zeros(y_test.shape)
    y_prediction = y_prediction.numpy()
 
    with torch.no_grad():
        for patient in range(len(y_test)):
            test_state = torch.tile(
                initial_state.state_value, (1, 1))
            
            decoder_output = torch.empty(
                (qst.num_available_features[qst.testing_data_indices_reverse_mapping[patient]]+1, len(qst.disease_names)))
            # track number of encoded features
            feature_rank = 1
            list_features = ['No information']
            # before anything has been encoded
            for index, name in enumerate(qst.disease_names):
                disease_pred = diseases_decoders[name](test_state)
                decoder_output[0, index] = torch.exp(disease_pred[:, 1])
            # apply trained encoders for each level in the tree
            for level in qst.order_test_features[patient].keys():
                feature_group = shuffle(
                    qst.order_test_features[patient][level])
                for feature_name in feature_group:
                    test_state = encoders[feature_name](test_state,
                                                        X_test[
                                                            patient, qst.feature_names.index(
                                                                feature_name)].view(-1,
                                                                                    1))

                    for index, name in enumerate(qst.disease_names):
                        disease_pred = diseases_decoders[name](test_state)
                        decoder_output[feature_rank, index] = torch.exp(
                            disease_pred[:, 1])
                    feature_rank += 1
                    list_features.append(feature_name)
                    
           
            y_pre, value_2 = torch.sort(decoder_output[len(X_test[0])-1],dim=0, descending=True )
          
            y_pre_2 = y_pre.numpy()
          
            larger = value_2[0].numpy()
       
            
            if y_pre_2[0] >= 0.5:
                y_prediction[patient, larger] = 1


        
        y_test_2 = y_test.numpy()

        # calculate the accuracy by comparing y test and y prediction for every disaese
        for i in range(len(y_test_2[0,:])):
            print('le taux de similitude est de ',(y_test_2[:,i]==y_prediction[:,i]).sum()/len(y_prediction[:,0])*100)
                  
    return
def train_and_test_modules(qst_obj):
    """
    Trains and tests encoder/decoder modules
    original function with the epoct_simplified
    :param qst_obj : questionnaire object dataset
    :return:
    """
    # set state_size
    STATE_SIZE = 30
    # get preprocessed train, valid and test sets
    (X_train, X_valid, X_test,
     y_train, y_valid, y_test) = preprocess_data_epoct(qst_obj, valid_size=0.2, test_size=0.2, fraction = 1/5)
                                                                                               
                                                                                               
    # instanciate modules
    initial_state = InitState(STATE_SIZE)
    encoders = {name: EpoctEncoder(STATE_SIZE) for name in qst_obj.feature_names}
    single_disease_decoders = {name: EpoctBinaryDecoder(STATE_SIZE) for name in qst_obj.disease_names}

    train_modules_epoct_data_all_levels_simplified(qst_obj, X_train, y_train, X_valid, y_valid,
                                                                  initial_state,
                                                    encoders,
                                                    single_disease_decoders)
    
    get_accuracy(qst_obj, X_test, y_test, initial_state, encoders,single_disease_decoders)
    
    #get_heatmaps_epoct(qst_obj, X_test, y_test, initial_state, encoders,
               # single_disease_decoders)
    
    return


def cleaning_function(df):
    """
    Clean the data set eCare
    :param df : the dataframe
    :return df: the dataframe cleaned
    """
    # delete consultation ids
    df = df.drop('consultation', axis=1)
    
    # delete the anonymous user code
    df = df.drop('slot_id', axis=1)  
    
    #delete because columns with only 0 or nan values
    df = df.drop('uti_vomiting', axis=1)              
    df = df.drop('classify_uti_complicated', axis=1) 
    
    #delete because not usefull for the prediction disease
    df = df.drop('hf', axis=1)                  
    
    #clean the date
    df['created'] = pd.to_datetime(df['created'], errors='coerce').dt.month
    df=df.drop('anaphyl_confirmed',axis=1)
    
    # delete columns with 87% of missing data
    df = df[ df.columns[df.isna().sum()/df.shape[0] < 0.87]] 
 
    #clean age and gender
    df['a_age'] = df['a_age'].replace('normal','nan')
    df['a_gender2'] = df['a_gender2'].replace('574143ff-2257-438e-932f-291250d6c2cf','nan')
    df['a_gender2'] = df['a_gender2'].replace('normal','nan')
    df['a_gender2'] = df['a_gender2'].replace('male','0')
    df['a_gender2'] = df['a_gender2'].replace('female','1')
   
    #put age and gender in type float
    df['a_age'] = df['a_age'].astype(float)
    df['a_gender2'] = df['a_gender2'].astype(float)
    
    #remplace strange value with nan
    df['classify_diarrhoea_acute_without'] = df['classify_diarrhoea_acute_without'].replace('green','nan') 
    df['classify_dysentery_complicated'] = df['classify_dysentery_complicated'].replace('green','nan')
    df['classify_dysentery_non_compl'] = df['classify_dysentery_non_compl'].replace('none','nan') 
    df['classify_ear_mastoiditis'] = df['classify_ear_mastoiditis'].replace('none','nan')
    df['source'] = df['source'].replace('abdominal','nan')
    df['source'] = df['source'].replace('coartem','nan')
    df['source'] = df['source'].replace('chaud','nan')
    df['source'] = df['source'].replace('eau','nan')
    df['source'] = df['source'].replace('sro','nan')
    df['source'] = df['source'].replace('bas','nan')
    df['classify_eye_conjunctivitis_vira'] = df['classify_eye_conjunctivitis_vira'].replace(
        '["1fafe800-fd5b-11e3-a3ac-0800200c9a66"]','nan') 
    df['classify_eye_bacterial_conjuncti'] = df['classify_eye_bacterial_conjuncti'].replace(to_replace = 39.6, value = 'nan')

    df['abdo_inguin_mass'] = df['abdo_inguin_mass'].replace('574143ff-2257-438e-932f-291250d6c2cf','nan')
    
    df['cough_ds'] = df['cough_ds'].replace({'["1fafe800-fd5b-11e3-a3ac-0800200c9a66"]':'nan',
                                        '[1fafe800-fd5b-11e3-a3ac-0800200c9a66]':'nan'})
    df['hydration_thirst'] = df['hydration_thirst'].replace({'9f1e50be-ea44-45bf-baba-15e6079b2268':'nan', 
                                                         '574143ff-2257-438e-932f-291250d6c2cf':'nan',
                                                        '5ad025d5-b7fe-4fe6-ac2d-aa4c75914cb4':'nan'})
    
    #makes the column cough_ds uniform
    df['cough_ds'] = df['cough_ds'].replace({'["unable-to-talk"]':'unable-to-talk',
                                            '[unable-to-talk]':'unable-to-talk',
                                            '["grunting"]':'grunting',
                                            '[grunting]':'grunting',
                                            '[stridor]':'stridor',
                                            '["stridor"]': 'stridor',
                                            '["cyanosis"]':'cyanosis',
                                            '[cyanosis]':'cyanosis',
                                            '["unable-to-talk","grunting"]':'[unable-to-talk,grunting]',
                                            '["grunting","stridor"]':'[stridor,grunting]',
                                            })
    #put the columns in type float
    df['source']=df['source'].dropna()

    df['classify_diarrhoea_acute_without'] = df['classify_diarrhoea_acute_without'].astype(float)
    df['classify_dysentery_complicated'] = df['classify_dysentery_complicated'].astype(float)
    df['classify_dysentery_non_compl'] = df['classify_dysentery_non_compl'].astype(float)
    df['classify_ear_mastoiditis'] = df['classify_ear_mastoiditis'].astype(float)
    df['classify_eye_conjunctivitis_vira'] = df['classify_eye_conjunctivitis_vira'].astype(float)
    df['classify_eye_bacterial_conjuncti'] = df['classify_eye_bacterial_conjuncti'].astype(float)

    df['abdo_inguin_mass'] = df['abdo_inguin_mass'].astype(float)
    df['source'] = df['source'].astype(float)

    return df
    

def sort_datas_by_countries(df_cleaned):
    """
    sort data by countries
    :param df_cleaned : the data frame cleaning
    :return: dataframe for each countries
    """
    df_RCA=df_cleaned[df_cleaned['source']==0]
    df_Niger=df_cleaned[df_cleaned['source']==1.]
    df_Nigeria=df_cleaned[df_cleaned['source']==2.]
    df_Tanzania=df_cleaned[df_cleaned['source']==3]
    df_Mali=df_cleaned[df_cleaned['source']==4]
    df_Tchad=df_cleaned[df_cleaned['source']==5]
    df_Niger2=df_cleaned[df_cleaned['source']==6]
    df_Kenya=df_cleaned[df_cleaned['source']==7]
    df_RCA2=df_cleaned[df_cleaned['source']==8]
    
    return df_RCA,df_Niger,df_Nigeria,df_Tanzania,df_Mali,df_Tchad,df_Niger2,df_Kenya,df_RCA2

def targets_features_separation(df_cleaned):
    """
    The original function is in epoct_simplified code ( Cecile  Trottet master thesis)
    adapted for the dataset eCare
    Normalize, drop the unbalanced columns
    :param qst_obj : questionnaire object dataset
    :return: features and targets 
    """
     
    classify=df_cleaned.filter(regex='classify')
    hisdx=df_cleaned.filter(regex='hisdx')
    done=df_cleaned.filter(regex='done')
    target_labels = df_cleaned.filter(regex='classify_cough_pneumonia|classify_anemia|classify_cough_bronchiolitis|classify_cough_urti|classify_malaria|classify_very_severe_disease_mal').columns
   
    target_labels=df_cleaned[target_labels]

    features_labels=df_cleaned.drop(columns=classify)
    features_labels=features_labels.drop(columns=hisdx)
    features_labels=features_labels.drop(columns=done)
   
    well_processed_targets = []
    very_unbalanced_targets = []
    well_processed_features = []
    very_unbalanced_features = []
    targets_to_process = []
    num_patients = len(df_cleaned)
  
    # find proportion in each target and feature  class
    for feature in features_labels:
        number_zeros = len(df_cleaned[feature][df_cleaned[feature] == 0])
        number_ones = len(df_cleaned[feature][df_cleaned[feature] == 1])
        if number_ones / num_patients < 0.001 or number_zeros / num_patients < 0.001:
            very_unbalanced_features.append(feature)
        else:
            well_processed_features.append(feature)
    features=df_cleaned[well_processed_features].copy()

    for target in target_labels:
        number_zeros = len(df_cleaned[target][df_cleaned[target] == 0])
        number_ones = len(df_cleaned[target][df_cleaned[target] == 1])

        if number_ones / num_patients < 0.001 or number_zeros / num_patients < 0.001:
            very_unbalanced_targets.append(target)
        else:
            well_processed_targets.append(target)

    # drop targets with too unbalanced categories
    targets = df_cleaned[well_processed_targets].copy()
    #features = df.drop(columns=target_labels)
    #features.drop(columns=['lab_malaria_test_any_done'],inplace=True)
    features.drop(columns=['lab_malaria_test_any_positive'],inplace=True)
    
    return features,targets