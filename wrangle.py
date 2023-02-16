import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model as m
import os
import seaborn as sns
from math import sqrt
from scipy import stats
from pydataset import data
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split



def get_dnd():
    
    '''
    This function is used to acquire the dnd_stats.csv from the local file. If the file does not already exist,
    the function will create the file.
    ''' 
    
    if os.path.isfile('dnd_stats.csv'):
        
        return pd.read_csv('dnd_stats.csv')
    
    else:
        
        df = pd.read_csv("dnd_stats.csv")
    
        return df  

def the_split(df, stratify= None):

    """ This functions is used to split the data into 3 different datasets: train, validate(val), and test.
        It then then returns the seperate datasets and prints the shape for each of them.
    """
        
    # train/validate/test split and is reproducible due to random_state = 123
    train_validate, test= train_test_split(df, test_size= .2, random_state= 7)
    train, val= train_test_split(train_validate, test_size= .3, random_state= 7)
    
    print(f'Train shape: {train.shape}\n' )
    
    print(f'Validate shape: {val.shape}\n' )
    
    print(f'Test shape: {test.shape}')
    
    return train, val, test

def prep_dnd(df):
    dummy_df = pd.get_dummies(df[['race']], drop_first= True)
    df = pd.concat( [df, dummy_df], axis=1 )
    df.rename(columns= {'race_half.elf': 'race_half_elf', 'race_half.orc': 'race_half_orc'}, inplace= True)
    train, val, test= the_split(df)
    return train, val, test


def the_split(df, stratify= None):

    """ This functions is used to split the data into 3 different datasets: train, validate(val), and test.
        It then then returns the seperate datasets and prints the shape for each of them.
    """
        
    # train/validate/test split and is reproducible due to random_state = 123
    train_validate, test= train_test_split(df, test_size= .2, random_state= 7)
    train, val= train_test_split(train_validate, test_size= .3, random_state= 7)
    
    print(f'Train shape: {train.shape}\n' )
    
    print(f'Validate shape: {val.shape}\n' )
    
    print(f'Test shape: {test.shape}')
    
    return train, val, test




def new_split(df, stratify= None):

    """ This functions is used to split the data into 3 different datasets: train, validate(val), and test.
        It then then returns the seperate datasets and prints the shape for each of them.
        Different from the previously used `six_splt` because it does not print the dataset shapes
    """
        
    # train/validate/test split and is reproducible due to random_state = 123
    train_validate, test= train_test_split(df, test_size= .2, random_state= 7)
    train, val= train_test_split(train_validate, test_size= .3, random_state= 7)
    

    return train, val, test
