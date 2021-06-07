###################### Function Files for regression exercises (Explore) #############################
from math import sqrt
from scipy import stats
from pydataset import data
from datetime import datetime


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import sklearn.metrics

import numpy as np
import pandas as pd
import seaborn as sns
from env import host, user, password, sql_connect



###################### Graph pairplot with regression Line #############################

def plot_variable_pairs(df):
    '''
    This function displays a pair plot off all the columns relationships and adds 
    a red regression line to the scatter plots
    '''
    sns.pairplot(df, corner = True, kind = 'reg', plot_kws={'line_kws':{'color':'red'}})
    

###################### Triple Visualization Function #############################


    
def plot_categorical_and_continuous_vars(df, cols, cats):
    for col in df[cols]:
        sns.relplot(data = df, x = df[col], y = cats, kind = 'scatter')
        plt.show()

    for col in df[cols]:
        sns.jointplot(data = df, x =df[col], y = cats, kind = 'scatter')
        plt.show()
        
    sns.heatmap(df.corr(), cmap ='RdBu', center = 0, annot = True, annot_kws={"size": 15})
    plt.show()