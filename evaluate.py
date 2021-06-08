import math

import pandas as pd
import numpy as np
import seaborn as sns 
from pydataset import data 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


###################Function File for Evaluating Regression Model ####################


################### Function for Plotting Residuals ####################


def plot_residuals(df, y, yhat):
    '''
    This function takes in a target variable form a dataframe and there regression model 
    predictions (yhat) and plots there residuals
    '''
    print('----------------Residuals Histogram---------------')
    df.residuals.plot.hist()
    plt.show()
    
    print('------------Baseline Residuals Histogram----------')
    df.baseline_residuals.plot.hist()
    plt.show()
    
    print('-----------------------------Comparison Histogram--------------------------')
    fig, ax = plt.subplots(figsize=(10,8))
    ax.hist(df.baseline_residuals, label='Baseline Residuals', alpha = .7)
    ax.hist(df.residuals, label='Model Residuals', alpha = .7)
    ax.legend()
    plt.show()
    
################### Function for Model Error ####################

def regression_errors(df, y, yhat):
    '''
    This functions is an evaluation tool to calculate all the scores for a 
    Linear Regression model
    '''
    ##Calculate sse
    sse = (df.residuals ** 2).sum()
    
    ##Calculate ess
    ess = ((yhat - y.mean())**2).sum()
    
    #Calculate tss
    tss = ((y - y.mean())**2).sum()
    
    ##Calculate mse
    n = df.shape[0]
    mse = sse / n
    
    ##Calculate rmse
    rmse = math.sqrt(mse)

    print(f'''
    Regression Model Error
    
    Sum of Squared Errors / sse: {sse}
    Explained Sum of Squares / ess: {ess}
    Total Sum of Squares / tss: {tss}
    Mean of Squared Errors / mse: {mse}
    Root Mean of Squared Errors / rmse: {rmse}
    
    ''')
    
################### Function for Baseline Model Error ####################

def baseline_mean_errors(df):
    '''
    This function is an evaluation tool for a Regression model to calculate the error of the 
    baseline using the SSE, MSE, and RMSE metrics
    '''
    
    n = df.shape[0]
    
    sse_baseline = (df.baseline_residuals ** 2).sum()
    mse_baseline = sse_baseline / n
    rmse_baseline = math.sqrt(mse_baseline)

    print(f'Baseline scores --> sse: {sse_baseline} mse: {mse_baseline} rmse: {rmse_baseline}')

################### Better than the Baseline?? ####################

def better_than_the_baseline(df):
    '''
    This function is an evaluation tool to determine if the model error is better than
    the baseline error for a built regression model using a pandas dataframe
    '''
    n = df.shape[0]
    
    sse_baseline = (df.baseline_residuals ** 2).sum()
    mse_baseline = sse_baseline / n
    rmse_baseline = math.sqrt(mse_baseline)
    
    ##Calculate sse
    sse = (df.residuals ** 2).sum()
    
    ##Calculate mse
    n = df.shape[0]
    mse = sse / n
    
    ##Calculate rmse
    rmse = math.sqrt(mse)
    
    if rmse < rmse_baseline:
        return True
    else:
        return False


    

    
    
    