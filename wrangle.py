###################### Function Files for regression exercises #############################
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


#############################Connect To SQL database Function##############################

def sql_connect(db, user=user, host=host, password=password):
    '''
    This function allows me to connect the Codeup database to pull SQL tables
    Using private information from my env.py file.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#############################Return a cleaned telco_churn Dataframe##############################


def wrangle_telco():
    '''
    This function is going to acquire the neccessary columns customer_id, monthly_charges, tenure, 
    and total_charges from the telco_churn database for all customers with a 2-year contract.

    The function will then clean the dataframe using pandas ridding the dataframe of missing values
    and resetting the index
    '''

    sql_query = '''
    Select customer_id, monthly_charges, tenure, total_charges
    from customers
    where contract_type_id = 3;
    '''

    df = pd.read_sql(sql_query, sql_connect('telco_churn'))

    df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)

    df = df.reset_index()

    return df