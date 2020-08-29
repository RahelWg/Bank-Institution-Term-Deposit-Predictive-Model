#Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def my_data():
    data = pd.read_csv("bank-additional-full.csv", sep= ';')
    df = data.copy()
    df.drop(['duration', 'euribor3m', 'emp.var.rate'], axis=1, inplace=True)
    return df

# Function to define the upper and lower whisker for outlier removal
def find_normal_boundaries(df, var, distance):
    # calculate the boundaries outside which sit the outliers
    upper_whisker = df[var].mean() + distance * df[var].std()
    lower_whisker = df[var].mean() - distance * df[var].std()
    return upper_whisker, lower_whisker

# For removing columns with outliers
def remove_outliers(df):
    cols = [ 'age', 'campaign', 'previous']
    for i in cols:
        upper_limit, lower_limit = find_normal_boundaries(df, i, 1.5)
        # Substitute the high values with upper whisker
        # and low values with lower whisker
        df[i]= np.where(df[i] > upper_limit, upper_limit,                      
                       np.where(df[i] < lower_limit, lower_limit, df[i]))
        return df

# Function to define the oversampler for unbalanced data
def Over_sampler(df):
      # Oversampling df on condition
      from sklearn.utils import resample
      # Separate majority and minority classes
      yes_total = len(df[df.y_yes == 1].index)
      no_total = len(df[df.y_yes == 0].index)
      if (yes_total >= no_total):
          df_majority = df[df.y_yes==1]
          df_minority = df[df.y_yes==0]
      else:
          df_majority = df[df.y_yes==0]
          df_minority = df[df.y_yes==1]
      majority = max(yes_total, no_total)
      minority = min(yes_total, no_total)
  # Upsample minority class
      df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=majority,    # final minority class size
                                 random_state=minority) # reproducible results
  # Combine majority class with upsampled minority class
      return pd.concat([df_majority, df_minority_upsampled])
  

