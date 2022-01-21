
"""## **DATA CLEANING**

- **missing_values_treatment**: this function removes missing values if these are more than 49% of data, else replaced by attribute mean

- **outlier-treatment**: this function, firstly, separates the data that doesn't come under Interquantile range (are treated as outliers), and then replaces those outlier values with attribute median. We can replace it with mean also. Both works similar, but mode works differently. Mean and median's value is similar.

We can call both methods or call either of both. Action will affect accuracy
- the data, only cleaned from *missing_values_treatment* method but not with *outlier_treatment*, may contain outliers if data has.

- the data, only cleaned from *outlier_treatment* method but not with *missing_values_treatment* method, may contain missing values if data has.

if both methods are called off then we'll have noise free data
"""


import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")


def features_(target_column, data):
    # Encoding object type features intpo numrical ones
    # Categorical boolean mask
    categorical_feature_mask = data.dtypes==object
    # filtering categorical columns using mask and turning it into a list
    categorical_cols = data.columns[categorical_feature_mask].tolist()
    # apply LabelEncoder on categorical feature columns
    data[categorical_cols] = data[categorical_cols].apply(lambda col: LabelEncoder().fit_transform(col))
   
    attributes = []    #getting all attributes of dataset except target attribute
    for col in data.columns:
        if col!=target_column:
            attributes.append(col)
    return attributes

#missing values treatment
def missing_values_treatment(data, features, target_col):
  #checking each column, whether it contains missing values or not. If yes, then check for the criteria i.e., are the values greater than 49% of data? 
  #If yes, then remove that attribute, else replacing those values with mean
  for feature in features:
    missing_values = data[feature].isnull().sum()
    if missing_values>((49*len(data))//100):
      data = data.drop(feature, axis=1)
    else:
      data[feature].fillna(data[feature].median(), inplace=True)
  
  return data        #returns dataframe without any missing value



#outliers treatment
def outliers_treatment(data, features, target_col):
  #whatever data is entered is going to be checked for outliers
  to_process_data = data.drop(target_col, axis=1)

  #calculting quantiles, Q1-->25%  & Q3-->75%. Data values that lie outside IQR are treated as outliers.
  #the values less than Q1 and greater than Q3 are treated as outliers.
  Q1 = to_process_data.quantile(0.25)
  Q3 = to_process_data.quantile(0.75)

  #Calculating the range under which data must lie
  IQR = Q3 - Q1
  
  #below print statement returns  data with boolean values (either true or false); true represents outlier
  #print((to_process_data < (Q1 - 1.5 * IQR)) | (to_process_data > (Q3 + 1.5 * IQR)))

  #clean_data contains data without outliers
  clean_data = data[~((to_process_data < (Q1 - 1.5 * IQR)) | (to_process_data > (Q3 + 1.5 * IQR))).any(axis=1)]

  #outliers contains outlier data; If we remove these values from data then it will reduce the quantity of data
  outliers = data[((to_process_data < (Q1 - 1.5 * IQR)) | (to_process_data > (Q3 + 1.5 * IQR))).any(axis=1)] 

  #removing outliers
  #replacing outliers with attribute median
  out_cols = []   # this contains column names of outliers data
  for col in to_process_data.columns:
      out_cols.append(col)
  for attribute in out_cols:
      outliers[attribute] = clean_data[attribute].median()  #calculating the median of each column and putting it in outliers data in same column
  outlier_free_data = pd.concat([clean_data,outliers], axis=0)   

  return outlier_free_data    #returns dataframe; outlier free  (No data is removed. All are reserved but modified with appropriate values)


