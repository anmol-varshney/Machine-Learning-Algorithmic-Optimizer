
"""# **---------------------------------------------------------------------------------------------------**

## **DATA TRANSFORMATION**

- **normalization**: this function normalizes the data by applying Standard Scaling on the data that scales values closer to 0. It is needed so that the data become normally distributed instead of skewed in either direction.

- **Discretization**: this function is applied to the features that are continuous in nature, to conver it into discrete ones. This helps in classification tasks where the label is also discrete. 
This fucntion is encoding the values as 'ordinal' by diving the data in 8 bins of particular features. This returns discretized data; ordinal values.
"""


import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import ExtraTreesClassifier
import warnings
warnings.filterwarnings("ignore")


#Normalization
def Normalization(data, features, target_col):
  X = data.drop(target_col, axis=1)
  columns = [col for col in X.columns]

  #scaling big values into smaller ones; scaled value close to 0
  X = StandardScaler().fit_transform(X.astype(float))
  normalize_data = pd.DataFrame(X,columns=columns)
  normalize_data[target_col] = data[target_col]

  return normalize_data    #returns normalized data with smaller values


#Discretization
def Discretization(data, features, target_col):
  #Instantiating the method; providing n_bins=8, the data is going to be divided into 8 bins and then the ordinal values will be assigned.
  #this method is useful if we have to do classification task
  discretizer = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform') 
  X = data.drop(target_col, axis=1)
  columns = [col for col in X.columns]
  discrete_data = discretizer.fit_transform(X)
  discrete_data = pd.DataFrame(discrete_data, columns=columns)
  discrete_data[target_col] = data[target_col]

  return discrete_data      #returns dataframe with discrete features


#feature selection
#for classification dataset
def Attribute_selection(data, features, target_col, k=10):
  X = data.drop(target_col, axis=1)
  y = data[target_col]
  bestfeatures = ExtraTreesClassifier()
  bestfeatures.fit(X,y)
  feat_importances = pd.Series(bestfeatures.feature_importances_,index=X.columns)
  best_features = feat_importances.nlargest(k)
  best_features = [feat for feat in best_features.index]
  bestfeatures_data = data[best_features]
  bestfeatures_data[target_col] = data[target_col]
  bestfeatures_data = shuffle(bestfeatures_data)    
  bestfeatures_data.reset_index(inplace=True, drop=True)

  # print('-----------------BEST FEATURES------------------')
  # print(best_features)
  # print('-----------------BEST FEATURES DATA------------------')

  return bestfeatures_data, best_features