
"""# **---------------------------------------------------------------------------------------------------**

# DATA REDUCTION

## **DATA REDUCTION**

- **attribute_subset_selection**: this function selects the subset of best k features out of all features given in the dataset. This methods returns the dataframe of selected features along with target feature.

- **dimensionality_reduction**: this function is generating a dataframe with features on which PCA is applied, along with target feature. The n_comp is taking value provided by the user, else takes the default value, set as 6.

 **The data to be passed in the *dimensionality_reduction* method should be normalized.**
"""

import pandas as pd
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import warnings
warnings.filterwarnings("ignore")

def Attribute_subset_selection(data, features, target_column, k=10):
  # Sequential Forward Selection(sfs)
  X = data.drop([target_column], axis=1)
  y = data[target_column]
  sfs = SFS(estimator=LogisticRegression(),
            k_features=k,
            forward=True,
            floating=False,
            scoring = 'r2',
            cv = 0)
  sfs_data = sfs.fit(X, y)
  attribute_subset = [attribute for attribute in sfs.k_feature_names_]

  attribute_subset_data = data[attribute_subset]
  attribute_subset_data[target_column] = data[target_column]
  attribute_subset_data = shuffle(attribute_subset_data)    
  attribute_subset_data.reset_index(inplace=True, drop=True)
  # print('-----------------BEST ATTRIBUTES SUBSET ------------------')
  # print(attribute_subset)
  # print('-----------------BEST ATTRIBUTES SUBSET DATA------------------')
  return attribute_subset_data, attribute_subset


def Dimensionality_reduction(data, features, target_col, n_comp=6):
  #dimensionality reduction (here, pca) is applied on independent features only
  X = data.drop([target_col], axis=1)
  y = data[target_col]

  #proving the number of components, as default=6
  pca = PCA(n_components=n_comp, random_state=0)
  pca.fit(X)
  pca_data = pca.transform(X)   #returns data in array format

  #converting the data from array fromat to dataframe along with target feature
  pc = ['PC'+str(i+1) for i in range(pca_data.shape[1])]
  dimensionality_reducted_data = pd.DataFrame(pca_data, columns=pc)
  dimensionality_reducted_data[target_col] = y
  dimensionality_reducted_data = shuffle(dimensionality_reducted_data)    
  dimensionality_reducted_data.reset_index(inplace=True, drop=True)
  #this is returning reduced and compressed data

  return dimensionality_reducted_data, 0   #returns data with features names as [PC1, PC2, ....., PCn] with 'target' as a target feature

