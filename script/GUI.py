from hashlib import new
import streamlit as st
import pandas as pd
import sys
sys.path.append('C:/Users/Acer/Desktop/VProject/')
sys.path.append('C:/Users/Acer/Desktop/VProject/datasets/')
from utils.data_cleaning import features_, outliers_treatment, missing_values_treatment
from utils.data_transformation import Normalization, Discretization, Attribute_selection
from utils.data_reduction import Attribute_subset_selection, Dimensionality_reduction
from utils.data_splitting import Split_data, space_res, space
from utils import ml_models
import warnings
warnings.filterwarnings("ignore")


st.markdown('<p style="background-color:#85B19E;color:#F1FAF6;text-align: center;font-size:38px;border-radius:3%;"><b>{}</b></p>'.format('MACHINE LEARNING GUI'), unsafe_allow_html=True)
space()
df=''
target=''
features=''
data = st.file_uploader("Upload your csv data file", type='csv')
if data:
    df = pd.read_csv('C:\\Users\\Acer\\Desktop\\VProject\\datasets\\'+str(data.name))
if st.checkbox('Show data') and data:
    st.text("")
    st.write("Shape of data: ",df.shape)
    st.text("")
    st.write(df)
    space()
    target = st.text_input("Enter the target variable")
    features = features_(target, df)
preprocessing = st.selectbox('Choose data preprocessing technique', ['None', 'Data preprocessing', 'Image preprocessing'])
if preprocessing=='None':
    pass
if preprocessing=='Data preprocessing':
    space()
    st.markdown('<p style="text-align: left;font-size:18px;"><b>{}</b></p>'.format('Data Cleaning'), unsafe_allow_html=True)
    cleaning = st.radio('Choose either: ', ['None', 'Missing Data', 'Noisy Data', 'Both'])
    st.write('<p style="text-align: left;"><b>{}</b>{}</p>'.format('Note: ','If you choose "Both" then this will first treat the missing values (if any) then treat outliers (if any).'), unsafe_allow_html=True)
    if cleaning=='None':
        temp_data=df

    if cleaning=='Noisy Data':
        temp_data = outliers_treatment(data=df, features=features, target_col=target)
        st.info("Your data is now noise free! Outliers have been treated.")
        
    if cleaning=='Missing Data':
        temp_data = missing_values_treatment(data=df, features=features, target_col=target)
        st.info("Your data is now free from null values! Missing values have been treated.")
        
    if cleaning=='Both':
        missing_treated_data = missing_values_treatment(data=df, features=features, target_col=target)
        temp_data =  outliers_treatment(data=missing_treated_data, features=features, target_col=target)
        st.info("Your data is now cleaned completely! Missing value & Outliers have been treated.")
        
    space()
    st.markdown('<p style="text-align: left;font-size:18px;"><b>{}</b></p>'.format('Data Transformations'), unsafe_allow_html=True)
    transformer = st.radio('Choose either: ', ['None', 'Normalization', 'Discretization', 'Attribute Selection'])
    st.write('<p style="text-align: left;"><b>{}</b>{}</p>'.format('Note: ','If you choose "Discretization" or "Attribute Selection", make sure that the data is free from missing/null values'), unsafe_allow_html=True)
    if transformer=='None':
        new_temp_data = temp_data

    if transformer=='Normalization':
        new_temp_data = Normalization(data=temp_data, features=features, target_col=target)
        st.info("Your data is normalized! All values of instances have been scaled.")
        st.write(new_temp_data)
        
    if transformer=='Discretization':
        new_temp_data = Discretization(data=temp_data, features=features, target_col=target)
        st.info("Your data is discretized! All values have been made discrete.")
        
    if transformer=='Attribute Selection':
        num_attributes = st.slider('Select number of best attributes you want', min_value=2, max_value=len(features), step=1)
        new_temp_data = Attribute_selection(data=temp_data, features=features, target_col=target, k=num_attributes)[0]
        st.info("Attributes are selected from the the set of attributes of dataset.") 
        st.write("Your selected attributes are: ", Attribute_selection(data=temp_data, features=features, target_col=target, k=num_attributes)[1])
        
    space()
    st.markdown('<p style="text-align: left;font-size:18px;"><b>{}</b></p>'.format('Data Reduction'), unsafe_allow_html=True)
    reduction_technique = st.radio('Choose either: ', ['None', 'Attribute Subset Selection', 'Dimensionality Reduction'])
    st.write('<p style="text-align: left;"><b>{}</b>{}</p>'.format('Note: ','To perform any of the data reduction techniques, make sure that the data is free from missing/null values'), unsafe_allow_html=True)
    if reduction_technique=='None':
        new_new_temp_data = new_temp_data

    if reduction_technique=='Attribute Subset Selection':
        c = 0
        for col in new_temp_data.columns:
            if col not in df.columns:
                c = c+1
        if c==0:
            num_attributes = st.slider('Select number of best attributes you want', min_value=2, max_value=len(features), step=1)
        else:  
            temp_features_len = len([i for i in new_temp_data.columns])
            num_attributes = st.slider('Select number of best attributes you want', min_value=2, max_value=temp_features_len, step=1)
        new_new_temp_data = Attribute_subset_selection(data=new_temp_data, features=features, target_column=target, k=num_attributes)
        st.info("The data has been reducted! Subset of best Attributes is selected from the the set of attributes of dataset.")
        st.write("Your selected subset of attributes are: ", Attribute_subset_selection(data=temp_data, features=features, target_column=target, k=num_attributes)[1])

    if reduction_technique=='Dimensionality Reduction':
        c = 0
        for col in new_temp_data.columns:
            if col not in df.columns:
                c = c+1
        if c==0:
            n_components = st.slider('Select number of components for PCA', min_value=2, max_value=len(features), step=1)
        else: 
            temp_features_len = len([i for i in new_temp_data.columns])
            n_components = st.slider('Select number of components for PCA', min_value=2, max_value=temp_features_len, step=1)  
        new_new_temp_data = Dimensionality_reduction(data=new_temp_data, features=features, target_col=target, n_comp=n_components)
        st.info("The data has been reducted by reducing the dimensionality. The dimensionality of your dataset has been reduced into selected number of components.")

    if preprocessing=='Image preprocessing':
        pass
if st.checkbox('Start Model Training'):
    #models
    algo = st.selectbox('Select your Machine Learning algorithm:', ['None', 'Logistic Regression', 'Linear Regression', 'CatBoost', 'AdaBoost', 'Random Forest', 'Decision Tree',  'Support Vector Machine', 'K Means Clustering', 'K Nearest Neighbors', 'Naive Bayes'])
    ratio = st.slider('How much data do you want in training set?', min_value=0.6, max_value=0.9, step=0.1)
    splitted_data = Split_data(new_new_temp_data[0], target, split_ratio=abs(1.0-ratio))
    X_train, X_test, y_train, y_test = splitted_data[0], splitted_data[1],splitted_data[2], splitted_data[3]
    if algo=='None':
        st.warning('No algorithm has been selected! Please select any one algorithm.')
    if algo=='Logistic Regression':
        matrix = ml_models.Logistic_regression(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test)  
        space_res(matrix)

    if algo=='Linear Regression':
        acc = ml_models.Linear_regression(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test)    
        if st.button('Display Accuracy'):
            space()
            st.write('<p style="text-align: left;"><b>{}</b></p>'.format('ACCURACY'), unsafe_allow_html=True)
            st.info(acc)
            
    if algo=='Support Vector Machine':
        matrix = ml_models.Support_vector_classifier(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test)  
        space_res(matrix)
            
    if algo=='AdaBoost':
        matrix = ml_models.Adaboost_classifier(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test)  
        space_res(matrix)
            
    if algo=='CatBoost':
        matrix = ml_models.Catboost_classifier(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test)  
        space_res(matrix)   
            
    if algo=='K Means Clustering':
        matrix = ml_models.K_means_classifier(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test)  
        if st.button('Display C'):
            pass
            
    if algo=='Naive Bayes':
        matrix = ml_models.Naive_bayes_classifier(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test)  
        space_res(matrix)
            
    if algo=='K Nearest Neighbors':
        matrix = ml_models.K_neighnors_classifier(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test)  
        space_res(matrix)

    if algo=='Decision Tree':
        matrix = ml_models.Decision_tree_classifier(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test)  
        space_res(matrix)
            
    if algo=='Random Forest':
        matrix = ml_models.Random_forest_classifier(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test)    
        space_res(matrix)
            
