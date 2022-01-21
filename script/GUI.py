BASE_DIR = 'https://github.com/Code-with-Palak/Machine-Learning-Algorithmic-Optimizer/'
dataset_path = BASE_DIR+'datasets/'
from hashlib import new
import streamlit as st
import pandas as pd
import sys
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR+'datasets/')
from utils.data_cleaning import features_, outliers_treatment, missing_values_treatment
from utils.data_transformation import Normalization, Discretization, Attribute_selection
from utils.data_reduction import Attribute_subset_selection, Dimensionality_reduction
from utils.data_splitting import Split_data, space_res, space
from utils import ml_models
from pandas_profiling import ProfileReport
import re
import webbrowser
import warnings
warnings.filterwarnings("ignore")


st.markdown('<p style="background-color:#85B19E;color:#F1FAF6;text-align: center;font-size:38px;border-radius:3%;"><b>{}</b></p>'.format('MACHINE LEARNING GUI'), unsafe_allow_html=True)
space()
df=''
target=''
features=''
data = st.file_uploader("Upload your csv data file", type='csv')

if data:
    data_name = data.name
    df = pd.read_csv(dataset_path+str(data_name))
if st.checkbox('Show data'):
    st.text("")
    st.write("Shape of data: ",df.shape)
    st.write("Number of samples in dataset: ", df.shape[0])
    st.write("Number of attributes in dataset: ", df.shape[1])
    st.write("Attributes: ", list(df.columns))
    st.text("")
    st.write(df)
    space()
    
eda = st.checkbox('Exploratory Data Analysis')
if eda and data:
    st.text("")
    
    
    rep = ProfileReport(df, explorative=True)
    rep.to_file(BASE_DIR+'EDA and Performance reports\\visualization{}.html'.format(' of '+ re.sub('.csv', '', str(data_name))))
    st.success('The Data visualization has been done.')
    webbrowser.open_new_tab(BASE_DIR+'EDA and Performance reports\\visualization{}.html'.format(' of '+re.sub('.csv','', str(data.name))))
    # st.write('The Visualization report for the selected dataset has been saved to this location {}'.format(BASE_DIR+'EDA and Performance reports/visualization{}.html'.format(' of '+re.sub('.csv','', str(data_name)))))
    st.write('<p style="text-align: left;"><b>{}</b>{}</p>'.format('Note: ', 'Please deselect the Exploratory Data Analysis checkbox for smoother performance.'), unsafe_allow_html=True)
    space()

if data:
    target = st.text_input("Enter the target variable")
    features = features_(target, df)
preprocessing = st.selectbox('Choose data preprocessing technique', ['None', 'Data preprocessing'])
if preprocessing=='None':
    pass
if preprocessing=='Data preprocessing':
    space()
    st.markdown('<p style="text-align: center;font-size:38px;"><b>{}</b></p>'.format('DATA PREPROCESSING'), unsafe_allow_html=True)
    
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
        if st.checkbox('View normalized data'):
            st.text("")
            st.write(new_temp_data)
        
    if transformer=='Discretization':
        new_temp_data = Discretization(data=temp_data, features=features, target_col=target)
        st.info("Your data is discretized! All values have been made discrete.")
        if st.checkbox('View discretized data'):
            st.text("")
            st.write(new_temp_data)
        
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
        if len(new_temp_data.columns)==len(df.columns):
            num_attributes = st.slider('Select number of best attributes you want', min_value=2, max_value=len(features), step=1)
        else: 
            temp_features_len = len([i for i in new_temp_data.columns])
            num_attributes = st.slider('Select number of best attributes you want', min_value=2, max_value=temp_features_len, step=1)
        new_new_temp_data = Attribute_subset_selection(data=new_temp_data, features=features, target_column=target, k=num_attributes)[0]
        st.info("The data has been reducted! Subset of best Attributes is selected from the the set of attributes of dataset.")
        st.write("Your selected subset of attributes are: ", Attribute_subset_selection(data=temp_data, features=features, target_column=target, k=num_attributes)[1])
        
            
    if reduction_technique=='Dimensionality Reduction':
        if len(new_temp_data.columns)==len(df.columns):
            n_components = st.slider('Select number of components for PCA', min_value=2, max_value=len(features), step=1)
        else: 
            temp_features_len = len([i for i in new_temp_data.columns])
            n_components = st.slider('Select number of components for PCA', min_value=2, max_value=temp_features_len, step=1)  
        new_new_temp_data = Dimensionality_reduction(data=new_temp_data, features=features, target_col=target, n_comp=n_components)[0]
        st.info("The data has been reducted by reducing the dimensionality. The dimensionality of your dataset has been reduced into selected number of components.")
        if st.checkbox('View reducted data'):
            st.text("")
            st.write(new_new_temp_data)
   
if st.checkbox('Start model deployment'):
    #models
    algo = st.selectbox('Select your Machine Learning algorithm:', ['None', 'AdaBoost', 'Decision Tree', 'Gradient Boosting Classifier', 'K Nearest Neighbors', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'XGBoost Classifier'])
    ratio = st.slider('How much data do you want in training set?', min_value=0.6, max_value=0.9, step=0.1)
    
    splitted_data = Split_data(new_new_temp_data, target, split_ratio=abs(1.0-ratio))
    X_train, X_test, y_train, y_test = splitted_data[0], splitted_data[1],splitted_data[2], splitted_data[3]
    if algo=='None':
        st.warning('No algorithm has been selected! Please select any one algorithm.')
    if algo=='Logistic Regression':
        report, measure, mat = ml_models.Logistic_regression(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test, data_name=data_name, algorithm_selected = algo)
        space_res(report, measure, mat)
            
    if algo=='Support Vector Machine':
        report, measure, mat = ml_models.Support_vector_classifier(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test,data_name=data_name, algorithm_selected = algo)  
        space_res(report, measure, mat)
        
    if algo=='XGBoost Classifier':
        report, measure, mat = ml_models.XGB_Classifier(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test, data_name=data_name, algorithm_selected = algo)
        space_res(report, measure, mat)
            
    if algo=='Gradient Boosting Classifier':
        report, measure, mat = ml_models.Gradient_Classifier(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test, data_name=data_name, algorithm_selected = algo)  
        space_res(report, measure, mat)
        
    if algo=='AdaBoost':
        report, measure, mat = ml_models.Adaboost_classifier(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test, data_name=data_name, algorithm_selected = algo)  
        space_res(report, measure, mat)
            
    # if algo=='CatBoost':
    #     report, measure, mat = ml_models.Catboost_classifier(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test, data_name=data_name, algorithm_selected = algo)  
    #     space_res(report, measure, mat)   
            
    if algo=='K Nearest Neighbors':
        report, measure, mat = ml_models.K_neighnors_classifier(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test, data_name=data_name, algorithm_selected = algo)  
        space_res(report, measure, mat)

    if algo=='Decision Tree':
        report, measure, mat = ml_models.Decision_tree_classifier(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test, data_name=data_name, algorithm_selected = algo)
        space_res(report, measure, mat)
            
    if algo=='Random Forest':
        report, measure, mat = ml_models.Random_forest_classifier(X_train=X_train,  X_test=X_test, y_train=y_train, y_test=y_test, data_name=data_name, algorithm_selected = algo)  
        space_res(report, measure, mat)
        