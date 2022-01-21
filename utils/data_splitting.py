from sklearn.model_selection import train_test_split
import streamlit as st

def Split_data(data, target_column, split_ratio = 0.2):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test

def space_res(matrix):
    if st.button('Display Confusion Matrix'):
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.write('<p style="text-align: center;font-size:25px;"><b>{}</b></p>'.format('CONFUSION MATRIX'), unsafe_allow_html=True)
        st.write(matrix.figure_)
def space():
    st.text('')
    st.text('')