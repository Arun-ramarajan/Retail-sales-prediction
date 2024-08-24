import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LeakyReLU
import numpy as np
#from keras.models import load_model

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


sales=pd.read_csv('C:/Users/LENOVO/Desktop/Guvi/Project/Final project/preprocessed/final_project_no_skew.csv')
x=sales.drop('Weekly_Sales', axis=1)
y=sales['Weekly_Sales']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
scaler=StandardScaler().fit(x_train)
x_train_ss=scaler.transform(x_train)
x_test_ss=scaler.transform(x_test)




model_path = 'C:/Users/LENOVO/Desktop/Guvi/Project/Final project/model_aug_23(100).h5'
custom_objects = {'LeakyReLU': LeakyReLU}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

tf.keras.backend.clear_session()
# model.compile(optimizer="adam",loss="mean_squared_error",
#                metrics=[tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError()])

type_encode={'A':1, 'B':2, 'C':3}
holiday_encode={'Holiday':1, 'Working day':0}



with st.sidebar:
    select = option_menu(menu_title=None, options=["Home", "Prediction"], icons=['house', 'check-circle'])

if select=='Home':

    st.title("ANN Predictive Modeling of Retail Sales and Markdown Impact")
    st.markdown(
        ''' <h4 style="color: green;">This project aims to predict the weekly sales of stores and its department from the impact of markdowns,
                 store size, type, CPI, unemployment, Temperature by ANN model </h4>''',
                   unsafe_allow_html=True)
    

    st.subheader(':blue[ Skills used:]')
    st.markdown("""
            * **Python:** Used python libraries pandas, matplotlib, seaborn and scikit-learn to understand and visualize the data.
            * **Preprocessing:** Used techniques to handle null, missing values, outliers and skewness
            * **Feature Engineering:** created new features from the data for model to understand the data
            * **EDA:** used matplotlib and seaborn to find outliers and skewness of the data
            * **Tensorflow:** used tensorflow for ANN model
            * **Web application:** used streamlit to create web application 
            * **App deploy:** Web application is deployed with AWS

                """, unsafe_allow_html=True)

else:

    col1, col2, col3 = st.columns(3)

    with col1:
        date= st.selectbox(label='select a Date', options=range(1,32))
        store= st.text_input(label="Enter a store number")
        size= st.text_input(label="Enter a store size")
        markdown1= st.text_input(label="Enter the markdown1 value")
        markdown4= st.text_input(label="Enter the markdown4 value")
        unemployment= st.text_input(label="Enter the unemployment value")



    with col2:
        month= st.selectbox(label='select a Month', options=range(1,13))
        dept= st.text_input(label="Enter a department number")
        temp= st.text_input(label="Enter the temperature")
        markdown2= st.text_input(label="Enter the markdown2 value")
        markdown5= st.text_input(label="Enter the markdown5 value")
        holiday= st.selectbox(label="Enter Holiday or Working day", options=['Holiday', 'Working day'])


    with col3:
        year= st.text_input(label="Enter a year")
        type= st.selectbox(label="select a store type", options=['A', 'B', 'C'])
        fuel= st.text_input(label="Enter the fuel price")
        markdown3= st.text_input(label="Enter the markdown3 value")
        cpi= st.text_input(label="Enter the CPI value")

    button=st.button(':orange[**Predict weekly sales**]')


    if button:
        if not all([date, month, year, store, dept, type, size, temp, fuel,  markdown1, markdown2, markdown3,
                    markdown4, markdown5, cpi, unemployment, holiday]):
            st.error("Select all required fields")

        else:
            Type=type_encode.get(type)
            Holiday=holiday_encode.get(holiday)

            pred=[store, dept, Type, size, temp, fuel, markdown1, markdown2, markdown3, markdown4, markdown5, cpi, unemployment, Holiday,
                   date, month, year]
            
            te=pd.DataFrame([pred])
            te = np.array(te, dtype=np.float32)
            te_ss=scaler.transform(te)
            # te_df=pd.DataFrame(te_ss)
            predictions= model.predict(te_ss)
            value = predictions[0, 0]

            # Convert to integer
            int_value = int(round(value))




            st.subheader(f":green[Predicted Selling Price :] {int_value}")
            

