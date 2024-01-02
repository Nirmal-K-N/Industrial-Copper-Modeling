import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler

# Regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

with st.sidebar:
    selected = option_menu("Main Menu", ["Home","Predict"], 
                icons=['house','cloud'], menu_icon="cast", default_index=0)
         
if selected == "Home":
    st.header("Industrial Copper Modeling",divider='grey')
    st.write("""**Overall, this project will equip you with practical skills and experience in data analysis, machine learning modeling, 
             and creating interactive web applications, and provide you with a solid foundation to tackle real-world problems in the manufacturing domain.**""")
    st.subheader(":orange[Problem Statement]",divider='grey')
    st.write("""**The copper industry deals with less complex data related to sales and pricing.
            However, this data may suffer from issues such as skewness and noisy data, which
            can affect the accuracy of manual predictions. Dealing with these challenges manually
            can be time-consuming and may not result in optimal pricing decisions. A machine
            learning regression model can address these issues by utilizing advanced techniques
            such as data normalization, feature scaling, and outlier detection, and leveraging
            algorithms that are robust to skewed and noisy data.**""")
    st.write("""**Another area where the copper industry faces challenges is in capturing the leads. A
            lead classification model is a system for evaluating and classifying leads based on
            how likely they are to become a customer . You can use the STATUS variable with
            WON being considered as Success and LOST being considered as Failure and
            remove data points other than WON, LOST STATUS values.**""")
    st.subheader(":orange[Skills take away from this Project]",divider='grey')
    st.text("> Python scripting,")
    st.text("> Data Preprocessing,")
    st.text("> EDA,")
    st.text("> Streamlit")
    
elif selected == "Predict":
    t1,t2 = st.tabs(['**Price**','**Status**'])

    with t1:
        st.subheader(":orange[Enter the values to predict the price]")

        #Regression Model
        pr = [1670798778, 1668701718, 628377, 640665, 611993, 1668701376, 164141591, 1671863738, 1332077137, 640405, 1693867550, 1665572374, 1282007633, 1668701698,
              628117, 1690738206, 628112, 640400, 1671876026, 164336407, 164337175, 1668701725, 1665572032, 611728, 1721130331, 1693867563, 611733, 1690738219,
              1722207579, 929423819, 1665584320, 1665584662, 1665584642]
        w_max = 1980.0
        w_min = 700.0
        t_min = 0.18
        t_max = 6.449999999999999
        a = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 2., 5., 39., 69., 70., 65., 58., 68.]
        it = [4, 5, 3, 1, 2, 0]
        s = [6, 0, 5, 1, 2, 7, 4, 3]
        c = ['28.0', '25.0', '30.0', '32.0', '38.0', '78.0', '27.0', '77.0', '113.0', '79.0', '26.0', '39.0', '40.0', '84.0', '80.0', '107.0', '89.0']
        qt_min = -73.38885726499998
        qt_max = 151.52817317499998

        W = st.number_input('Enter Width',min_value=w_min, max_value=w_max, value=None,placeholder = 'Width')
        Q = st.number_input('Enter Quantity Tons',min_value=qt_min, max_value=qt_max, value=None,placeholder = 'Quantity Tons')
        T = st.number_input('Enter Thickness',min_value=t_min, max_value=t_max, value=None,placeholder = 'Thickness')
        PR = st.selectbox("Select Product Reference",pr,index=None,placeholder="Product Reference",key='sp_pr')
        A = st.selectbox("Select Application",a,index=None,placeholder="Application",key='sp_a')
        IT = st.selectbox("Select Item Type",it,index=None,placeholder="Item Type",key='sp_it')
        S = st.selectbox("Select Status",s,index=None,placeholder="Status",key='sp_s')
        C = st.selectbox("Select Country",c,index=None,placeholder="Country",key='sp_c')
        
        with open(r"C:\Guvi\Project\Industrial Copper Modeling\rfg_model_regressor.pkl", 'rb') as file:
            loaded_model = pickle.load(file)
        # Now you can use the loaded_model for predictions
        # Example input data (replace this with your own input data)

        new_input = [[Q,C,S,IT,A,T,W,PR]]  # ['quantity tons','country','status','item type','application','thickness','width','product_ref']
        if Q is not None and C is not None and S is not None and IT is not None and A is not None and T is not None and W is not None and PR is not None:
            # Make predictions using the best model
            prediction = loaded_model.predict(new_input)
            # Print the predicted result
            st.write("Predicted Selling Price:", *prediction)

    with t2:
        st.subheader(":orange[Enter the values to predict the status]")

        pr = [1670798778, 1668701718, 628377, 640665, 611993, 1668701376, 164141591, 1671863738, 1332077137, 640405, 1693867550, 1665572374, 1282007633, 1668701698,
              628117, 1690738206, 628112, 640400, 1671876026, 164336407, 164337175, 1668701725, 1665572032, 611728, 1721130331, 1693867563, 611733, 1690738219,
              1722207579, 929423819, 1665584320, 1665584662, 1665584642]
        w_max = 7.590852123688581
        w_min = 6.551080335043404
        t_min = -1.7147984280919266
        t_max = 1.864080130807681
        a = [2.30258509, 3.71357207, 3.33220451, 4.07753744, 2.7080502, 1.38629436, 3.63758616, 4.02535169, 3.73766962, 3.25809654, 3.29583687, 2.94443898,
             2.99573227, 4.18965474, 3.36729583, 3.09104245, 3.68887945, 3.21887582, 4.20469262, 4.36944785, 1.09861229, 0.69314718, 1.60943791, 3.66356165,
             4.2341065, 4.24849524, 4.17438727, 4.06044301, 4.21950771]
        it = [4, 5, 3, 1, 2, 0]
        sp_min = 5.5
        sp_max = 7.2
        c = ['28.0', '25.0', '30.0', '32.0', '38.0', '78.0', '27.0', '77.0', '113.0', '79.0', '26.0', '39.0', '40.0', '84.0', '80.0', '107.0', '89.0']
        qt_min = -11.512925464970229
        qt_max = 5.020771569211873

        W = st.number_input('Enter Width',min_value=w_min, max_value=w_max, value=None,placeholder = 'Width')
        Q = st.number_input('Enter Quantity Tons',min_value=qt_min, max_value=qt_max, value=None,placeholder = 'Quantity Tons')
        T = st.number_input('Enter Thickness',min_value=t_min, max_value=t_max, value=None,placeholder = 'Thickness')
        PR = st.selectbox("Select Product Reference",pr,index=None,placeholder="Product Reference",key='s_pr')
        A = st.selectbox("Select Application",a,index=None,placeholder="Application",key='s_a')
        IT = st.selectbox("Select Item Type",it,index=None,placeholder="Item Type",key='s_it')
        SP = st.number_input('Enter Selling Price',min_value=sp_min, max_value=sp_max, value=None,placeholder = 'Selling Price')
        C = st.selectbox("Select Country",c,index=None,placeholder="Country",key='s_c')

        with open(r"C:\Guvi\Project\Industrial Copper Modeling\rfg_model_regressor.pkl", 'rb') as file:
            loaded_model = pickle.load(file)
        # Now you can use the loaded_model for predictions
        # Example input data (replace this with your own input data)

        new_input = [[Q,C,IT,A,T,W,PR,SP]]  # ['quantity tons','country','item type','application','thickness','width','product_ref','selling price']
        if Q is not None and C is not None and SP is not None and IT is not None and A is not None and T is not None and W is not None and PR is not None:
            with open(r"C:\Guvi\Project\Industrial Copper Modeling\rfg_model_classifier.pkl", 'rb') as file:
                loaded_model = pickle.load(file)
            # Make predictions using the best model
            prediction = loaded_model.predict(new_input)
            # Print the predicted result
            st.write("Predicted Status:", *prediction)

