import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
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

#Read the data
df = pd.read_csv("C:\Guvi\Project\Industrial Copper Modeling\Copper_Set.xlsx - Result 1.csv")

#Cleaning the data
condition = df['quantity tons'] == 'e'
# Use the condition to drop rows
df.drop(df[condition].index, inplace=True)
#Transforming the data types
df['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce')
df['quantity tons'] = pd.to_numeric(df['quantity tons'])
df['customer'] = df['customer'].astype(str)
df['country'] = df['country'].astype(str)
df['status'] = df['status'].astype(str)
df['item type'] = df['item type'].astype(str)
df['application'] = pd.to_numeric(df['application'], errors='coerce')
df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce')
df['width'] = pd.to_numeric(df['width'], errors='coerce')
df['material_ref'] = df['material_ref'].astype(str)
df['product_ref'] = pd.to_numeric(df['product_ref'], errors='coerce')
df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce')
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')

df.replace('nan', np.nan, inplace=True)

df.drop(['material_ref','id','item_date','delivery date','customer'],axis=1,inplace=True)

df = df.dropna(subset=[])
df['country'].bfill(inplace=True)
df['status'].bfill(inplace=True)
df[['thickness','selling_price']] = df[['thickness','selling_price']].fillna(df[['thickness','selling_price']].mean())
df['application'].fillna(df['application'].mode().iloc[0], inplace=True)

df = df.query("status != 'Offerable'")
df = df.query("`item type` != 'SLAWR'")
df = df[df["application"]<=80]

#Removing Outliers
# Calculate the IQR
Q1 = df['thickness'].quantile(0.25)
Q3 = df['thickness'].quantile(0.75)
IQR = Q3 - Q1
# Set a scaling factor (adjust as needed)
k = 1.5
# Define the lower and upper bounds
lower_bound = Q1 - k * IQR
upper_bound = Q3 + k * IQR
# Clip the outliers to the bounds
df['thickness'] = df['thickness'].clip(lower=lower_bound, upper=upper_bound)

# Calculate the IQR
Q1 = df['quantity tons'].quantile(0.25)
Q3 = df['quantity tons'].quantile(0.75)
IQR = Q3 - Q1
# Set a scaling factor (adjust as needed)
k = 1.5
# Define the lower and upper bounds
lower_bound = Q1 - k * IQR
upper_bound = Q3 + k * IQR
# Clip the outliers to the bounds
df['quantity tons'] = df['quantity tons'].clip(lower=lower_bound, upper=upper_bound)

# Calculate the IQR
Q1 = df['width'].quantile(0.25)
Q3 = df['width'].quantile(0.75)
IQR = Q3 - Q1
# Set a scaling factor (adjust as needed)
k = 1.5
# Define the lower and upper bounds
lower_bound = Q1 - k * IQR
upper_bound = Q3 + k * IQR
# Clip the outliers to the bounds
df['width'] = df['width'].clip(lower=lower_bound, upper=upper_bound)

df = df[df["selling_price"]<=20000000]
# Calculate the IQR
Q1 = df['selling_price'].quantile(0.25)
Q3 = df['selling_price'].quantile(0.75)
IQR = Q3 - Q1
# Set a scaling factor (adjust as needed)
k = 1.5
# Define the lower and upper bounds
lower_bound = Q1 - k * IQR
upper_bound = Q3 + k * IQR
# Clip the outliers to the bounds
df['selling_price'] = df['selling_price'].clip(lower=lower_bound, upper=upper_bound)

#LabelEncoding
model = LabelEncoder()
df['status'] = model.fit_transform(df['status'])
df['item type'] = model.fit_transform(df['item type'])

df_price = df.copy()
df_status = df.copy()

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
        X = df_price.drop(['selling_price',],axis=1)
        y = df_price['selling_price']
        SS = StandardScaler()
        SS.fit_transform(X)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        # Choose the best model (RandomForestRegressor in this case)
        best_model = RandomForestRegressor()
        # Train the best model on the entire training dataset
        best_model.fit(x_train, y_train)
        # Now, you can input new values for prediction

        pr = X['product_ref'].unique()
        w_max = X.width.max()
        w_min = X.width.min()
        t_min = X.thickness.min()
        t_max = X.thickness.max()
        a = X.application.unique()
        it = X['item type'].unique()
        s = X['status'].unique()
        c = X['country'].unique()
        qt_min = X['quantity tons'].min()
        qt_max = X['quantity tons'].max()

        W = st.number_input('Enter Width',min_value=w_min, max_value=w_max, value=None,placeholder = 'Width')
        Q = st.number_input('Enter Quantity Tons',min_value=qt_min, max_value=qt_max, value=None,placeholder = 'Quantity Tons')
        T = st.number_input('Enter Thickness',min_value=t_min, max_value=t_max, value=None,placeholder = 'Thickness')
        PR = st.selectbox("Select Product Reference",pr,index=None,placeholder="Product Reference",key='sp_pr')
        A = st.selectbox("Select Application",a,index=None,placeholder="Application",key='sp_a')
        IT = st.selectbox("Select Item Type",it,index=None,placeholder="Item Type",key='sp_it')
        S = st.selectbox("Select Status",s,index=None,placeholder="Status",key='sp_s')
        C = st.selectbox("Select Country",c,index=None,placeholder="Country",key='sp_c')
        

        new_input = [[Q,C,S,IT,A,T,W,PR]]  # ['quantity tons','country','status','item type','application','thickness','width','product_ref']
        if Q is not None and C is not None and S is not None and IT is not None and A is not None and T is not None and W is not None and PR is not None:
            # Make predictions using the best model
            prediction = best_model.predict(new_input)
            # Print the predicted result
            st.write("Predicted Result:", *prediction)

    with t2:
        st.subheader(":orange[Enter the values to predict the status]")
        
        df_status['quantity tons'] = np.log(df_status['quantity tons'])
        df_status['application'] = np.log(df_status['application'])
        df_status['thickness'] = np.log(df_status['thickness'])
        df_status['width'] = np.log(df_status['width'])
        df_status['selling_price'] = np.log(df_status['selling_price'])
        df_status = df_status.dropna(subset=['quantity tons'])

        X = df_status.drop(['status'],axis=1)
        y = df_status['status']

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        # Choose the best model (RandomForestRegressor in this case)
        best_model = RandomForestClassifier()
        # Train the best model on the entire training dataset
        best_model.fit(x_train, y_train)
        # Now, you can input new values for prediction

        pr = X['product_ref'].unique()
        w_max = X.width.max()
        w_min = X.width.min()
        t_min = X.thickness.min()
        t_max = X.thickness.max()
        a = X.application.unique()
        it = X['item type'].unique()
        sp_min = X.selling_price.min()
        sp_max = X.selling_price.max()
        c = X['country'].unique()
        qt_min = X['quantity tons'].min()
        qt_max = X['quantity tons'].max()

        W = st.number_input('Enter Width',min_value=w_min, max_value=w_max, value=None,placeholder = 'Width')
        Q = st.number_input('Enter Quantity Tons',min_value=qt_min, max_value=qt_max, value=None,placeholder = 'Quantity Tons')
        T = st.number_input('Enter Thickness',min_value=t_min, max_value=t_max, value=None,placeholder = 'Thickness')
        PR = st.selectbox("Select Product Reference",pr,index=None,placeholder="Product Reference",key='s_pr')
        A = st.selectbox("Select Application",a,index=None,placeholder="Application",key='s_a')
        IT = st.selectbox("Select Item Type",it,index=None,placeholder="Item Type",key='s_it')
        SP = st.number_input('Enter Selling Price',min_value=sp_min, max_value=sp_max, value=None,placeholder = 'Selling Price')
        C = st.selectbox("Select Country",c,index=None,placeholder="Country",key='s_c')

        new_input = [[Q,C,IT,A,T,W,PR,SP]]  # ['quantity tons','country','item type','application','thickness','width','product_ref','selling price']
        if Q is not None and C is not None and SP is not None and IT is not None and A is not None and T is not None and W is not None and PR is not None:
            # Make predictions using the best model
            prediction = best_model.predict(new_input)
            # Print the predicted result
            st.write("Predicted Result:", *prediction)

