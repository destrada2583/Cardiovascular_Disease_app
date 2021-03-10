#loading required libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style = "whitegrid")
import pickle
import urllib.request
import importlib
import altair
import time
from PIL import Image

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

from log_reg import *


@st.cache
def load_data():
    # assume data file will always be the same per training
    data = pickle.load(open('./df.pkl', 'rb'))
    return data

def load_data():
    # assume data file will always be the same per training
    data = pickle.load(open('./cardio.pkl', 'rb'))
    return data

# callig model saved in Jupyter notebook
pickle_in = open('logisticRegr.pkl', 'rb')
classifier = pickle.load(pickle_in)

st.sidebar.image('cardio.jpeg', width=None)


st.sidebar.title("Navigation")
rad = st.sidebar.radio("Go To",["Home","Dataset","Data Exploration | Demographics","Data Exploration | Lifestyle","Correlation Chart","Models","Cardiovascular Predictor"])



if rad == "Home":
    st.title("Cardiovascular Disease App")
    st.subheader("Cardiovascular Disease ")
    st.write("Cardiovascular disease or CVD for short is defined as a group of conditions that involve the heart and blood vessels. When one hears cardiovascular disease, they think of complications like heart attacks, strokes, or angina which is a form of chest pain. Health experts have stated that in the US 1 person dies every 37 seconds from cardiovascular disease and the number of lives lost will increase annually without intervention.")
    st.write("According to the American Heart Association, as of 2019, cardiovascular disease was the number one cause of death in the U.S.")
    st. write("Problem statement: Classify patients as healthy or suffering from cardiovascular disease based on the attributes within the dataset.")
    st.image('cardio2.png', width=None)

if rad == "Dataset":

    st.write("All of the dataset values were collected at the moment of medical examination.  There were no duplicates in the dataset and over 69,000 records.")
    st.subheader('Dataset')

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.1)
        progress.progress(i+1)
    
    df

    st.balloons()

    st.subheader("Data Statistics Table")
    st.write(" ")
    df.describe().T

if rad == "Data Exploration | Demographics":
    
    st.subheader('Cardio Disease Presence by Age')
    plt.figure(figsize = (12, 8))
    sns.countplot(x= 'AGE_RANGE', hue= 'cardio', data= cardio, palette = "YlGnBu")
    plt.xlabel('Age Range', fontsize=17)
    plt.legend( ['Disease Not Present', 'Disease Present'], fontsize=17)
    plt.ylabel('Number of Patients', fontsize=17)
    plt.tick_params(labelsize=17)
    st.pyplot(plt)

    st.subheader('Cardio Disease Presence by Gender')
    legend_name = ['Male', 'Female']
    plt.figure(figsize = (12, 8))
    sns.countplot(x='cardio', hue='gender', data= df, palette = "YlGnBu")
    plt.xlabel('Disease Presence', fontsize=17)
    sns.set_context("poster")
    plt.legend(legend_name, fontsize=17)
    plt.ylabel('Number of Patients', fontsize=17)
    plt.tick_params(labelsize=17)
    st.pyplot(plt)

    st.subheader('Cardio Disease Presence by Age and Weight')
    plt.figure(figsize=(12, 8))
    sns.set_context("poster")
    sns.boxplot(x='AGE_RANGE', y="weight", hue='cardio', data=cardio, palette="YlGnBu")
    plt.xlabel('Age Range', fontsize=17)
    plt.ylabel('Weight', fontsize=17)
    disease_label = ['Disease Not Present', 'Disease Present']
    plt.legend(disease_label,  fontsize=17) 
    plt.tick_params(labelsize=17) 
    st.pyplot(plt) 

    st.subheader('Cardio Disease Presence Based on Age and BMI')
    plt.figure(figsize=(12, 8))
    sns.set_context("poster")
    sns.scatterplot(x='AGE_RANGE', y="bmi", hue='cardio', data=cardio, palette="YlGnBu", s=125, edgecolors='k')
    plt.xlabel('Age Range', fontsize=17)
    plt.ylabel('BMI', fontsize=17)
    plt.legend(disease_label , fontsize=17)
    plt.tick_params(labelsize=17)
    st.pyplot(plt)

if rad == "Data Exploration | Lifestyle":
    
    st.subheader('Smoking and Alcohol Consumption')
    fig, ax =plt.subplots(1,2,figsize=(24,16))
    sns.set_context("poster")
    sns.countplot(x='smoke', hue='cardio', data=cardio, palette="YlGnBu",ax=ax[0])
    sns.countplot(x='alco',hue='cardio',data=cardio, palette="YlGnBu",ax=ax[1])
    ax[0].set_title("Smoker vs Heart Disease", fontsize=32)
    ax[1].set_title("Alcohol Consumption vs Heart Disease", fontsize=32)
    ax[0].set_xlabel("Smoker", fontsize=25)
    ax[1].set_xlabel("Alcohol Consumption", fontsize=25)
    ax[0].set_xticklabels(["No","Yes"], fontsize = 25)
    ax[1].set_xticklabels(["No","Yes"], fontsize=25)
    ax[0].legend(["Disease Not Present", "Disease Present"],fontsize=25,loc="upper right")
    ax[1].legend(["Disease Not Present","Disease Present"], fontsize=25,loc="upper right")
    st.pyplot(plt)

    st.subheader('Excercise and Cholesterol')
    fig, ax =plt.subplots(1,2,figsize=(24,16))
    sns.set_context("poster")
    sns.countplot(x='active', hue='cholesterol', data=cardio, palette="YlGnBu",ax=ax[0])
    sns.countplot(x='active',hue='gluc',data=cardio, palette="YlGnBu",ax=ax[1])
    ax[0].set_title("Excercise vs Cholesterol", fontsize=32)
    ax[1].set_title("Excercise vs Glucose Levels", fontsize=32)
    ax[0].set_xlabel("Excercise", fontsize=25)
    ax[1].set_xlabel("Excercise", fontsize=25)
    ax[0].set_xticklabels(["No","Yes"], fontsize = 25)
    ax[1].set_xticklabels(["No","Yes"], fontsize=25)
    ax[0].legend(["Normal", "Above Normal", "Very High"],fontsize=25,loc="upper right")
    ax[1].legend(["Normal","Above Normal", "Very High"], fontsize=25,loc="upper right")
    st.pyplot(plt)
 
if rad == "Correlation Chart":
    st.subheader('Correlation Between Collected Features')
    plt.figure(figsize=(35,25))
    cardio_plot = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True, annot_kws={"size":30})
    sns.set_context("poster")
    plt.tick_params(labelsize=30)
    st.pyplot(plt)

if rad == "Models":
   st.title('Models')
   st.write("[SVM] (https://github.com/curlylady321/Final_Project_CardioVascular-Disease/blob/main/Cardio_data_analysis.ipynb)") 
   st.write("[Random Forrest] (https://nbviewer.jupyter.org/github/jenniferfernandezcadiz/Final_Project/blob/main/Random_Forest_JF.ipynb)")
   st.write("[Logistic Regression] (https://nbviewer.jupyter.org/github/destrada2583/Final_Cardio/blob/main/Cardiovascular_Disease_Detection.ipynb)")

if rad == "Cardiovascular Predictor":
    st.title('Cardiovascular Disease Prediction')
    name = st.text_input("Name:")
    age = st.number_input("Age:")
    gender = st.number_input("Gender:  1 female | 2 male")
    bmi =  st.number_input("BMI | Body mass index (weight in kg/(height in m)^2):")
    st.markdown("[BMI Calculator] (https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/english_bmi_calculator/bmi_calculator.html)")
    ap_hi = st.number_input("Systolic blood pressure:")
    ap_lo = st.number_input("Diastolic blood pressure:")
    cholesterol = st.number_input("Cholesterol: 1 normal | 2 above normal | 3 well above normal")
    gluc = st.number_input("Glucose: 1 normal | 2 above normal | 3 well above normal")
    smoke = st.number_input("Smoking:  0 no | 1 yes")
    alco = st.number_input("Alcohol Intake:  0 no | 1 yes")
    active = st.number_input("Physical activity: 0 no | 1 yes")
    submit = st.button('Predict')

    if submit:
        prediction = classifier.predict([[age, gender, bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
        if prediction == 0:
            st.write(name,"-- Based on your inputs, you are NOT prone to Cardiovascular Disease")
            
        else:
            st.write(name,"-- Based on your inputs, you are prone to Cardiovascular Disease.")
            st.write("For More Information on Cardiovascular Disease, please visit " 
            "[CDC: Heart Disease Facts] (https://www.cdc.gov/heartdisease/facts.htm) and [Cardiovascular Disease Prevention] (https://www.cdc.gov/heartdisease/prevention.htm)")


st.sidebar.title("About")                          
st.sidebar.info("Using Cardiovascular Data provided by Kaggle, we created three "
                "Classification Models to inform and predict Cardiovascular Disease.")  
st.sidebar.info('app created by: Brittney Portes   |   Diana Estrada   |   Jenny Fernandez.')              
               
            
