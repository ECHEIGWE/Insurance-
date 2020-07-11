# Core Pkg
import streamlit as st
import os

# EDA Pkgs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg') 
import joblib

@st.cache
def load_data(dataset):
	df=pd.read_csv(dataset)
	return df
#load model
def load_prediction_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

#to get value
def get_value(val, my_dict):
	for key, value in my_dict.items():
		if val == key:
			return value
#to get key
def get_key(val, my_dict):
	for key, value in my_dict.items():
		if val == key:
			return key

def main():
	st.title ('Medical Insurance')
	st.subheader('Prepared by ECHE')

	activities = ['EDA','Prediction','About']
	choices = st.sidebar.selectbox("Select Activity",activities)
	if choices == 'EDA':
		st.subheader("Exploratory Data Analysis (EDA)")

		data = load_data('data/Medical_data.csv')
		st.dataframe(data.head(10))
		if st.checkbox('Show Summary'):
			st.write(data.describe())

		if st.checkbox('Show Shape'):
			st.write(data.shape)
		if st.checkbox('NaN Sum'):
			st.write(data.isna().sum())
		if st.checkbox('Data Type'):
			st.write(data.dtypes)

	if choices == 'Prediction':
		st.subheader("Prediction")
		sex_label = {'male': 0, 'female': 1}
		smoker_label = {'no': 0, 'yes': 1}
		region_label = {'southwest': 0, 'southeast': 1, 'northeast': 2, 'northwest': 3}

#Machine learning User Input

		age = st.number_input ('Age',min_value = 18, max_value=64, value =25)
		sex = st.selectbox('Sex', tuple(sex_label.keys()))
		bmi = st.number_input('BMI', min_value=16,max_value=53, value=16)
		children = st.selectbox('Children', [0,1,2,3,4,5])
		smoker = st.radio('Smoker', tuple(smoker_label.keys()))
		region = st.selectbox('Region', tuple(region_label.keys()))

		#User Input 
		k_sex = get_value(sex, sex_label)
		k_smoker = get_value(smoker, smoker_label)
		k_region =  get_value(region, region_label)

		#USER RESULT
		selected = [age, sex , bmi, children, smoker, region]
		vectorized = [age, k_sex , bmi, children, k_smoker, k_region]
		sample_data = np.array(vectorized).reshape(1,-1)
		st.info(selected)
		json_form = {"age": age, "sex":sex ,"bmi": bmi,"children": children, "smoker":smoker,"region": region}
		st.json(json_form)
		#st.write(vectorized)

		#MAKE EVALUATION 
		st.subheader('EVALUATION')
		if st.checkbox ('Make Evaluation'):
			my_model_list = ("DecisionTree", "Lasso","Linear ReGression")

			model_choice = st.selectbox("Model Choice", my_model_list)

		if st.button('Evalaute'):
			if model_choice =='DecisionTree':
				predictor = load_prediction_model("models/dec_med_model.pkl")
				prediction = predictor.predict(sample_data)
				#st.write(prediction)
			if model_choice =='Lasso':
				predictor = load_prediction_model("models/la_med_model.pkl")
				prediction = predictor.predict(sample_data)
				#st.write(prediction)
			if model_choice =='Linear ReGression':
				predictor = load_prediction_model("models/lin_med_model.pkl")
				prediction = predictor.predict(sample_data)
				#st.write(prediction)
			st.success('The Evalaution is: {}'.format(prediction))

	if choices =='About':
		st.subheader("About")
		st.markdown(""" 
			This is simple App designed to predict the cost of Insurance based on 
			various parameters(features) such as Age, Sex, BMI, Children, Smoker and Region.

			**Data Source**: Kaggle Datasets """)

if __name__ == '__main__':
	main()