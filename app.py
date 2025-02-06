import streamlit as st
import tensorflow
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from tensorflow.keras.models import load_model

with open('label_encoder_gender.pkl','rb') as file:
    le=pickle.load(file)

with open('ohe_geography.pkl','rb') as file:
    ohe=pickle.load(file)

model=load_model('model.h5')
st.title("CHURN PREDICTION APP")

geography=st.selectbox('Geography',ohe.categories_[0])
gender=st.selectbox('Gender',le.classes_)
age=st.slider('Age',18,100)
balance=st.number_input('Balance')
estimated_salary=st.number_input('Estimated Salary')
credit_score=st.number_input('Credit Score')
tenure=st.slider('tenure',0,10)
num_of_products=st.slider('number of products',0,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])


#input data
input_data=pd.DataFrame({
'CreditScore':[credit_score],
'Gender':le.transform([gender]),
'Age':[age],
'Tenure':[tenure],
'Balance':[balance],
'NumOfProducts':[num_of_products],
'HasCrCard':[has_cr_card],
'IsActiveMember':[is_active_member],
'EstimatedSalary':[estimated_salary],
})
geo_encoded=ohe.transform([[geography]])
geo_encoded=pd.DataFrame(geo_encoded,columns=ohe.get_feature_names_out())

input_data=pd.concat([input_data,geo_encoded],axis=1)

#predicting 
pred=model.predict(input_data)
pred_prob=pred[0][0]
st.write(f'{pred_prob} is the calculated probability')
if pred_prob>=0.5:
    st.write("The person is likely to Churn")
else:
    st.write('the person will not churn')
