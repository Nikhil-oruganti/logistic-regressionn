
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

st.title("predicting the attorney")

st.sidebar.header("enter input data")

def user_input_features():
    CLMSEX = st.sidebar.selectbox("Gender",('1','0'))
    CLMINSUR = st.sidebar.selectbox("insurence",('1','0'))
    SEATBELT = st.sidebar.selectbox("selbelt",('1','0'))
    CLMAGE = st.sidebar.number_input("enter the age")
    LOSS = st.sidebar.number_input("enter the loss")
    dataframe = {
        "CLMSEX":CLMSEX,
        "CLMINSUR":CLMINSUR,
        "SEATBELT":SEATBELT,
        "CLMAGE":CLMAGE,
        "LOSS":LOSS}
    features = pd.DataFrame(dataframe,index=[0])
    return features

df = user_input_features()
st.subheader("claimants information")
st.write(df)

claimants = pd.read_csv("claimants.csv")
claimants.drop(["CASENUM"],inplace=True,axis=1)
claimants = claimants.dropna()

X= claimants.iloc[:,[1,2,3,4,5]]
Y = claimants.iloc[:,0]
classifier = LogisticRegression()
classifier.fit(X,Y)

predicted= classifier.predict(df)
probability = classifier.predict_proba(df)

st.subheader("predicted")

if predicted==0:
    st.write("the attorney is appointed")
    
else:
    st.write("The attorney is not appointed")
    
    
st.subheader("predicted probability")
st.write(probability)


    


