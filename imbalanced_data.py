#)welcome to the portal
import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import math
from pymongo import MongoClient
import pymongo
import streamlit as st

#)title
st.title(':orange[💮Evaluation metrics in dataframe🌳]')

#)reading the dataset
df = pd.read_csv(r"D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\data\loan_approval_dataset.csv")

#) checkbox with df
st.subheader("\n:green[loan approval dataset🌝]\n")
if (st.checkbox("original data")):
    #)showing original dataframe
    st.markdown("\n#### :red[1.1 original dataframe]\n")
    data = df.head(5)
    st.dataframe(data.style.applymap(lambda x: 'color:purple'))

#) checking the columns
#df.columns

#)converting catagorical column into the numerical value
le = LabelEncoder()
cols = [' education',' self_employed',' loan_status']
for col in cols:
    df[col] = le.fit_transform(df[col])

#)education column value counts
df[' education'].value_counts()

#)self_employed column value counts
df[' self_employed'].value_counts()

#)loan_status column value counts
df[' loan_status'].value_counts()

#)loan_status column value count in percentage
df[' loan_status'].value_counts(normalize=True)

#)barplot to compare the classes in loan_status column
df[' loan_status'].value_counts(normalize= True).plot.bar(rot=0,stacked=False)
plt.show()

# creating connection between jupyter notebook to mongodb
connection = pymongo.MongoClient('mongodb://127.0.0.1:27017/')

#) database and collections in mongodb
db = connection['D']
col = db['lab']

#) without applying the hyperparameter
X =df.drop([' loan_status'],axis=1)
y = df[' loan_status']

#)splitting training and testing data
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#) Ml models
models = [LogisticRegression(),
          KNeighborsClassifier(),
          SVC(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          AdaBoostClassifier(),
          GradientBoostingClassifier()
         ]
for model in models:
        model.fit(x_train,y_train)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        
        model_type = type(model).__name__
        #)training data
        accuracy = accuracy_score(y_train,train_pred)
        accuracy = (math.ceil(accuracy*1000)/1000)
        precision = precision_score(y_train,train_pred)
        precision = (math.ceil(precision*1000)/1000)
        recall = recall_score(y_train,train_pred)
        recall = (math.ceil(recall*1000)/1000)
        f1score = f1_score(y_train,train_pred)
        f1score = (math.ceil(f1score*1000)/1000)
        
        #)training dictionary
        train_dict = {'sampler':'no sampler','model_type':model_type,'Evaluation':'train','accuracy':accuracy,'precision':precision,
                      'recall': recall, 'f1score': f1score}
        db.col.insert_one(train_dict)
        
        #)testing data
        accuracy = accuracy_score(y_test,test_pred)
        accuracy = (math.ceil(accuracy*1000)/1000)
        precision = precision_score(y_test,test_pred)
        precision = (math.ceil(precision*1000)/1000)
        recall = recall_score(y_test,test_pred)
        recall = (math.ceil(recall*1000)/1000)
        f1score = f1_score(y_test,test_pred)
        f1score = (math.ceil(f1score*1000)/1000)
        
        #)testing dictionary
        test_dict = {'sampler':'no sampler','model_type':model_type,'Evaluation': 'test','accuracy':accuracy,'precision':precision,
                      'recall': recall, 'f1score': f1score}
        
        db.col.insert_one(test_dict)
        
#) random over sampling
X =df.drop([' loan_status'],axis=1)
y = df[' loan_status']
model = RandomUnderSampler()
X,y = model.fit_resample(X,y)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

models = [LogisticRegression(),
          KNeighborsClassifier(),
          SVC(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          AdaBoostClassifier(),
          GradientBoostingClassifier()
         ]
for model in models:
        model.fit(x_train,y_train)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        
        model_type = type(model).__name__
        
        #)training data
        accuracy = accuracy_score(y_train,train_pred)
        accuracy = (math.ceil(accuracy*1000)/1000)
        precision = precision_score(y_train,train_pred)
        precision = (math.ceil(precision*1000)/1000)
        recall = recall_score(y_train,train_pred)
        recall = (math.ceil(recall*1000)/1000)
        f1score = f1_score(y_train,train_pred)
        f1score = (math.ceil(f1score*1000)/1000)

        #)training dictionary
        train_dict = {'sampler':'RandomUnderSampler','model_type':model_type,'Evaluation':'train','accuracy':accuracy,'precision':precision,
                      'recall': recall, 'f1score': f1score}
        db.col.insert_one(train_dict)
        
        #)testing data
        accuracy = accuracy_score(y_test,test_pred)
        accuracy = (math.ceil(accuracy*1000)/1000)
        precision = precision_score(y_test,test_pred)
        precision = (math.ceil(precision*1000)/1000)
        recall = recall_score(y_test,test_pred)
        recall = (math.ceil(recall*1000)/1000)
        f1score = f1_score(y_test,test_pred)
        f1score = (math.ceil(f1score*1000)/1000)
        
        #)testing dictionary
        test_dict = {'sampler':'RandomUnderSampler','model_type':model_type,'Evaluation': 'test','accuracy':accuracy,'precision':precision,
                      'recall': recall, 'f1score': f1score}
        
        db.col.insert_one(test_dict)        

#)random over sampling
X =df.drop([' loan_status'],axis=1)
y = df[' loan_status']
model = RandomOverSampler()
X,y = model.fit_resample(X,y)


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

models = [LogisticRegression(),
          KNeighborsClassifier(),
          SVC(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          AdaBoostClassifier(),
          GradientBoostingClassifier()
         ]
for model in models:
        model.fit(x_train,y_train)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
    
        model_type = type(model).__name__
        
        #)training data
        accuracy = accuracy_score(y_train,train_pred)
        accuracy = (math.ceil(accuracy*1000)/1000)
        precision = precision_score(y_train,train_pred)
        precision = (math.ceil(precision*1000)/1000)
        recall = recall_score(y_train,train_pred)
        recall = (math.ceil(recall*1000)/1000)
        f1score = f1_score(y_train,train_pred)
        f1score = (math.ceil(f1score*1000)/1000)
        #)training dictionary
        train_dict = {'sampler':'RandomOverSampler','model_type':model_type,'Evaluation':'train','accuracy':accuracy,'precision':precision,
                      'recall': recall, 'f1score': f1score}
        db.col.insert_one(train_dict)
        
        #)testing data
        accuracy = accuracy_score(y_test,test_pred)
        accuracy = (math.ceil(accuracy*1000)/1000)
        precision = precision_score(y_test,test_pred)
        precision = (math.ceil(precision*1000)/1000)
        recall = recall_score(y_test,test_pred)
        recall = (math.ceil(recall*1000)/1000)
        f1score = f1_score(y_test,test_pred)
        f1score = (math.ceil(f1score*1000)/1000)
        
        #)testing dictionary
        test_dict = {'sampler':'RandomOverSampler','model_type':model_type,'Evaluation': 'test','accuracy':accuracy,'precision':precision,
                      'recall': recall, 'f1score': f1score}
        
        db.col.insert_one(test_dict)       
        
for doc in db.col.find():
    print(doc)

#)transforming the mongodb data into pandas
cursor = db.col.find()
list_cur = list(cursor)
b_df = pd.DataFrame(list_cur)

#) checkbox with df
if (st.checkbox("label encoding")):
    #)showing original dataframe
    st.markdown("\n#### :red[ML models with label encoding]\n")
    data = b_df
    st.dataframe(data.style.applymap(lambda x: 'color:purple'))

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
#)reading the dataset
df = pd.read_csv(r"D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\data\loan_approval_dataset.csv")
#)converting catagorical column into the numerical value
cols = [' education',' self_employed',' loan_status']
df = pd.get_dummies(df,columns=cols,drop_first='True',dtype='int')

#)splitting training and testing data
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

X =df.drop([' loan_status_ Rejected'],axis=1)
y = df[' loan_status_ Rejected']

# creating connection between jupyter notebook to mongodb
connection = pymongo.MongoClient('mongodb://127.0.0.1:27017/')

#) database and collections in mongodb
db = connection['C']
col = db['dummy']

#) Ml models
models = [LogisticRegression(),
          KNeighborsClassifier(),
          SVC(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          AdaBoostClassifier(),
          GradientBoostingClassifier()
         ]
for model in models:
        model.fit(x_train,y_train)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        
        model_type = type(model).__name__
        #)training data
        accuracy = accuracy_score(y_train,train_pred)
        accuracy = (math.ceil(accuracy*1000)/1000)
        precision = precision_score(y_train,train_pred)
        precision = (math.ceil(precision*1000)/1000)
        recall = recall_score(y_train,train_pred)
        recall = (math.ceil(recall*1000)/1000)
        f1score = f1_score(y_train,train_pred)
        f1score = (math.ceil(f1score*1000)/1000)
        
        #)training dictionary
        train_dict = {'sampler':'no sampler','model_type':model_type,'Evaluation':'train','accuracy':accuracy,'precision':precision,
                      'recall': recall, 'f1score': f1score}
        db.col.insert_one(train_dict)
        
        #)testing data
        accuracy = accuracy_score(y_test,test_pred)
        accuracy = (math.ceil(accuracy*1000)/1000)
        precision = precision_score(y_test,test_pred)
        precision = (math.ceil(precision*1000)/1000)
        recall = recall_score(y_test,test_pred)
        recall = (math.ceil(recall*1000)/1000)
        f1score = f1_score(y_test,test_pred)
        f1score = (math.ceil(f1score*1000)/1000)
        
        #)testing dictionary
        test_dict = {'sampler':'no sampler','model_type':model_type,'Evaluation': 'test','accuracy':accuracy,'precision':precision,
                      'recall': recall, 'f1score': f1score}
        
        db.col.insert_one(test_dict)
        
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


#) random under sampling
model = RandomUnderSampler()
X,y = model.fit_resample(X,y)

models = [LogisticRegression(),
          KNeighborsClassifier(),
          SVC(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          AdaBoostClassifier(),
          GradientBoostingClassifier()
         ]
for model in models:
        model.fit(x_train,y_train)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        
        model_type = type(model).__name__
        
        #)training data
        accuracy = accuracy_score(y_train,train_pred)
        accuracy = (math.ceil(accuracy*1000)/1000)
        precision = precision_score(y_train,train_pred)
        precision = (math.ceil(precision*1000)/1000)
        recall = recall_score(y_train,train_pred)
        recall = (math.ceil(recall*1000)/1000)
        f1score = f1_score(y_train,train_pred)
        f1score = (math.ceil(f1score*1000)/1000)

        #)training dictionary
        train_dict = {'sampler':'RandomUnderSampler','model_type':model_type,'Evaluation':'train','accuracy':accuracy,'precision':precision,
                      'recall': recall, 'f1score': f1score}
        db.col.insert_one(train_dict)
        
        #)testing data
        accuracy = accuracy_score(y_test,test_pred)
        accuracy = (math.ceil(accuracy*1000)/1000)
        precision = precision_score(y_test,test_pred)
        precision = (math.ceil(precision*1000)/1000)
        recall = recall_score(y_test,test_pred)
        recall = (math.ceil(recall*1000)/1000)
        f1score = f1_score(y_test,test_pred)
        f1score = (math.ceil(f1score*1000)/1000)
        
        #)testing dictionary
        test_dict = {'sampler':'RandomUnderSampler','model_type':model_type,'Evaluation': 'test','accuracy':accuracy,'precision':precision,
                      'recall': recall, 'f1score': f1score}
        
        db.col.insert_one(test_dict)        

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#)random over sampling
model = RandomOverSampler()
X,y = model.fit_resample(X,y)
df = X
df['loan_status_ Rejected'] = y
df['loan_status_ Rejected'].value_counts()

from imblearn.over_sampling import RandomOverSampler

model = RandomOverSampler()
X,y = model.fit_resample(X,y)

models = [LogisticRegression(),
          KNeighborsClassifier(),
          SVC(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          AdaBoostClassifier(),
          GradientBoostingClassifier()
         ]
for model in models:
        model.fit(x_train,y_train)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
    
        model_type = type(model).__name__
        
        #)training data
        accuracy = accuracy_score(y_train,train_pred)
        accuracy = (math.ceil(accuracy*1000)/1000)
        precision = precision_score(y_train,train_pred)
        precision = (math.ceil(precision*1000)/1000)
        recall = recall_score(y_train,train_pred)
        recall = (math.ceil(recall*1000)/1000)
        f1score = f1_score(y_train,train_pred)
        f1score = (math.ceil(f1score*1000)/1000)
        #)training dictionary
        train_dict = {'sampler':'RandomOverSampler','model_type':model_type,'Evaluation':'train','accuracy':accuracy,'precision':precision,
                      'recall': recall, 'f1score': f1score}
        db.col.insert_one(train_dict)
        
        #)testing data
        accuracy = accuracy_score(y_test,test_pred)
        accuracy = (math.ceil(accuracy*1000)/1000)
        precision = precision_score(y_test,test_pred)
        precision = (math.ceil(precision*1000)/1000)
        recall = recall_score(y_test,test_pred)
        recall = (math.ceil(recall*1000)/1000)
        f1score = f1_score(y_test,test_pred)
        f1score = (math.ceil(f1score*1000)/1000)
        
        #)testing dictionary
        test_dict = {'sampler':'RandomOverSampler','model_type':model_type,'Evaluation': 'test','accuracy':accuracy,'precision':precision,
                      'recall': recall, 'f1score': f1score}
        
        db.col.insert_one(test_dict)       
        
for doc in db.col.find():
    print(doc)

#)transforming the mongodb data into pandas
cursor = db.col.find()
list_cur = list(cursor)
a_df = pd.DataFrame(list_cur)

#) checkbox with df
if (st.checkbox("dummies")):
    #)showing original dataframe
    st.markdown("\n#### :red[ML models with dummies]\n")
    data = a_df
    st.dataframe(data.style.applymap(lambda x: 'color:green'))