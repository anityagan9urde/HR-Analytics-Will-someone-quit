#This python file will create a model and save it locally
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

hr = pd.read_csv("aug_train.csv")
hr.isna().sum()/len(hr)
hr_train = hr.fillna(0)

hr_train['relevent_experience'] = hr_train['relevent_experience'].replace('Has relevent experience',1)
hr_train['relevent_experience'] = hr_train['relevent_experience'].replace('No relevent experience',0)

from sklearn.preprocessing import OrdinalEncoder

edu_lv = [0,'Primary School','High School','Graduate','Masters','Phd']
uni = [0,'no_enrollment', 'Part time course', 'Full time course']
comp_size = [0,'<10','10/49','50-99','100-500','500-999','1000-4999','5000-9999','10000+']

enc = OrdinalEncoder(categories=[uni])
ordi1 = pd.DataFrame(enc.fit_transform(hr_train[["enrolled_university"]]))
ordi1 = ordi1.rename(columns={0:"University"})

enc = OrdinalEncoder(categories=[edu_lv])
ordi2 = pd.DataFrame(enc.fit_transform(hr_train[["education_level"]]))
ordi2 = ordi2.rename(columns={0:"Education level"})

enc = OrdinalEncoder(categories=[comp_size])
ordi3 = pd.DataFrame(enc.fit_transform(hr_train[["company_size"]]))
ordi3 = ordi3.rename(columns={0:"Company size"})

hr_train= pd.get_dummies(hr_train, columns=['gender', 'major_discipline', 'company_type'])
hr_train['experience'] = hr_train['experience'].replace('>20',21)
hr_train['experience'] = hr_train['experience'].replace('<1',0.5)
hr_train['last_new_job'] = hr_train['last_new_job'].replace('>4',5)
hr_train['last_new_job'] = hr_train['last_new_job'].replace('never',0)

hr_train = hr_train.drop(columns=['enrollee_id',
                       'enrolled_university',
                       'education_level',
                       'company_size',
                       'target',
                       'city'])

X = pd.concat([hr_train, ordi1, ordi2, ordi3], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, hr['target'], test_size = 0.2, random_state=46)

from sklearn.svm import SVC
svclassifier = SVC(C=411)#C selected on the basis of trial and error
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_val)

from sklearn.metrics import classification_report
print(classification_report(y_val, y_pred))

import joblib
joblib.dump(svclassifier, 'svm_model.pkl')

regressor = joblib.load('svm_model.pkl')

model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Model Trained!!")

