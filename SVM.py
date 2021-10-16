#This python file will create a model and save it locally
import numpy as np
import pandas as pd
import joblib

df = pd.read_csv("aug_train.csv")
df.isna().sum()/len(df)
df_train = df.fillna(0)

df_train['relevent_experience'] = df_train['relevent_experience'].replace('Has relevent experience',1)
df_train['relevent_experience'] = df_train['relevent_experience'].replace('No relevent experience',0)

from sklearn.preprocessing import OrdinalEncoder

education = [0,'Primary School','High School','Graduate','Masters','Phd']
university = [0,'no_enrollment', 'Part time course', 'Full time course']
company = [0,'<10','10/49','50-99','100-500','500-999','1000-4999','5000-9999','10000+']

enc = OrdinalEncoder(categories=[university])
ordinal_1 = pd.DataFrame(enc.fit_transform(df_train[["enrolled_university"]]))
ordi1 = ordinal_1.rename(columns={0:"University"})

enc = OrdinalEncoder(categories=[education])
ordinal_2 = pd.DataFrame(enc.fit_transform(df_train[["education_level"]]))
ordi2 = ordinal_2.rename(columns={0:"Education level"})

enc = OrdinalEncoder(categories=[company])
ordinal_3 = pd.DataFrame(enc.fit_transform(df_train[["company_size"]]))
ordi3 = ordinal_3.rename(columns={0:"Company size"})

df_train= pd.get_dummies(df_train, columns=['gender', 'major_discipline', 'company_type'])
df_train['experience'] = df_train['experience'].replace('>20',21)
df_train['experience'] = df_train['experience'].replace('<1',0.5)
df_train['last_new_job'] = df_train['last_new_job'].replace('>4',5)
df_train['last_new_job'] = df_train['last_new_job'].replace('never',0)

df_train = df_train.drop(columns=['enrollee_id',
                       'enrolled_university',
                       'education_level',
                       'company_size',
                       'target',
                       'city'])

X = pd.concat([df_train, ordinal_1, ordinal_2, ordinal_3], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, df['target'], test_size = 0.2, random_state=46)

from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf', C=411)#C selected on the basis of trial and error
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_val)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_val, y_pred))

#To save the model locally before running the api:

import joblib
joblib.dump(svclassifier, 'svm_model.pkl')

svclassifier = joblib.load('svm_model.pkl')

model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')

print("Model Trained!!")

