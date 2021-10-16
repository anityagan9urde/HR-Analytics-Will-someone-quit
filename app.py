#This python api uses the saved model to predict results from given input to the api
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
svc = joblib.load('svm_model.pkl')
model_columns = joblib.load("model_columns.pkl")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    d = None
    if request.method == 'POST':
        print('POST received')
        d = request.form.to_dict()
    else:
        print('GET received')
        d = request.args.to_dict()

    if svc:
        df = pd.DataFrame([d])
        df.isna().sum()/len(df)
        df_train = df.fillna(0)

        df_train['relevent_experience'] = df_train['relevent_experience'].replace('Has relevent experience', 1)
        df_train['relevent_experience'] = df_train['relevent_experience'].replace('No relevent experience', 0)

        from sklearn.preprocessing import OrdinalEncoder

        education = [0,'Primary School','High School','Graduate','Masters','Phd']
        university = [0,'no_enrollment', 'Part time course', 'Full time course']
        company = [0,'<10','10/49','50-99','100-500','500-999','1000-4999','5000-9999','10000+']

        enc = OrdinalEncoder(categories=[university])
        ordinal_1 = pd.DataFrame(enc.fit_transform(df_train[["enrolled_university"]]))
        ordinal_1 = ordinal_1.rename(columns={0:"University"})

        enc = OrdinalEncoder(categories=[education])
        ordinal_2 = pd.DataFrame(enc.fit_transform(df_train[["education_level"]]))
        ordinal_2 = ordinal_2.rename(columns={0:"Education level"})

        enc = OrdinalEncoder(categories=[company])
        ordinal_3 = pd.DataFrame(enc.fit_transform(df_train[["company_size"]]))
        ordinal_3 = ordinal_3.rename(columns={0:"Company size"})

        df_train = pd.get_dummies(df_train, columns=['gender', 'major_discipline', 'company_type'])

        df_train = df_train.drop(columns=['enrolled_university',
                                       'education_level',
                                       'company_size',
                                       'city'])

        X = pd.concat([df_train, ordinal_1, ordinal_2, ordinal_3], axis=1)
        query = X.reindex(columns=model_columns, fill_value=0)

        prediction = svc.predict(query)

        if int(prediction)==1:
            output = 'The employee will leave.'
        elif int(prediction)==0:
            output = 'The employee will not leave.'

        return render_template('index.html', prediction_text='{}'.format(output))



if __name__ == "__main__":
    svc = joblib.load("svm_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    app.run(debug=True)
