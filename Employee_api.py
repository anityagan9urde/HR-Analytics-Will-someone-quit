#This python api uses the saved model to predict results from given input to the api
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
svcclassifier = joblib.load('svm_model.pkl')

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

    if svcclassifier:
        df = pd.DataFrame([d])
        df.isna().sum()/len(df)
        hr_train = df.fillna(0)
        hr_train['relevent_experience'] = hr_train['relevent_experience'].replace('Has relevent experience', 1)
        hr_train['relevent_experience'] = hr_train['relevent_experience'].replace('No relevent experience', 0)
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

        hr_train = hr_train.drop(columns=['enrollee_id',
                                       'enrolled_university',
                                       'education_level',
                                       'company_size',
                                       'city'])

        X = pd.concat([hr_train, ordi1, ordi2, ordi3], axis=1)
        query = X.reindex(columns=model_columns, fill_value=0)
        prediction = svcclassifier.predict(query)

        if int(prediction)==1:
            output = 'The employee will leave.'
        else:
            output = 'The employee will not leave.'

        return render_template('index.html', prediction_text='{}'.format(output))



if __name__ == "__main__":
    svcclassifier = joblib.load("svm_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    app.run(debug=True)
