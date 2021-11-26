# HR-Analytics: Predict whether someone will quit the job
- This is a Flask API developed by me to determine if a Data Scientist would leave their current job provided their previous information.
> Deploy the API on Heroku by clicking the button below.<br><br> 
[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://hr-analytics-will-someone-quit.herokuapp.com)

### Running the project locally:

##### - Clone this repository. Open CMD. Ensure that you are in the project home directory. Create the machine learning model by running [SVM.py](https://github.com/AnityaGan9urde/HR-Analytics-Will-someone-quit/edit/main/SVM.py) as such:

`python SVM.py`

##### - This will create a serialized version of our model into a file svm_model.pkl or use the pretrained model saved with the same name.

##### - Now, run [Employee_api.py](https://github.com/AnityaGan9urde/HR-Analytics-Will-someone-quit/edit/main/Employee_api.py) using below command to start Flask API

`python Employee_api.py`

##### - Open any browser and paste this URL: http://localhost:5000 to run the file as an app.
<hr>

### Following images show how the API will look when run properly:<br><br>  
<img src="https://user-images.githubusercontent.com/68852047/120635173-cb947580-c489-11eb-9e0d-5907ca7de54c.png" style="width: 750px"></img><br>
<img src="https://user-images.githubusercontent.com/68852047/120635186-cfc09300-c489-11eb-9c7e-a825d0d6c310.png" style="width: 750px"></img>
<hr>

### Dataset:
- The dataset used for training the SVM model was taken from Kaggle.
- Link: HR Analytics: Job Change of Data Scientists: https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists
- Features:
  - *enrollee_id*: Unique ID of the candidate
  - *city*: City code
  - *city_development_index*: Developement index of the city (scaled)
  - *gender*: Gender of the candidate
  - *relevent_experience*: Relevant experience of the candidate
  - *enrolled_university*: Type of University course enrolled (if any)
  - *education_level*: Education level of the candidate
  - *major_discipline*: Education major discipline of the candidate
  - *experience*: Candidate total experience in years
  - *company_size*: No. of employees in current employer's company
  - *company_type*: Type of the current employer
  - *last_new_job*: Difference in years between previous job and current job
  - *training_hours*: Training hours completed
- Target: 
  - 0 : Not looking for job change, 
  - 1 : Looking for a job change<br>
### Model Used:
- The dataset shows that this is clearly a **classification** task and can be solved by a myriad of classification algorithms such as Logistic Regression, Decision Trees and even Random Forests.
- I chose **Support Vector Machines(SVMs)** because of the flexibility it shows during training.
- After implementing **Cross Validation using StratifiedKFold** and doing parameter search using **Grid Search CV** on both **SVM** and **Random Forest**, I found out that the **SVM** *performed slightly better* in understanding the correlation between the features and the target.
- Following is the table for their individual scores.  

| **Scoring Parameter** | **Random Forest** | **SVM** |  
|:-------------------:|:---------------:|:-----:|  
| *Accuracy* | 0.77 | 0.77 |  
| *Precision(wgt)* | 0.75 | 0.75 |  
| *Recall(wgt)* | 0.77 | 0.77 |  
| *F1-score(wgt)* | 0.75 | 0.76 |  

- Recall for each class:

| **Classes** | **Random Forest** | **SVM** |
|:----:|:---:|:---:|
| 0 | 0.92 | 0.90 |
| 1 | 0.32 | 0.38 |

- Hyper Parameters chosen for:
  - **SVM**: {'C': 411, 'kernel': 'rbf'}
  - **Random Forest**: {'criterion': 'entropy', 'max_depth': 9, 'max_features': 'sqrt', 'n_estimators': 425}
- Provided the dataset was slightly unbalanced, the SVM model gave a better Recall score for the negative classes as compared to Random Forest. 
- Hence, I chose SVM as the model to use for the API.<br>
### API:
- I have made an *API for the SVM model* so that users can interact and use the model with ease. 
- To make the API work I have used the **Flask** library which are mostly used for such tasks.
- I have also connected a **HTML** form to the flask app to take in user input and a **CSS** file to decorate it.<br>
### Deployment:
- The Flask API was deployed on the **Heroku** cloud platform so that anyone with the link to the app can access it online.
- I have connected this GitHub repository to the Heroku dyno so that it can be run on the cloud.
- I have used the **Gunicorn** package which lets Python applications run on any web server. The `Procfile` and `requirements.txt` should be defined with all the details required before the deployment.<br>
### What did I learn:
- *Data Wrangling* using **Pandas**
- *Feature Engineering* to fit our data to our model
- Selecting the right model using *cross-validation*
- *Hyperparameter tuning*
- *Unbalanced datasets* are hard to work with and *finding the right scoring method*
- Saving the model and using it again with **Pickle**
- Making a flask app
- A little frontend web development
- Making the app live by deploying it on cloud platforms 

