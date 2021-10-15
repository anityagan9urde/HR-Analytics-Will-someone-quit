# HR-Analytics: Predict whether someone will quit the job- SVM model to determine if someone is bound to leave or not
[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://hr-analytics-will-someone-quit.herokuapp.com)

* Following image shows the Flask api which contains the form to fill up the employee information:

![Annotation 2021-06-03 162622](https://user-images.githubusercontent.com/68852047/120635173-cb947580-c489-11eb-9e0d-5907ca7de54c.png)

* This image shows the results displayed:

![Annotation 2021-06-03 162648](https://user-images.githubusercontent.com/68852047/120635186-cfc09300-c489-11eb-9c7e-a825d0d6c310.png)


### Running the project locally:

##### Open CMD. Ensure that you are in the project home directory. Create the machine learning model by running [SVM.py](https://github.com/AnityaGan9urde/HR-Analytics-Will-someone-quit/edit/main/SVM.py) as such:

`python SVM.py`

##### This would create a serialized version of our model into a file svm_model.pkl

##### Run [Employee_api.py](https://github.com/AnityaGan9urde/HR-Analytics-Will-someone-quit/edit/main/Employee_api.py) using below command to start Flask API

`python Employee_api.py`

##### By default, flask will run on port 5000.

##### Navigate to URL http://localhost:5000 on a browser.

##### You should be able to view the homepage.

##### Enter valid values in all the input boxes and hit Predict.

##### If everything goes well, you should be able to see the predicted vaule on the HTML page and you know if an employee will quit or not.
