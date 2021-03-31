# HR Analytics

## Project Overview:

* Using an imbalanced HR dataset from Big Data / Data Science - Company  
* Performing a binary Classification including Model Comparison, Validation, Evaluation & Classbalancing 
* Deploying the ML Model on Streamlit with additional functionalities like connection to MySQL and Tableau Public


## Introduction:

"A company which is active in Big Data and Data Science wants to hire data scientists among people who successfully pass some courses which conduct by the company. Many people signup for their training. Company wants to know which of these candidates are really wants to work for the company after training or looking for a new employment because it helps to reduce the cost and time as well as the quality of training or planning the courses and categorization of candidates. Information related to demographics, education, experience are in hands from candidates signup and enrollment." (Link to the data in the Resources)

### Features

enrollee_id : Unique ID for candidate

city: City code

city_ development _index : Developement index of the city (scaled)

gender: Gender of candidate

relevent_experience: Relevant experience of candidate

enrolled_university: Type of University course enrolled if any

education_level: Education level of candidate

major_discipline :Education major discipline of candidate

experience: Candidate total experience in years

company_size: No of employees in current employer's company

company_type : Type of current employer

lastnewjob: Difference in years between previous job and current job

training_hours: training hours completed

target: 0 – Not looking for job change, 1 – Looking for a job change


## Project Scope:

This project aims to provide a minimal valuable application to support the potential HR department of the unnamed company to predict which Data Scientists are looking for a new job opportunity. This application should include functionalities to load, save and visualize data and predictions to enable an more efficient recruiting process.

## App Preview:

The application was built with Streamlit and contains the following functionalities:
* Choosing where to load the data from (SQL or CSV)
* Inspecting the predictions (including probabilities) and corresponding features in a dataframe format 
* Storing the results in SQL, Google Sheet or download as CSV 
* Access to an embedded Tableau Public Dashboard (connected with the Google Sheet, dynamically updates once a day)

![Streamlit App Overview](https://github.com/Reik96/HR_Analytics/blob/master/pictures/Streamlit_Page1.JPG)
![Streamlit App Tableau Integration](https://github.com/Reik96/HR_Analytics/blob/master/pictures/Streamlit_Tableau.JPG)



## Conclusion:

### Best Algorithm

After validating and comparing different classifiers (Logistic Regression, Random Forest, XGBoosting, KNN and SVM), the Logistic Regression achieved the best results when it comes to detect the underrepresented class.

### Most important Features

It seems like the most important feature that decides whether the Data Scientists are looking for a new job or not is the city they are living in. <br>

![Feature Importance](https://github.com/Reik96/HR_Analytics/blob/master/pictures/Feature_Importance.jpeg)

###  Predictions:

With an AUC of over 76 % the Model generally can distinguish the classes quite good.<br>
A closer look on the classification report and confusion matrix reveals, that the Model is good in detecting class 0 but underperforming in detecting class 1. <br>

![Classification Report](https://github.com/Reik96/HR_Analytics/blob/master/pictures/Classification_Report.JPG)

![Confusion Matrix](https://github.com/Reik96/HR_Analytics/blob/master/pictures/Confusion_Matrix.jpeg)<br>

### Final Thoughts

After finishing my first project where I actually delivered a ML Service it is time now to wrap up the benefits and potential weaknesses of it.
I tried to provide a solution that comes as close as possible to a reliable tool for the potential HR department. The developed tool is able to detect over 77% of job seekers.
For the Testset, it reduces the amount of individuals in question by 63%. That would lead to a reduced costs, effort and time consumption during the recruiting process.<br> 
However there is still room for improvement, since the model has only a 51% chance, that Data Scientists marked as class 1 are really looking for a new job. Furthermore are the functionalities provided in the Streamlit-App not fully matured (e.g. the embedded Tableau Dashboard can only be accessed via API to refresh it on demand when using Tableau Server and Tableau Online).<br> 
All in all my developed Application would be a good MVP to start with for the unknown company to hire new Data Scientists. 


## Resources: 
* Data<br>
https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists/tasks?taskId=3015<br>
* Feature Name Function <br>
https://johaupt.github.io/scikit-learn/tutorial/python/data%20processing/ml%20pipeline/model%20interpretation/columnTransformer_feature_names.html<br>
* CSV Downloader Function <br>
https://github.com/Jcharis/Streamlit_DataScience_Apps/blob/master/File_Downloader_App/app.py
* Streamlit Basics<br>
https://github.com/dataprofessor/code/tree/master/streamlit
