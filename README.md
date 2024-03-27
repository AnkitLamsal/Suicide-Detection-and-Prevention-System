# Suicidal Post Detection and Mental Health Conversation 

## This is an application oriented program code which detects whether a tweet or post is Suicidal or not and based on that provides Mental Health Conversation. 

### Datasets
### Suicidal Post Detection from Kaggle : https://www.kaggle.com/datasets/aunanya875/suicidal-tweet-detection-dataset/data
### Mental Health Conversational Data from Kaggle : https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data/data

### Approach and models  : 
#### A xgboost classifier classifies whether the given post or tweet is suicidal or not 
#### for mental health conversation, a bidirectional LSTM is used to predict the intent of input data and to generate the responses based on that. 

### Its applications: 
#### Can be used as a web crawler. 

### Problem Background: 
#### The rate and trend of people with suicidal thoughts and showing their desire to post or tweet as a cry for help before taking severe action is very high.
#### Sometimes all that a person need is a just a friendly chat. 
#### All the help that is present nowadays in terms of mental health and help against depression and suicide is as per need to know, 
#### person by their will have to go and search or talk . . . 
#### but this program can be implemented as a crawler, it would detect any potential suicidal tweets and hit a message to the respective person. 
