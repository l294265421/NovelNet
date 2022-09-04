# NovelNet
Modeling User Repeat Consumption Behavior for Online Novel Recommendation (RecSys 2022)

We implemented our model based on the session-recommedndation framework [session-rec](https://github.com/rn5l/session-rec), and detailed usage can be found there.

## Requirements
anaconda
python==3.7.1
torch==1.4.0 
scipy==1.6.2 
python-dateutil==2.8.1 
pytz==2021.1 
certifi==2020.12.5 
numpy==1.20.2 
dill==0.3.3 
pyyaml==5.4.1 
networkx==2.5.1 
scikit-learn==0.24.2 
numexpr==2.7.3 
keras==2.3.1 
six==1.15.0 
theano==1.0.3 
pandas==1.2.4 
psutil==5.8.0 
pympler==0.9 
tensorflow==2.3.0 
tables==3.6.1 
scikit-optimize==0.8.1 
python-telegram-bot==13.5

## Dataset
Datasets can be downloaded from: https://www.dropbox.com/sh/ur9amfhf9mag213/AAAtI7SWJft1WZZiR03nyDNCa?dl=0.
- Unzip the dataset file to the data folder.

## Running NovelNet
python run_confg.py conf/in_for_fiction_rec_more/84.yml conf/out

