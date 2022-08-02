# Welcome to the incredible world of Airbnb in Seattle


# 1. Installations

For the development of this project we use diferent kind of libreries the list of those it is listing next:

- import numpy as np
- import pandas as pd
- import matplotlib.pyplot as plt
- from sklearn.linear_model import LinearRegression
- from sklearn.ensemble import RandomForestRegressor
- from sklearn.tree import DecisionTreeRegressor
- from sklearn.model_selection import train_test_split
- from sklearn.metrics import r2_score, mean_squared_error
- from sklearn.model_selection import StratifiedKFold,GridSearchCV
- from sklearn.preprocessing import StandardScaler,MinMaxScaler 
- import seaborn as sns;sns.set_theme(style="darkgrid")
- import math
- from scipy import stats
- from sklearn.impute import SimpleImputer
- from sklearn.feature_selection import VarianceThreshold
- %matplotlib inline

Also this project where produce in python 3 and the version of the libreries was the newest of day write this readme.

# 2.Project Motivation

As a process to improve the skills as a data scientist take in on a course of Udacity, the decision to explore the dataset of Airbnb in Seattle and make some insides in the information as: 

### Questions
#### Q1
What time of the year has the most bookings and prices?
#### Q2
Indentify which variables have influences on the price of each Airbnb?
#### Q3
Can we predict the price of each properties offert to help host to have more bookings?

# File Descriptions
The Airbnb seattle 
- Listings, including full descriptions and average review score
- Reviews, including unique id for each reviewer and detailed comments
- Calendar, including listing id and the price and availability for that day

