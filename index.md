# Airbnb Seattle, this is might help to get more booking 😉

![1_Z5xW8y_nEaOhnrpgKsrkIw](https://user-images.githubusercontent.com/60525865/182408108-aea6d3d1-3210-4f91-b3cc-9bd394da7155.jpeg)


## Introduction

Airbnb nowdays is one of the most popular host reservation sites where you can book an house, apartament, loft and others types of places, it can be in anywhere in the planet. This blog will be about the Airbnb seattle data base which brings three different files that has information for all available home places and hosts in seattle:

- Listings, including full descriptions and average review score
- Reviews, including unique id for each reviewer and detailed comments
- Calendar, including listing id and the price and availability for that day

The description of this data base is on here [Seattle Airbnb](https://www.kaggle.com/datasets/airbnb/seattle)

For this reason we would like to answer the next couple of questions about this data set
### Questions
#### Q1
What time of the year has the most bookings and prices?
#### Q2
Indentify which variables have influences on the price of each Airbnb?
#### Q3
Can we predict the price of each properties offert to help host to have more bookings?

Before to answer this question this dataset need to be clening, this part just will be mention the principal clean process that has to be made.

## Data Wrangling
First, we identify the shape of the dataset
```markdown
Dimension of the dataset for Calendar is 1393570 rows and 4 columns
Dimension of the dataset for Reviews is 84849 rows and 6 columns
Dimension of the dataset for Listings is 3818 rows and 92 columns
```
With the shape we like to see the null values that might have the dataset

## Cleaning

## 📅 Calendar
![Calendar](https://user-images.githubusercontent.com/60525865/182382443-dee04bdb-06e6-404e-92a2-0f768016b064.png)

It sees that calendar has null in prices, then for make a solution of that problem we drop the nulls even that in others dataset can be those nulls but the price of the place they are missing it is not in either of the others data base when we look up.

Furthermore, we create months name and days of the week varibles to be viable to answer the first question.
## 🔎 Reviews

![Reviews](https://user-images.githubusercontent.com/60525865/182391635-88652b39-d611-463a-b150-e2621f344bc3.png)

It sees that reviews has null in comments but it is just not necessary because the question will not answer about satistifation of the positive or negative language of a book place.

## 📝 Listing

![listing_cleaning](https://user-images.githubusercontent.com/60525865/182394336-5848279c-1552-45e7-99a9-d7416d632824.png)

It sees that many variables has missing values but the most missing values are in:
```markdown
monthly_price    0.602672
square_feet      0.974594
license          1.000000
```
That they have more that the 60% of the information missing for tha reason this variables will not take into a count because even that the uses of an imputation method will produce that the distribution change and might get a stroger weight at time to predict.

![image](https://user-images.githubusercontent.com/60525865/182400479-e5f9ae0c-414c-4efc-b9e4-0df21a4017cb.png)

Also the dataset has information as reviews of comments that it will not need it, as the image shows so every comments will remove and also every url information

![box_plot_accomodate](https://user-images.githubusercontent.com/60525865/182403669-c357482a-61d3-43e0-9fdb-cc56aeb31d2e.png)

Creating a box plot of accommodates which is the number of day booking, and price we can see that the price has some outliers and might generate problems at time to predict the model, for thar reason we use a z score methodology to remove this outliers and no use it in the dataset.

```markdown
Number of outliers: 81
```

## Q1
What time of the year has the most bookings and prices?

![price_vs_available_month](https://user-images.githubusercontent.com/60525865/182405466-00e2d3f7-af02-4777-b9a0-a8102af69122.png)

In the first plot, it sees that in the middle of May the price start to goes up and the available get donw that behavior keeps happen through June, July, August and September, where July has the expensive booking and lowest available, that time in the year it is summer so that makes a lot sense. The others seasons of the year Seattle has less visits, it can be because the weather it is colder, where January it is cheaper and higher available because it is winter.

![price_vs_available_day](https://user-images.githubusercontent.com/60525865/182405499-2cdc8e18-9665-46bd-a9cd-1b4f9b8e2054.png)

The second plot shows that during the week the time of the weekends it is where the price is higher and the available goes down that means the people that take on a reservation in Airbnb that go for the weekend and not for all week apparently.

## Q2
Indentify which variables have influences on the price of each Airbnb

![correlation_matriz_p1](https://user-images.githubusercontent.com/60525865/182405839-f1b519c9-f106-49b8-a029-fed8d74c7115.png)

At produce a correlation matriz we can see the stong and week relation ship between variables with the price

![image](https://user-images.githubusercontent.com/60525865/182406257-502078a3-fefa-4514-b594-a433d3868912.png)

Where accommodates, cleaning_fee, bedrooms are the most strong variables with price that means that people when pay for a place consider those items importances and the options that brings.

![image](https://user-images.githubusercontent.com/60525865/182408919-5ce10a7e-2d9a-4d44-9fab-2e61104576b7.png)

In the other hand looking into the negative correlation it sees that reviews_per_month, number_of_reviews and calculated_host_listings_count has weeks relationship and if we take a plot about this variables this what we get:

![reviews_per_month vs price](https://user-images.githubusercontent.com/60525865/182409698-31551797-3571-47a3-9936-a348ee578e08.png)

![number_of_reviews vs price](https://user-images.githubusercontent.com/60525865/182409711-5f3bc6a2-3faf-4e95-a248-b5d8b25925c9.png)


As we said before some variables have weak relationship as:
- reviews_per_month that does not have sense because with less reviews the price is higher and that has to be reverse the relationship
- the same happend with number_of_reviews or calculated_host_listings_count
if we look at the scatters plots of this variables we can see that the relationships do not exist and it will help in the question 3 include that variables

## Q3
Can we predict the price of each properties offert to help host to have more bookings?

To predict the price we take into the account to impute the values missing and the categorical variables that we have, for the first thing we impute values: 

![image](https://user-images.githubusercontent.com/60525865/182523495-a52813af-9c9f-4a81-b890-8762688672d0.png)

For some information we use the mean because those variebles are express in money, but some we use mode because we cannot have half rooms or something like that.

After all that we proceed to dummies all categorical variables that we are gonna use in the models to predict prices and we get a this inside into the models, that left us with:
```markdown
Dimension of the dataset for is 3737 rows and 41 column
```
We use three models, first one lineal regression, Decision Tree Regressor and Random Forest Regressor

### lineal regression

![image](https://user-images.githubusercontent.com/60525865/182413061-378d25d2-1982-474e-b962-b72477ef3ea1.png)

Get deep in the variables that have strong relatitionship with price at time to predict the price the are not a variable that has the stronger relationship but requires_license, availability_30, availability_90 and host_identity_verified	have the stronger it meand that the price is influence for availables days and certification of the place that really it is trustworthy. 

```markdown
print(r2_score(ytrain, ytrain_pred))
print(mean_squared_error(ytrain, ytrain_pred))
print(r2_score(ytest, ytest_pred))
print(mean_squared_error(ytest, ytest_pred))

0.6173492343016038
1757.3920106804337
0.5975286586007006
1537.5763056795515
```
We can see that the model has a good prediction does not show a subestimation or overestimation.

![lm_regre](https://user-images.githubusercontent.com/60525865/182523685-1894c34c-9578-4e0b-8496-13dcee7bc28b.png)


### Decision Tree Regression

![image](https://user-images.githubusercontent.com/60525865/182414541-4d695e35-6a9e-4b8e-abba-d499bb6630f8.png)

It this case, the strogers relationships are in variables that we see in the analysis of the variables that we are gonna uses in the model, for exmple cleaning_fee, bedrooms, room_type_Private, room_type_Shared_room and bathrooms, have the importance to predict price and make sense because the fee of cleaning and the number of bedrooms can predict a better price

```markdown
print(r2_score(ytrain, ytrain_pred))
print(mean_squared_error(ytrain, ytrain_pred))
print(r2_score(ytest, ytest_pred))
print(mean_squared_error(ytest, ytest_pred))

0.6284948369628475
1706.2038390447688
0.5412383135084035
1752.6244145747703
```
We can see that the model has a good prediction does not show a subestimation or overestimation.

![decision_tree](https://user-images.githubusercontent.com/60525865/182523695-d41c096f-f6c0-46a0-a9e9-d30311854544.png)


### Random Forest Regression

![image](https://user-images.githubusercontent.com/60525865/182416630-2d3a131d-fb22-4369-86f4-5c5ba860b68b.png)

It this case, the strogers relationships are in variables that we see in the analysis of the variables that we are gonna uses in the model, for exmple bedrooms, cleaning_fee, and accommodates, have the importance to predict price and make sense because the fee of cleaning and the number of bedrooms can predict a better price even the days of the place

```markdown
print(r2_score(ytrain, ytrain_pred))
print(mean_squared_error(ytrain, ytrain_pred))
print(r2_score(ytest, ytest_pred))
print(mean_squared_error(ytest, ytest_pred))

0.7155278571784438
1306.489143287873
0.6235268253758193
1438.254537611428
```

We can see that the model has a good prediction does not show a subestimation or overestimation.

![random_forest](https://user-images.githubusercontent.com/60525865/182523710-e4752653-5b61-487c-a7f2-aec8b2e3bf00.png)


As a result to implement differents kind of models we have good prediction of the price to booking a Airbnb. Even though the models have more than 50% to predict new information the stronger result it is the random forest it which can predict the 62%, 12% more.

## Conclusions

According with the idea of explore the incredible word of Airbnb in seatle, it can see that the purpose our three question can be answer, just be cleaning, analyzing and modeling the dataset. Even though it can answer those question, this dataset has so much to offer as prediction for location, neighborhood, even the street of each Airbnb, it could predict o analyze which it is the better place to stay in price and location, that just and idea, as the tittle said welcome to the incredible world of Seattle Airbnb.
