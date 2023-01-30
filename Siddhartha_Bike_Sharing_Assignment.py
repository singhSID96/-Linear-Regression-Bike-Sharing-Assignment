#!/usr/bin/env python
# coding: utf-8

# ## Bike Sharing Assignment Case Study - Linear Regression

# #### Importing the Library

# In[106]:


# importing the Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# importing the library for models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.metrics import r2_score


# In[107]:


# supress warnings
import warnings
warnings.filterwarnings("ignore")


# In[108]:


# importing the data set

data = pd.read_csv("day.csv")


# ### Step 1 : Data Understanding

# In[109]:


data.head()


# In[110]:


data.info()


# In[111]:


data.describe()


# In[112]:


data.shape


# In[113]:


data.isnull().sum()


# #### From the above analysis we can conclude that there is no null values in the dataset.

# In[114]:


# analyzing the statistical values of numerical variable
data.describe()


# In[115]:


data["instant"].value_counts()


# In[116]:


# Instant column look like an Index type column which is not useful in model building so simply we can drop it.
data.drop(['instant'], axis=1 , inplace=True)


# In[117]:


data.head()


# In[118]:


# dteday column provide similar information like yr, mnth, weekday etc so simply we can drop it
data.drop(['dteday'], axis=1 , inplace=True)


# In[119]:


data.head()


# * cnt is the target variable and addition of casual and registered is equal to cnt variable. So keeping casual and registerated variable makes model complicated so straightly we can drop it

# In[120]:


# dropping the casual and registered columns
data.drop(['casual','registered'], axis=1 , inplace=True)


# In[121]:


data.head()


# By analyzing the data, we can say season,mnth,holiday,weekday,weathersit are categorical variables, so better replace with more meaningful name
# 

# In[122]:


data['season'] = data['season'].map({1:'spring', 2:'summer', 3:'fall', 4:'winter'})
data['mnth'] = data['mnth'].map({1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'june',7:'july',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'})
data['weekday'] = data['weekday'].map({0:'sun',1:'mon',2:'tue',3:'wed',4:'thu',5:'fri',6:'sat'})
data['weathersit'] = data['weathersit'].map({1:"Clear_Few Clouds_Partly Clouds",2:"Mist_cloudy",3:"Light snow_Light rain_Thunderstorm",4:"Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog"})


# In[123]:


data.head()


# In[124]:


# checking the datatype of the variable again after mapping values
data.info()


# ### Step 2 : Data Visualisation - Applying EDA on dataset

# In[125]:


# create box plot for category column with cnt variablr
# create barplot for category column with cnt variable
# create pairplot for numeric varable
# create heatmap for numeric variable


# In[126]:


sns.pairplot(data)
plt.show()


# In[127]:


# segregating the column
category_column = ['season','yr','mnth','holiday', 'weekday','workingday','weathersit']
numerical_column = ['temp', 'atemp' , 'hum' , 'windspeed']


# #### For the categorical column we can create Box plot w.r.t to CNT

# In[128]:


# Looping the category_column to create the box plot and using the For loop make a code efficient and simplier

plt.figure(figsize=(20, 15))
for i,j in enumerate(category_column):
    plt.subplot(3,3,i+1)
    sns.boxplot(x = j, y = 'cnt', data = data)
    plt.xticks(rotation='45')
plt.show()


# * Target varaible cnt is high in summer and fall season
# * Target variable is more in month june to october
# * Cnt is more in 2019 as compared to 2018
# * People prefer to rent bike when the sky is clear or few clouds

# ### Ploting BarPlot Yearwise

# In[129]:


def bar_categorical(column):
    plt.figure(figsize = (14,6))
    sns.barplot(column,'cnt',data=data, hue='yr',palette='Set1')
    plt.legend(title='yr', loc='upper left', labels=['2018', '2019'] ,fontsize="x-large" ,labelcolor = ['darkred','blue'])
    plt.show()


# In[130]:


bar_categorical('season')


# * Fall season seems to have more renting of bike and the users is increased from 2018 to 2019
# 

# In[131]:


bar_categorical('mnth')


# * The users is more in month june, july, aug ,sep and the count is increase from 2018 to 2019
# 

# In[132]:


bar_categorical('holiday')


# * More people prefer to rent bike on working days and during holiday people prefer to stay home.
# 

# In[133]:


bar_categorical('weekday')


# * Thu, Fir, Sat and Sun have more number of users as compared to the start of the week.
# 

# In[134]:


bar_categorical('workingday')


# * Booking seemed to be almost equal either on working day or non-working day. But, the count increased from 2018 to 2019.
# 

# In[135]:


bar_categorical('weathersit')


# * People prefer to rent bike when the wheather is clear or few clouds
# 

# ### Analyzing Numerical Variables

# In[136]:


# ploting pair plot for categorical variable
sns.pairplot(data, vars=['temp','atemp','hum','windspeed',"cnt"])
plt.show()


# In[137]:


#plotiing thr heat map for checking corelation
plt.figure(figsize = (14, 8))
sns.heatmap(data.corr(), annot = True, cmap="RdYlBu")
plt.show()


# * The coorelation between temp and atemp is very high 0.99. So we ignore 1 variable during RFE method.
# * The coreation between cnt with yr and temp is also pretty good.

# ### Step 3 : Preparing the Data

# In[138]:


data.info()


# In[139]:


## check the data again
data.head()


# Create Dummy Variable for season, mnth, weekday, weathersit

# In[140]:


df_seasons = pd.get_dummies(data.season,drop_first=True)
df_months = pd.get_dummies(data.mnth,drop_first=True)
df_weekdays = pd.get_dummies(data.weekday,drop_first=True)
df_weathersit = pd.get_dummies(data.weathersit,drop_first=True)


# In[141]:


df_seasons


# In[142]:


df_months


# In[143]:


df_weekdays


# In[144]:


df_weathersit


# In[145]:


# concating all the dataframe to the original data
data = pd.concat([data,df_seasons,df_months,df_weekdays,df_weathersit],axis=1)


# In[146]:


data.head()


# In[147]:


# Dropping season, mnth, weekday, weathersit columns as we have already created dummy variable out of it.
data.drop(['season','mnth','weekday','weathersit'], axis = 1, inplace = True)


# In[148]:


data.head()


# In[149]:


data.info()


# ### Step 4 : Split the dataset into Training and Test data

# In[150]:


# splitting the dataframe into test and train

# we need same row for test and train data respectively
np.random.seed(0)
data_train, data_test = train_test_split(data, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[151]:


# checking the train data
data_train.head()


# In[152]:


data_train.shape


# In[153]:


# checking the test data
data_test.head()


# In[154]:


data_test.shape


# #### Scaling the Numerical Variable

# In[155]:


# Using the minmaxscaler for sclaing the numerical variable..
scaler = MinMaxScaler()


# In[156]:


# checking the columns again
data.columns


# In[157]:


numerical_var = ['temp', 'atemp', 'hum', 'windspeed' ,'cnt']
data_train[numerical_var] = scaler.fit_transform(data_train[numerical_var])


# In[158]:


data_train.head()


# In[159]:


data_train.describe()


# In[160]:


## creating the X anf y variable
y_train = data_train.pop('cnt')
X_train = data_train


# In[161]:


X_train.head()


# In[162]:


y_train.head()


# ### Analyzing the Corelation matrix among the variables

# In[163]:


plt.figure(figsize=(30,25))
sns.heatmap(data.corr() , annot = True, cmap='RdYlBu')
plt.show()


# * CNT target variable shows good corelation with temp, atemp , yr. While there is a good negative corelation with spring also.
# * CNT also show good coreation with jan, feb and light snow_light rain_thunderstorm variable.

# #### Selecting the Top15 variable through RFE method

# In[164]:


# RFE method for feature Elimination

lm = LinearRegression()
lm.fit(X_train,y_train)

rfe = RFE(lm,n_features_to_select = 15)
rfe.fit(X_train,y_train)


# In[165]:


# zipping the columns name with ranking 
list(zip(X_train.columns, rfe.support_,rfe.ranking_))


# In[166]:


# selecting the RFE columns
column = X_train.columns[rfe.support_]
column


# ### Step 5 Building the Linear Model
# 

# In[167]:


X_train_RFE = X_train[column]


# In[168]:


X_train_lm_1 = sm.add_constant(X_train_RFE)
lr_1 = sm.OLS(y_train,X_train_lm_1).fit()
print(lr_1.summary())


# #### Calculate VIF

# In[169]:


## Creating the generic function to calculate VIF
def calculate_VIF(df):
    vif = pd.DataFrame()
    vif['feature'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values ,i) for i in range(df.shape[1])]
    vif['VIF'] = round(vif['VIF'],2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return vif 


# In[170]:


calculate_VIF(X_train_RFE)


# In[171]:


# As humidity shows very high VIF values hence we can drop it
X_train_new = X_train_RFE.drop(['hum'], axis = 1)


# ##### Building the second model

# In[172]:


X_train_lm_2 = sm.add_constant(X_train_new)
lr_2 = sm.OLS(y_train,X_train_lm_2).fit()
print(lr_2.summary())


# In[173]:


# calculating the VIF
calculate_VIF(X_train_new)


# In[174]:


# As summer shows  high P values hence we can drop it
X_train_new = X_train_new.drop(['summer'], axis = 1)


# ##### Building the third Model

# In[175]:


X_train_lm_3 = sm.add_constant(X_train_new)
lr_3 = sm.OLS(y_train,X_train_lm_3).fit()
print(lr_3.summary())


# In[176]:


# calculate VIF
calculate_VIF(X_train_new)


# * The model seems good as all the VIF value should be less than 5. Let's make the model better
# 
# 

# In[177]:


# As nov shows  high P values hence we can drop it
X_train_new = X_train_new.drop(['nov'], axis = 1)


# Buiding the fourth model

# In[178]:


X_train_lm_4 = sm.add_constant(X_train_new)
lr_4 = sm.OLS(y_train,X_train_lm_4).fit()
print(lr_4.summary())


# In[179]:


# calculate VIF
calculate_VIF(X_train_new)


# In[180]:


# As dec shows  high P values hence we can drop it
X_train_new = X_train_new.drop(['dec'], axis = 1)


# building the fifth model

# In[181]:


X_train_lm_5 = sm.add_constant(X_train_new)
lr_5 = sm.OLS(y_train,X_train_lm_5).fit()
print(lr_5.summary())


# In[182]:


# calculate VIF
calculate_VIF(X_train_new)


# In[183]:


# As Jan shows  high P values hence we can drop it
X_train_new = X_train_new.drop(['jan'], axis = 1)


# Building the sixth model

# In[184]:


X_train_lm_6 = sm.add_constant(X_train_new)
lr_6 = sm.OLS(y_train,X_train_lm_6).fit()
print(lr_6.summary())


# In[185]:


# calculate VIF
calculate_VIF(X_train_new)


# * We can cosider the above model i.e lr_6, as it seems to have very low multicolinearity between the predictors and the p-values for all the predictors is significant.
# * F-Statistics value of 248.7 (which is greater than 1) and the Prob (F-statistic) of 1.16e-186 i.e almost equals to zero, states that the overall model is significant

# In[186]:


#checking the coefficient parameter
lr_6.params


# ### Step 6 : Residual Analaysis

# In[187]:


# calculating the prediction value of y for the lr_6 model
y_train_pred = lr_6.predict(X_train_lm_6)


# ##### Distribution Of Error Terms

# In[188]:


# plotting the histogram graph of the residual
plt.figure(figsize=(10,6))
sns.distplot(y_train - y_train_pred ,bins =30)
plt.xlabel("Residual Error")
plt.show()


# * As per the above graph,errors are normally distributed
# 

# #### Homoscedasticity Of Model

# In[189]:


residual = y_train - y_train_pred
sns.scatterplot(y_train,residual)
plt.plot(y_train,(y_train - y_train), '-r')
plt.xlabel('Count')
plt.ylabel('Residual')
plt.show()


# * Model is distributed over the line. And there is no visible pattern is there.
# 

# #### Multicolinearity Of Model

# In[190]:


calculate_VIF(X_train_new)


# * All the values is less than 5. So the model is having no multi colinearity
# 
# 

# #### Linearity Of Model

# In[191]:


plt.figure(figsize=(8,6))
plt.scatter(y_train, y_train_pred, c='crimson')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
p1 = max(max(y_train_pred), max(y_train))
p2 = min(min(y_train_pred), min(y_train))
plt.plot([p1, p2], [p1, p2], 'b-')


# * The actual value and predicted value are showing linearly corelation.
# 

# ### Step 7 : Making Final Prediction

# In[192]:


# Applying scaling on the test dataset as we did earlier
numerical_vars = ['temp', 'atemp', 'hum', 'windspeed','cnt']

data_test[numerical_vars] = scaler.transform(data_test[numerical_vars])


# In[193]:


data_test.head()


# In[194]:


data_test.describe()


# In[195]:


# prepare the X and  y data for test
y_test = data_test.pop('cnt')
X_test = data_test


# In[196]:


columns_test = X_train_new.columns
X_test = X_test[columns_test]


# In[197]:


X_test.head()


# In[198]:


y_test.head()


# In[199]:


X_test_lm_6 = sm.add_constant(X_test)
y_test_pred = lr_6.predict(X_test_lm_6)  # predicing the values using the model


# In[200]:


# checking the r2 score
r2 = r2_score(y_test, y_test_pred) 


# In[201]:


r2


# ### Step 8 : Model Evaluation

# In[202]:


# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_test_pred)
plt.xlabel('y_test', fontsize = 18)
plt.ylabel('y_pred', fontsize = 16) 


# In[203]:


lr_6.params


# Write the Equation
# 
# cnt = 0.251899 + 0.234092 x yr + -0.098556 x holiday + 0.451455 x temp + -0.139817 x windspeed + -0.110795 x spring + 0.047278 x winter + -0.072719 x july + 0.057705 x sep + -0.286408 x Light snow_Light rain_Thunderstorm + -0.081133 x Mist_cloudy

# In[204]:


# Calculating Adjusted-R^2 value for the test dataset

adjusted_r2 = round(1-(1-r2)*(X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1),4)
print(adjusted_r2)


# In[205]:


# Visualizing the fit on the test data
# plotting a Regression plot

plt.figure()
sns.regplot(x=y_test, y=y_test_pred, ci=68, fit_reg=True,scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title('y_test vs y_pred', fontsize=20)
plt.xlabel('y_test', fontsize=18)
plt.ylabel('y_pred', fontsize=16)
plt.show()


# The model is giving the accurate prediction as the graph between test data and predicted data is linear.
# 
# 

# ### Final Summary
# 

# ##### Comparison
# 
# * Train dataset R^2 : 0.833
# * Test dataset R^2 : 0.8070
# * Train dataset Adjusted R^2 : 0.830
# * Test dataset Adjusted R^2 : 0.7977
# 
# #### Demand of bike depend on the below factors :
# 
# * yr, holiday, temp, windspeed, spring, winter, july, sep,Light snow_Light rain_Thunderstorm,Mist_cloudy

# In[ ]:




