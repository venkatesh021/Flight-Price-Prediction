import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle


#Importing data
data=pd.read_excel(r"C:\Users\Mani chandhar Reddy\OneDrive\Desktop\Flight Price Prediction Using ML\Data\Data_Train.xlsx")
# Handle missing values
data.fillna(0, inplace=True)
#Viewing the sample of data:
print(data.head())
#Finding unique values in categorical data:
category=[]
category.append("Airline")
category.append("Source")
category.append("Destination")
category.append("Additional_Info")
for i in category:
    unique_values = data[i].unique()
    print(f"Unique values in {i}:")
    print(unique_values)
#we now split the Date column to extract the "Date","Month" and "year"
data.Date_of_Journey=data.Date_of_Journey.str.split('/')
print(data.Date_of_Journey)
#Splitting the Data columns
data['Date']=data.Date_of_Journey.str[0]
data['Month']=data.Date_of_Journey.str[1]
data['Year']=data.Date_of_Journey.str[2]
#Checking maximum no of stops:
Stops=data.Total_Stops.unique()
print(Stops)
#Based on Maximum no of stops we can split the cities from minimum 2 to maximum 6:
data.Route=data.Route.str.split('→')
print(data.Route)
data['City1']=data.Route.str[0]
data['City2']=data.Route.str[1]
data['City3']=data.Route.str[2]
data['City4']=data.Route.str[3]
data['City5']=data.Route.str[4]
data['City6']=data.Route.str[5]
#Similarly we can split the departure time :
data.Dep_Time=data.Dep_Time.str.split(':')
data['Dep_Time_Hour']=data.Dep_Time.str[0]
data['Dep_Time_Mins']=data.Dep_Time.str[1]
#for the arrival date and arrival time separation, we split the ‘Arrival_Time’ column, and create ‘Arrival_date’ column. We also split the time and divide it into ‘Arrival_time_hours’ and ‘Arrival_time_minutes’, similar to what we did with the ‘Dep_time’ column
data.Arrival_Time=data.Arrival_Time.str.split(' ')
data['Arrival_date']=data.Arrival_Time.str[1]
data['Time_of_Arrival']=data.Arrival_Time.str[0]
data['Time_of_Arrival']=data.Time_of_Arrival.str.split(':')
data['Arrival_Time_Hour']=data.Time_of_Arrival.str[0]
data['Arrival_Time_Mins']=data.Time_of_Arrival.str[1]
#Same process for Duration column also:
data.Duration=data.Duration.str.split(' ')
data['Travel_Hours']=data.Duration.str[0]
data['Travel_Hours']=data['Travel_Hours'].str.split('h')
data['Travel_Hours']=data['Travel_Hours'].str[0]
data.Travel_Hours=data.Travel_Hours
data['Travel_Mins']=data.Duration.str[1]
data.Travel_Mins=data.Travel_Mins.str.split('m')
data.Travel_Mins=data.Travel_Mins.str[0]
#Now lets replace the "Non-stop" into 0:
data.Total_Stops.replace('non-stop',0,inplace=True)
data.Total_Stops=data.Total_Stops.str.split(' ')
data.Total_Stops=data.Total_Stops.str[0]
#In Additional info column, we find that 'I' is capital in No Info so we replace it with "No info" to merge it into single category.
data.Additional_Info.replace('No Info','No info',inplace=True)
#Cheking Missing values:
print(data.isnull().sum())
#Now drop columns which has majority of null values:
data.drop(['City4','City5','City6'],axis=1,inplace=True)
data.drop(['Date_of_Journey','Route','Dep_Time','Arrival_Time','Duration','Time_of_Arrival'],axis=1,inplace=True)
print(data.isnull().sum())
#Now Treating Missing Values:
data['City3'].fillna('None',inplace=True)
data['Arrival_date'].fillna(data['Date'],inplace=True)
data['Travel_Mins'].fillna(0,inplace=True)
#After treating all the missing values lets check the Info of data:
print(data.info())
#Now lets change the data types of columns which consists of integer values:
data.Date=data.Date.astype('int64')
data.Month=data.Month.astype('int64')
data.Year=data.Year.astype('int64')
data.Dep_Time_Hour=data.Dep_Time_Hour.astype('int64')
data.Dep_Time_Mins=data.Dep_Time_Mins.astype('int64')
data.Arrival_date=data.Arrival_date.astype('int64')
data.Arrival_Time_Hour=data.Arrival_Time_Hour.astype('int64')
data.Arrival_Time_Mins=data.Arrival_Time_Mins.astype('int64')
#for conversion of Travel_Hours we have to rectify a not convertable issue at index value 6474 where Travel_Hours=5m:
data.drop(index=6474,inplace=True,axis=0)
data.Travel_Hours=data.Travel_Hours.astype('int64')
data.Travel_Mins=data.Travel_Mins.astype('int64')
#Here Total_stops is not converting into int since it has a NAN Value some where in data so im changing it into 0:
data['Total_Stops'].fillna(0, inplace=True)
data.Total_Stops=data.Total_Stops.astype('int64')
print(data.info())
#Now seperate the columns based on their data types:
categorical=['Airline','Source','Destination','Additional_Info','City1','City2','City3']
Numerical=['Total_Stops','Price','Date','Month','Year','Dep_Time_Hour','Dep_Time_Mins','Arrival_date','Arrival_Time_Hour','Arrival_Time_Mins','Travel_Hours','Travel_Mins']
#Now lets use label Encoding to convert Categorical data into integer values:
le=LabelEncoder()
data.Airline=le.fit_transform(data.Airline)
data.Source=le.fit_transform(data.Source)
data.Destination=le.fit_transform(data.Destination)
data.Total_Stops=le.fit_transform(data.Total_Stops)
data.City1=le.fit_transform(data.City1)
data.City2=le.fit_transform(data.City2)
data.City3=le.fit_transform(data.City3)
data.Additional_Info=le.fit_transform(data.Additional_Info)
print(data.head())
#Now lets remove the unwanted or columns which does not impact the price:
data1=data[['Airline','Source','Destination','Date','Month','Year','Dep_Time_Hour','Dep_Time_Mins','Arrival_date','Arrival_Time_Hour','Arrival_Time_Mins','Price']]
print(data1.head())
print(data1.describe())
print(data1.info())
#Visual Analysis:
for col in categorical:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=col, palette='viridis')
    plt.xticks(rotation=90)
    plt.title(f'Count of {col}')
    plt.show()
#Here we are having some issues for visualizing multiple plots in single page so we are visualizing each plot individually.
#Lets visualize Distribution of Price column:
plt.figure(figsize=(15, 8))
g = sns.displot(data=data, x='Price', kde=True, color='blue', bins=30)
g.set(xlabel='Price', ylabel='Density')
plt.title('Distribution of Price')
plt.show()
#Now lets check the Corrleation Using HeatMap
plt.figure(figsize = (20,10))
sns.heatmap(data.corr(),annot=True)
plt.show()
#Now lets find out outiliers in price column using boxplot:
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Price'])
plt.xlabel('Price')
plt.title('Distribution of Price')
plt.show()
#Scaling Data:
y=data1['Price']
x=data1.drop(columns=['Price'],axis=1)
ss=StandardScaler()
x_scaled=ss.fit_transform(x)
x_scaled=pd.DataFrame(x_scaled,columns=x.columns)
print(x_scaled.head())
#Splitting data into train and test:
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.head())
#Training the Model in multiple algorithms:
rfr=RandomForestRegressor()
gb=GradientBoostingRegressor()
ad=AdaBoostRegressor()
for i in [rfr,gb,ad]:
    i.fit(x_train,y_train)
    y_pred=i.predict(x_test)
    test_score=r2_score(y_test,y_pred)
    train_score=r2_score(y_train, i.predict(x_train))
    if abs(train_score-test_score)<=0.2:
        print(i)
        print("R2 score is",r2_score(y_test,y_pred))
        print("R2 for train data" ,r2_score(y_train, i.predict(x_train)))
        print("Mean Absolute Error is" ,mean_absolute_error(y_pred,y_test))
        print( "Mean Squared Error is",mean_squared_error(y_pred,y_test))
        print( "Root Mean Sqaured Error is", (mean_squared_error(y_pred,y_test,squared=False)))
knn=KNeighborsRegressor()
lr=LinearRegression()
svr=SVR()
dt=DecisionTreeRegressor()
for i in [knn, svr, dt]:
    i.fit(x_train, y_train)
    y_pred = i.predict(x_test)
    test_score = r2_score(y_test, y_pred)
    train_score = r2_score(y_train, i.predict(x_train))
    if abs(train_score - test_score) <= 0.1:
        print(i)
        print('R2 Score is', test_score)
        print('R2 Score for train data', train_score)
        print('Mean Absolute Error is', mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error is', mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error is', mean_squared_error(y_test, y_pred, squared=False))
#Checking cross validation for RandomForestRegressor:
# defining the parameters for the randomforest
n_estimators = [int(x) for x in np.linspace(start = 100, stop=1200,num=12)] # number of trees
max_features = ['sqrt'] # number feature for every split
max_depth = [int(x) for x in np.linspace(5,30,num=6)]
min_samples_split=[2,5,10,15,100]
min_samples_leaf=[1,2,5,10]
# creating a random grid
random_grid = {
    'n_estimators': n_estimators,
    'max_features' : max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf
}
random_search = RandomizedSearchCV(lr, random_grid, cv=5, n_iter=10)
random_forest_cv = RandomizedSearchCV(estimator = rfr, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
random_forest_cv.fit(x_train,y_train)
#Testing the model:
#Testing using RandomForestRegressor and GradientBoostingRegressor():
param_grid={'n_estimators':[10,30,50,70,100],'max_depth':[None,1,2,3],'max_features':['sqrt']}
rfr=RandomForestRegressor()
rf_res=RandomizedSearchCV(estimator=rfr,param_distributions=param_grid,cv=3,verbose=2,n_jobs=-1)
rf_res.fit(x_train,y_train)
gb=GradientBoostingRegressor()
gb_res=RandomizedSearchCV(estimator=gb,param_distributions=param_grid,cv=3,verbose=2,n_jobs=-1)
gb_res.fit(x_train,y_train)
# Calculate accuracy of the models
rfr=RandomForestRegressor(n_estimators=10,max_features='sqrt',max_depth=None)
rfr.fit(x_train,y_train)
y_train_pred=rfr.predict(x_train)
y_test_pred=rfr.predict(x_test)
# Print the accuracies
print("rfr train accuracy",r2_score(y_train_pred,y_train))
print("rfr test accuracy",r2_score(y_test_pred,y_test))
gb=GradientBoostingRegressor(n_estimators=10,max_features='sqrt',max_depth=None)
gb.fit(x_train,y_train)
y_train_pred=rfr.predict(x_train)
y_test_pred=rfr.predict(x_test)
# Print the accuracies
print("gb train accuracy",r2_score(y_train_pred,y_train))
print("gb test accuracy",r2_score(y_test_pred,y_test))
Price=data.Price
price_list=pd.DataFrame({'price':Price})
print(price_list)
#Saving the model:
file = open('Flight_random_forest_model.pkl','wb')
pickle.dump(random_forest_cv,file)
model=open('Flight_random_forest_model.pkl','rb')
random_forest = pickle.load(model)






