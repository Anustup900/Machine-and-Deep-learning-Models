import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
test=pd.read_csv(r"C:\Users\Anustup\Desktop\Test.csv")
train=pd.read_csv(r"C:\Users\Anustup\Desktop\train.csv")
train['Type']='Train' #Create a flag for Train and Test Data set
test['Type']='Test'
fullData = pd.concat([train,test],axis=0) #Combined both Train and Test Data set
fullData.columns # This will show all the column names
fullData.head(10) # Show first 10 records of dataframe
fullData.describe() #You can look at summary of numerical fields by using describe() function
ID_col = ['REF_NO']
target_col = ["Age"]
cat_cols = ['location','status','gender','skill','Reason_of_login','Education','Income','self_employed', 'self_employed_partner']
num_cols= list(set(list(fullData.columns))-set(cat_cols)-set(ID_col)-set(target_col))
other_col=['Type'] #Test and Train Data set identifier
num_cat_cols = num_cols+cat_cols # Combined numerical and Categorical variables

#Create a new variable for each variable having missing value with VariableName_NA 
# and flag missing value with 1 and other with 0

for var in num_cat_cols:
    if fullData[var].isnull().any()==True:
        fullData[var+'_NA']=fullData[var].isnull()*1 
#Impute numerical missing values with mean
fullData[num_cols] = fullData[num_cols].fillna(fullData[num_cols].mean(),inplace=True)
#Impute categorical missing values with -9999
fullData[cat_cols] = fullData[cat_cols].fillna(value = -9999)
#create label encoders for categorical features
for var in cat_cols:
 number = LabelEncoder()
 fullData[var] = number.fit_transform(fullData[var].astype('str'))

#Target variable is also a categorical so convert it
fullData["Account.Status"] = number.fit_transform(fullData["Account.Status"].astype('str'))

train=fullData[fullData['Type']=='Train']
test=fullData[fullData['Type']=='Test']

train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
Train, Validate = train[train['is_train']==True], train[train['is_train']==False]
features=list(set(list(fullData.columns))-set(ID_col)-set(target_col)-set(other_col))
x_train = Train[list(features)].values
y_train = Train["Account.Status"].values
x_validate = Validate[list(features)].values
y_validate = Validate["Account.Status"].values
x_test=test[list(features)].values
random.seed(100)
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)
status = rf.predict_proba(x_validate)
fpr, tpr, _ = roc_curve(y_validate, status[:,1])
roc_auc = auc(fpr, tpr)
print roc_auc

final_status = rf.predict_proba(x_test)
test["Account.Status"]=final_status[:,1]
test.to_csv('C:/Users/Analytics Vidhya/Desktop/model_output.csv',columns=['REF_NO','Account.Status'])
