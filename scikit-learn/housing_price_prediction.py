#!/usr/bin/env python
# coding: utf-8

# In[8]:

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#this line can be written only in jupyter notebook so that while plotting graphs
#you dont see OUT[25] on the left hand side.
get_ipython().run_line_magic('matplotlib', 'inline')

#one way of importing dataset using sklearn
from sklearn.datasets import fetch_california_housing


# In[9]:
housing=fetch_california_housing()
# In[10]:
housing


# In[11]:
#DESCR function gives you complete description about the dataset from where
#it is fetch

housing.DESCR


# In[12]:
housing.data


# In[13]:
#this will give the name of all the attributes
housing.feature_names


# In[14]:
housing.target


# In[15]:

housing=pd.read_csv(r'C:\Users\Lenovo\Desktop\ML course\scikit-learn\housing.csv')


# In[16]:
housing


# In[17]:
#gives you starting five rows
housing.head()


# In[18]:
#gives you ending five rows
housing.tail()


# In[19]:
#it gives info about each dataattribute how many notnull,memoryusage
housing.info()


# In[20]:
#Data Visualization,this will give you complete correlation matrix
housing.corr()


# In[21]:
#plotiing histogram for each atrribute
housing.hist(figsize=(30,35),bins=50)


# In[22]:
#dividing dataset into train,test
from sklearn.model_selection import train_test_split
#??train_test_split       #to see description of function

train_set,test_set=train_test_split(housing,test_size=0.2,random_state=2)


# In[23]:
train_set.size


# In[24]:
test_set.size


# In[25]:


#Assume housing price depends upon median_income of people staying thier
#as we can see in corr matrix income is higly corelated to price
housing['median_income'].hist()
#we can see that income ranges from 0-14+ so we can create categories of income.


# In[26]:
#standard scaling
housing['income_cat']=np.ceil(housing['median_income']/1.5)


# In[28]:
housing['income_cat'].hist()
'''as we can see in histogram after 5 and more the values are low so 
instead of having 1-10 category we will create >5 and <5 category.'''


# In[30]:
#categorizing the data
housing['income_cat'].value_counts()


# In[31]:
#where condition if true it keeps the original value ,if false then replace the original value with new value specified
#thus combining all values greater than 5 to 5.0
housing['income_cat'].where(housing['income_cat']<5,5.0,inplace=True)


# In[32]:
housing['income_cat'].value_counts()


# In[33]:
housing['income_cat'].hist()


# In[34]:
#not a random sampling now
from sklearn.model_selection import StratifiedShuffleSplit
#to split data based on income
#n_split=1 no of times the split is going to take place 
new_split=StratifiedShuffleSplit(n_splits=1,test_size=0.2)


# In[35]:
for train_index,test_index in new_split.split(housing,housing['income_cat']):
    s_train_set=housing.loc[train_index]
    s_test_set=housing.loc[test_index]


# In[36]:
s_train_set.hist()


# In[37]:
get_ipython().run_line_magic('pinfo2', 'StratifiedShuffleSplit')


# In[38]:
housing=s_train_set.copy()  


# In[39]:
housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.1)


# In[40]:
housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.3,s=housing['population']/100,c='median_house_value',figsize=(19,15),cmap=plt.get_cmap('coolwarm'),colorbar=True)


# In[41]:
housing.corr()


# In[42]:

#to plot scatter plot for all the attributes mention creating a scatter matrix
from pandas.plotting import scatter_matrix
scatter_matrix(housing[['housing_median_age','total_rooms','median_income','median_house_value']],figsize=(12,8))


# In[43]:
#Processing Data to prepare it for training
housing=s_train_set.copy()


# In[44]:
housing.drop('median_house_value',axis=1,inplace=True)


# In[45]:
housing.tail()


# In[46]:
#separating output feature
housing_labels=s_train_set['median_house_value'].copy()


# In[47]:
housing.tail()


# In[48]:
#handle missing values
incomplete_columns=housing[housing.isnull().any(axis=1)]


# In[49]:
#delete the row which has null value but since inplace=False original dataset will remain as itis
incomplete_columns.dropna(subset=['total_bedrooms'])


# In[50]:
#dropping column since it is not important
incomplete_columns.drop('total_bedrooms',axis=1)


# In[51]:
#fill the empty row with median value
incomplete_columns['total_bedrooms'].fillna(housing['total_bedrooms'].median(),inplace=True)


# In[52]:
incomplete_columns


# In[53]:
#filling missing value for all the numeric columns not only one 
from sklearn.preprocessing import Imputer
get_ipython().run_line_magic('pinfo', 'Imputer')
fill=Imputer(strategy='median')


# In[54]:
housing.tail()
#the function imputer can be applied to only numeric columns so we have to remove ocean_proximity


# In[55]:
numeric_housing=housing.drop('ocean_proximity',axis=1)


# In[56]:
numeric_housing.tail()


# In[57]:
#filling missing value for all the numeric columns not only one 
from sklearn.preprocessing import Imputer
#?Imputer
fill=Imputer(strategy='median')
fill.fit(numeric_housing)
fill.statistics_


# In[58]:
X_train=fill.transform(numeric_housing)


# In[59]
#transform function return an np array thats why
transfromed_housing=pd.DataFrame(X_train,columns=numeric_housing.columns,index=list(housing.index.values))
fill.fit_transform(transfromed_housing)


# In[60]:
transfromed_housing.loc[incomplete_columns.index.values]


# In[61]:
#converting ocean values to numeric
housing_cat=housing[['ocean_proximity']]
housing_cat.tail()


# In[62]:
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le_encode=le.fit_transform(housing_cat)


# In[63]:
le_encode


# In[64]:
le.classes_


# In[65]:
#replacg each ocean,inland into separate attributes and using
#0or1 for presence or not
from sklearn.preprocessing import OrdinalEncoder
order=OrdinalEncoder()
ocean_en= order.fit_transform(housing_cat)


# In[66]:
ocean_en[:10]


# In[67]:
order.categories_


# In[68]:
from sklearn.preprocessing import OneHotEncoder
onehat = OneHotEncoder(sparse=False)
newattri=onehat.fit_transform(housing_cat)


# In[69]:


newattri


# In[70]:


onehat.categories_


# In[71]:
#custom encoder in order to perform all the transformation like
#missingvalues chartonumeric in one 
from sklearn.base import BaseEstimator,TransformerMixin


# In[72]:
#initializing columns index
rooms_i,bed_i,population_i,household_i=3,4,5,6


# In[87]:
#writing our own transformer like fit_transform method
class CustomTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True):
        self.add_bedrooms_per_room=add_bedrooms_per_room
        
    def fit(self,x,y=None):
        return self
    
    def transform(self , x , y=None):
        rooms_per_household = x[:,rooms_i]/x[:,household_i]
        population_per_household = x[:,population_i]/x[:,household_i]
        if self.add_bedrooms_per_room:
            bedrooms_per_household = x[:,bed_i]/x[:,household_i]
            return np.c_[x,rooms_per_household,population_per_household,bedrooms_per_household]
        else:
            return np.c_[x,rooms_per_household,population_per_household]
        

cust_fea=CustomTransformer(add_bedrooms_per_room=False)
housing_with_custom_features=cust_fea.transform(housing.values)


# In[88]:
housing_with_custom_features


# In[91]:
housing_custom_df=pd.DataFrame(housing_with_custom_features,
columns=list(housing.columns)+['rooms_per_household','population_per_household'])


# In[93]:


housing_custom_df.head()


# In[94]:


#buildng a numerical pipeline to have easy transformartion  of numeric data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[102]:


num_pipeline=Pipeline([
    ('selector',DataAtrributeSelector(num_attributes)),
    ('imputer',Imputer(strategy='median')),
    ('custom_transform',CustomTransformer()),
    ('std_scaler',StandardScaler())
])

cat_pipeline=Pipeline([
    ('selector',DataAtrributeSelector(num_attributes)),
    ('one_hot_encoder',OneHOtEncoder(sparse=False)),
    
])


housing_num = num_pipeline.fit_transform(numeric_housing)


# In[111]:


class DataAtrributeSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names=attribute_names
    
    def fit(self , x , y=None):
        return self
    
    def transform(self,x,y=None):
        return x[self.attribute_names].values


# In[112]:
num_attributes=list(numeric_housing)


# In[113]:
num_attriutes


# In[114]:
cat_attributes = ['ocean_proximity']


# In[119]:
num_pipeline=Pipeline([
    ('selector',DataAtrributeSelector(num_attributes)),
    ('imputer',Imputer(strategy='median')),
    ('custom_transform',CustomTransformer()),
    ('std_scaler',StandardScaler())
])

cat_pipeline=Pipeline([
    ('selector',DataAtrributeSelector(cat_attributes)),
    ('one_hot_encoder',OneHotEncoder(sparse=False)),
    
])


# In[120]:
from sklearn.pipeline import FeatureUnion


# In[121]:
housing_pipeline = FeatureUnion(transformer_list=[('num_pipeline',num_pipeline),('cat_pipeline',cat_pipeline),
    
])


# In[122]:
housing_processed=housing_pipeline.fit_transform(housing)


# In[125]:
housing_processed


# In[126]:
housing_processed.shape


# In[127]:
housing.shape


# In[129]:
from sklearn.linear_model import LinearRegression
li=LinearRegression()
li.fit(housing_processed,housing_labels)


# In[130]:
sample_data=housing.iloc[:10]
sample_labels=housing_labels[:10]


# In[131]:
sample_processed_data=housing_pipeline.transform(sample_data)


# In[132]:
li.predict(sample_processed_data)


# In[133]:
list(sample_labels)


# In[134]:
#since we are not getting correct predictions we are going to minimize mse
from sklearn.metrics import mean_squared_error


# In[135]:
housing_predictions=li.predict(housing_processed)


# In[136]:
lin_mse=mean_squared_error(housing_labels,housing_predictions)


# In[137]:
lin_rmse=np.sqrt(lin_mse)


# In[138]:
#average error while making prediction
lin_rmse


# In[142]:
#trying other algorithm whicch can perform better
#lets try decisiontree
from sklearn.tree import DecisionTreeRegressor
dtree_reg = DecisionTreeRegressor()
dtree_reg.fit(housing_processed,housing_labels)
    


# In[148]:
dtreepred=dtree_reg.predict(sample_processed_data)


# In[144]:
#as we can see now decision tree has performed well on our data
list(sample_labels)


# In[149]:
#also we can check mse
#yeahhhh no error
dtreepred=dtree_reg.predict(housing_processed)
dtree_mse=mean_squared_error(housing_labels,dtreepred)
dtree_rmse=np.sqrt(dtree_mse)
dtree_rmse


# In[151]:
from sklearn.model_selection import cross_val_score


# In[154]:
#di will divide our dataset into 10 diff subsets and it will train on one subset and use other values to find the prdeiction 
scores=cross_val_score(dtree_reg , housing_processed , housing_labels , scoring="neg_mean_squared_error", cv=10)


# In[155]:
scores


# In[156]:
dtree_rmse_scores=np.sqrt(-scores)


# In[160]:
dtree_rmse_scores
dtree_rmse_scores.mean()


# In[158]:
scores=cross_val_score(li, housing_processed , housing_labels , scoring="neg_mean_squared_error", cv=10)
li_scores=np.sqrt(-scores)
li_scores


# In[159]:
#linear mean is less it is much better than dtree because for the new value it can predict better
li_scores.mean()


# In[161]:
from sklearn.ensemble import RandomForestRegressor
red_forest=RandomForestRegressor()
red_forest.fit(housing_processed,housing_labels)


# In[162]:
#this was the error made in predictions on the same training data
housing_pred= red_forest.predict(housing_processed)
forest_mse=mean_squared_error(housing_labels,housing_pred)
forest_rmse=np.sqrt(forest_mse)
forest_rmse


# In[163]:
scores=cross_val_score(red_forest, housing_processed , housing_labels , scoring="neg_mean_squared_error", cv=10)
forest_scores=np.sqrt(-scores)
forest_scores


# In[165]:
#thus mean is less thats why random forest is far more better model than li and dtree
forest_scores.mean()


# In[166]:
get_ipython().run_line_magic('pinfo2', 'DecisionTreeRegressor')


# In[ ]:




