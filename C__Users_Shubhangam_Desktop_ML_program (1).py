#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd


# In[31]:


India_Menu_path = 'C:/Users/Shubhangam/Desktop/ML/CSV files/India_Menu.csv'


# In[32]:


India_menu_data=pd.read_csv(India_Menu_path)


# In[33]:


India_menu_data.describe()


# In[95]:


India_menu_data.columns


# In[28]:


India_menu_data = India_menu_data.dropna(axis=0)


# In[36]:


y = India_menu_data.Price


# In[82]:


India_menu_features = ['Per_Serve_Size', 'Energy _(kCal)', 'Protein_(g)','Total_fat_(g)', 'Sat_Fat+(g)', 'Trans fat (g)',
       'Cholesterols (mg)', 'Total carbohydrate (g)', 'Total Sugars (g)']


# In[83]:


X=India_menu_data[India_menu_features]


# In[84]:


X.describe()


# In[85]:


X.head()


# In[86]:


from sklearn.tree import DecisionTreeRegressor


# In[97]:


Indian_Menu_model = DecisionTreeRegressor(random_state=5)


# In[96]:


Indian_Menu_model.fit(x,y)


# In[73]:


print("Making predictions for the following 5 Menues:")


# In[91]:


print(X.head())


# In[92]:


print("The predictions are")


# In[93]:


print(Indian_Menu_model.predict(x.head()))


# In[ ]:




