#!/usr/bin/env python
# coding: utf-8

# ![image-2.png](attachment:image-2.png)

# ### **1-Read the data**

# In[80]:


import numpy
import pandas as pd
train_data = pd.read_csv("C:/Users/Effat/Desktop/Internships/CODSOFT/Fraud_Detection/fraudTrain.csv")


# In[81]:


train_data.head()


# In[82]:


train_data.info()


# In[83]:


train_data.columns


# ### **2- Exploratory Data Analysis (EDA)**

# In[84]:


train_data.isnull().sum()


# In[85]:


train_data.describe().T


# **Target Variable Distribution**

# In[86]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot distribution of the target variable
sns.countplot(x='is_fraud', data=train_data)
plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions')
plt.xlabel('Is Fraud')
plt.ylabel('Count')
plt.show()

# Display percentage of fraud
fraud_percentage = train_data['is_fraud'].value_counts(normalize=True) * 100
print("Fraudulent transactions percentage:\n", fraud_percentage)


# In[87]:


sns.histplot(train_data['amt'], bins=50, kde=True)
plt.title('Transaction Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()


# In[88]:


# Dividing train dataset into 2 sections - fraudulent and non-fraudulent

fraud = train_data[train_data.is_fraud == 1]
not_fraud = train_data[train_data.is_fraud == 0]


# ### By Category

# In[89]:


# Creating a 1x2 grid for subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.set_theme()

# Getting the counts of fraudulent and non-fraudulent activities per category
cat_fraud = fraud.category.value_counts().reset_index()
cat_fraud.columns = ["Category", "Counts"]
cat_not_fraud = not_fraud.category.value_counts().reset_index()
cat_not_fraud.columns = ["Category", "Counts"]

# Plotting the number of fraudulent and non-fraudulent transactions per category
sns.barplot(x="Category", y="Counts", data=cat_fraud, ax=axes[0])
axes[0].set_title("Number of fraudulent transactions per category")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Number of transactions")
axes[0].tick_params(axis="x", rotation=90)

sns.barplot(x="Category", y="Counts", data=cat_not_fraud, ax=axes[1])
axes[1].set_title("Number of non-fraudulent transactions per category")
axes[1].set_xlabel("Category")
axes[1].set_ylabel("Number of transactions")
axes[1].tick_params(axis="x", rotation=90)

plt.tight_layout()


# ### By Gender

# In[90]:


# Creating a 1x2 grid for subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.set_theme()

# Getting the counts of fraudulent and non-fraudulent activities per gender
g_fraud = fraud.gender.value_counts().reset_index()
g_fraud.columns = ["Gender", "Counts"]
g_not_fraud = not_fraud.gender.value_counts().reset_index()
g_not_fraud.columns = ["Gender", "Counts"]

# Plotting the number of fraudulent and non-fraudulent transactions per gender
sns.barplot(x="Gender", y="Counts", data=g_fraud, ax=axes[0])
axes[0].set_title("Number of fraudulent transactions per gender")
axes[0].set_xlabel("Gender")
axes[0].set_ylabel("Number of transactions")
axes[0].bar_label(axes[0].containers[0])

sns.barplot(x="Gender", y="Counts", data=g_not_fraud, ax=axes[1])
axes[1].set_title("Number of non-fraudulent transactions per gender")
axes[1].set_xlabel("Gender")
axes[1].set_ylabel("Number of transactions")
axes[1].bar_label(axes[1].containers[0])

plt.tight_layout()


# **We notice that there is no significant differences between the number of fraud victims with respect to gender. Women are involved in more transactions than men - 709863 transactions for the former compared to 586812 for the latter. Hence, around 0.64% of transactions involving men are fraudulent compared to 0.53% for women.**

# ### **3-Preprocessing**

# In[91]:


train_data.drop(columns=['Unnamed: 0','cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'],inplace=True)


# In[92]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train_data["merchant"] = encoder.fit_transform(train_data["merchant"])
train_data["category"] = encoder.fit_transform(train_data["category"])
train_data["gender"] = encoder.fit_transform(train_data["gender"])
train_data["job"] = encoder.fit_transform(train_data["job"])


# In[93]:


train_data


# ### **4-Train the Model**

# In[94]:


X = train_data.drop(columns=["is_fraud"], inplace = False)
Y = train_data["is_fraud"]


# In[95]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[96]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier


# In[97]:


model1 = DecisionTreeClassifier()


# In[98]:


model1.fit(X_train, y_train)
y_pred1_1 = model1.predict(X_test)

accuracy = accuracy_score(y_test, y_pred1_1)
precision = precision_score(y_test, y_pred1_1)
recall = recall_score(y_test, y_pred1_1)
f1 = f1_score(y_test, y_pred1_1)
print(f"\n Accuracy: {accuracy}")
print(f" Precision: {precision}")
print(f" Recall: {recall}")
print(f" F1 Score: {f1}")


# # **5-Undersampling**

# In[99]:


normal = train_data[train_data['is_fraud']==0]
fraud = train_data[train_data['is_fraud']==1]


# In[100]:


normal_sample = normal.sample(n=fraud.shape[0])


# In[101]:


new_data = pd.concat([normal_sample,fraud], ignore_index=True)


# In[102]:


new_data.head()


# In[103]:


new_data['is_fraud'].value_counts()
X = new_data.drop('is_fraud', axis = 1)
y= new_data['is_fraud']


# In[104]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[105]:


model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)
y_pred1_2 = model2.predict(X_test)
print(f"\n Accuaracy: {accuracy_score(y_test, y_pred1_2)}")
print(f"\n Precision: {precision_score(y_test, y_pred1_2)}")
print(f"\n Recall: {recall_score(y_test, y_pred1_2)}")
print(f"\n F1 Score: {f1_score(y_test, y_pred1_2)}")


# # **6-Ouversampling**

# In[106]:


X = train_data.drop('is_fraud', axis = 1)
y= train_data['is_fraud']


# In[107]:


from imblearn.over_sampling import SMOTE


# In[108]:


X_ouver, y_ouver = SMOTE().fit_resample(X,y)
y_ouver.value_counts()


# In[109]:


X_train, X_test, y_train, y_test = train_test_split(X_ouver, y_ouver, test_size = 0.2, random_state = 0)


# In[110]:


model3 = DecisionTreeClassifier()
model3.fit(X_train, y_train)
y_pred1_3 = model3.predict(X_test)
print(f"\n Accuaracy: {accuracy_score(y_test, y_pred1_3)}")
print(f"\n Precision: {precision_score(y_test, y_pred1_3)}")
print(f"\n Recall: {recall_score(y_test, y_pred1_3)}")
print(f"\n F1 Score: {f1_score(y_test, y_pred1_3)}")


# # **7-Lets Test the model**

# In[111]:


test_data = pd.read_csv("C:/Users/Effat/Desktop/Internships/CODSOFT/Fraud_Detection/fraudTest.csv")
test_data


# In[112]:


test_data.drop(columns=['Unnamed: 0','cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'],inplace=True)
encoder = LabelEncoder()
test_data["merchant"] = encoder.fit_transform(test_data["merchant"])
test_data["category"] = encoder.fit_transform(test_data["category"])
test_data["gender"] = encoder.fit_transform(test_data["gender"])
test_data["job"] = encoder.fit_transform(test_data["job"])


# In[113]:


test_data


# In[114]:


X_test = test_data.drop(columns=["is_fraud"], inplace = False)
Y_test = test_data["is_fraud"]


# In[115]:


y_pred1 = model1.predict(X_test)
print(f"\n Accuaracy: {accuracy_score(Y_test, y_pred1)}")
print(f"\n Precision: {precision_score(Y_test, y_pred1)}")
print(f"\n Recall: {recall_score(Y_test, y_pred1)}")
print(f"\n F1 Score: {f1_score(Y_test, y_pred1)}")


# In[116]:


y_pred2 = model2.predict(X_test)
print(f"\n Accuaracy: {accuracy_score(Y_test, y_pred2)}")
print(f"\n Precision: {precision_score(Y_test, y_pred2)}")
print(f"\n Recall: {recall_score(Y_test, y_pred2)}")
print(f"\n F1 Score: {f1_score(Y_test, y_pred2)}")


# In[117]:


y_pred3 = model3.predict(X_test)
print(f"\n Accuaracy: {accuracy_score(Y_test, y_pred3)}")
print(f"\n Precision: {precision_score(Y_test, y_pred3)}")
print(f"\n Recall: {recall_score(Y_test, y_pred3)}")
print(f"\n F1 Score: {f1_score(Y_test, y_pred3)}")


# # **8-Build the final model**

# In[118]:


import numpy as np
from scipy import stats
combined_predictions = np.vstack((y_pred1, y_pred2, y_pred3)).T
final_predictions = stats.mode(combined_predictions, axis=1)[0].flatten()


# In[119]:


print(f"\n Accuaracy: {accuracy_score(Y_test, final_predictions)}")
print(f"\n Precision: {precision_score(Y_test, final_predictions)}")
print(f"\n Recall: {recall_score(Y_test, final_predictions)}")
print(f"\n F1 Score: {f1_score(Y_test, final_predictions)}")

