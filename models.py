# -*- coding: utf-8 -*-
"""
Created on Wed May  7 16:46:55 2025

@author: borlu
"""
# Dataset: Bank Marketing Dataset from UCI Machine Learning Repository
# URL: https://archive.ics.uci.edu/dataset/222/bank+marketing

#import modules
import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Exploratory Data Analysis

#import the dataset to a pandas dataframe 
data = pd.read_csv('bank-additional-full.csv',sep=';')
#print(data.columns)

#numerical data 
describe_data = data.describe()
data_types = data.dtypes

"""
sns.boxplot(x='y', y='duration', data=data)
plt.title("Duration Distribution by y")
plt.show()
"""

"""
sns.boxplot(x='y', y='emp.var.rate', data=data)
plt.title("Employment Variation Rate vs Campaign Response (y)")
plt.xlabel('Campaign Response')
plt.ylabel('Employment Variation Rate')
plt.show()
"""

"""
sns.boxplot(x='y', y='age', data=data)
plt.title("Age Distribution by y")
plt.show()
"""

#categorical data 

#print(min(data['age']))
#bins = np.linspace(min(data['age']), max(data['age']), 7)
#group_names = ['Young', 'Young Adult', 'Adult', 'Middle-Aged', 'Old', 'Very Old']
#data['binned_age'] = pd.cut(data['age'], bins, labels=group_names, include_lowest=True)

#sns.histplot(data['binned_age'])
#plt.title("Age Distribution")
#plt.xlabel("Age Group")
#plt.ylabel("Count")
#plt.show()
#data = data.drop(columns=['binned_age'])

"""
sns.histplot(data['day_of_week'])
plt.title("Days Distribution in Dataset")
plt.xlabel('Days')
plt.show()
"""

"""
sns.histplot(data['month'])
plt.title("Month Distribution in Dataset")
plt.xlabel('Months')
plt.show()
"""

vc_job = data['job'].value_counts().to_frame()
vc_marital = data['marital'].value_counts().to_frame()
vc_education = data['education'].value_counts().to_frame()
vc_telephone = data['contact'].value_counts().to_frame()
vc_poutcome = data['poutcome'].value_counts().to_frame()


"""no>>>yes"""
"""
sns.countplot(x='y', data=data)
plt.title("Target(y) Distribution")
plt.show()
"""

"""
sns.countplot(x='poutcome', hue='y', data=data)
plt.title("Previous Attendance vs. Attendance(yes/no)")
plt.xlabel('Previous Attendance')
plt.xticks(rotation=45)
plt.show()
"""

#Data Preprocessing

#get rid of unknown values with mode
data['job'] = data['job'].replace('unknown',data['job'].mode()[0])
#check values
print(data['job'].unique())

data['marital'] = data['marital'].replace('unknown', data['marital'].mode()[0])
print(data['marital'].unique())

data['education'] = data['education'].replace('unknown', data['education'].mode()[0])
print(data['education'].unique())

data['default'] = data['default'].replace('unknown', data['default'].mode()[0])
print(data['default'].unique())

data['housing'] = data['housing'].replace('unknown', data['housing'].mode()[0])
print(data['education'].unique())

data['loan'] = data['loan'].replace('unknown', data['loan'].mode()[0])
print(data['loan'].unique())


#dropping dublicate(yes/no)
ohe_data = pd.get_dummies(data).astype(int)
ohe_data = ohe_data.drop(columns=['default_no','housing_no',
                                  'loan_no','y_no'])

x = ohe_data.iloc[:,0:54]
y = ohe_data.iloc[:,54:]


#Under Sampling
# The following sampling approach is adapted from:
# Sahil Chachra, "Handling Imbalance Dataset", GitHub (2020)
# https://github.com/SahilChachra/Handling-Imbalanced-Dataset

def splittingData(X_rec,y_rec):
    X_train, X_test, y_train, y_test = train_test_split(X_rec, y_rec, test_size=0.3, random_state=0)
    return X_train,X_test,y_train,y_test

c_class_0, c_class_1 = ohe_data.y_yes.value_counts()

data_class_0 = ohe_data[ohe_data['y_yes']==0]
data_class_1 = ohe_data[ohe_data['y_yes']==1]

data_class_0_under = data_class_0.sample(c_class_1)
data_test_under = pd.concat([data_class_0_under, data_class_1], axis=0)

#Correlation 
correlation = ohe_data[['duration','pdays','previous','emp.var.rate','euribor3m',
                        'nr.employed','poutcome_success','poutcome_nonexistent','y_yes']].corr()
#HeatMap
#fig, ax = plt.subplots(figsize=(8, 5))
#axis_corr = sns.heatmap(
#correlation,
#vmin=-1, vmax=1, center=0,
#cmap='coolwarm',
#square=True
#)
#plt.show()

#Normalization
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Model Training
#splitting data into train and test
# Same technique as in Sahil Chachra's imbalance handling notebook (GitHub)
X_0_under = data_test_under.drop('y_yes', axis=1).values
Y_0_under = data_test_under['y_yes'].values
x_under_train, x_under_test, y_under_train, y_under_test = splittingData(X_0_under, Y_0_under)

#Random Forest
rf = RandomForestClassifier(criterion='entropy',class_weight='balanced',
                            min_samples_split=5, min_samples_leaf=2, random_state=0)

rf_model = rf.fit(x_under_train,y_under_train)
y_pred_rf = rf_model.predict(x_under_test)

# Model Evaluation
cm_rf = confusion_matrix(y_under_test, y_pred_rf)

RF_SCORES = {    
    'Precision': round(precision_score(y_under_test, y_pred_rf), 4),
    'Recall': round(recall_score(y_under_test, y_pred_rf), 4),
    'F1-Score': round(f1_score(y_under_test, y_pred_rf), 4),
    'Accuracy': round(accuracy_score(y_under_test, y_pred_rf), 4)
}

# ROC curve implementation adapted from:
# scikit-learn documentation, "Plotting ROC curves"
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
pos_label = 1  

# Precision-Recall Curve
PrecisionRecallDisplay.from_estimator(
    rf_model,  
    x_under_test,
    y_under_test,
    pos_label=pos_label,
    ax=axs[0],
    name="Random Forest"
)
axs[0].set_title("Precision-Recall Curve")
axs[0].legend()

# ROC Curve
RocCurveDisplay.from_estimator(
    rf_model,
    x_under_test,
    y_under_test,
    pos_label=pos_label,
    ax=axs[1],
    name="Random Forest",
    plot_chance_level=True,
)
axs[1].set_title("ROC Curve")
axs[1].legend()

fig.suptitle("Random Forest Model Evaluation", fontsize=14)
plt.tight_layout()
plt.show()


# Model Comparison & Insights

#1. SVM
svc = SVC(kernel='linear',class_weight='balanced',random_state=0)
svm_model = svc.fit(x_under_train, y_under_train)
y_pred_svm = svm_model.predict(x_under_test)

cm_svm = confusion_matrix(y_under_test, y_pred_svm)

SVM_SCORES = {
    'Precision': round(precision_score(y_under_test, y_pred_svm), 4),
    'Recall': round(recall_score(y_under_test, y_pred_svm), 4),
    'F1-Score': round(f1_score(y_under_test, y_pred_svm), 4),
    'Accuracy': round(accuracy_score(y_under_test, y_pred_svm), 4)
}

# Same approach as used earlier (see Scikit-learn ROC example)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
pos_label = 1  

# Precision-Recall Curve
PrecisionRecallDisplay.from_estimator(
    svm_model,  
    x_under_test,
    y_under_test,
    pos_label=pos_label,
    ax=axs[0],
    name="SVM"
)
axs[0].set_title("Precision-Recall Curve")
axs[0].legend()

# ROC Curve
RocCurveDisplay.from_estimator(
    svm_model,
    x_under_test,
    y_under_test,
    pos_label=pos_label,
    ax=axs[1],
    name="SVM",
    plot_chance_level=True,
)
axs[1].set_title("ROC Curve")
axs[1].legend()

fig.suptitle("SVM Model Evaluation", fontsize=14)
plt.tight_layout()
plt.show()

#2. KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn_model = knn.fit(x_under_train,y_under_train)
y_pred_knn = knn_model.predict(x_under_test)

cm_knn = confusion_matrix(y_under_test, y_pred_knn)

KNN_SCORES = {
    'Precision': round(precision_score(y_under_test, y_pred_knn), 4),
    'Recall': round(recall_score(y_under_test, y_pred_knn), 4),
    'F1-Score': round(f1_score(y_under_test, y_pred_knn), 4),
    'Accuracy': round(accuracy_score(y_under_test, y_pred_knn), 4)
}

# Same approach as used earlier (see Scikit-learn ROC example)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
pos_label = 1  

# Precision-Recall Curve
PrecisionRecallDisplay.from_estimator(
    knn_model,  
    x_under_test,
    y_under_test,
    pos_label=pos_label,
    ax=axs[0],
    name="KNN"
)
axs[0].set_title("Precision-Recall Curve")
axs[0].legend()

# ROC Curve
RocCurveDisplay.from_estimator(
    knn_model,
    x_under_test,
    y_under_test,
    pos_label=pos_label,
    ax=axs[1],
    name="KNN",
    plot_chance_level=True,
)
axs[1].set_title("ROC Curve")
axs[1].legend()

fig.suptitle("KNN Model Evaluation", fontsize=14)
plt.tight_layout()
plt.show()

#3. Decision Tree
tree = DecisionTreeClassifier(criterion = 'entropy', splitter = 'best',
                              max_leaf_nodes= 10, class_weight='balanced',random_state=0)
model_tree = tree.fit(x_under_train,y_under_train)
y_pred_tree = model_tree.predict(x_under_test)

cm_tree = confusion_matrix(y_under_test, y_pred_tree)

DT_SCORES = {
    'Precision': round(precision_score(y_under_test, y_pred_tree), 4),
    'Recall': round(recall_score(y_under_test, y_pred_tree), 4),
    'F1-Score': round(f1_score(y_under_test, y_pred_tree), 4),
    'Accuracy': round(accuracy_score(y_under_test, y_pred_tree), 4)
}

plt.figure()
plot_tree(model_tree)

# Same approach as used earlier (see Scikit-learn ROC example)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
pos_label = 1  

# Precision-Recall Curve
PrecisionRecallDisplay.from_estimator(
    model_tree,  
    x_under_test,
    y_under_test,
    pos_label=pos_label,
    ax=axs[0],
    name="Decision Tree"
)
axs[0].set_title("Precision-Recall Curve")
axs[0].legend()

# ROC Curve
RocCurveDisplay.from_estimator(
    model_tree,
    x_under_test,
    y_under_test,
    pos_label=pos_label,
    ax=axs[1],
    name="Decision Tree",
    plot_chance_level=True,
)
axs[1].set_title("ROC Curve")
axs[1].legend()

fig.suptitle("Decision Tree Model Evaluation", fontsize=14)
plt.tight_layout()
plt.show()

#4. Logistic Regression
model_lg = LogisticRegression(class_weight='balanced',random_state=0) 
model_lg = model_lg.fit(x_under_train, y_under_train)
y_pred_lg = model_lg.predict(x_under_test)

cm_lg = confusion_matrix(y_under_test, y_pred_lg)

LR_SCORES = {
    'Precision': round(precision_score(y_under_test, y_pred_lg), 4),
    'Recall': round(recall_score(y_under_test, y_pred_lg), 4),
    'F1-Score': round(f1_score(y_under_test, y_pred_lg), 4),
    'Accuracy': round(accuracy_score(y_under_test, y_pred_lg), 4)
}

# Same approach as used earlier (see Scikit-learn ROC example)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
pos_label = 1  

# Precision-Recall Curve
PrecisionRecallDisplay.from_estimator(
    model_lg,  
    x_under_test,
    y_under_test,
    pos_label=pos_label,
    ax=axs[0],
    name="Logistic Regression"
)
axs[0].set_title("Precision-Recall Curve")
axs[0].legend()

# ROC Curve
RocCurveDisplay.from_estimator(
    model_lg,
    x_under_test,
    y_under_test,
    pos_label=pos_label,
    ax=axs[1],
    name="Logistic Regression",
    plot_chance_level=True,
)
axs[1].set_title("ROC Curve")
axs[1].legend()

fig.suptitle("Logistic Regression Model Evaluation", fontsize=14)
plt.tight_layout()
plt.show()


#Model Comparison Bar Plots
ind = np.arange(len(KNN_SCORES.keys()))  
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

plt.bar(ind - 1.5*width, list(KNN_SCORES.values()), color='maroon', width=width, edgecolor="black", label="KNN")
plt.bar(ind - 0.5*width, list(DT_SCORES.values()), color='darkgreen', width=width, edgecolor='black', label='DT')
plt.bar(ind + 0.5*width, list(RF_SCORES.values()), color='darkblue', width=width, edgecolor='black', label='RF')
plt.bar(ind + 1.5*width, list(LR_SCORES.values()), color='purple', width=width, edgecolor='black', label='LR')

ax.set_xticks(ind)
ax.set_xticklabels(KNN_SCORES.keys(), rotation=45, fontsize=10)

ax.set_xlabel("Metrics")
ax.set_ylabel("Scores")
ax.set_title("Model Comparison", fontsize=14)


plt.legend(title="Models", loc="upper left")
plt.tight_layout()
plt.show()

















