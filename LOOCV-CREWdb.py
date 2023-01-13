#!/usr/bin/env python
# coding: utf-8

#  ### TESTING ML MODELS WITH LEAVE-ONE-OUT CROSS VALIDATION
# * Mollah Lab: CREWdb
# * Last Updated: 1/13/23

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns


# ### Data Pre-Processing

# In[7]:


#Reading in bulk data
data = pd.read_csv("./data/dataset_cleaned_397.csv").rename(columns={'REW (Convert to Numbers)': 'REW'})
data.head()


# In[8]:


data.replace("\#", np.nan)


# In[9]:


#Dropping columns 'ID' and 'SYMBOL' because they are not features
data.drop(['ID (REMOVE)', 'SYMBOL', 'HGNC approved name'], axis = 1, inplace = True)
data.replace('#', np.nan, inplace = True)


# In[10]:


#Checking where all data in the columns are NaN
to_check = data.columns[5:]
data[to_check].isna().all(1).sum()
data.dropna(how='all', subset = to_check, inplace = True)
print('Final Shape of Dataset:', data.shape)
data.replace(np.nan, '-', inplace = True)


# ### Categorical Encoding

# In[12]:


X = data.iloc[:, 1:]


# In[13]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', sparse = False)
X_encoded = encoder.fit_transform(X)
X_encoded.shape


# ### Class Rebalancing

# In[14]:


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from collections import Counter

y_df = data['REW']

X = X_encoded
y = y_df.values

y_ = LabelEncoder().fit_transform(y)
oversample = SMOTE(random_state=42)
X_smote, y_smote = oversample.fit_resample(X, y_)


# ### Functions for Tested Models

# In[ ]:


#Naive Bayes Model
from sklearn.model_selection import LeaveOneOut

loocv = LeaveOneOut()

model_cv =GaussianNB()

predicted_prob_dict_cv = dict()
auc_s_cv = []
accuracy_s_cv = []
f1_s_cv = []
ytest_cv = []
ypred_cv = []
    
#score_id
scores_id_cv = []
scores_cv = []
sample_ids_cv = np.array(data.index)

y_pred_proba_cv_list = []

for fold, (train_idx, test_idx) in enumerate(loocv.split(X_smote)):   
  X_train_cv = X_smote[train_idx]
  X_test_cv = X_smote[test_idx]

  anova_filter_cv = SelectKBest(f_classif, k= 8)
  var_filter_cv = VarianceThreshold(threshold= .8 * (1 - .8))
  anova_pipeline_cv = make_pipeline(var_filter_cv, anova_filter_cv, model)
  anova_pipeline_cv.fit(X_train_cv, y_smote[train_idx])
  y_pred_prob_cv = anova_pipeline_cv.predict_proba(X_test_cv)
  y_pred_cv = anova_pipeline.predict(X_test_cv)
  
  
  #auc_cv = roc_auc_score(y_smote[test_idx], y_pred_prob_cv.flatten(), multi_class='ovr')
  y_pred_proba_cv_list.append(y_pred_prob_cv.flatten())
  #auc_s_cv.append(auc_cv)
  accuracy_s_cv.append(accuracy_score(y_smote[test_idx], y_pred_cv))
  f1_s_cv.append(f1_score(y_smote[test_idx], y_pred_cv, average="weighted"))
  scores_id_cv.append(test_idx)
  scores_cv.append(y_pred_prob_cv)


scores_id_cv = np.concatenate(scores_id_cv, axis=0) 
scores_cv = np.concatenate(scores_cv, axis=0) 
map_ids = []
for i in range(len(sample_ids_cv)):
  map_ids.append(np.where(scores_id_cv==i)[0][0])
map_scores = scores_cv[map_ids]
y_predict_cv = np.argmax(map_scores, axis=1)

confusion_mat = confusion_matrix(y_, y_predict_cv)

sns.set(style="white")
labs=['Eraser','Writer', 'Reader']
# Generate a large random dataset
plt.figure(figsize=(4, 4))
sns.set(font_scale=1.2)
ax = sns.heatmap(confusion_mat, annot=True, square=True, xticklabels=labs, yticklabels=labs, cmap='RdPu', fmt="d",cbar_kws={"shrink": 0.85})
ax.xaxis.tick_top()
ax.yaxis.tick_left()
ax.set_xticklabels(labs, fontsize=14)
ax.set_yticklabels(labs, fontsize=14)
plt.xlabel('Naive Bayes', fontsize=14, labelpad=11)
plt.yticks(rotation=0) 
plt.yticks(rotation=0) 

plt.savefig("nb_loocv", format="png",bbox_inches='tight', dpi=600)

accuracy_cv = np.mean(accuracy_s_cv)
f1_cv = np.mean(f1_s_cv)
auc_Score_cv = roc_auc_score(y_smote, y_pred_proba_cv_list, multi_class='ovr')
predicted_prob_dict_cv[fold] = y_pred_prob_cv


# In[ ]:


print("Accuracy for Naive Bayes" , accuracy_cv)
print("F1-score for Naive Bayes" , f1_cv)
print("AUC score for Naive Bayes" , auc_Score_cv)


# In[ ]:


#ANN Model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut

    
auc_s_loocv = []
accuracy_s_loocv = []
f1_s_loocv = []
    
    #score_id
scores_id_loocv = []
scores_loocv = []
sample_ids_loocv = np.array(data.index)
            
loocv = LeaveOneOut()

clf = MLPClassifier(hidden_layer_sizes = (100,100,), random_state=0, max_iter=5000)
    
for fold, (train_idx, test_idx) in enumerate(loocv.split(X_smote)):
  X_train = X_smote[train_idx]
  X_test = X_smote[test_idx]
  anova_filter = SelectKBest(f_classif, k= 9)
  var_filter = VarianceThreshold(threshold= .8 * (1 - .8))
  anova_knn = make_pipeline(var_filter, anova_filter, clf)
  anova_knn.fit(X_train, y_smote[train_idx])
  y_pred_prob = anova_knn.predict_proba(X_test)
  y_pred = anova_knn.predict(X_test)

  auc_s_loocv.append(y_pred_prob.flatten())
  accuracy_s_loocv.append(accuracy_score(y_smote[test_idx], y_pred))
  f1_s_loocv.append(f1_score(y_smote[test_idx], y_pred, average='weighted'))      
  scores_id_loocv.append(test_idx)
  scores_loocv.append(y_pred_prob)
        
scores_id_loocv = np.concatenate(scores_id_loocv, axis=0) 
scores_loocv = np.concatenate(scores_loocv, axis=0) 
map_ids_loocv = []
for i in range(len(sample_ids_loocv)):
    map_ids_loocv.append(np.where(scores_id_loocv==i)[0][0])
    map_scores_loocv = scores_loocv[map_ids_loocv]
y_predict_loocv = np.argmax(map_scores_loocv, axis=1)
    
confusion_mat = confusion_matrix(y_, y_predict_loocv)
sns.set(style="white")
labs=['Eraser','Writer', 'Reader']
    # Generate a large random dataset
plt.figure(figsize=(4, 4))
sns.set(font_scale=1.2)
ax = sns.heatmap(confusion_mat, annot=True, square=True, xticklabels=labs, yticklabels=labs, cmap='RdPu', fmt="d",cbar_kws={"shrink": 0.85})
ax.xaxis.tick_top()
ax.yaxis.tick_left()
ax.set_xticklabels(labs, fontsize=14)
ax.set_yticklabels(labs, fontsize=14)
plt.xlabel('Deep Neural Net', fontsize=14, labelpad=11)


plt.yticks(rotation=0) 
plt.xticks(rotation=0) 
plt.savefig("dnn_loocv", format="png",bbox_inches='tight', dpi=600)
plt.show()
            
#avg_score = np.mean(auc_s)
accuracy_ann = np.mean(accuracy_s_loocv)
f1_ann = np.mean(f1_s_loocv)
auc_ann = roc_auc_score(y_smote, auc_s_loocv, multi_class='ovr')


# In[ ]:


print("Accuracy for ANN" , accuracy_ann)
print("F1-score for ANN" , f1_ann)
print("AUC score for ANN" ,  roc_auc_score(y_smote, auc_s_loocv, multi_class='ovr'))


# In[14]:


#KNN Model
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt


def knn_kfolds(X, y, n_folds, n_neighbors, new_data, old_y, random_state=None):
    
    auc_s = []
    accuracy_s = []
    f1_s = []
    ytest = []
    ypred = []
    
    scores_id = []
    scores = []
    sample_ids = np.array(new_data.index)
    
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state = random_state)
    
    model = KNeighborsClassifier(n_neighbors= n_neighbors)
    
    loocv = LeaveOneOut()
    y_pred_proba_cv_list = []
    
    
    for fold, (train_idx, test_idx) in enumerate(loocv.split(X,y)):
        
        scaler = preprocessing.StandardScaler().fit(X)
        X_train_scaled = scaler.transform(X[train_idx])
        X_test_scaled = scaler.transform(X[test_idx])
        
        anova_filter = SelectKBest(f_classif, k= 9)
        var_filter = VarianceThreshold(threshold= .8 * (1 - .8))
        anova_knn = make_pipeline(var_filter, anova_filter, model)
        anova_knn.fit(X_train_scaled, y[train_idx])
        y_pred_prob = anova_knn.predict_proba(X_test_scaled)
        y_pred = anova_knn.predict(X_test_scaled)
        y_pred_proba_cv_list.append(y_pred_prob.flatten())
        
        
        accuracy_s.append(accuracy_score(y[test_idx], y_pred))
        f1_s.append(f1_score(y[test_idx], y_pred, average="weighted"))
        scores_id.append(test_idx)
        scores.append(y_pred_prob)
        
    scores_id = np.concatenate(scores_id, axis=0) 
    scores = np.concatenate(scores, axis=0) 
    map_ids = []
    for i in range(len(sample_ids)):
        map_ids.append(np.where(scores_id==i)[0][0])
    map_scores = scores[map_ids]
    y_predict = np.argmax(map_scores, axis=1)
    
    confusion_mat = confusion_matrix(old_y, y_predict)

    sns.set(style="white")
    labs=['Eraser','Writer', 'Reader']
    
    # Generate a large random dataset
    plt.figure(figsize=(4, 4))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(confusion_mat, annot=True, square=True, xticklabels=labs, yticklabels=labs, cmap='RdPu', fmt="d",cbar_kws={"shrink": 0.85})
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    ax.set_xticklabels(labs, fontsize=14)
    ax.set_yticklabels(labs, fontsize=14)
    plt.xlabel('K-Neighbors Classifier', fontsize=14, labelpad=11)

    plt.yticks(rotation=0) 
    plt.xticks(rotation=0) 
    plt.show()
    
    avg_score = roc_auc_score(y_smote, y_pred_proba_cv_list, multi_class='ovr')
    accuracy = np.mean(accuracy_s)
    f1 = np.mean(f1_s)

    return [avg_score, accuracy, f1, confusion_mat]


# In[15]:


#Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

def decision_tree(X, y, n_folds, max_depth, new_data, old_y, random_state=None):
    
    auc_s = [] 
    accuracy_s = []
    f1_s = []
    
    #score_id
    scores_id = []
    scores = []
    sample_ids = np.array(new_data.index)
            
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state = random_state)

    model = DecisionTreeClassifier(max_depth= max_depth)
    
    loocv = LeaveOneOut()
    y_pred_proba_cv_list = []
    
    for fold, (train_idx, test_idx) in enumerate(loocv.split(X,y)):
        
        scaler = preprocessing.StandardScaler().fit(X)
        X_train_scaled = scaler.transform(X[train_idx])
        X_test_scaled = scaler.transform(X[test_idx])
        
 
        anova_filter = SelectKBest(f_classif, k= 9)
        var_filter = VarianceThreshold(threshold=(.8 * (1 - .8)))
        anova_knn = make_pipeline(anova_filter, model)
        anova_knn.fit(X_train_scaled, y[train_idx])
        y_pred_prob = anova_knn.predict_proba(X_test_scaled)
        y_pred = anova_knn.predict(X_test_scaled)
        
        y_pred_proba_cv_list.append(y_pred_prob.flatten())

        accuracy_s.append(accuracy_score(y[test_idx], y_pred))
        f1_s.append(f1_score(y[test_idx], y_pred, average='weighted'))
     
        scores_id.append(test_idx)
        scores.append(y_pred_prob)
        
    scores_id = np.concatenate(scores_id, axis=0) 
    scores = np.concatenate(scores, axis=0) 
    map_ids = []
    for i in range(len(sample_ids)):
        map_ids.append(np.where(scores_id==i)[0][0])
    map_scores = scores[map_ids]
    y_predict = np.argmax(map_scores, axis=1)
    
    confusion_mat = confusion_matrix(old_y, y_predict)

    sns.set(style="white")
    labs=['Eraser','Writer', 'Reader']
    
    # Generate a large random dataset
    plt.figure(figsize=(4, 4))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(confusion_mat, annot=True, square=True, xticklabels=labs, yticklabels=labs, cmap='RdPu', fmt="d",cbar_kws={"shrink": 0.85})
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    ax.set_xticklabels(labs, fontsize=14)
    ax.set_yticklabels(labs, fontsize=14)
    plt.xlabel('Decision Tree', fontsize=14, labelpad=11)

    plt.yticks(rotation=0) 
    plt.xticks(rotation=0) 
    plt.show()
        
    avg_score = roc_auc_score(y_smote, y_pred_proba_cv_list, multi_class='ovr')
    accuracy = np.mean(accuracy_s)
    f1 = np.mean(f1_s)

    return [avg_score, accuracy, f1, confusion_mat]


# In[16]:


#Random Forest Model
from sklearn.ensemble import RandomForestClassifier

def random_forest(X, y, n_folds, max_depth, new_data, old_y, random_state=None):
    
    auc_s = []
    accuracy_s = []
    f1_s = []
    
    #score_id
    scores_id = []
    scores = []
    sample_ids = np.array(new_data.index)
            
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state = random_state)

    model = RandomForestClassifier(max_depth= max_depth)
    
    loocv = LeaveOneOut()
    y_pred_proba_cv_list = []
    
    for fold, (train_idx, test_idx) in enumerate(loocv.split(X,y)):
        
        scaler = preprocessing.StandardScaler().fit(X)
        X_train_scaled = scaler.transform(X[train_idx])
        X_test_scaled = scaler.transform(X[test_idx])
        
 
        anova_filter = SelectKBest(f_classif, k= 9)
        var_filter = VarianceThreshold(threshold= .8 * (1 - .8))
        anova_knn = make_pipeline(var_filter, anova_filter, model)
        anova_knn.fit(X_train_scaled, y[train_idx])
        y_pred_prob = anova_knn.predict_proba(X_test_scaled)
        y_pred = anova_knn.predict(X_test_scaled)

        y_pred_proba_cv_list.append(y_pred_prob.flatten())

        accuracy_s.append(accuracy_score(y[test_idx], y_pred))
        f1_s.append(f1_score(y[test_idx], y_pred, average='weighted'))
        
        
        scores_id.append(test_idx)
        scores.append(y_pred_prob)
        
    scores_id = np.concatenate(scores_id, axis=0) 
    scores = np.concatenate(scores, axis=0) 
    map_ids = []
    for i in range(len(sample_ids)):
        map_ids.append(np.where(scores_id==i)[0][0])
    map_scores = scores[map_ids]
    y_predict = np.argmax(map_scores, axis=1)
    
    confusion_mat = confusion_matrix(old_y, y_predict)

    sns.set(style="white")
    labs=['Eraser','Writer', 'Reader']
    
    # Generate a large random dataset
    plt.figure(figsize=(4, 4))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(confusion_mat, annot=True, square=True, xticklabels=labs, yticklabels=labs, cmap='RdPu', fmt="d",cbar_kws={"shrink": 0.85})
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    ax.set_xticklabels(labs, fontsize=14)
    ax.set_yticklabels(labs, fontsize=14)
    plt.xlabel('Random Forest', fontsize=14, labelpad=11)

    plt.yticks(rotation=0) 
    plt.xticks(rotation=0) 
    plt.show()
        
    avg_score = roc_auc_score(y_smote, y_pred_proba_cv_list, multi_class='ovr')
    accuracy = np.mean(accuracy_s)
    f1 = np.mean(f1_s)

            
    return [avg_score, accuracy, f1, confusion_mat]


# In[17]:


#SVM Model 

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def svm_f(X, y, n_folds, new_data, old_y, random_state=None):
    
    auc_s = []
    accuracy_s = []
    f1_s = []
    
    #score_id
    scores_id = []
    scores = []
    sample_ids = np.array(new_data.index)
    
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state = random_state)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    
    loocv = LeaveOneOut()
    y_pred_proba_cv_list = []
    
    for fold, (train_idx, test_idx) in enumerate(loocv.split(X,y)):
        
        scaler = preprocessing.StandardScaler().fit(X)
        X_train_scaled = scaler.transform(X[train_idx])
        X_test_scaled = scaler.transform(X[test_idx])
        
 
        anova_filter = SelectKBest(f_classif, k= 9)
        var_filter = VarianceThreshold(threshold= .8 * (1 - .8))
        anova_knn = make_pipeline(var_filter, anova_filter, clf)
        anova_knn.fit(X_train_scaled, y[train_idx])
        y_pred_prob = anova_knn.predict_proba(X_test_scaled)
        y_pred = anova_knn.predict(X_test_scaled)
        
        y_pred_proba_cv_list.append(y_pred_prob.flatten())

        accuracy_s.append(accuracy_score(y[test_idx], y_pred))
        f1_s.append(f1_score(y[test_idx], y_pred, average='weighted'))
        
        scores_id.append(test_idx)
        scores.append(y_pred_prob)
        
    scores_id = np.concatenate(scores_id, axis=0) 
    scores = np.concatenate(scores, axis=0) 
    map_ids = []
    for i in range(len(sample_ids)):
        map_ids.append(np.where(scores_id==i)[0][0])
    map_scores = scores[map_ids]
    y_predict = np.argmax(map_scores, axis=1)
    
    confusion_mat = confusion_matrix(old_y, y_predict)

    sns.set(style="white")
    labs=['Eraser','Writer', 'Reader']
    # Generate a large random dataset
    plt.figure(figsize=(4, 4))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(confusion_mat, annot=True, square=True, xticklabels=labs, yticklabels=labs, cmap='RdPu', fmt="d",cbar_kws={"shrink": 0.85})
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    ax.set_xticklabels(labs, fontsize=14)
    ax.set_yticklabels(labs, fontsize=14)
    plt.xlabel('Support Vector Machine', fontsize=14, labelpad=11)

    plt.yticks(rotation=0) 
    plt.xticks(rotation=0) 
    plt.show()
            
    avg_score = roc_auc_score(y_smote, y_pred_proba_cv_list, multi_class='ovr')
    accuracy = np.mean(accuracy_s)
    f1 = np.mean(f1_s)

    return [avg_score, accuracy, f1, confusion_mat]


# In[18]:


#Feed-Forward Neural Network Model
from sklearn.neural_network import MLPClassifier

def fnn_f(X, y, n_folds, new_data, old_y, random_state=None):
    
    auc_s = []
    accuracy_s = []
    f1_s = []
    
    #score_id
    scores_id = []
    scores = []
    sample_ids = np.array(new_data.index)
            
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state = random_state)

    clf = MLPClassifier(random_state=0, max_iter=5000)
    
    loocv = LeaveOneOut()
    y_pred_proba_cv_list = []
    
    for fold, (train_idx, test_idx) in enumerate(loocv.split(X,y)):
        
        scaler = preprocessing.StandardScaler().fit(X)
        X_train_scaled = scaler.transform(X[train_idx])
        X_test_scaled = scaler.transform(X[test_idx])
        
 
        anova_filter = SelectKBest(f_classif, k= 9)
        var_filter = VarianceThreshold(threshold= .8 * (1 - .8))
        anova_knn = make_pipeline(var_filter, anova_filter, clf)
        anova_knn.fit(X_train_scaled, y[train_idx])
        y_pred_prob = anova_knn.predict_proba(X_test_scaled)
        y_pred = anova_knn.predict(X_test_scaled)

        y_pred_proba_cv_list.append(y_pred_prob.flatten())

        accuracy_s.append(accuracy_score(y[test_idx], y_pred))
        f1_s.append(f1_score(y[test_idx], y_pred, average='weighted'))
        
        scores_id.append(test_idx)
        scores.append(y_pred_prob)
        
    scores_id = np.concatenate(scores_id, axis=0) 
    scores = np.concatenate(scores, axis=0) 
    map_ids = []
    for i in range(len(sample_ids)):
        map_ids.append(np.where(scores_id==i)[0][0])
    map_scores = scores[map_ids]
    y_predict = np.argmax(map_scores, axis=1)
    
    confusion_mat = confusion_matrix(old_y, y_predict)

    sns.set(style="white")
    labs=['Eraser','Writer', 'Reader']
    # Generate a large random dataset
    plt.figure(figsize=(4, 4))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(confusion_mat, annot=True, square=True, xticklabels=labs, yticklabels=labs, cmap='RdPu', fmt="d",cbar_kws={"shrink": 0.85})
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    ax.set_xticklabels(labs, fontsize=14)
    ax.set_yticklabels(labs, fontsize=14)
    plt.xlabel('Feed-Forward Neural Network', fontsize=14, labelpad=11)

    plt.yticks(rotation=0) 
    plt.xticks(rotation=0) 
    plt.show()
            
    avg_score = roc_auc_score(y_smote, y_pred_proba_cv_list, multi_class='ovr')
    accuracy = np.mean(accuracy_s)
    f1 = np.mean(f1_s)

    return [avg_score, accuracy, f1, confusion_mat]


# In[19]:


#Logistic Regression Model
from sklearn.linear_model import LogisticRegression

def lr(X, y, n_folds, new_data, old_y, random_state=None):
    
    auc_s = []
    accuracy_s = []
    f1_s = []
            
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state = random_state)

    clf = LogisticRegression(penalty ='l2')
    
    #score_id
    scores_id = []
    scores = []
    sample_ids = np.array(new_data.index)
    
    loocv = LeaveOneOut()
    y_pred_proba_cv_list = []
    
    for fold, (train_idx, test_idx) in enumerate(loocv.split(X,y)):
        
        scaler = preprocessing.StandardScaler().fit(X)
        X_train_scaled = scaler.transform(X[train_idx])
        X_test_scaled = scaler.transform(X[test_idx])
        
 
        anova_filter = SelectKBest(f_classif, k= 9)
        var_filter = VarianceThreshold(threshold= .8 * (1 - .8))
        anova_knn = make_pipeline(var_filter, anova_filter, clf)
        anova_knn.fit(X_train_scaled, y[train_idx])
        y_pred_prob = anova_knn.predict_proba(X_test_scaled)
        y_pred = anova_knn.predict(X_test_scaled)

        y_pred_proba_cv_list.append(y_pred_prob.flatten())

        accuracy_s.append(accuracy_score(y[test_idx], y_pred))
        f1_s.append(f1_score(y[test_idx], y_pred, average='weighted'))
        
        scores_id.append(test_idx)
        scores.append(y_pred_prob)
        
    scores_id = np.concatenate(scores_id, axis=0) 
    scores = np.concatenate(scores, axis=0) 
    map_ids = []
    for i in range(len(sample_ids)):
        map_ids.append(np.where(scores_id==i)[0][0])
    map_scores = scores[map_ids]
    y_predict = np.argmax(map_scores, axis=1)
    
    confusion_mat = confusion_matrix(old_y, y_predict)

    sns.set(style="white")
    labs=['Eraser','Writer', 'Reader']
    # Generate a large random dataset
    plt.figure(figsize=(4, 4))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(confusion_mat, annot=True, square=True, xticklabels=labs, yticklabels=labs, cmap='RdPu', fmt="d",cbar_kws={"shrink": 0.85})
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    ax.set_xticklabels(labs, fontsize=14)
    ax.set_yticklabels(labs, fontsize=14)
    plt.xlabel('Logistic Regression', fontsize=14, labelpad=11)

    plt.yticks(rotation=0) 
    plt.xticks(rotation=0) 
    plt.show()
            
    avg_score = roc_auc_score(y_smote, y_pred_proba_cv_list, multi_class='ovr')
    accuracy = np.mean(accuracy_s)
    f1 = np.mean(f1_s)

            
    return [avg_score, accuracy, f1, confusion_mat]


# ### Model Evaluation

# In[ ]:


#Running the ML models on the generated dataset

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

results = {}
auc = []
accuracy = []
f1 = []
confusion_matrices = []

knn_values = []
dt_values = []
rf_values = []

k = range(1,10)
for i in k:
    knn = knn_kfolds(X_smote, y_smote, 160, i, data, y_, random_state=5)
    dt = decision_tree(X_smote, y_smote, 160, i, data, y_, random_state=5)
    rf = random_forest(X_smote, y_smote, 160, i, data, y_, random_state=5) 

    knn_values.append(knn[0])
    dt_values.append(dt[0])
    rf_values.append(rf[0])

svm = svm_f(X_smote, y_smote, 160, data, y_, random_state=5)
fnn = fnn_f(X_smote, y_smote, 160, data, y_, random_state=5)
lr_s = lr(X_smote, y_smote, 160, data, y_, random_state=5)

auc.append(max(knn_values))
auc.append(max(rf_values))
auc.append(max(dt_values))
auc.append(fnn[0])
auc.append(lr_s[0])
auc.append(svm[0])

accuracy.append(knn[1])
accuracy.append(dt[1])
accuracy.append(rf[1])
accuracy.append(fnn[1])
accuracy.append(lr_s[1])
accuracy.append(svm[1])

f1.append(knn[2])
f1.append(dt[2])
f1.append(rf[2])
f1.append(fnn[2])
f1.append(lr_s[2])
f1.append(svm[2])


confusion_matrices.append(knn[3])
confusion_matrices.append(dt[3])
confusion_matrices.append(rf[3])
confusion_matrices.append(fnn[3])
confusion_matrices.append(lr_s[3])
confusion_matrices.append(svm[3])

results['auc'] = auc
#results['accuracy'] = accuracy
results['f1'] = f1
results['accuracy'] = accuracy
results['confusion_matrix'] = confusion_matrices


# ### Plotting Confusion Matrices

# In[ ]:


#Plotting the generated confusion matrices from cell above

sns.set(style="white")
labs=['Eraser','Writer', 'Reader']
models = ["K-Neighbors Classifier", "Decision Tree", "Random Forest", "Support Vector Machine", "Feed-Forward Neural Network", "Logistic Regression"]
i = 0
img_file ="loocv"
for m in results['confusion_matrix']:
    plt.figure(figsize=(4, 4))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(m, annot=True, square=True, xticklabels=labs, yticklabels=labs, cmap='RdPu', fmt="d",cbar_kws={"shrink": 0.85})
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    ax.set_xticklabels(labs, fontsize=14)
    ax.set_yticklabels(labs, fontsize=14)
    plt.xlabel(models[i], fontsize=14, labelpad=11)

    plt.yticks(rotation=0) 
    plt.xticks(rotation=0) 
    plt.savefig(img_file + str(i), format="png",bbox_inches='tight', dpi=600)
    i = i + 1

