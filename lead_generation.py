#Import libraries 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt 
import seaborn as sns 

#load data in to pandas dataframe 
df = pd.read_csv(r'C:\Users\jahna\Downloads\Data_Science_Internship - Dump.csv')

#view the first 5 rows of dataframe
df.head()

#checking for duplicated values 
df.duplicated().sum()

#checking for missing values
df.isnull().sum()

#according to the assignment 
df.replace({'9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0': np.nan})

#checking again 
df.isnull().sum()

df = df[df.status != 'OPPORTUNITY']
df = df[df.status != 'CONTACTED']
df = df[df.status != 'PROCESSING']
df = df[df.status != 'IMPORTANT']

df['status'].value_counts()

print (100*df['status'].value_counts()/len(df['status']))

def Distribution(df,lists,cat_list,num_list):
    for i in lists:
        print(data[i].value_counts())
        for i in cat_lists:
            plt.figure()
            data[i].value_counts().plot(kind='bar',color='blue')
            plt.plot()

def VisualizeConvertedNumeric(df,col):
    dataframe = pd.concat([df[num_list], previous['status']], axis=1)
    plt.figure(figsize=(7,5))
    sns.distplot(dataframe.loc[dataframe.status == 'WON',col], hist=False, color='black',label='WON')
    sns.distplot(dataframe.loc[dataframe.status == 'LOST',col], hist=False, color='black',label='LOST')
    plt.legend()

plt.figure(figsize=(10,8))
sns.heatmap(df[custSoc].corr(method='pearson'),annot=True)
plt.show()

def Data_Cleaning(df):
    cat_list = []
    num_list = []
    for column in df:
        if is_string_dtype(df[column]):
            cat_list.append(column)
        if is_numeric_dtype(df[column]):
            num_list.append(column)
    return(cat_list,num_list)
cat_list,num_list = Data_Cleaning(data)

def One_Hot_Encoding(df, lists):
    ohe = OneHotEncoder(handle_unknown='error')
    ohc = ohe.fit_transform(df[lists]).toarray()
    ohfn = ohe.get_feature_names(lists)
    ohedf = pd.DataFrame(ohc, columns=ohfn).astype(int).reset_index(drop=True)
    df_transformed = pd.concat([df,ohedf], axis=1).reset_index(drop=True)
    df = df_transformed.drop(labels=lists, axis=1)
    return(df)

df.info()

corr = data_encoded1.corr(method='pearson')
corr.unstack().drop_duplicates().sort_values(ascending = False).reset_index()[:20]

corr.unstack().drop_duplicates().sort_values(ascending=False).reset_index()[-15:]

cols = list(X.columns)
model = LogisticRegression()

#rfe
rfe = RFE(estimator = model, n_features_to_select=15)
X_rfe = rfe.fit_transform(X,Y)

#fitting data to model 
model.fit(X_rfe,Y)
temp = pd.Series(rfe.support_,index=cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)

def SMOTEOversampling(X_train12, X_test12, y_train12, y_test12):
    print(sorted(Counter(y_train12).items()))
    X_train12=np.array(X_train12)
    y_train12=np.array(y_train12)
    smote = SMOTE()
    Xc12, yc12 = smote.fit_resample(X_train12,y_train12)
    print(sorted(Counter(yc12).items()))
    return(Xc12,yc12,X_test12,y_test12)

param_grid=[
    {'penalty':['l1','l2','elasticnet','none'],
     'C':np.logspace(-1,0.01,1,10,100),
     'solver':['lbfgs','newton-cg','liblinear','sag','saga'],
     'max_iter':[100,500,1000]}
]
logModel = LogisticRegression()
clf = GridSearchCV (logModel, param_grid=param_grid,cv=3,verbose=True,n_jobs=-1)
best_clf=clf.fit(X_train, y_train)

def LogsiticRegressionModel(X_train,y_train,X_test,y_test,x,y):
    logreg=LogsiticRegression(C=0.01,max_iter=500,penalty='none')
    logreg.fit(X_train,y_train)
    y_pred=logreg.predict(X_test)
    logreg_accuracy=metrics.accuracy_score(y_pred,y_test)
    logref_f1_score=metrics.f1_score(y_pred,y_test,average='micro')
    logreg_recall_score=metrics.recall_score(y_pred,y_test,average='micro')
    print('logreg smote: 'logreg_accuracy)
    print('logreg f1 score: 'logreg_f1_score)
    print('logreg recall score: 'logreg_recall_score)
    logreg_mse=np.sqrt(mean_squared_error(y_test,y_pred))
    print('logreg rmse for prediction: %.4f: '%logreg_mse)

    y_train_pred = logreg.predict(X_train)
    print('Accuracy for train data')
    print (classification_report(y_train, y_train_pred))
    print('Accuracy for test data')
    print (classification_report(y_test, y_pred))
    X=StandardScaler().fit_transform(Xc1)
    metrics.plot_confusion_matrix(logreg, X_test, y_test,cmap=plt.cm.Blues,normalize='true')
    return(y_pred,logreg)

def StatsLogisticRegression(Xc,yc,X_test1,y_test1,x,y):
    X_train_c = add_constant(Xc)
    X_test_c = add_constant(X_test1)
    model = sm.Logit(np.ravel(yc),X_train_c).fit()
    print (model_summary())
    ORs = np.exp(model_params)
    print(ORs)
    pred=model.predict(exog=X_test_c)
    stat_mse=np.sqrt(mean_squared_error(y_test1,pred))
    print('logreg rmse for prediction: %.4f' %stat_mse)

    fpr, tpr, thresholdsr = roc_curve(y_true=list(y_test1), y_score=list(pred))
    auc = roc_auc_score(y_true=list(y_test1), y_score=list(pred))
    stat.roc(fpr=fpr, tpr=tpr, auc=auc, shade_auc=True, per_class=True, legendpos='upper center',legendanchor=(0.5,1.00),legendcols=3)
    plt.figure(figsize=(5,5))
    sns.scatterplot(x=fpr,y=tpr)
    sns.lineplot(x=fpr,y=tpr)
    sns.lineplot(x=[0,1], y=[0,1],color='green')
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')

    return(model_pred)

leads=[]
for i in range(len(finaltestdata)):
    if finaltestdata.Predicted_prob[i] <=35:
        leads.append("Cold Leads")
    elif (finaltestdata.Predicted_prob[i] > 35) & (finaltestdata.Predicted_prob[i] <= 70):
        leads.append("Warm Leads")
    else:
        leads.append("Hot Leads ")
