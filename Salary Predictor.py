
# coding: utf-8

# In[1]:

import networkx as nx
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# In[2]:

G = nx.read_gpickle('email_prediction.txt')
def salary_predictions():
    df=pd.DataFrame(index=G.nodes())
    df['Department']=pd.Series(nx.get_node_attributes(G,'Department'))
    df['Salary']=pd.Series(nx.get_node_attributes(G,'ManagementSalary'))
    df['Salary'].fillna(2,inplace=True)
    df['connection']=pd.Series(G.edges())
    
    df['clust']=(nx.clustering(G)).values()
    df['deg']=(nx.degree(G)).values()
    df['deg_cent']=(nx.degree_centrality(G)).values()
    df['close_cent']=(nx.closeness_centrality(G)).values()
    df['betw_cent']=(nx.betweenness_centrality(G,normalized=True,endpoints=False)).values()
    
    df_train=df[df['Salary']!=2]
    df_predict=df[df['Salary']==2]
    
    X_train=df_train[['clust','deg','deg_cent','close_cent','betw_cent']]
    y_train=df_train['Salary']
    X_test=df_predict[['clust','deg','deg_cent','close_cent','betw_cent']]
    
    scaler=MinMaxScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    
    clf=LogisticRegression().fit(X_train_scaled,y_train)
    y_predict=clf.predict_proba(X_test_scaled)
    ans=pd.Series(y_predict[:,1],index=X_test.index)
    return ans


# In[ ]:



