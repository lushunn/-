# -*- coding: utf-8 -*-
"""
Created on Wed May  9 14:58:19 2018

@author: Dell
"""


import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
train_behavior_A=pd.read_csv(".//data//train_scene_A//train_behavior_A.csv")
train_target_A=pd.read_csv(".//data//train_scene_A//train_target_A.csv")
#探索性数据分析
#统一变量数据类型
train_behavior_A.dtypes[train_behavior_A.dtypes=='object']
objectype_list=list(train_behavior_A.dtypes[train_behavior_A.dtypes=='object'].index)
train_behavior_A[objectype_list].head
train_behavior_A[objectype_list].describe()#查看未定数据类型的列
train_behavior_A[objectype_list]=train_behavior_A[objectype_list].fillna('novalue')#标记缺失值
#查看实值的分布状况
tmp=np.array(train_behavior_A.count())
plt.hist(tmp)
tmp=pd.DataFrame(train_behavior_A.count())
tmp=tmp[tmp[0]>=8000]#缺失值多的变量直接舍去
ix=list(tmp.index)
train_behavior_A_target= pd.merge(train_behavior_A[ix], train_target_A,how = 'left',on='ccx_id')
#objectype_list=list(train_behavior_A_target.dtypes[train_behavior_A_target.dtypes=='object'].index)

#除了var1，都是分类变量，对var1分箱
train_behavior_A_target[['var1']].hist()#查看 var1分布
train_behavior_A_target['var1_bin']= pd.qcut(train_behavior_A_target['var1'], 8)
del train_behavior_A_target['var1']
#填充缺失值
train_behavior_A_target.columns[train_behavior_A_target.isnull().any()]
for i in ['var10', 'var158', 'var159', 'var160', 'var161', 'var162', 'var163']:
    train_behavior_A_target[i]=train_behavior_A_target[i].fillna('novalue')
###查看各变量的iv值
iv=[]
for i in['var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8',
       'var9', 'var10', 'var19', 'var158', 'var159', 'var160', 'var161',
       'var162', 'var163',  'var1_bin']:
    eps=0.00001
    a=pd.crosstab(train_behavior_A_target[i],train_behavior_A_target['target'])+eps
    b=train_behavior_A_target['target'].value_counts()+eps
    c=a/b
    c['woe']=np.log(c[1]/c[0])
    c['iv']=(c[1]-c[0])*c['woe']
    iv.append( c['iv'].sum())

iv=pd.DataFrame({'var':['var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8',
       'var9', 'var10', 'var19', 'var158', 'var159', 'var160', 'var161',
       'var162', 'var163',  'var1_bin'],'value':iv})

iv['value'].sort_values(ascending=False).plot(kind='bar')    
iv['value'].sort_values(ascending=False).index
iv= iv.ix[[2, 4, 15, 12, 1, 11, 10, 3, 14, 13, 16, 9, 5, 6, 8, 7, 0]]
iv.head(12)
iv=iv.head(12)
#去掉iv最低的5个变量
#var4 ,var6,var163,var160,加上特征选择的'var1','var3'	,'var5'

    
##部分分类变量取值较多影响模型效果，将频次累计值达到95%分为以后取值命名为lesscount,探究变量处理后的影响  
for i in ['var4','var6']:
    tmp=train_behavior_A_target[i].value_counts()
    tmp1=tmp.cumsum()
    list1=tmp1[tmp1>0.95*21245].index
    list2=['lessCategorical' for i in range(0,len(list1))]
    map=dict(zip(list1,list2))
    train_behavior_A_target[i]=train_behavior_A_target[i].replace(map)

iv2=[]
for i in['var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8',
       'var9', 'var10', 'var19', 'var158', 'var159', 'var160', 'var161',
       'var162', 'var163',  'var1_bin']:
    eps=0.00001
    a=pd.crosstab(train_behavior_A_target[i],train_behavior_A_target['target'])+eps
    b=train_behavior_A_target['target'].value_counts()+eps
    c=a/b
    c['woe']=np.log(c[1]/c[0])
    c['iv']=(c[1]-c[0])*c['woe']
    iv2.append( c['iv'].sum())

iv2=pd.DataFrame({'var':['var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8',
       'var9', 'var10', 'var19', 'var158', 'var159', 'var160', 'var161',
       'var162', 'var163',  'var1_bin'],'value':iv2})

iv2['value'].sort_values(ascending=False).plot(kind='bar')    
iv2['value'].sort_values(ascending=False).index
iv2= iv2.ix[[4, 2, 15, 12, 1, 11, 10, 3, 14, 13, 16, 9, 5, 6, 8, 7, 0]]

###最后以不进行分箱的方式提取特征
train_behavior_A=pd.read_csv(".//data//train_scene_A//train_behavior_A.csv")
train_target_A=pd.read_csv(".//data//train_scene_A//train_target_A.csv")
train_behavior_A_target= pd.merge(train_behavior_A[ix], train_target_A,how = 'left',on='ccx_id')
tmp=np.array(train_behavior_A.count())
plt.hist(tmp)
tmp=pd.DataFrame(train_behavior_A.count())
tmp=tmp[tmp[0]>=4249]#缺失值多的变量直接舍去 0.3#4249
ix=list(tmp.index)
train_behavior_A=train_behavior_A[ix]
train_behavior_A_target= pd.merge(train_behavior_A, train_target_A,how = 'left',on='ccx_id')#iv值变小 ，建议，不进行‘lesscount‘处理
#插补缺失值
train_behavior_A_target=train_behavior_A_target.fillna('novalue')
a= (train_behavior_A_target.columns.tolist())[1:-1]
notbadvar=[]

#过滤只含两个值的变量（包括缺失值）
for i in a:
    if len(train_behavior_A[i].unique())>2:
        notbadvar.append(i)



iv2=[]
for i in notbadvar :
    eps=0.00001
    a=pd.crosstab(train_behavior_A_target[i],train_behavior_A_target['target'])+eps
    b=train_behavior_A_target['target'].value_counts()+eps
    c=a/b
    c['woe']=np.log(c[1]/c[0])
    c['iv']=(c[1]-c[0])*c['woe']
    iv2.append( c['iv'].sum())

iv2=pd.DataFrame({'var':notbadvar,'value':iv2})    
#只留下iv值大于0.02的特征
finalvar=iv2[(iv2['value']>0.02)]
finalvar=finalvar['var'].tolist()
finalvar.append('ccx_id')
finalvar.append('var3')
finalvar.append('var5')
