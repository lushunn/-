# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:37:49 2018

@author: Dell
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import seaborn as sns 
train_consumer_A=pd.read_csv(".//data//train_scene_A//train_consumer_A.csv")

train_target_A=pd.read_csv(".//data//train_scene_A//train_target_A.csv")
train_consumer_A['V_11'][train_consumer_A['V_11']=='0000-00-00 00:00:00']=np.nan#0000-00-00 00:00:00设为缺失值
train_consumer_A[['V_7','V_11']]=train_consumer_A[['V_7','V_11']].apply(pd.to_datetime)
train_consumer_A.info()
#缺失值处理
##v8v10v14为分类变量，将缺失值标记为other
train_consumer_A['V_8'][train_consumer_A['V_8'].isnull()]='other'
train_consumer_A['V_10'][train_consumer_A['V_10'].isnull()]='other'
train_consumer_A['V_14'][train_consumer_A['V_14'].isnull()]='other'
#v12为nan的标为0
train_consumer_A['V_12'][train_consumer_A['V_12'].isnull()]=0

train_consumer_A_gp=train_consumer_A['ccx_id'].groupby(train_consumer_A['ccx_id']).count()
#变量衍生，提取每个样本数值变量的均值最大值最小值，分类变量的频率最高的取值，不同取值个数，最高频数
def createfeature_object(col,df,df1):
    unique=df[col].groupby(df['ccx_id']).apply(lambda x: x.describe()[1])
    top=df[col].groupby(df['ccx_id']).apply(lambda x: x.describe()[2])
    freq=df[col].groupby(df['ccx_id']).apply(lambda x: x.describe()[3])
    tmp=pd.concat([unique, top,freq], axis=1)
    tmp.columns=[col+'_unique',col+'_top',col+'_freq']
    df1=pd.concat([df1, tmp], axis=1)
    return df1
for i in ['V_1','V_2','V_3','V_8','V_14']:
     train_consumer_A_gp=createfeature_object(i,train_consumer_A,train_consumer_A_gp)
def createfeature_float(col,df,df1):
    mean=df[col].groupby(df['ccx_id']).apply(lambda x: x.describe()[1])
    Min=df[col].groupby(df['ccx_id']).apply(lambda x: x.describe()[3])
    Max=df[col].groupby(df['ccx_id']).apply(lambda x: x.describe()[-1])
    tmp=pd.concat([mean,Min,Max], axis=1)
    tmp.columns=[col+'_mean',col+'_Min',col+'_Max']
    df1=pd.concat([df1, tmp], axis=1)
    return df1
for i in ['V_4','V_5','V_6','V_9','V_10','V_12','V_13']:
    train_consumer_A_gp=createfeature_float(i,train_consumer_A,train_consumer_A_gp)

#spyder预览时间变量取值较多的数据框会崩溃
##生成timediff变量，即v7和v11的差值
timediff=pd.concat([train_consumer_A['ccx_id'],train_consumer_A['V_7']-train_consumer_A['V_11']],axis=1)
timediff_mean=timediff[0].groupby(timediff['ccx_id']).apply(lambda x: x.describe()[1])
timediff_Min=timediff[0].groupby(timediff['ccx_id']).apply(lambda x: x.describe()[3])
timediff_Max=timediff[0].groupby(timediff['ccx_id']).apply(lambda x: x.describe()[-1])
timediff_1=pd.concat([ timediff_mean,timediff_Min,timediff_Max], axis=1)
timediff_1.columns=['timediff_mean','timediff_Min','timediff_Max']
result=pd.concat([ train_consumer_A_gp,timediff_1,], axis=1)
#提取月份
month=pd.concat([train_consumer_A['ccx_id'],train_consumer_A['V_7'].apply(lambda x: x.month)],axis=1)
month=month.groupby(month['ccx_id']).apply(lambda x: x.mode().iloc[0])
result=pd.concat([ result,month], axis=1)
result.info()
for i in ['timediff_mean','timediff_Min','timediff_Max']:
     result[i]=result[i].apply(lambda x: x.total_seconds())
     result[i].fillna(train_consumer_A_gp[i].meadian(), inplace=True)#将timediff转化成秒数

result.to_csv('train_consumer_A_test.csv')#保存结果

del result['Unnamed: 0']
del result['ccx_id.2']
tmp=list(result.columns)
tmp[0]='count'
result.columns=tmp
objectype_list=list(result.dtypes[result.dtypes=='object'].index)
#对分类变量woe编码
result_target= pd.merge(result, train_target_A,how = 'left',on='ccx_id')
for i in objectype_list:
    eps=0.00001
    a=pd.crosstab(result_target[i],result_target['target'])+eps
    b=result_target['target'].value_counts()+eps
    c=a/b
    c['woe']=np.log(c[1]/c[0])
    c['iv']=(c[1]-c[0])*c['woe']
    ix=c.index
    iv_value=list(c['iv'])
    map=dict(zip(ix,iv_value))
    result_target[i]=result_target[i].replace(map)
    print(c['iv'].sum())
result_target.info()
#数值型变量用中位数插补缺失值
for i in list(result_target.columns[result_target.count()!=21245]): 
    result_target[i].fillna(result_target[i].median(), inplace=True)
#得出特征重要性得分
from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
X=result_target.iloc[:,1:-1]
y=result_target.iloc[:,-1]
rf = RandomForestClassifier(random_state=42)
rf_param = {'max_depth':[20], 'min_samples_split':[100],'n_estimators':[2000]}
rf_select = model_selection.GridSearchCV(rf, rf_param , cv=5, n_jobs=25, verbose=1, scoring='roc_auc')
rf_select.fit(X,y)
print('Top N Features Best RF Params:' + str(rf_select.best_params_))
print('Top N Features Best RF Score:' + str(rf_select.best_score_))
feature_imp_sorted_rf = pd.DataFrame({'feature': list(X),'importance': rf_select.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
feature_imp_sorted_rf.plot(kind='bar')
features_top_9 = feature_imp_sorted_rf.head(9)['feature']
#重要性前9的变量
''''
['V_3_top',
 'V_5_mean',
 'V_6_mean',
 'V_2_top',
 'V_10_Max',
 'V_10_mean',
 'V_13_mean',
 'V_13_Max',
 'timediff_mean']
'''
##对这9个变量，结合RPM模型，进行特征衍生
train_consumer_A=pd.read_csv(".//data//train_scene_A//train_consumer_A.csv")
train_target_A=pd.read_csv(".//data//train_scene_A//train_target_A.csv")
train_consumer_A['V_11'][train_consumer_A['V_11']=='0000-00-00 00:00:00']=np.nan#0000-00-00 00:00:00设为缺失值
train_consumer_A[['V_7','V_11']]=train_consumer_A[['V_7','V_11']].apply(pd.to_datetime)
train_consumer_A.info()
#缺失值处理
##v8v10v14为分类变量，将缺失值标记为other
train_consumer_A['V_8'][train_consumer_A['V_8'].isnull()]='other'
train_consumer_A['V_10'][train_consumer_A['V_10'].isnull()]='other'
train_consumer_A['V_14'][train_consumer_A['V_14'].isnull()]='other'
#v12为nan的标为0
train_consumer_A['V_12'][train_consumer_A['V_12'].isnull()]=0

#考虑RFM模型，以及前述变量重要性得分
##提取F 即消费频次和购买频率
train_consumer_A_gp=train_consumer_A['ccx_id'].groupby(train_consumer_A['ccx_id']).count()
over = datetime(2017, 6, 1)
train_consumer_A['datediff']=(over-train_consumer_A['V_7']).apply(lambda x: x.days)
tmp1=train_consumer_A['datediff'].groupby(train_consumer_A['ccx_id']).min()
tmp2=train_consumer_A['datediff'].groupby(train_consumer_A['ccx_id']).max()
tmp1=train_consumer_A_gp/(tmp2-tmp1)
tmp1[tmp1==float('inf')]=0
train_consumer_A_gp=pd.concat([train_consumer_A_gp,tmp1],axis=1)
train_consumer_A_gp.columns=['count','pinglv']
#提取R 即最后一次消费离基准时间的天数
##由于时间变量最近的时间是2017-5-31,而behavior表v19变量只含有2017-06-01取值，所以以2017-06-01算出最近消费相隔天数的基准天
over = datetime(2017, 6, 1)
train_consumer_A['datediff']=(over-train_consumer_A['V_7']).apply(lambda x: x.days)
tmp=train_consumer_A['datediff'].groupby(train_consumer_A['ccx_id']).min()
train_consumer_A_gp=pd.concat([train_consumer_A_gp,tmp],axis=1)
train_consumer_A_gp.columns=['count','pinglv', 'datediff']
#提取M 即消费总金额
tmp=train_consumer_A['V_5'].groupby(train_consumer_A['ccx_id']).sum()
tmp1=train_consumer_A['V_5'].groupby(train_consumer_A['ccx_id']).mean()
train_consumer_A_gp=pd.concat([train_consumer_A_gp,tmp,tmp1],axis=1)
train_consumer_A_gp.columns=['count','pinglv' ,'datediff','v5_sum','v5_mean']
#对重要性较高的变量进行提取
##对V12提取
tmp=train_consumer_A['V_12'].groupby(train_consumer_A['ccx_id']).sum()
train_consumer_A_gp=pd.concat([train_consumer_A_gp,tmp],axis=1)
train_consumer_A_gp.columns=['count','pinglv' ,'datediff','v5_sum','v5_mean','v12_sum']
#v3,v2
tmp=train_consumer_A['V_2'].groupby(train_consumer_A['ccx_id']).apply(lambda x: x.describe()[2])
tmp1=train_consumer_A['V_3'].groupby(train_consumer_A['ccx_id']).apply(lambda x: x.describe()[2])
train_consumer_A_gp=pd.concat([train_consumer_A_gp,tmp,tmp1],axis=1)
train_consumer_A_gp.columns=['count','pinglv' ,'datediff','v5_sum','v5_mean','v12_sum','v2_top','3_top']
#V_10
tmp=train_consumer_A['V_10'].groupby(train_consumer_A['ccx_id']).sum()
tmp1=train_consumer_A['V_10'].groupby(train_consumer_A['ccx_id']).mean()
train_consumer_A_gp=pd.concat([train_consumer_A_gp,tmp,tmp1],axis=1)
train_consumer_A_gp.columns=['count','pinglv' ,'datediff','v5_sum','v5_mean','v12_sum','v2_top','3_top','V_10_SUM','V_10_MEAN']

##按季节
train_consumer_A['month']=train_consumer_A['V_7'].apply(lambda x: x.month)
train_consumer_A['quarter']=train_consumer_A['month']
train_consumer_A['quarter'][train_consumer_A['month']<4]=1
train_consumer_A['quarter'][(train_consumer_A['month']>=4)&( train_consumer_A['month']<7)]=2
train_consumer_A['quarter'][(train_consumer_A['month']>=7)&( train_consumer_A['month']<10)]=3
train_consumer_A['quarter'][(train_consumer_A['month']>=10)&( train_consumer_A['month']<=12)]=4
train_consumer_A['fake']=1
tmp=train_consumer_A.pivot_table(index='ccx_id',columns='quarter',values='fake',aggfunc=np.sum)
tmp=tmp.fillna(0)
train_consumer_A_gp=pd.concat([train_consumer_A_gp,tmp],axis=1)
train_consumer_A_gp.columns=['count','pinglv' ,'datediff','v5_sum','v5_mean','v12_sum','v2_top','v3_top','V_10_SUM','V_10_MEAN','q1','q2','q3','q4']
