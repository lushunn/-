# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:59:50 2018

@author: Dell
"""
import pandas as pd
train_consumer_A=pd.read_csv("./train/scene_A/train_consumer_A.csv")
train_behavior_A=pd.read_csv("./train/scene_A/train_behavior_A.csv")
train_ccx_A=pd.read_csv("./train/scene_A/train_ccx_A.csv")
train_target_A=pd.read_csv("./train/scene_A/train_target_A.csv")
import numpy as np

#提取变量

train_behavior_A=train_behavior_A[['ccx_id','var1',
                                   'var3','var5',
 'var4',
 'var6',
 'var11',
 'var12',
 'var13',
 'var16',
 'var17',
 'var18',
 'var155',
 'var156',
 'var158',
 'var159',
 'var160',
 'var163',
 'var431',
 'var440',
 'var441',
 'var443',
 'var449',
 'var450',
 'var452',
 'var458',
 'var789',
 'var843',
 'var844',
 'var969',
 'var970',
 'var978',
 'var1163',
 'var1166',
 'var1271',
 'var1272',
 'var1273',
 'var1274',
 'var1275',
 'var1276',
 'var1277',
 'var1278',
 'var1279',
 'var1280',
 'var1281',
 'var1282',
 'var1283',
 'var1284',
 'var1285',
 'var1286',
 'var1287',
 'var1288',
 'var1289',
 'var1290',
 'var1291',
 'var1292',
 'var1293',
 'var1294',
 'var1295',
 'var1298',
 'var1299',
 'var1300',
 'var1301',
 'var1302',
 'var1303',
 'var1304',
 'var1307',
 'var1308',
 'var1309',
 'var1310',
 'var1311',
 'var1312',
 'var1610',
 'var1619',
 'var1620',
 'var1622',
 'var1642',
 'var1643',
 'var1644',
 'var1645',
 'var1646',
 'var1734',
 'var1735',
 'var1737',
 'var1738',
 'var1806',
 'var1807',
 'var1968',
 'var1986',
 'var1989',
 'var2244']]


#插补缺失值
for i in ['var4','var6','var3','var5','var11', 'var12', 'var13','var18']:
    train_behavior_A[i][train_behavior_A[i].isnull()]='novalue'   

for i in[  'var11',
 'var12',
 
 'var16',
 'var17',
 'var18',
 'var155',
 'var156',
 'var158',
 'var159',
 'var160',
 'var163',
 'var431',
 'var440',
 'var441',
 'var443',
 'var449',
 'var450',
 'var452',
 'var458',
 'var789',
 'var843',
 'var844',
 'var969',
 'var970',
 'var978',
 'var1163',
 'var1166',
 'var1271',
 'var1272',
 'var1273',
 'var1274',
 'var1275',
 'var1276',
 'var1277',
 'var1278',
 'var1279',
 'var1280',
 'var1281',
 'var1282',
 'var1283',
 'var1284',
 'var1285',
 'var1286',
 'var1287',
 'var1288',
 'var1289',
 'var1290',
 'var1291',
 'var1292',
 'var1293',
 'var1294',
 'var1295',
 'var1298',
 'var1299',
 'var1300',
 'var1301',
 'var1302',
 'var1303',
 'var1304',
 'var1307',
 'var1308',
 'var1309',
 'var1310',
 'var1311',
 'var1312',
 'var1610',
 'var1619',
 'var1620',
 'var1622',
 'var1642',
 'var1643',
 'var1644',
 'var1645',
 'var1646',
 'var1734',
 'var1735',
 'var1737',
 'var1738',
 'var1806',
 'var1807',
 'var1968',
 'var1986',
 'var1989',
 'var2244']:
     train_behavior_A[i][train_behavior_A[i].isnull()]=-1    
#var16反映年份的变量数据杂乱，进行清洗，规整化
tmp=train_behavior_A['var16'][train_behavior_A['var16']>2015].index.tolist()
for i in tmp:
    train_behavior_A.ix[i,'var16']=int(str(train_behavior_A.ix[i,'var16'])[:4])

b=train_behavior_A['var16'].unique()
b.sort()
c=[i for i in range(-1,20)]
b=dict(zip(b,c))
train_behavior_A['var16']=train_behavior_A['var16'].replace(b)

from sklearn.model_selection import train_test_split
train_behavior,test__behavior=train_test_split(train_behavior_A ,test_size=0.3, random_state=0)


train_behavior_t= pd.merge(train_behavior, train_target_A,how = 'left',on='ccx_id')
test_behavior_t=pd.merge(test__behavior, train_target_A,how = 'left',on='ccx_id')

#进行因子化编码

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in['var4','var6','var3','var5','var11', 'var12', 'var13','var18']:
    le.fit(pd.concat([train_behavior_t[i],test_behavior_t[i]]))
    train_behavior_t[i]=le.transform(train_behavior_t[i])
    test_behavior_t[i]=le.transform(test_behavior_t[i])

#consumer_A表     


#考虑RFM模型，以及前述变量重要性得分
##提取F 即消费频次

#提取R 即最后一次消费离基准时间的天数
##由于时间变量最近的时间是2017-5-31,而behavior表v19变量只含有2017-06-01取值，所以以2017-06-01算出最近消费相隔天数的基准天
from datetime import datetime
def chuli_consumer(test_consumer_A,over = datetime(2017, 6, 1)):
    #test_consumer_A['V_11'][test_consumer_A['V_11']=='0000-00-00 00:00:00']=np.nan#0000-00-00 00:00:00设为缺失值
    test_consumer_A[['V_7']]=test_consumer_A[['V_7']].apply(pd.to_datetime)
    test_consumer_A['V_8'][test_consumer_A['V_8'].isnull()]='novalue' 
    test_consumer_A['V_10'][test_consumer_A['V_10'].isnull()]='novalue' 
    test_consumer_A['V_14'][test_consumer_A['V_14'].isnull()]='novalue' 
    test_consumer_A['V_12'][test_consumer_A['V_12'].isnull()]=-1
    test_consumer_A_gp=test_consumer_A['ccx_id'].groupby(test_consumer_A['ccx_id']).count()
    test_consumer_A['datediff']=(over-test_consumer_A['V_7']).apply(lambda x: x.days)
    tmp1= test_consumer_A['datediff'].groupby( test_consumer_A['ccx_id']).min()
    tmp2= test_consumer_A['datediff'].groupby( test_consumer_A['ccx_id']).max()
    tmp1= test_consumer_A_gp/(tmp2-tmp1)
    tmp1[tmp1==float('inf')]=0
    test_consumer_A_gp=pd.concat([test_consumer_A_gp,tmp1],axis=1)
    test_consumer_A_gp.columns=['count','pinglv']
    #test_consumer_A['datediff']=(over-test_consumer_A['V_7']).apply(lambda x: x.days)
    tmp=test_consumer_A['datediff'].groupby(test_consumer_A['ccx_id']).min()
    test_consumer_A_gp=pd.concat([test_consumer_A_gp,tmp],axis=1)
    test_consumer_A_gp.columns=['count','pinglv', 'datediff']
    #提取M 即消费总金额
    tmp=test_consumer_A['V_5'].groupby(test_consumer_A['ccx_id']).sum()
    tmp1=test_consumer_A['V_5'].groupby(test_consumer_A['ccx_id']).mean()
    test_consumer_A_gp=pd.concat([test_consumer_A_gp,tmp,tmp1],axis=1)
    test_consumer_A_gp.columns=['count','pinglv' ,'datediff','v5_sum','v5_mean']
    #对重要性较高的变量进行提取
    ##对V12提取
    tmp=test_consumer_A['V_12'].groupby(test_consumer_A['ccx_id']).sum()
    test_consumer_A_gp=pd.concat([test_consumer_A_gp,tmp],axis=1)
    test_consumer_A_gp.columns=['count','pinglv' ,'datediff','v5_sum','v5_mean','v12_sum']
    #v3,v2
    tmp=test_consumer_A['V_2'].groupby(test_consumer_A['ccx_id']).apply(lambda x: x.describe()[2])
    tmp1=test_consumer_A['V_3'].groupby(test_consumer_A['ccx_id']).apply(lambda x: x.describe()[2])
    test_consumer_A_gp=pd.concat([test_consumer_A_gp,tmp,tmp1],axis=1)
    test_consumer_A_gp.columns=['count','pinglv' ,'datediff','v5_sum','v5_mean','v12_sum','v2_top','3_top']
    #V_10
    tmp=test_consumer_A['V_10'].groupby(test_consumer_A['ccx_id']).sum()
    tmp1=test_consumer_A['V_10'].groupby(test_consumer_A['ccx_id']).mean()
    test_consumer_A_gp=pd.concat([test_consumer_A_gp,tmp,tmp1],axis=1)
    test_consumer_A_gp.columns=['count','pinglv' ,'datediff','v5_sum','v5_mean','v12_sum','v2_top','3_top','V_10_SUM','V_10_MEAN']
    
    
    ##按季节
    test_consumer_A['month']=test_consumer_A['V_7'].apply(lambda x: x.month)
    test_consumer_A['quarter']=test_consumer_A['month']
    test_consumer_A['quarter'][test_consumer_A['month']<4]=1
    test_consumer_A['quarter'][(test_consumer_A['month']>=4)&( test_consumer_A['month']<7)]=2
    test_consumer_A['quarter'][(test_consumer_A['month']>=7)&( test_consumer_A['month']<10)]=3
    test_consumer_A['quarter'][(test_consumer_A['month']>=10)&(test_consumer_A['month']<=12)]=4
    test_consumer_A['fake']=1
    tmp=test_consumer_A.pivot_table(index='ccx_id',columns='quarter',values='fake',aggfunc=np.sum)
    tmp=tmp.fillna(0)
    test_consumer_A_gp=pd.concat([test_consumer_A_gp,tmp],axis=1)
    test_consumer_A_gp.columns=['count','pinglv' ,'datediff','v5_sum','v5_mean','v12_sum','v2_top','v3_top','V_10_SUM','V_10_MEAN','q1','q2','q3','q4']
    return test_consumer_A_gp

train_consumer=pd.merge(train_behavior_t[['ccx_id']],train_consumer_A,how = 'left',on='ccx_id')
test_consumer=pd.merge(test_behavior_t[['ccx_id']],train_consumer_A,how = 'left',on='ccx_id')
test_consumer_A_gp=chuli_consumer(test_consumer)
train_consumer_A_gp=chuli_consumer(train_consumer)

train_consumer_A_gp['ccx_id']=train_consumer_A_gp.index
train_consumer_A_gp=pd.merge(train_consumer_A_gp, train_target_A,how = 'left',on='ccx_id')
test_consumer_A_gp['ccx_id']=test_consumer_A_gp.index
test_consumer_A_gp=pd.merge(test_consumer_A_gp, train_target_A,how = 'left',on='ccx_id')


#pca_left_colname=['count','pinglv', 'datediff', 'v5_sum', 'v5_mean', 'v12_sum', 'V_10_SUM', 'V_10_MEAN', 'q1', 'q2', 'q3', 'q4','ccx_id', 'target']

#进行因子化编码
for i in ['v2_top','v3_top']:
    le.fit(pd.concat([ train_consumer_A_gp[i], test_consumer_A_gp[i]]))
    train_consumer_A_gp[i]=le.transform(train_consumer_A_gp[i])
    test_consumer_A_gp[i]=le.transform(test_consumer_A_gp[i])

#ccx
#ccx
def chuli_consumer(train_ccx_A=train_ccx_A,train_target_A=train_target_A,over=datetime(2017, 6, 1)):
    #月份只有2017.1-2017.5的数据，只保留月份作为分类变量
    
    train_ccx_A[['var_06']]=train_ccx_A[['var_06']].apply(pd.to_datetime)
    train_ccx_A['datediff']=(over-train_ccx_A['var_06']).apply(lambda x: x.days)
    query = train_ccx_A.groupby(train_ccx_A['ccx_id']).size()#查询次数
    query = query.reset_index() #index 改为 column
    datediff=train_ccx_A['datediff'].groupby(train_ccx_A['ccx_id']).min()
    datediff=datediff.reset_index()
    query=pd.merge(query,datediff,on='ccx_id',how='left')
    query.columns = ['ccx_id','query','datediff']
    tmp1= train_ccx_A['datediff'].groupby( train_ccx_A['ccx_id']).min()
    tmp2= train_ccx_A['datediff'].groupby( train_ccx_A['ccx_id']).max()
    query['query']= list( query['query'].tolist()/(tmp2-tmp1))
    query['query'][query['query']==float('inf')]=0       
    for i in['var_01', 'var_02', 'var_03', 'var_04', 'var_05']:
        query[i]=list(train_ccx_A[i].groupby(train_ccx_A['ccx_id']).apply(lambda x: x.describe()[2]))
    df3 = pd.merge(train_target_A, query,on='ccx_id',how='left')#target与ccx合并
    df4 = df3.drop(['target'], axis = 1) #只有数据没有target


    df4=df4.fillna(0)
    return df4

def chuli_consumer(train_ccx_A=train_ccx_A,train_target_A=train_target_A,over=datetime(2017, 6, 1)):
    #月份只有2017.1-2017.5的数据，只保留月份作为分类变量
    
    train_ccx_A[['var_06']]=train_ccx_A[['var_06']].apply(pd.to_datetime)
    train_ccx_A['datediff']=(over-train_ccx_A['var_06']).apply(lambda x: x.days)
    query = train_ccx_A.groupby(train_ccx_A['ccx_id']).size()#查询次数
    query = query.reset_index() #index 改为 column

    datediff=train_ccx_A['datediff'].groupby(train_ccx_A['ccx_id']).min()
    datediff=datediff.reset_index()
    query=pd.merge(query,datediff,on='ccx_id',how='left')
    query.columns = ['ccx_id','query','datediff']
    tmp1= train_ccx_A['datediff'].groupby( train_ccx_A['ccx_id']).min()
    tmp2= train_ccx_A['datediff'].groupby( train_ccx_A['ccx_id']).max()
    query['query']= list( query['query'].tolist()/(tmp2-tmp1))
    query['query'][query['query']==float('inf')]=0     
    df = pd.get_dummies(train_ccx_A) #变成哑变量形式one-hot编码 特征增加了
    df2 = df.groupby(['ccx_id'],as_index=False).sum() #根据id汇总 加和
    

    df3 = pd.merge(df2,query,on='ccx_id',how='left')#query 和 df2合并
    df3 = pd.merge(train_target_A,df3,on='ccx_id',how='left')#target与ccx合并
    df4 = df3.drop(['target'], axis = 1) #只有数据没有target


    df4=df4.fillna(0)

    df4 = df4.set_index("ccx_id")
    
    return df4


train_ccx=pd.merge(train_behavior_t[['ccx_id']],train_ccx_A,how = 'left',on='ccx_id')
train_ccx=train_ccx.dropna()
test_ccx=pd.merge(test_behavior_t[['ccx_id']],train_ccx_A,how = 'left',on='ccx_id')
test_ccx=test_ccx.dropna()
train_ccx=chuli_consumer(train_ccx_A=train_ccx,train_target_A=train_behavior_t[['ccx_id','target']])
train_ccx['ccx_id']=train_ccx.index
tmp=list(test_consumer_A_gp.index)
tmp1=[0 for i in range(len (tmp))]
test_ccx=chuli_consumer(test_ccx,train_target_A=test_behavior_t[['ccx_id','target']])
test_ccx['ccx_id']=test_ccx.index

tmp=list(set(train_ccx.columns)&set(test_ccx.columns))
train_ccx=train_ccx[tmp]
test_ccx=test_ccx[tmp]

#对于一些特征取值，没有同时存在于train表和test表，则就生成相应的全0特征
for i in list(set(train_ccx.columns)-set(test_ccx.columns)):
    test_ccx[i]=0
for i in list(set(test_ccx.columns)-set(train_ccx.columns)):
    train_ccx[i]=0

 #划分测试训练集
train_X=pd.merge(train_consumer_A_gp,train_behavior_t,on='ccx_id')
train_X=pd.merge(train_X,train_ccx,on='ccx_id').drop('ccx_id',axis=1)

train_y=train_X['target_x']
del train_X['target_x']
del train_X['target_y']

test_X=pd.merge(test_consumer_A_gp,test_behavior_t,on='ccx_id')
test_X=pd.merge(test_X,test_ccx,on='ccx_id').drop('ccx_id',axis=1)
test_y=test_X['target_x']
del test_X['target_x']
del test_X['target_y']

test_X=test_X[train_X.columns.tolist()]







#运行测试 

from sklearn import model_selection
from sklearn.metrics import roc_curve
from sklearn.metrics import auc 
import xgboost as xgb
import lightgbm as lgb
#xgb
xgb_param= {'n_estimators': [260],'max_depth':[4], 'min_child_weight':[4],'gamma':[ 0.1],'subsample': [0.8],'colsample_bytree':[0.6],'reg_alpha':[0.5], 'reg_lambda':[ 0.05],'learning_rate': [0.01]}

xgm = xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
xgm_grid=model_selection.GridSearchCV(xgm,xgb_param, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc')

xgb_param= {'n_estimators': 260,'max_depth':4, 'min_child_weight':4,'gamma':0.1,'subsample':0.8,'colsample_bytree':0.6,'reg_alpha':0.5, 'reg_lambda':0.05,'learning_rate': 0.01,'scale_pos_weight':2.047} 
  
  
xgm=xgb.XGBClassifier(**xgb_param)


  
xgm.fit(train_X,train_y)


print(' Best  Params:' + str(xgm_grid.best_params_))
print(' Best Score:' + str(xgm_grid.best_score_))


pred_xgb=xgm.predict_proba(test_X)

fpr, tpr, thresholds = roc_curve(test_y, pred_xgb[:,1], pos_label=1)
print('xgb auc:'+str(auc(fpr, tpr)) )

lgm_param={'num_leaves':4,'max_depth':3,'feature_fraction':0.3,'bagging_fraction':0.6,'lambda_l1':0.8,'lambda_l2':0.6,'metric':'auc','n_estimators':900,'learning_rate': 0.01}

lgm = lgb.LGBMClassifier(**lgm_param)
lgm_grid=model_selection.GridSearchCV(lgm,lgm_param, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc')

lgm_grid.fit(train_X,train_y)

print(' Best  Params:' + str(lgm_grid.best_params_))
print(' Best Score:' + str(lgm_grid.best_score_))


pred_lgm=lgm_grid.predict_proba(test_X)

fpr, tpr, thresholds = roc_curve(test_y, pred_lgm[:,1], pos_label=1)
print('lgm auc:'+str(auc(fpr, tpr)) )
# 0.6434127743508056
####

from sklearn.ensemble import GradientBoostingClassifier
gbm_param={'n_estimators':450,'max_depth':4, 'min_samples_split':1200, 'min_samples_leaf':45,'max_features':60,'learning_rate': 0.01}

gbm = GradientBoostingClassifier(**gbm_param)
gbm_grid=model_selection.GridSearchCV(gbm,gbm_param, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc')

gbm_grid.fit(train_X,train_y)


print(' Best  Params:' + str(gbm_grid.best_params_))
print(' Best Score:' + str(gbm_grid.best_score_))


pred_gbm=gbm_grid.predict_proba(test_X)

fpr, tpr, thresholds = roc_curve(test_y, pred_gbm[:,1], pos_label=1)
print('gbm auc:'+str(auc(fpr, tpr)) )
#0.6414269716374987


pred_final=(0.9*pred_lgm+0.035*pred_xgb+0.065*pred_gbm)
fpr, tpr, thresholds = roc_curve(test_y, pred_final[:,1], pos_label=1)
print('predict auc:'+str(auc(fpr, tpr)) )
