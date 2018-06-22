# -*- coding: utf-8 -*-


###
#python文件载入部分
###
#加载验证数据集
import pandas as pd
test_consumer_A=pd.read_csv("./test/scene_A/test_consumer_A.csv")
test_consumer_B=pd.read_csv("./test/scene_B/test_consumer_B.csv")
test_behavior_A=pd.read_csv("./test/scene_A/test_behavior_A.csv")
test_behavior_B=pd.read_csv("./test/scene_B/test_behavior_B.csv")
test_ccx_A=pd.read_csv("./test/scene_A/test_ccx_A.csv")
# 自测代码时读取的验证集数据样本
# test_consumer_A=pd.read_csv("./testdemo/scene_A/test_consumer_A.csv")
# test_consumer_B=pd.read_csv("./testdemo/scene_B/test_consumer_B.csv")
# test_behavior_A=pd.read_csv("./testdemo/scene_A/test_behavior_A.csv")
# test_behavior_B=pd.read_csv("./testdemo/scene_B/test_behavior_B.csv")
# test_ccx_A=pd.read_csv("./testdemo/scene_A/test_ccx_A.csv")
###
###
#行为表
###
###
train_consumer_A=pd.read_csv("./train/scene_A/train_consumer_A.csv")
train_behavior_A=pd.read_csv("./train/scene_A/train_behavior_A.csv")
train_ccx_A=pd.read_csv("./train/scene_A/train_ccx_A.csv")
train_target_A=pd.read_csv("./train/scene_A/train_target_A.csv")
import numpy as np
####改变数据类型
tmp=['ccx_id','var1',
    'var3','var5',
 'var4',
 'var6',
 'var11',
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
 'var440',
 'var441',
 'var443',
 'var449',
 'var450',
 'var452',
 'var789',
 'var843',
 'var844',
 'var969',
 'var970',
 'var978',
 'var1163',
 'var1166',
 'var1283',
 'var1284',
 'var1285',
 'var1286',
 'var1287',
 'var1288',
 'var1289',
 'var1290',
 'var1291',
 'var1301',
 'var1302',
 'var1303',
 'var1304',
 'var1307',
 'var1308',
 'var1309',
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
 'var2244']
test_behavior_A=test_behavior_A[tmp]
train_behavior_A=train_behavior_A[tmp]
test_behavior_B=test_behavior_B[tmp]
for i in ['var4','var6','var3','var5','var11', 'var12', 'var18']:
    train_behavior_A[i][train_behavior_A[i].isnull()]='novalue'
    test_behavior_A[i][test_behavior_A[i].isnull()]='novalue'
    test_behavior_B[i][test_behavior_B[i].isnull()]='novalue'
tmp=[   'var11',
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
 'var440',
 'var441',
 'var443',
 'var449',
 'var450',
 'var452',
 'var789',
 'var843',
 'var844',
 'var969',
 'var970',
 'var978',
 'var1163',
 'var1166',
 'var1283',
 'var1284',
 'var1285',
 'var1286',
 'var1287',
 'var1288',
 'var1289',
 'var1290',
 'var1291',
 'var1301',
 'var1302',
 'var1303',
 'var1304',
 'var1307',
 'var1308',
 'var1309',
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
 'var2244']
for i in tmp:
     train_behavior_A[i][train_behavior_A[i].isnull()]=-1
     test_behavior_A[i][test_behavior_A[i].isnull()]=-1
     test_behavior_B[i][test_behavior_B[i].isnull()]=-1
     
test_behavior_A['var16'][test_behavior_A['var16']=='null']=-1
test_behavior_A['var16']  = test_behavior_A['var16'].astype(int)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in['var4','var6','var3','var5','var11', 'var12', 'var18']:
    le.fit(pd.concat([train_behavior_A[i],test_behavior_A[i],test_behavior_B[i]]))
    train_behavior_A[i]=le.transform(train_behavior_A[i])
    test_behavior_A[i]=le.transform(test_behavior_A[i])
    test_behavior_B[i]=le.transform(test_behavior_B[i])
     
train_behavior_t= pd.merge(train_behavior_A, train_target_A,how = 'left',on='ccx_id')


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

test_consumer_A_gp=chuli_consumer(test_consumer_A)
train_consumer_A_gp=chuli_consumer(train_consumer_A)
over = datetime(2017, 1, 1)
test_consumer_B_gp=chuli_consumer(test_consumer_B,over)
train_consumer_A_gp['ccx_id']=train_consumer_A_gp.index
train_consumer_A_gp=pd.merge(train_consumer_A_gp, train_target_A,how = 'left',on='ccx_id')
test_consumer_A_gp['ccx_id']=test_consumer_A_gp.index
test_consumer_B_gp['ccx_id']=test_consumer_B_gp.index

for i in ['v2_top','v3_top']:
    le.fit(pd.concat([ train_consumer_A_gp[i], test_consumer_A_gp[i],test_consumer_B_gp[i]]))
    train_consumer_A_gp[i]=le.transform(train_consumer_A_gp[i])
    test_consumer_A_gp[i]=le.transform(test_consumer_A_gp[i])
    test_consumer_B_gp[i]=le.transform(test_consumer_B_gp[i])

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
    df = pd.get_dummies(train_ccx_A) #变成哑变量形式one-hot编码 特征增加了
    df2 = df.groupby(['ccx_id'],as_index=False).sum() #根据id汇总 加和
    

    df3 = pd.merge(df2,query,on='ccx_id',how='left')#query 和 df2合并
    df3 = pd.merge(train_target_A,df3,on='ccx_id',how='left')#target与ccx合并
    df4 = df3.drop(['target'], axis = 1) #只有数据没有target


    df4=df4.fillna(0)

    df4 = df4.set_index("ccx_id")
    
    return df4



train_ccx_A=chuli_consumer(train_ccx_A)
train_ccx_A['ccx_id']=train_ccx_A.index
tmp=list(test_consumer_A_gp.index)
tmp1=[0 for i in range(len (tmp))]
fake_target=pd.DataFrame({'ccx_id':tmp,'target':tmp1})
test_ccx_A=chuli_consumer(test_ccx_A,train_target_A=fake_target)
test_ccx_A['ccx_id']=test_ccx_A.index

for i in list(set(train_ccx_A.columns)-set(test_ccx_A.columns)):
    test_ccx_A[i]=0
for i in list(set(test_ccx_A.columns)-set(train_ccx_A.columns)):
    train_ccx_A[i]=0



train_X=pd.merge(train_consumer_A_gp,train_behavior_A,on='ccx_id')
train_X=pd.merge(train_X,train_ccx_A,on='ccx_id').drop('ccx_id',axis=1)
train_y=train_X['target']
del train_X['target']
test_X=pd.merge(test_consumer_A_gp,test_behavior_A,on='ccx_id')
test_X=pd.merge(test_X,test_ccx_A,on='ccx_id').drop('ccx_id',axis=1)


train_X_B=pd.merge(train_consumer_A_gp,train_behavior_A,on='ccx_id')
del train_X_B['target']
test_X_B=pd.merge(test_consumer_B_gp,test_behavior_B,on='ccx_id')


#
from sklearn.ensemble import GradientBoostingClassifier
#运行测试 
import xgboost as xgb
import lightgbm as lgb
#xgb
xgb_param= {'n_estimators': 25,'max_depth':3,
 'min_child_weight':5,'gamma': 0.1,'subsample':  0.8, 'colsample_bytree': 0.8,'reg_alpha': 0.05, 'reg_lambda': 0.05,'learning_rate': 0.1}
xgm = xgb.XGBClassifier(**xgb_param)
test_X=test_X[train_X.columns.tolist()]
xgm.fit(train_X,train_y)
pred_xgb=xgm.predict_proba(test_X)
pred_xgb=pred_xgb


xgm = xgb.XGBClassifier(**xgb_param)
xgm.fit(train_X_B,train_y)
pred_xgb_B=xgm.predict_proba(test_X_B)


#lgb


lgm_param ={'num_leaves':3,'num_leaves':3,'max_depth':3,'feature_fraction':0.3,'bagging_fraction':0.6,'lambda_l1':0.8,'lambda_l2':0.6,'metric':'auc'}

lgm=lgb.LGBMClassifier(**lgm_param)
lgm.fit(train_X,train_y)
pred_lgm=lgm.predict_proba(test_X)

lgm=lgb.LGBMClassifier(**lgm_param)
lgm.fit(train_X_B,train_y)
pred_lgm_B=lgm.predict_proba(test_X_B)


#gbm

gbm_param = {'n_estimators':30,'max_depth':5, 'min_samples_split':1200, 'min_samples_leaf':30,'max_features':87}
gbm=GradientBoostingClassifier(**gbm_param )
gbm.fit(train_X,train_y)
pred_gbm=gbm.predict_proba(test_X)
gbm_param = {'n_estimators':120,'max_depth':5, 'min_samples_split':800,'min_samples_leaf':60}
gbm=GradientBoostingClassifier(**gbm_param )
gbm.fit(train_X_B,train_y)
pred_gbm_B=gbm.predict_proba(test_X_B)

pred_final=(pred_lgm+pred_xgb+pred_gbm)/3
pred_final_B=(pred_lgm_B+pred_xgb_B+pred_gbm_B)/3
predict_result_A=pd.DataFrame({'ccx_id':list(test_ccx_A.index),'prob':pred_final[:,1]})
predict_result_B=pd.DataFrame({'ccx_id':list(test_consumer_B_gp['ccx_id']),'prob':pred_final_B[:,1]})
###
#python文件结束部分
###
# 保存预测的结果 predict_result_A predict_result_B为您构建的模型预测出的概率和唯一索引构成的DataFrame
predict_result_A.to_csv('./predict_result_A.csv',encoding='utf-8',index=False)
predict_result_B.to_csv('./predict_result_B.csv',encoding='utf-8',index=False)

