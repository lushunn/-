

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
test_behavior_A=test_behavior_A[['ccx_id','var1','var6','var4','var3','var5']]
train_behavior_A=train_behavior_A[['ccx_id','var1','var6','var4','var3','var5']]
test_behavior_B=test_behavior_B[['ccx_id','var1','var6','var4','var3','var5']]
for i in ['var3','var4','var6']:
    test_behavior_A[i][test_behavior_A[i].isnull()]='novalue'
    test_behavior_B[i][test_behavior_B[i].isnull()]='novalue'
    train_behavior_A[i][train_behavior_A[i].isnull()]='novalue'   
    

for i in ['var4','var6']:
    tmp=train_behavior_A[i].value_counts()
    tmp1=tmp.cumsum()
    list1=tmp1[tmp1>0.95*21245].index
    list2=['lessCategorical' for i in range(0,len(list1))]
    map=dict(zip(list1,list2))
    train_behavior_A[i]=train_behavior_A[i].replace(map)
    test_behavior_A[i]=test_behavior_A[i].replace(map)
    test_behavior_B[i]=test_behavior_B[i].replace(map)

train_behavior_A_target= pd.merge(train_behavior_A, train_target_A,how = 'left',on='ccx_id')
#iv编码
for i in['var3','var4','var5','var6']:
    eps=0.00001
    a=pd.crosstab(train_behavior_A_target[i],train_behavior_A_target['target'])+eps
    b=train_behavior_A_target['target'].value_counts()+eps
    c=a/b
    c['woe']=np.log(c[1]/c[0])
    c['iv']=(c[1]-c[0])*c['woe']
    ix=c.index
    iv_value=list(c['iv'])
    map=dict(zip(ix,iv_value))
    train_behavior_A[i]=train_behavior_A[i].replace(map)
    test_behavior_A[i]=test_behavior_A[i].replace(map)
    test_behavior_A[i][test_behavior_A[i].apply(lambda x : type(x)!=float)]=float(0)
    test_behavior_A[i]=test_behavior_A[i].astype(float)
    test_behavior_B[i]=test_behavior_B[i].replace(map)
    test_behavior_B[i][test_behavior_B[i].apply(lambda x : type(x)!=float)]=float(0)
    test_behavior_B[i]=test_behavior_B[i].astype(float)
#consumer_A表     


#考虑RFM模型，以及前述变量重要性得分
##提取F 即消费频次

#提取R 即最后一次消费离基准时间的天数
##由于时间变量最近的时间是2017-5-31,而behavior表v19变量只含有2017-06-01取值，所以以2017-06-01算出最近消费相隔天数的基准天
from datetime import datetime
def chuli_consumer(test_consumer_A,over = datetime(2017, 6, 1)):
    test_consumer_A['V_11'][test_consumer_A['V_11']=='0000-00-00 00:00:00']=np.nan#0000-00-00 00:00:00设为缺失值
    test_consumer_A[['V_7','V_11']]=test_consumer_A[['V_7','V_11']].apply(pd.to_datetime)
    test_consumer_A['V_8'][test_consumer_A['V_8'].isnull()]='other'
    test_consumer_A['V_10'][test_consumer_A['V_10'].isnull()]='other'
    test_consumer_A['V_14'][test_consumer_A['V_14'].isnull()]='other'
    test_consumer_A['V_12'][test_consumer_A['V_12'].isnull()]=0
    test_consumer_A_gp=test_consumer_A['ccx_id'].groupby(test_consumer_A['ccx_id']).count()
    test_consumer_A_gp.columns=['count']
    
    test_consumer_A['datediff']=(over-test_consumer_A['V_7']).apply(lambda x: x.days)
    tmp=test_consumer_A['datediff'].groupby(test_consumer_A['ccx_id']).min()
    test_consumer_A_gp=pd.concat([test_consumer_A_gp,tmp],axis=1)
    test_consumer_A_gp.columns=['count', 'datediff']
    #提取M 即消费总金额
    tmp=test_consumer_A['V_5'].groupby(test_consumer_A['ccx_id']).sum()
    tmp1=test_consumer_A['V_5'].groupby(test_consumer_A['ccx_id']).mean()
    test_consumer_A_gp=pd.concat([test_consumer_A_gp,tmp,tmp1],axis=1)
    test_consumer_A_gp.columns=['count', 'datediff','v5_sum','v5_mean']
    #对重要性较高的变量进行提取
    ##对V12提取
    tmp=test_consumer_A['V_12'].groupby(test_consumer_A['ccx_id']).sum()
    test_consumer_A_gp=pd.concat([test_consumer_A_gp,tmp],axis=1)
    test_consumer_A_gp.columns=['count', 'datediff','v5_sum','v5_mean','v12_sum']
    #v3,v2
    tmp=test_consumer_A['V_2'].groupby(test_consumer_A['ccx_id']).apply(lambda x: x.describe()[2])
    tmp1=test_consumer_A['V_3'].groupby(test_consumer_A['ccx_id']).apply(lambda x: x.describe()[2])
    test_consumer_A_gp=pd.concat([test_consumer_A_gp,tmp,tmp1],axis=1)
    test_consumer_A_gp.columns=['count', 'datediff','v5_sum','v5_mean','v12_sum','v2_top','v3_top']
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
    eps=0.00001
    a=pd.crosstab(train_consumer_A_gp[i],train_consumer_A_gp['target'])+eps
    b=train_consumer_A_gp['target'].value_counts()+eps
    c=a/b
    c['woe']=np.log(c[1]/c[0])
    c['iv']=(c[1]-c[0])*c['woe']
    ix=c.index
    iv_value=list(c['iv'])
    map=dict(zip(ix,iv_value))
    train_consumer_A_gp[i]=train_consumer_A_gp[i].replace(map)
    test_consumer_A_gp[i]=test_consumer_A_gp[i].replace(map)
    test_consumer_A_gp[i][test_consumer_A_gp[i].apply(lambda x : type(x)!=float)]=float(0)
    test_consumer_A_gp[i]=test_consumer_A_gp[i].astype(float)
    test_consumer_B_gp[i][test_consumer_B_gp[i].apply(lambda x : type(x)!=float)]=float(0)
    test_consumer_B_gp[i]=test_consumer_B_gp[i].astype(float)
#ccx
#ccx
def chuli_consumer(train_ccx_A,train_target_A=train_target_A,over=datetime(2017, 6, 1)):
    #月份只有2017.1-2017.5的数据，只保留月份作为分类变量
    
    train_ccx_A[['var_06']]=train_ccx_A[['var_06']].apply(pd.to_datetime)
    train_ccx_A['datediff']=(over-train_ccx_A['var_06']).apply(lambda x: x.days)
    query = train_ccx_A.groupby(train_ccx_A['ccx_id']).size()#查询次数
    query = query.reset_index() #index 改为 column
    datediff=train_ccx_A['datediff'].groupby(train_ccx_A['ccx_id']).min()
    datediff=datediff.reset_index()
    query=pd.merge(query,datediff,on='ccx_id',how='left')
    query.columns = ['ccx_id','query','datediff']
    df = pd.get_dummies(train_ccx_A) #变成哑变量形式one-hot编码 特征增加了
    df2 = df.groupby(['ccx_id'],as_index=False).sum() #根据id汇总 加和
    

    df3 = pd.merge(df2,query,on='ccx_id',how='left')#query 和 df2合并
    df3 = pd.merge(train_target_A,df3,on='ccx_id',how='left')#target与ccx合并
    df4 = df3.drop(['target'], axis = 1) #只有数据没有target


    df4[np.isnan(df4)] = 0


    col =['ccx_id',  'var_01_C2', 'var_01_C3', 'var_02_T1',
       'var_02_T2', 'var_02_T3', 'var_02_T4', 'var_02_T5', 'var_02_T6',
       'var_02_T7', 'var_02_T77', 'var_02_T9', 'var_03_R0', 'var_03_R2',
       'var_03_R3', 'var_03_R4', 'var_03_R5', 'var_03_R6', 'var_04_B1',
       'var_04_B2', 'var_04_B3', 'var_04_B4', 'var_04_B5', 'var_04_B7',
       'var_05_A0', 'var_05_A1', 'var_05_A2', 'var_05_A3', 'query',
       'datediff_y']
    df4 = df4[col]
    







    df4 = df4.set_index("ccx_id")
    
    return df4
train_ccx_A=chuli_consumer(train_ccx_A)
train_ccx_A['ccx_id']=train_ccx_A.index
tmp=list(test_consumer_A_gp.index)
tmp1=[0 for i in range(len (tmp))]
fake_target=pd.DataFrame({'ccx_id':tmp,'target':tmp1})
test_ccx_A=chuli_consumer(test_ccx_A,train_target_A=fake_target)
test_ccx_A['ccx_id']=test_ccx_A.index

train_X=pd.merge(train_consumer_A_gp,train_behavior_A,on='ccx_id')
train_X=pd.merge(train_X,train_ccx_A,on='ccx_id').drop('ccx_id',axis=1)
train_y=train_X['target']
del train_X['target']
test_X=pd.merge(test_consumer_A_gp,test_behavior_A,on='ccx_id')
test_X=pd.merge(test_X,test_ccx_A,on='ccx_id').drop('ccx_id',axis=1)


train_X_B=pd.merge(train_consumer_A_gp,train_behavior_A,on='ccx_id')
del train_X_B['target']
test_X_B=pd.merge(test_consumer_B_gp,test_behavior_B,on='ccx_id')
list1=['v5_mean',
 'v3_top',
 'v5_sum',
 'datediff',
 'var6',
 'v2_top',
 'var1',
 'var4',
 'v12_sum',
 'count',
 'var5',
 'var3',
 'datediff_y',
 'query',
 'var_01_C2',
 'var_01_C3',
 'var_03_R0',
 'var_05_A0',
 'var_03_R5',
 'var_03_R4',
 'var_02_T1',
 'var_05_A3',
 'var_04_B1',
 'var_03_R2',
 'var_05_A2',
 'var_02_T2',
 'var_04_B5',
 'var_03_R3',
 'var_05_A1',
 'var_04_B2',
 'var_04_B7',
 'var_04_B4',
 'var_04_B3',
 'var_03_R6',
 'var_02_T77',
 'var_02_T5']
train_X=train_X[list1]
test_X=test_X[list1]
#
from sklearn.ensemble import GradientBoostingClassifier
#运行测试 
import xgboost as xgb
import lightgbm as lgb
#xgb
xgb_param= {
 'max_depth':5,
 'min_child_weight':9,'gamma':0.7,'subsample':0.9,
 'colsample_bytree':0.4,'reg_alpha':1,'learning_rate':0.09
}
xgm = xgb.XGBClassifier(**xgb_param)
xgm.fit(train_X,train_y)
pred_xgb=xgm.predict_proba(test_X)
pred_xgb=pred_xgb


xgm = xgb.XGBClassifier(**xgb_param)
xgm.fit(train_X_B,train_y)
pred_xgb_B=xgm.predict_proba(test_X_B)


#lgb


lgm_param = {'num_leaves' : 10,'max_depth':6,'max_bin':400,'min_data_in_leaf':45,
             'feature_fraction ':0.0, 'bagging_fraction':1.0,
             'lambda_l1':0.0,'lambda_l2':0.6,'n_estimators':100}
lgm=lgb.LGBMClassifier(**lgm_param)
lgm.fit(train_X,train_y)
pred_lgm=lgm.predict_proba(test_X)

lgm=lgb.LGBMClassifier(**lgm_param)
lgm.fit(train_X_B,train_y)
pred_lgm_B=lgm.predict_proba(test_X_B)


#gbm

gbm_param = {'n_estimators':120,'max_depth':5, 'min_samples_split':800,'min_samples_leaf':60,'max_features':18}
gbm=GradientBoostingClassifier(**gbm_param )
gbm.fit(train_X,train_y)
pred_gbm=gbm.predict_proba(test_X)
gbm_param = {'n_estimators':120,'max_depth':5, 'min_samples_split':800,'min_samples_leaf':60,'max_features':10}
gbm=GradientBoostingClassifier(**gbm_param )
gbm.fit(train_X_B,train_y)
pred_gbm_B=gbm.predict_proba(test_X_B)

pred_final=(0.33375*pred_gbm+0.33375*pred_lgm+0.33290*pred_xgb)
pred_final_B=(pred_gbm_B+pred_lgm_B+pred_xgb_B)/3
predict_result_A=pd.DataFrame({'ccx_id':list(test_ccx_A.index),'prob':pred_final[:,1]})
predict_result_B=pd.DataFrame({'ccx_id':list(test_consumer_B_gp['ccx_id']),'prob':pred_final_B[:,1]})
###
#python文件结束部分
###
# 保存预测的结果 predict_result_A predict_result_B为您构建的模型预测出的概率和唯一索引构成的DataFrame
predict_result_A.to_csv('./predict_result_A.csv',encoding='utf-8',index=False)
predict_result_B.to_csv('./predict_result_B.csv',encoding='utf-8',index=False)