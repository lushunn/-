# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:04:05 2018

@author: dongjing
"""

import os
os.chdir('C:\\Users\\dongjing\\Desktop\\game\\ml\\data\\train_scene_A')
os.getcwd()

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
train_consumer_A=pd.read_csv("./train/scene_A/train_consumer_A.csv")
train_behavior_A=pd.read_csv("./train/scene_A/train_behavior_A.csv")
train_ccx_A=pd.read_csv("./train/scene_A/train_ccx_A.csv")
train_target_A=pd.read_csv("./train/scene_A/train_target_A.csv")
from datetime import datetime
over=datetime(2017, 6, 1)
train_ccx_A[['var_06']]=train_ccx_A[['var_06']].apply(pd.to_datetime)
train_ccx_A['datediff']=(over-train_ccx_A['var_06']).apply(lambda x: x.days)
#月份只有2017.1-2017.5的数据，只保留月份作为分类变量
train_ccx_A['var_06'].describe()
month = [re.split('-',str(line))[1] for line in train_ccx_A['var_06']]
train_ccx_A['var_06'] = month



def chuli_consumer(train_ccx_A=train_ccx_A,train_target_A=train_target_A,over=datetime(2017, 6, 1)):
    #月份只有2017.1-2017.5的数据，只保留月份作为分类变量
    
    train_ccx_A[['var_06']]=train_ccx_A[['var_06']].apply(pd.to_datetime)
    train_ccx_A['datediff']=(over-train_ccx_A['var_06']).apply(lambda x: x.days)
    query = train_ccx_A.groupby(train_ccx_A['ccx_id']).size()#查询次数
    query = query.reset_index() #index 改为 column
     #最后一次消费距分析时间距离
    datediff=train_ccx_A['datediff'].groupby(train_ccx_A['ccx_id']).min()
    datediff=datediff.reset_index()
    query=pd.merge(query,datediff,on='ccx_id',how='left')
    query.columns = ['ccx_id','query','datediff']
	#衍生变量 消费频率
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


