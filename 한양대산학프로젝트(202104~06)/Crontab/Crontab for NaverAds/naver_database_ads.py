#!/usr/bin/env python
# coding: utf-8

# # ↓ 집행중인 캠페인 이름 작성

# In[470]:


## 집행중인 캠페인 목록

camps_list = ['nyw', 'Hoon', 'jungwon', 'sumin']


# In[471]:


import time
import random
import requests
import json
import signaturehelper
import pandas as pd
from pandas import DataFrame
import datetime
from datetime import date, timedelta


BASE_URL = 'https://api.naver.com'
API_KEY = ''
SECRET_KEY = ''
CUSTOMER_ID = ''    #smartmind


# In[472]:


def get_header(method, uri, api_key, secret_key, customer_id):
    timestamp = str(round(time.time() * 1000))
    signature = signaturehelper.Signature.generate(timestamp, method, uri, SECRET_KEY)
    return {'Content-Type': 'application/json; charset=UTF-8', 'X-Timestamp': timestamp,
          'X-API-KEY': API_KEY, 'X-Customer': str(CUSTOMER_ID), 'X-Signature': signature}



def get_campaign_list(CUSTOMER_ID):
    uri = '/ncc/campaigns'
    method = 'GET'
    r = requests.get(BASE_URL + uri, params={'Id':CUSTOMER_ID}, headers=get_header(method, uri, API_KEY, SECRET_KEY, CUSTOMER_ID))
    CampID_list = []
    CampName_list = []

    for i in range(len(r.json())):
        if r.json()[i]['name'] in camps_list:
            CampID_list.append(r.json()[i]['nccCampaignId'])
            CampName_list.append(r.json()[i]['name'])
            
    return CampID_list, CampName_list



def get_adgroup_list(cmp_list):
    uri = '/ncc/adgroups'
    method = 'GET'
    adg_camp_match = {}
    adgroup_Id_list = []
    adgroup_Name_list = []
    
    for i in range(len(cmp_list)):
        r = requests.get(BASE_URL + uri, params={'nccCampaignId':cmp_list[i]},
                         headers=get_header(method, uri, API_KEY, SECRET_KEY, CUSTOMER_ID))
        
        for j in range(len(r.json())):
#             if r.json()[j]['status'] == 'ELIGIBLE':
            adgroup_Id_list.append(r.json()[j]['nccAdgroupId'])
            adgroup_Name_list.append(r.json()[j]['name'])
            adg_camp_match[r.json()[j]['nccAdgroupId']] = r.json()[j]['nccCampaignId']     
                
    return adgroup_Id_list, adgroup_Name_list, adg_camp_match



# def get_ads_list(grp_list):
#     uri = '/ncc/ads'
#     method = 'GET'
#     ad_adg_match = {}
#     ad_Id_list = []
#     ad_name_list = []
    
#     for i in range(len(grp_list)):
#         r = requests.get(BASE_URL + uri, params={'nccAdgroupId':grp_list[i]},
#                          headers=get_header(method, uri, API_KEY, SECRET_KEY, CUSTOMER_ID))
        
#         for j in range(len(r.json())):
# ##             if r.json()[j]['status'] == 'ELIGIBLE' and r.json()[j]['nccAdgroupId'] in grp_list:
#             if r.json()[j]['nccAdgroupId'] in grp_list:
#                 ad_Id_list.append(r.json()[j]['nccAdId'])
#                 ad_name_list.append(r.json()[j]['ad']['headline'])
#                 ad_adg_match[r.json()[j]['nccAdId']] = r.json()[j]['nccAdgroupId']
                
#     return ad_Id_list, ad_name_list, ad_adg_match



def get_ads_stat(nad_list, days_list):
    uri = '/stats'
    method = 'GET'
    r = requests.get(BASE_URL + uri, params={'ids': nad_list, 
                                             'fields': '["impCnt", "clkCnt", "salesAmt", "ctr", "cpc", "avgRnk", "ccnt", "pcNxAvgRnk", "mblNxAvgRnk", "crto"]', 
                                             'timeRange': days_list,
                                             'breakdown':'hh24'}, 
                     headers=get_header(method, uri, API_KEY, SECRET_KEY, CUSTOMER_ID))
    return r.json()



def match_id_name(id_list, name_list):
    new_dict = dict(zip(id_list, name_list))
    return new_dict


# # Loading Data

# In[473]:


camp_Id_list = get_campaign_list(CUSTOMER_ID)[0]
camp_Name_list = get_campaign_list(CUSTOMER_ID)[1]
camp_data = match_id_name(camp_Id_list, camp_Name_list)

Adgroup_Id_list = get_adgroup_list(camp_Id_list)[0]
Adgroup_Name_list = get_adgroup_list(camp_Id_list)[1]
adg_camp_match = get_adgroup_list(camp_Id_list)[2]
adgroup_data = match_id_name(Adgroup_Id_list, Adgroup_Name_list)



### 불러올 DataFrame 컬럼 목록 생성
fields_data = ["date", "campaign_id", "campaign_name", "adgroup_id", "adgroup_name"]
fields_stat = ["name", "impCnt", "clkCnt", "salesAmt", "ctr", "cpc", "avgRnk", "ccnt", "pcNxAvgRnk", "mblNxAvgRnk", "crto"]


# ## last7days stat

# In[474]:


### today 기준, last7days 리스트 생성
days_list = []
for i in reversed(range(1, 8)):
    day = (date.today() - timedelta(days=i)).strftime('%Y-%m-%d')
    during = '{'+'"since":"'+day+'", "until":"'+day+'"}'
    days_list.append(during)


# In[475]:


df_01=DataFrame(None,columns=fields_data)
df_02=DataFrame(None,columns=fields_stat)

for i in range(len(days_list)):
    thisday = days_list[i][10:20]
    oneday_data = get_ads_stat(Adgroup_Id_list, days_list[i])['data']
    
    for j in range(len(oneday_data)):
        Adgroup_id = oneday_data[j]['id']
        Adgroup_name = adgroup_data.get(Adgroup_id)
        Campaign_id = adg_camp_match.get(Adgroup_id)
        Campaign_name = camp_data.get(Campaign_id)

        infos = {'campaign_id':Campaign_id, 'campaign_name':Campaign_name, 'adgroup_id':Adgroup_id, 'adgroup_name':Adgroup_name,
                 'date':thisday}

        for k in range(len(oneday_data[j]['breakdowns'])):
            df_02=df_02.append(DataFrame(oneday_data[j]['breakdowns'][k],index=[k]))
            df_01=df_01.append(DataFrame(infos, index=[k]))


df1 = pd.concat([df_01, df_02], axis=1)
df1['name'] = df1['name'].str.split('시').str[0]             ### "name"컬럼 전처리 (00시~01시 -> 00)
df1.rename(columns = {'name' : 'hour'}, inplace = True)     ### "name"컬럼명 변경 (name -> hour)


# ## today stat

# In[476]:


df_03=DataFrame(None,columns=fields_data)
df_04=DataFrame(None,columns=fields_stat)
                
uri = '/stats'
method = 'GET'
r = requests.get(BASE_URL + uri,
                 params={'ids': Adgroup_Id_list, 'fields': '["impCnt", "clkCnt", "salesAmt", "ctr", "cpc", "avgRnk", "ccnt", "pcNxAvgRnk", "mblNxAvgRnk", "crto"]',
                         'datePreset': 'today',
                         'breakdown': 'hh24'},
                 headers=get_header(method, uri, API_KEY, SECRET_KEY, CUSTOMER_ID))
today_stat = r.json()

for m in range(len(today_stat['data'])):
    Adgroup_id =today_stat['data'][m]['id']
    Adgroup_name = adgroup_data.get(Adgroup_id)
    Campaign_id = adg_camp_match.get(Adgroup_id)
    Campaign_name = camp_data.get(Campaign_id)
    
    today = date.today().strftime('%Y-%m-%d')
    info = {'campaign_id':Campaign_id, 'campaign_name':Campaign_name, 'adgroup_id':Adgroup_id, 'adgroup_name':Adgroup_name,
            'date':today}
    
    for n in range(len(today_stat['data'][m]['breakdowns'])):
        df_04=df_04.append(DataFrame(today_stat['data'][m]['breakdowns'][n],index=[n]))
        df_03=df_03.append(DataFrame(info, index=[n]))
        

df2 = pd.concat([df_03, df_04], axis=1)
df2['name'] = df2['name'].str.split('시').str[0]             ### "name"컬럼 전처리 (00시~01시 -> 00)
df2.rename(columns = {'name' : 'hour'}, inplace = True)     ### "name"컬럼명 변경 (name -> hour)


# # Preprocessing

# ## 1) 값이 없는 시간대에 Null 추가한 DataFrame 생성

# ### last7days(df1) -> df_plus1

# In[477]:


fields_all = df1.columns.tolist()

hour_24 = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
           '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']


# In[478]:


df_plus1=DataFrame(None, columns=fields_all)

for i in df1['adgroup_id'].unique():
    
    for j in df1[df1['adgroup_id'] == i]['date'].unique():
        hourcheck_data = df1[(df1['adgroup_id'] == i) & (df1['date'] == j)]['hour'].tolist()

        if len(hourcheck_data) != 24:
            Adgroup_id = i
            Adgroup_name = adgroup_data.get(i)
            Campaign_id = adg_camp_match.get(i)
            Campaign_name = camp_data.get(Campaign_id)

            for k in hour_24:
                if k not in hourcheck_data:
                    plus_data = {'campaign_id':Campaign_id, 'campaign_name':Campaign_name,
                                 'adgroup_id':Adgroup_id, 'adgroup_name':Adgroup_name,
                                 'date':j, 'hour':k}
                    df_plus1=df_plus1.append(DataFrame(plus_data, index=[k]))


# ### today(df2) -> df_plus2

# In[479]:


df_plus2=DataFrame(None, columns=fields_all)

for i in df2['adgroup_id'].unique():
    hourcheck_data = df2[df2['adgroup_id'] == i]['hour'].tolist()

    if len(hourcheck_data) != 24:
        Adgroup_id = i
        Adgroup_name = adgroup_data.get(Adgroup_id)
        Campaign_id = adg_camp_match.get(Adgroup_id)
        Campaign_name = camp_data.get(Campaign_id)

        for j in hour_24:
            if (j not in hourcheck_data) and (j <= df2['hour'].max()):
                plus_data = {'campaign_id':Campaign_id, 'campaign_name':Campaign_name, 'adgroup_id':Adgroup_id, 
                             'adgroup_name':Adgroup_name, 'date':today, 'hour':j}
                df_plus2=df_plus2.append(DataFrame(plus_data, index=[j]))


# ## 2) 전체 데이터 concat (last7days + today) -> df

# In[480]:


df = pd.concat([df1,df_plus1, df2, df_plus2])


# ## 3) date_hour 생성 및 정렬

# In[481]:


df['date_hour'] = df['date'] + " " + df['hour'] + ":00:00"
df['date_hour'] = pd.to_datetime(df['date_hour'])
df = df.sort_values(by=['date', 'campaign_id', 'hour']).reset_index().drop('index', axis=1)
# df = df.drop_duplicates(subset=None, keep='first', ignore_index=False)


# In[482]:


df


# In[483]:


#df.to_csv('test_0513.csv')


# In[484]:


df[(df['date'] =='2021-05-21') & (df['campaign_name'] =='Hoon') ]


# In[485]:


# df.to_excel('test_0513.xlsx')
aa_naver_ads = df
aa_naver_ads


# In[486]:


from sqlalchemy.types import NVARCHAR, Float, Date, DateTime, Numeric, BigInteger
import pyodbc
from sqlalchemy import create_engine


# In[487]:


server = 'vodasql.database.windows.net'
database = 'GateWay-DSP' 
username = 'smartmind'
password = '1!tmakxmakdlsem' 
engn = create_engine(f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server")
col = { "date": Date,
"campaign_id": NVARCHAR(length="30"),
"campaign_name" :NVARCHAR(length="20"),
"adgroup_id" :NVARCHAR(length="30"),
"adgroup_name":NVARCHAR(length="30"),
"hour": NVARCHAR(length="30"),
"impCnt" :BigInteger,
"clkCnt" :BigInteger,
"salesAmt" :Numeric,
"ctr" :Float,
"cpc" :Numeric,
"avgRnk" :Float,
"ccnt" :Numeric,
"pcNxAvgRnk" :Float,
"mblNxAvgRnk" :Float,
"crto" :Numeric,
"date_hour" :DateTime
       
}


# In[495]:


from sqlalchemy import create_engine, select, delete, text

with engn.begin() as conn:
    conn.execute(text('''DELETE FROM dbo.aa_naver_ads
     WHERE date = (SELECT CONVERT(VARCHAR(50), GETDATE(), 23) AS 'TODAY')
         '''))


# In[490]:


import pandas as pd
from tqdm import tqdm

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def insert_with_progress(aa_naver_ads):
    con = create_engine(f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server", fast_executemany = True)
    aa_naver_ads.to_sql("aa_naver_ads", con=con, if_exists="append", index=False, dtype = col )
    
insert_with_progress(aa_naver_ads)


# In[1]:


import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
import numpy as np
import sys
from sqlalchemy import create_engine
import pyodbc 

driver = 'ODBC Driver 17 for SQL Server'
server = 'vodasql.database.windows.net'
database = 'GateWay-DSP'
username = 'smartmind'
password = '1!tmakxmakdlsem'

conn = pyodbc.connect(f'Driver={driver};'
                      f'Server={server};'
                      f'Database={database};'
                      f'uid={username};'
                      f'pwd={password}')

cursor = conn.cursor()

df1 = pd.read_sql_query('''SELECT * FROM dbo.aa_naver_ads 
                        ''',conn)
df1


# In[333]:


from sqlalchemy import create_engine, select, delete, text


# In[498]:


with engn.begin() as conn:
    conn.execute(text('''DELETE A 
FROM(
SELECT [campaign_name], date_hour, ROW_NUMBER() OVER(PARTITION BY [campaign_name], date_hour ORDER BY impCnt DESC) AS RN
FROM aa_naver_ads) A  WHERE RN > 1
         '''))

