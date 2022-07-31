from sqlalchemy import create_engine
import pyodbc 
from scipy.stats import beta, bernoulli
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
import random
import math
import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, parsemode, ReplyKeyboardMarkup
from telegram.ext import Updater, MessageHandler, Filters, CommandHandler, CallbackQueryHandler, Dispatcher,  ConversationHandler  # import modules
import time
import datetime
from functools import partial
import json
import logging
import sqlite3
import sys
import csv
import re
import requests
import urllib.parse as urlparse
from bs4 import BeautifulSoup as bs
import os
import threading
from datetime import datetime, timedelta
import numpy as np
import time
import signaturehelper
import pandas as pd
from pandas import DataFrame
from pandas import json_normalize
from datetime import date, timedelta


def get_header(method, uri, api_key, secret_key, customer_id):
    timestamp = str(round(time.time() * 1000))
    signature = signaturehelper.Signature.generate(timestamp, method, uri, secret_key)
    return {'Content-Type': 'application/json; charset=UTF-8', 'X-Timestamp': timestamp, 'X-API-KEY': api_key, 'X-Customer': str(customer_id), 'X-Signature': signature}


def RS(ads, ACTUAL_CTR): #Random_Selection
    n = 1000
    regret = 0 
    total_reward = 0
    regret_list = [] # list for collecting the regret values for each impression (trial)
    ctr = {}
    chosen_ads = [] # list for collecting the number of randomly choosen Ad
    impressions = {}
    clicks = {}
    for i in ads:
        ctr[f'{i}'] = []
        impressions[f'{i}'] = 0
        clicks[f'{i}'] = 0
    p = []
    for i in range(len(ads)):
        p.append(1/len(ads))
    for i in range(n):
        random_ad = np.random.choice(ads, p=p) # randomly choose the ad
        chosen_ads.append(random_ad) # add the value to list

        impressions[random_ad] += 1 # add 1 impression value for the choosen Ad
        did_click = bernoulli.rvs(ACTUAL_CTR[random_ad]) # simulate if the person clicked on the ad usind Actual CTR value

        if did_click:
            clicks[random_ad] += did_click # if person clicked add 1 click value for the choosen Ad

        # calculate the CTR values and add them to list
        ctr_0 = []

        for i in ads:
            if impressions[i] == 0:
                ctr_0.append(0)
            else:
                ctr_0.append(clicks[i]/impressions[i])

        for j in range(len(ads)):
            ctr[ads[j]].append(ctr_0[j])

        # calculate the regret and reward
        regret += max(ACTUAL_CTR.values()) - ACTUAL_CTR[random_ad]
        regret_list.append(regret)
        total_reward += did_click
    
    random_dict = {
        'reward':total_reward, 
        'regret_list':regret_list, 
        'ads_count':pd.Series(chosen_ads).value_counts(normalize=True)
    }
    return random_dict
    
def EG(ads,ACTUAL_CTR): # Epsilon_Greedy e값,시도횟수도 받을까?  고정할까?
    e = .15 # set the Epsilon value
    n_init = 1000 # number of impressions to choose the winning Ad

    # set the initial values for impressions and clicks 
    impressions = {}
    clicks = {}
    for i in ads:
        impressions[f'{i}'] = 0
        clicks[f'{i}'] = 0

    p = []
    for i in range(len(ads)):
        p.append(1/len(ads))

    for i in range(n_init):
        random_ad = np.random.choice(ads, p=p) # randomly choose the ad

        impressions[random_ad] += 1
        did_click = bernoulli.rvs(ACTUAL_CTR[random_ad])
        if did_click:
            clicks[random_ad] += did_click

    ctr_0 = []
    for i in ads:
        ctr_0.append(clicks[i]/impressions[i])

    win_index = np.argmax(ctr_0) # select the Ad number with the highest CTR

    regret = 0
    total_reward = 0
    regret_list = []
    chosen_ads = []
    ctr = {}
    impressions = {}
    clicks = {}
    for i in ads:
        ctr[f'{i}'] = []
        impressions[f'{i}'] = 0
        clicks[f'{i}'] = 0    

    p = np.full(len(ads), 1/len(ads))
    p[:] = e / (len(p) - 1)
    p[win_index] = 1 - e

    for i in range(n_init):    
        win_ad = np.random.choice(ads, p=p) # randomly choose the ad # p는 뽑힐 확률
        chosen_ads.append(win_ad) # add the value to list

        impressions[win_ad] += 1 # add 1 impression value for the choosen Ad
        did_click = bernoulli.rvs(ACTUAL_CTR[win_ad]) # simulate if the person clicked on the ad usind Actual CTR value

        if did_click:
            clicks[win_ad] += did_click # if person clicked add 1 click value for the choosen Ad

        for i in ads:
            if impressions[i] == 0:
                ctr_0.append(0)
            else:
                ctr_0.append(clicks[i]/impressions[i])

        for j in range(len(ads)):
            ctr[ads[j]].append(ctr_0[j])

        # calculate the regret and reward
        regret += max(ACTUAL_CTR.values()) - ACTUAL_CTR[win_ad]
        regret_list.append(regret)
        total_reward += did_click
    
    epsilon_dict = {
        'reward':total_reward, 
        'regret_list':regret_list, 
        'ads_count':pd.Series(chosen_ads).value_counts(normalize=True)}
    
    return epsilon_dict

def TS(ads,ACTUAL_CTR): #Thompson_Sampling
    n = 1000
    chosen_ads = []
    regret = 0 
    total_reward = 0
    regret_list = [] 
    index_list = [] 
    ctr = {}
    impressions = {}
    clicks = {}
    priors = {}
    for i in ads:
        ctr[f'{i}'] = []
        impressions[f'{i}'] = 0
        clicks[f'{i}'] = 0
        priors[f'{i}'] = 1  # prior 정보 없어서 1로 사용

    p = []
    for i in range(len(ads)):
        p.append(1/len(ads))
    win_ad = np.random.choice(ads, p=p) ## randomly choose the first shown Ad
    for i in range(n):    
        impressions[win_ad] += 1
        did_click = bernoulli.rvs(ACTUAL_CTR[win_ad])
        if did_click:
            clicks[win_ad] += did_click

        ctr_0 = []
        for j in range(len(ads)):
            if j == 0:
                ctr_0.append(random.betavariate(priors[ads[0]]+clicks[ads[j]], priors[ads[j+1]] + impressions[ads[j]] - clicks[ads[j]]))
            else:
                ctr_0.append(random.betavariate(priors[ads[0]]+clicks[ads[j]], priors[ads[j]] + impressions[ads[j]] - clicks[ads[j]]))
        win_ad = ads[np.argmax(ctr_0)]

        chosen_ads.append(win_ad)

        for k in range(len(ads)):
            ctr[ads[k]].append(ctr_0[k])
        
        regret += max(ACTUAL_CTR.values()) - ACTUAL_CTR[win_ad]
        regret_list.append(regret)    
        total_reward += did_click
    thompson_dict = {
        'reward':total_reward, 
        'regret_list':regret_list, 
        'ads_count':pd.Series(chosen_ads).value_counts(normalize=True)}

    return thompson_dict

def UCB(ads,ACTUAL_CTR): # Upper_Confidence_Bound
    n = 1000
    regret = 0 
    total_reward = 0
    regret_list = [] 
    index_list = [] 
    ctr = {}
    impressions = {}
    clicks = {}
    for i in ads:
        ctr[f'{i}'] = []
        impressions[f'{i}'] = 0
        clicks[f'{i}'] = 0

    for i in range(n):
        win_ad = 0
        max_upper_bound = 0
        for k in ads:
            if (impressions[k] > 0):
                CTR = clicks[k] / impressions[k]
                delta = math.sqrt(2 * math.log(i+1) / impressions[k])
                upper_bound = CTR + delta
                ctr[k].append(CTR)
            else:
                upper_bound = 1e400
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                win_ad = k
        index_list.append(win_ad)
        impressions[win_ad] += 1
        reward = bernoulli.rvs(ACTUAL_CTR[win_ad])
        
        clicks[win_ad] += reward
        total_reward += reward
        regret += max(ACTUAL_CTR.values()) - ACTUAL_CTR[win_ad]
        regret_list.append(regret)
    ucb1_dict = {
    'reward':total_reward, 
    'regret_list':regret_list, 
    'ads_count':pd.Series(index_list).value_counts(normalize=True)}
    return ucb1_dict


def KPI():
    BASE_URL = 'https://api.naver.com'
    user_id = 1459643056
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT api_key,secret_key,customer_id FROM naver WHERE telegram_id = {}'.format(user_id))
    fetch = cur.fetchone()
    api_key = fetch[0]
    secret_key = fetch[1]
    customer_id = fetch[2]
    cur.execute('SELECT telegram_id FROM naver WHERE api_key = "{}" AND secret_key = "{}" AND customer_id = {} '.format(fetch[0],fetch[1],fetch[2]))
    fetch = cur.fetchall()
    cur.execute('SELECT daily_budget,grp_id,w_cpc,w_cpm,w_ctr FROM kpi WHERE customer_id = {}'.format(customer_id))
    fet = cur.fetchall()
    maxbudget = int(fet[0][0])
    grp_id_list = fet[0][1]
    w1 = float(fet[0][2])
    w2 = float(fet[0][3])
    w3 = float(fet[0][4])
    send_list = []
    for i in range(len(fetch)):
        send_list.append(fetch[i][0])
    
    grp_id_list = grp_id_list.split(',')
    driver = 'ODBC Driver 17 for SQL Server'
    server = 'vodasql.database.windows.net'
    database = 'GateWay-DSP'
    username = 'smartmind'
    password = '1!tmakxmakdlsem'
    conn2 = pyodbc.connect(f'Driver={driver};'
                          f'Server={server};'
                          f'Database={database};'
                          f'uid={username};'
                          f'pwd={password}')
    cursor = conn2.cursor()
    
    
    now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    today = date.today().strftime('%Y-%m-%d')
    yesterday = (date.today() - timedelta(1)).strftime('%Y-%m-%d')
    df = pd.read_sql_query(f'''SELECT * FROM dbo.aa_naver_ads WHERE date = '{yesterday}' order by 'date_hour'
                            ''',conn2)
    data = df.copy()
    data = data.loc[:, ['date','hour','adgroup_id', 'cpc','clkCnt','impCnt','ctr', 'salesAmt']]

    data[['cpc', 'salesAmt','impCnt','clkCnt']] = data[['cpc', 'salesAmt','impCnt','clkCnt']].apply(pd.to_numeric)

    data.sort_values(by=['adgroup_id', 'date', 'hour'], inplace=True)
    data = data.reset_index().drop('index', axis=1)

    ##### CTR #####
    ctr_dict = {}
    ctrList = []
    ads = grp_id_list
    for i in grp_id_list:
        impressions = df.loc[df['adgroup_id'] == i, :]['impCnt'].astype('float').sum()
        clicks = df.loc[df['adgroup_id'] == i, :]['clkCnt'].astype('float').sum()
        ctr = math.ceil(clicks / impressions * 10000)/100
        ctrList.append(ctr)
        ctr_dict[i] = ctr
    check_num = 0
    for i in ctrList:
        if i >1:
            check_num += 1
    if check_num != 0 :
        min_max_scaler2 = MinMaxScaler()
        ctrList = np.array(ctrList)
        ctrList = min_max_scaler2.fit_transform(ctrList.reshape(-1, 1))
        ctrList2 = []
        for i in range(4):
            ctrList2.append(ctrList[i][0])
        ctrList = ctrList2
    ##### CPC #####

    cpc_list = []
    for i in grp_id_list:
        cost = data.loc[data['adgroup_id'] == i, :]['salesAmt'].sum()
        clicks = data.loc[data['adgroup_id'] == i, :]['clkCnt'].sum()
        if cost == 0 and clicks == 0:
            cpc = 0
        else:
            cpc = round(cost / clicks)
        cpc_list.append(cpc)

    ##### CPM #####

    cpm_list = []
    for i in grp_id_list:
        cost = data.loc[data['adgroup_id'] == i, :]['salesAmt'].sum()
        impressions = data.loc[data['adgroup_id'] == i, :]['impCnt'].sum()
        if cost == 0 and impressions == 0:
            cpm = 0
        else:
            cpm = round(cost / impressions * 1000)
        cpm_list.append(cpm)        
    cols = grp_id_list
    kpi_df = pd.DataFrame(None, columns=cols)
    
    kpi_df.loc['ctr'] = ctrList
    kpi_df.loc['cpm'] = cpm_list
    kpi_df.loc['cpc'] = cpc_list

    kpi_df = kpi_df.T

    kpi_df["1/cpc"] = 1 / kpi_df['cpc']
    kpi_df["1/cpm"] = 1 / kpi_df['cpm']

    kpi_df.replace([np.inf, -np.inf], 0, inplace=True)

    scaler_icpc = StandardScaler()
    scaler_icpm = StandardScaler()
    scaler_ctr = StandardScaler()
    
    scaler_icpc.fit(kpi_df["1/cpc"].values.reshape(-1,1))
    kpi_df["st_1/cpc"] = scaler_icpc.transform(kpi_df["1/cpc"].values.reshape(-1,1))

    scaler_icpm.fit(kpi_df["1/cpm"].values.reshape(-1,1))
    kpi_df["st_1/cpm"] = scaler_icpm.transform(kpi_df["1/cpm"].values.reshape(-1,1))

    scaler_ctr.fit(kpi_df["ctr"].values.reshape(-1,1))
    kpi_df["st_ctr"] = scaler_ctr.transform(kpi_df["ctr"].values.reshape(-1,1))
    
    min_max_scaler = MinMaxScaler()

    min_max_scaler.fit(kpi_df['st_1/cpc'].values.reshape(-1,1))
    kpi_df['minmax_1/cpc'] = min_max_scaler.transform(kpi_df['st_1/cpc'].values.reshape(-1,1))

    min_max_scaler.fit(kpi_df['st_1/cpm'].values.reshape(-1,1))
    kpi_df['minmax_1/cpm'] = min_max_scaler.transform(kpi_df['st_1/cpm'].values.reshape(-1,1))

    min_max_scaler.fit(kpi_df['st_ctr'].values.reshape(-1,1))
    kpi_df['minmax_ctr'] = min_max_scaler.transform(kpi_df['st_ctr'].values.reshape(-1,1))

    kpi_df["KPI"] = (w1* kpi_df["minmax_1/cpc"]) + (w2* kpi_df["minmax_1/cpm"]) + (w3 * kpi_df["minmax_ctr"])

    budget_list2 = (round(kpi_df['KPI'] * (1 / sum(kpi_df['KPI'])), 2) * maxbudget).tolist()
    budget_list =[]
    for i in budget_list2:
        if i >70:
            budget_list.append(int(i))
        else:
            budget_list.append(70)
    ads = {}
    for i in range(len(budget_list)):
        ads[grp_id_list[i]] = budget_list[i]

    for k,v in ads.items():
        uri = '/ncc/adgroups/' + k
        method = 'GET'
        r = requests.get(BASE_URL + uri, headers=get_header(method, uri, api_key, secret_key, customer_id))
        create = r.json()
        method = 'PUT'
        create['dailyBudget'] = v
        r = requests.put(BASE_URL + uri, params={'nccAdgroupId': k}, json=create, headers=get_header(method, uri, api_key, secret_key, customer_id))

    row = ''
    for k,v in ads.items():
        row += df.loc[df['adgroup_id'] == k,'adgroup_name'].unique()[0][:14]+' : '+str(ads.get(k))+'원\n'
    row = row.replace('#','_')
    
    for id in send_list:
        requests.get(f"https://api.telegram.org/bot1741553288:AAFjvybYZ3n4mnQJcUVbSgMIm64WGiTK6qo/sendmessage?chat_id={id}&text={today} KPI 예산분배 가중치\n클릭당 비용 : {w1}\n1000회 노출당 비용 : {w2}\n클릭률 : {w3}\n\n{row}")
    

def MAB():
    BASE_URL = 'https://api.naver.com'
    user_id = 1459643056
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT api_key,secret_key,customer_id FROM naver WHERE telegram_id = {}'.format(user_id))
    fetch = cur.fetchone()
    api_key = fetch[0]
    secret_key = fetch[1]
    customer_id = fetch[2]
    cur.execute('SELECT telegram_id FROM naver WHERE api_key = "{}" AND secret_key = "{}" AND customer_id = {} '.format(fetch[0],fetch[1],fetch[2]))
    fetch = cur.fetchall()
    cur.execute('SELECT daily_budget,grp_id FROM kpi WHERE customer_id = {}'.format(customer_id))
    fet = cur.fetchall()
    maxbudget = fet[0][0]
    grp_id_list = fet[0][1]
    send_list = []
    for i in range(len(fetch)):
        send_list.append(fetch[i][0])
    
    grp_id_list = grp_id_list.split(',')
    driver = 'ODBC Driver 17 for SQL Server'
    server = 'vodasql.database.windows.net'
    database = 'GateWay-DSP'
    username = 'smartmind'
    password = '1!tmakxmakdlsem'
    conn2 = pyodbc.connect(f'Driver={driver};'
                          f'Server={server};'
                          f'Database={database};'
                          f'uid={username};'
                          f'pwd={password}')
    cursor = conn2.cursor()
    
    
    now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    today = date.today().strftime('%Y-%m-%d')
    yesterday = (date.today() - timedelta(1)).strftime('%Y-%m-%d')
    df = pd.read_sql_query(f'''SELECT * FROM dbo.aa_naver_ads WHERE date = '{yesterday}' order by 'date_hour'
                            ''',conn2)
    ctr_dict = {}
    ctrList = []
    ads = grp_id_list
    check_num = 0
    for i in grp_id_list:
        impressions = df.loc[df['adgroup_id'] == i, :]['impCnt'].astype('float').sum()
        clicks = df.loc[df['adgroup_id'] == i, :]['clkCnt'].astype('float').sum()
        ctr = math.ceil(clicks / impressions * 10000)/100
        ctrList.append(ctr)
        ctr_dict[i] = ctr

    for i in ctrList:
        if i >1:
            check_num += 1
    if check_num != 0 :
        min_max_scaler2 = MinMaxScaler()
        ctrList = np.array(ctrList)
        ctrList = min_max_scaler2.fit_transform(ctrList.reshape(-1, 1))
        ctrList2 = []
        for i in range(4):
            ctrList2.append(ctrList[i][0])
        ctrList = ctrList2

    ACTUAL_CTR = dict(zip(ads, ctrList))
    random_dict = RS(ads,ACTUAL_CTR)
    epsilon_dict = EG(ads,ACTUAL_CTR)
    thompson_dict = TS(ads,ACTUAL_CTR)
    ucb1_dict = UCB(ads,ACTUAL_CTR)
    TotalReward = [random_dict['reward'], epsilon_dict['reward'], thompson_dict['reward'], ucb1_dict['reward']]
    Algorithms = ['RandomSelection', 'EpsilonGreedy', 'ThompsonSampling', 'Ucb1']
    comparisonList = dict(zip(Algorithms, TotalReward))
    for key, value in comparisonList.items():
        if value == max(comparisonList.values()):
            opt_algorithms = key
    if opt_algorithms == 'RandomSelection':
        algorithm = random_dict
    elif opt_algorithms == 'EpsilonGreedy':
        algorithm = epsilon_dict
    elif opt_algorithms == 'ThompsonSampling':
        algorithm = thompson_dict
    else:
        algorithm = ucb1_dict
#     print(comparisonList)
#     print(opt_algorithms)
#     print(algorithm['ads_count'])
    
    budget_list = {}
    for i in ads:
        budget = math.floor(round(algorithm['ads_count'][i] * int(maxbudget)/10))*10
        if budget < 70:
            budget = 70
        budget_list[i] = budget
    grp_ids = list(budget_list.keys())
    for k in budget_list:
        uri = '/ncc/adgroups/' + k
        method = 'GET'
        r = requests.get(BASE_URL + uri, headers=get_header(method, uri, api_key, secret_key, customer_id))
        create = r.json()
        method = 'PUT'
        create['dailyBudget'] = budget_list.get(k)
        r = requests.put(BASE_URL + uri, params={'nccAdgroupId': k}, json=create, headers=get_header(method, uri, api_key, secret_key, customer_id))
        #print("{}. response status_code = {}".format(k,r.status_code))
    #print(opt_algorithms)
    row = ''
    for k in budget_list:
        row += df.loc[df['adgroup_id'] == k,'adgroup_name'].unique()[0][:14]+' : '+str(budget_list.get(k))+'원\n'
    row = row.replace('#','_')
    row2 = ''
    for k,v in comparisonList.items():
        row2 += k+' : '+str(v)+'\n'
    cur.execute('SELECT date FROM history WHERE date = "{}"'.format(yesterday))
    fetch = cur.fetchall()
    if fetch ==[]:
        info = [yesterday, opt_algorithms, row2, row, now_time]
        ins_sql='insert into history values(?,?,?,?,?)'
        cur.executemany(ins_sql,[info])
        conn.commit()
        conn.close()
        #print('추가완료')
    else:
        pass

    for id in send_list:
        requests.get(f"https://api.telegram.org/bot1741553288:AAFjvybYZ3n4mnQJcUVbSgMIm64WGiTK6qo/sendmessage?chat_id={id}&text={today} MAB 예산분배\n\n{row2}\n선택된 알고리즘 : {opt_algorithms}\n\n{row}")

def main():
    user_id = 1459643056
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT choice FROM now WHERE telegram_id = {}'.format(user_id))
    fetch = cur.fetchone()
    choice = fetch[0]
    conn.close()
    MAB()
    
if __name__ == "__main__":
    main()
