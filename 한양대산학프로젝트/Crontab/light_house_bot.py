from sqlalchemy import create_engine
import pyodbc 
from scipy.stats import beta, bernoulli
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



# def error(update,error):
#     logger.warn('Update "%s" caused error "%s"', update,error)

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


def help_command(update, context) :
    chat_id = update.message.chat_id
    nickname = check_nickname(update, context)
    update.message.reply_text("문의 안받음")
    logger.info("닉: {}({}) > help-choose ".format(nickname,chat_id)) 

def get_header(method, uri, api_key, secret_key, customer_id):
    timestamp = str(round(time.time() * 1000))
    signature = signaturehelper.Signature.generate(timestamp, method, uri, secret_key)
    return {'Content-Type': 'application/json; charset=UTF-8', 'X-Timestamp': timestamp, 'X-API-KEY': api_key, 'X-Customer': str(customer_id), 'X-Signature': signature}


def check_id(update, context):
    try:
        id = update.message.chat.id
        return id
    except:
        id = update.channel_post.chat.id
        return id

def check_nickname(update, context):
    try:
        nickname = update.message.from_user.first_name
        return nickname
    except:
        nickname = update.channel_post.from_user.first_name
        return nickname
    
def get_message(update, context) :
    update.message.reply_text(time.time())
    #print(update.message)
    update.message.reply_text(update.message.text)

def login(update, context) :
    update.message.reply_text("로그인")

def register(update, context) :
    update.message.reply_text("가입")

def get_cmp_list(customer_id):
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM naver WHERE customer_id = {}'.format(customer_id))
    fet = cur.fetchall()
    customer_id = fet[0][2]
    api_key = fet[0][0]
    secret_key = fet[0][1]
    conn.close()
    uri = '/ncc/campaigns'
    method = 'GET'
    cmp_list = []
    r = requests.get(BASE_URL + uri, params={'Id':customer_id}, 
                     headers=get_header(method, uri, api_key, secret_key, customer_id))
    if r.status_code != 200:
        cmp_list = 0
        return cmp_list
    else:
        cmp_list = r.json()
        return cmp_list

def get_cmp_stat(c_id,num,cmp_list):
    customer_id = c_id
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM naver WHERE customer_id = {}'.format(customer_id))
    fet = cur.fetchall()
    customer_id = fet[0][2]
    api_key = fet[0][0]
    secret_key = fet[0][1]
    conn.close()
    uri = '/ncc/campaigns'
    method = 'GET'
   
    r = requests.get(BASE_URL + uri, params={'ids':cmp_list}, 
                     headers=get_header(method, uri, api_key, secret_key, customer_id))
    info = r.json()
    df = json_normalize(info[0])
    df.rename(columns={'nccCampaignId':'캠페인 아이디', 'name': '캠페인 이름', 'campaignTp':'캠페인 타입', 'deliveryMethod':'예산 배분형식', 'periodStartDt':'캠페인 시작일', 'periodEndDt':'캠페인 종료일', 'dailyBudget':'하루예산', 'status':'작동상태', 'expectCost':'예상비용', 'editTm':'수정된 시간'}, inplace=True)
    df['캠페인 시작일'] = (pd.to_datetime(df['캠페인 시작일']) + timedelta(hours=9))[0].strftime('%Y-%m-%d %H:%M:%S')
    df['캠페인 종료일'] = (pd.to_datetime(df['캠페인 종료일']) + timedelta(hours=9))[0].strftime('%Y-%m-%d %H:%M:%S')
    df['수정된 시간'] = (pd.to_datetime(df['수정된 시간']) + timedelta(hours=9))[0].strftime('%Y-%m-%d %H:%M:%S')
    df = df[['캠페인 아이디', '캠페인 이름','하루예산', '캠페인 타입', '예산 배분형식', '캠페인 시작일', '캠페인 종료일', '작동상태', '예상비용', '수정된 시간']].T
    df.rename({0:'info'}, axis=1, inplace=True)   

    return df


def get_cmp_stat2(c_id,cmp_list):
    customer_id = c_id
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM naver WHERE customer_id = {}'.format(customer_id))
    fet = cur.fetchall()
    customer_id = fet[0][2]
    api_key = fet[0][0]
    secret_key = fet[0][1]
    conn.close()
    uri = '/ncc/campaigns'
    method = 'GET'
    num=0
    r = requests.get(BASE_URL + uri, params={'ids':cmp_list}, 
                     headers=get_header(method, uri, api_key, secret_key, customer_id))
    info = r.json()
    for i in range(len(r.json())):
        if cmp_list == r.json()[i]['nccCampaignId']:
            info = r.json()[i]
    df = json_normalize(info)
    
    df.rename(columns={'nccCampaignId':'캠페인 아이디', 'name': '캠페인 이름', 'campaignTp':'캠페인 타입', 'deliveryMethod':'예산 배분형식', 'periodStartDt':'캠페인 시작일', 'periodEndDt':'캠페인 종료일', 'dailyBudget':'하루예산', 'status':'작동상태', 'expectCost':'예상비용', 'editTm':'수정된 시간'}, inplace=True)
    df['캠페인 시작일'] = (pd.to_datetime(df['캠페인 시작일']) + timedelta(hours=9))[0].strftime('%Y-%m-%d %H:%M:%S')
    df['캠페인 종료일'] = (pd.to_datetime(df['캠페인 종료일']) + timedelta(hours=9))[0].strftime('%Y-%m-%d %H:%M:%S')
    df['수정된 시간'] = (pd.to_datetime(df['수정된 시간']) + timedelta(hours=9))[0].strftime('%Y-%m-%d %H:%M:%S')
    df = df[['캠페인 아이디', '캠페인 이름','하루예산', '캠페인 타입', '예산 배분형식', '캠페인 시작일', '캠페인 종료일', '작동상태', '예상비용', '수정된 시간']].T
    df.rename({0:'info'}, axis=1, inplace=True)
    return api_key,secret_key,customer_id,num, df,r.json()

def get_grp_stat(c_id,num,grp_list):
    customer_id = c_id
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM naver WHERE customer_id = {}'.format(customer_id))
    fet = cur.fetchall()
    customer_id = fet[0][2]
    api_key = fet[0][0]
    secret_key = fet[0][1]
    conn.close()
    uri = '/ncc/campaigns'
    method = 'GET'
    cmp_list = []
    r = requests.get(BASE_URL + uri, params={'ids':cmp_list}, 
                     headers=get_header(method, uri, api_key, secret_key, customer_id))
    info = r.json()
    df = json_normalize(info[num]).T
    df.rename({0:'info'}, axis=1, inplace=True)
    return df


def get_grp_stat2(c_id,grp_list):
    customer_id = c_id
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM naver WHERE customer_id = {}'.format(customer_id))
    fet = cur.fetchall()
    customer_id = fet[0][2]
    api_key = fet[0][0]
    secret_key = fet[0][1]
    conn.close()
    uri = '/ncc/adgroups'
    method = 'GET'
    num=0
    r = requests.get(BASE_URL + uri, params={'adgroupId':grp_list}, 
                     headers=get_header(method, uri, api_key, secret_key, customer_id))
    info = r.json()
    for i in range(len(r.json())):
        if grp_list == r.json()[i]['nccAdgroupId']:
            info = r.json()[i]
            num=i
    df = json_normalize(info)
    df.rename(columns={'nccAdgroupId':'광고그룹 아이디', 'name': '광고그룹 이름', 'adgroupType':'광고그룹 타입', 'dailyBudget':'하루예산', 'bidAmt':'기본 입찰가', 'targetSummary.pcMobile':'PC/모바일', 'targetSummary.media':'매체', 'targetSummary.time':'요일/시간', 'targetSummary.region':'지역', 'status':'작동상태', 'expectCost':'예상비용', 'editTm':'수정된 시간'}, inplace=True)
    df['수정된 시간'] = (pd.to_datetime(df['수정된 시간']) + timedelta(hours=9))[0].strftime('%Y-%m-%d %H:%M:%S')
    df = df[['광고그룹 아이디', '광고그룹 이름', '광고그룹 타입', '하루예산', '기본 입찰가', 'PC/모바일', '매체', '요일/시간', '지역', '작동상태', '예상비용', '수정된 시간']].T
    df.rename({0:'info'}, axis=1, inplace=True)
    return api_key,secret_key,customer_id,num, df,r.json()
    
def get_grp_list(customer_id, cmp_list):
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM naver WHERE customer_id = {}'.format(customer_id))
    fet = cur.fetchall()
    customer_id = fet[0][2]
    api_key = fet[0][0]
    secret_key = fet[0][1]
    conn.close()
    
    uri = '/ncc/adgroups'
    method = 'GET'
    grp_list = []
    r = requests.get(BASE_URL + uri, params={'nccCampaignId':cmp_list},
                        headers=get_header(method, uri, api_key, secret_key, customer_id))
    grp_list = r.json()
    return grp_list

def get_grp_list2(customer_id, grp_list):
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM naver WHERE customer_id = {}'.format(customer_id))
    fet = cur.fetchall()
    customer_id = fet[0][2]
    api_key = fet[0][0]
    secret_key = fet[0][1]
    conn.close()
    uri = '/ncc/adgroups'
    method = 'GET'
    r = requests.get(BASE_URL + uri, params={'adgroupId':grp_list},
                        headers=get_header(method, uri, api_key, secret_key, customer_id))
    info = r.json()
    for i in range(len(r.json())):
        if grp_list == r.json()[i]['nccAdgroupId']:
            info = r.json()[i]
            num = i
    df = json_normalize(info)
    df.rename(columns={'nccAdgroupId':'광고그룹 아이디', 'name': '광고그룹 이름', 'adgroupType':'광고그룹 타입', 'dailyBudget':'하루예산', 'bidAmt':'기본 입찰가', 'targetSummary.pcMobile':'PC/모바일', 'targetSummary.media':'매체', 'targetSummary.time':'요일/시간', 'targetSummary.region':'지역', 'status':'작동상태', 'expectCost':'예상비용', 'editTm':'수정된 시간'}, inplace=True)
    df['수정된 시간'] = (pd.to_datetime(df['수정된 시간']) + timedelta(hours=9))[0].strftime('%Y-%m-%d %H:%M:%S')
    df = df[['광고그룹 아이디', '광고그룹 이름', '광고그룹 타입', '하루예산', '기본 입찰가', 'PC/모바일', '매체', '요일/시간', '지역', '작동상태', '예상비용', '수정된 시간']].T
    df.rename({0:'info'}, axis=1, inplace=True)
    return num, df,r.json()

def get_nad_list2(customer_id, grp_list,num):
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM naver WHERE customer_id = {}'.format(customer_id))
    fet = cur.fetchall()
    customer_id = fet[0][2]
    api_key = fet[0][0]
    secret_key = fet[0][1]
    conn.close()
    uri = '/ncc/ads'
    method = 'GET'
    nad_list = []
    r = requests.get(BASE_URL + uri, params={'nccAdgroupId':grp_list},
                        headers=get_header(method, uri, api_key, secret_key, customer_id))
    info = r.json()[int(num)]
    df = json_normalize(info)
    df.rename(columns={'nccAdId':'광고 아이디', 'inspectStatus':'검수상태', 'type':'광고 타입', 'status':'작동상태', 'ad.description':'광고 설명', 'ad.headline':'광고 제목', 'ad.mobile.display':'표시 URL (모바일)', 'ad.mobile.final':'연결 URL (모바일)', 'ad.pc.display':'표시 URL (PC)', 'ad.pc.final':'연결 URL (PC)', 'editTm':'수정된 시간'}, inplace=True)
    df['수정된 시간'] = (pd.to_datetime(df['수정된 시간']) + timedelta(hours=9))[0].strftime('%Y-%m-%d %H:%M:%S')
    df = df[['광고 아이디', '광고 제목', '광고 설명', '광고 타입', '연결 URL (PC)', '연결 URL (모바일)' ,'검수상태','작동상태', '수정된 시간']].T

    df.rename({0:'info'}, axis=1, inplace=True)
    return num, df,r.json()

def get_grp_info(num,customer_id, grp_list):
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM naver WHERE customer_id = {}'.format(customer_id))
    fet = cur.fetchall()
    customer_id = fet[0][2]
    api_key = fet[0][0]
    secret_key = fet[0][1]
    conn.close()
    uri = '/ncc/adgroups'
    method = 'GET'
    grp_list = []
    r = requests.get(BASE_URL + uri, params={'nccAdgroupId':grp_list},
                        headers=get_header(method, uri, api_key, secret_key, customer_id))
    grp_name = r.json()[int(num)]['name']
    cmp_id = r.json()[int(num)]['nccCampaignId']
    return grp_name, cmp_id

def get_nad_list(customer_id, grp_list):
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM naver WHERE customer_id = {}'.format(customer_id))
    fet = cur.fetchall()
    customer_id = fet[0][2]
    api_key = fet[0][0]
    secret_key = fet[0][1]
    conn.close()
    uri = '/ncc/ads'
    method = 'GET'
    nad_list = []
    r = requests.get(BASE_URL + uri, params={'nccAdgroupId':grp_list},
                        headers=get_header(method, uri, api_key, secret_key, customer_id))
    nad_list = r.json()
    return nad_list

def get_nad(customer_id, grp_list,num):
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM naver WHERE customer_id = {}'.format(customer_id))
    fet = cur.fetchall()
    customer_id = fet[0][2]
    api_key = fet[0][0]
    secret_key = fet[0][1]
    conn.close()
    uri = '/ncc/ads'
    method = 'GET'
    nad_list = []
    r = requests.get(BASE_URL + uri, params={'nccAdgroupId':grp_list},
                        headers=get_header(method, uri, api_key, secret_key, customer_id))
    nad_list = json.dumps(r.json()[num]['ad'], indent="\t", ensure_ascii= False)
    nad_name = r.json()[num]['ad']
    return nad_list , nad_name

def get_nad_grp(customer_id, nad_list):
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM naver WHERE customer_id = {}'.format(customer_id))
    fet = cur.fetchall()
    customer_id = fet[0][2]
    api_key = fet[0][0]
    secret_key = fet[0][1]
    conn.close()
    uri = '/ncc/ads'
    method = 'GET'
    
    r = requests.get(BASE_URL + uri, params={'ids':nad_list},
                        headers=get_header(method, uri, api_key, secret_key, customer_id))
    grp_id = r.json()[0]['nccAdgroupId']
    return grp_id

def get_nad_ids(customer_id, nad_list,num):
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM naver WHERE customer_id = {}'.format(customer_id))
    fet = cur.fetchall()
    customer_id = fet[0][2]
    api_key = fet[0][0]
    secret_key = fet[0][1]
    conn.close()
    uri = '/ncc/ads'
    method = 'GET'
    nad_list = []
    r = requests.get(BASE_URL + uri, params={'ids':nad_list},
                        headers=get_header(method, uri, api_key, secret_key, customer_id))
    nad_list = json.dumps(r.json()[num]['ad'], indent="\t", ensure_ascii= False)
    nad_name = r.json()[num]['ad']
    return nad_list , nad_name

def button(update, context):
    keyboard = []
    query = update.callback_query
    nick = query.message.chat.first_name
    chat_id = query.message.chat_id  
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM naver WHERE telegram_id = {}'.format(chat_id))
    fet = cur.fetchall()
    u_id = '1459643056'
    try:
        customer_id = fet[0][2]
        c_id = customer_id
        api_key = fet[0][0]
        secret_key = fet[0][1]
        conn.close()
    except:
        cur.execute('SELECT * FROM naver WHERE telegram_id = {}'.format(1876055116))
        fet = cur.fetchall()
        customer_id = fet[0][2]
        c_id = customer_id
        api_key = fet[0][0]
        secret_key = fet[0][1]
        conn.close()
    if query.data == 'b_start':
        context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text="Smartmind_Lighthouse_Bot")
        bt1 = InlineKeyboardButton("로그인", callback_data='login')
        bt2 = InlineKeyboardButton("계정등록", callback_data='register')
        reply_markup = InlineKeyboardMarkup([[bt1,bt2]])
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)

    
    if query.data == 'login':
        conn = sqlite3.connect('/datadrive/Crontab/login.db')
        cur = conn.cursor()
        cur.execute('SELECT * FROM naver WHERE customer_id = {}'.format(customer_id))
        fet = cur.fetchall()
        customer_id = fet[0][2]
        api_key = fet[0][0]
        secret_key = fet[0][1]
        cur2 = conn.cursor()
        cur2.execute('SELECT customer_id FROM naver WHERE telegram_id = {}'.format(chat_id))
        aa = cur2.fetchall()
        conn.close()
        if len(aa) > 0:
            context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text="광고 계정 목록" )
            for i in range(len(aa)):
                keyboard.append([InlineKeyboardButton("Naver : {}".format(aa[i][0]), callback_data='|'.join([str(aa[i][0]),'naver_cmp_list']))])
                #reply_markup = InlineKeyboardMarkup(keyboard)
            keyboard.append([InlineKeyboardButton("<<  뒤로가기", callback_data='b_start')])
            reply_markup = InlineKeyboardMarkup(keyboard)
            context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)
        else:
            context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text="등록된 광고계정이 없습니다")
            keyboard.append([InlineKeyboardButton("<<  뒤로가기", callback_data='b_start')])
            reply_markup = InlineKeyboardMarkup(keyboard)
            context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)

    if query.data == 'register':
        #context.bot.sendMessage(text='회원가입',chat_id=query.message.chat_id)
        context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text="가입할 플랫폼 선택" )
        bt1 = InlineKeyboardButton("네이버 광고", callback_data='r_naver')
        bt2 = InlineKeyboardButton("구글 광고(준비중)", callback_data='r_google')
        bt3 = InlineKeyboardButton("<<  뒤로가기", callback_data='b_start')
        reply_markup = InlineKeyboardMarkup([[bt1,bt2],[bt3]])
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)
        
          
    if query.data == 'r_naver':
        context.bot.sendMessage(text='네이버 광고 CUSTOMER_ID, API_KEY ,SECRET_KEY 순서대로 띄어쓰기 없이 쉼표로 구분해서 입력하세요(최소 120글자)\nex)1234567,a1s2dpw,q3c4we1',chat_id=query.message.chat_id)
        
    
    if query.data == 'r_google':
        context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text="구글은 준비 중입니다.")
        keyboard.append([InlineKeyboardButton("<<  뒤로가기", callback_data='register')])
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.bot.edit_message_reply_markup(chat_id=chat_id, message_id=query.message.message_id,reply_markup=reply_markup)
    
    
    if 'opti_shoose' in query.data:
        c_id, dname = query.data.split('|')
        conn = sqlite3.connect('/datadrive/Crontab/login.db')
        cur = conn.cursor()
        cur.execute('SELECT * FROM mab WHERE customer_id = {}'.format(c_id))
        fetch = cur.fetchall()
        max_daily_budget = fetch[0][1]
        edit_list = list(fetch[0][2].split(','))
        cur.execute('SELECT choice FROM now WHERE telegram_id = {}'.format(u_id))
        fet = cur.fetchone()
        choice = fet[0].upper()
        cur.execute('SELECT * FROM kpi WHERE customer_id = {}'.format(c_id))
        fetch = cur.fetchall()
        w1 = fetch[0][1]
        w2 = fetch[0][2]
        w3 = fetch[0][3]
        conn.close()
        uri = '/ncc/adgroups'
        method = 'GET'
        r = requests.get(BASE_URL + uri, params={'customerId':c_id}, 
                         headers=get_header(method, uri, api_key, secret_key, c_id))
        opti_list = r.json()
        if choice == 'MAB':
            context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=f'현재 적용 알고리즘 : {choice}\n현재 예산 총합 : {max_daily_budget}\n예산최적화 적용여부 설정(클릭해서 변경)')
        else:
            context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=f'현재 적용 알고리즘 : {choice}\n현재 예산 총합 : {max_daily_budget}\n\n현재 가중치\n클릭당 비용 : {w1}\n1000회 노출당 비용 : {w2}\n클릭률 : {w3}\n\n예산최적화 적용여부 설정(클릭해서 변경)')
        for i in range(len(opti_list)):
            if opti_list[i]['nccAdgroupId'] in edit_list:
                bt1 = InlineKeyboardButton("{}".format(opti_list[i]['name'][:14]),callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_del']))
                bt2 = InlineKeyboardButton("O", callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_del']))
            else:
                bt1 = InlineKeyboardButton("{}".format(opti_list[i]['name'][:14]), callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_add']))
                bt2 = InlineKeyboardButton("X", callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_add']))
            keyboard.append([bt1,bt2])
        bt5 = InlineKeyboardButton("MAB로 변경", callback_data='|'.join(['MAB',c_id,'change_mab']))
        bt6 = InlineKeyboardButton("KPI로 변경", callback_data='|'.join(['KPI',c_id,'change_mab']))
        keyboard.append([bt5,bt6])
        bt3 = InlineKeyboardButton("최대 예산 설정", callback_data='set_opti_budget')
        bt4 = InlineKeyboardButton("가중치 설정", callback_data='|'.join([c_id, 'opti_kpi'])) ####
        keyboard.append([bt3,bt4])
        bt7 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, 'naver_cmp_list']))
        keyboard.append([bt7])
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup) 
    
    if 'opti_kpi' in query.data:
        c_id, dname = query.data.split('|')
        conn = sqlite3.connect('/datadrive/Crontab/login.db')
        cur = conn.cursor()
        cur.execute('SELECT * FROM kpi WHERE customer_id = {}'.format(c_id)) 
        fet = cur.fetchall()
        conn.close()
        w1 = fet[0][1]
        w2 = fet[0][2]
        w3 = fet[0][3]
        daily_budget = fet[0][4]       ####
        context.bot.sendMessage(text='클릭당 비용, 1000회노출당 비용, 클릭률 순서대로 주고싶은 가중치를 입력하세요.\nex) 가중치:0.2,0.5,0.3',chat_id=query.message.chat_id)
        
    if 'opti_set' in query.data:
        c_id, dname = query.data.split('|')
        conn = sqlite3.connect('/datadrive/Crontab/login.db')
        cur = conn.cursor()
        cur.execute('SELECT * FROM mab WHERE customer_id = {}'.format(c_id))
        fetch = cur.fetchall()
        max_daily_budget = fetch[0][1]
        edit_list = list(fetch[0][2].split(','))
        cur.execute('SELECT choice FROM now WHERE telegram_id = {}'.format(u_id))
        fet = cur.fetchone()
        choice = fet[0].upper()
        cur.execute('SELECT * FROM kpi WHERE customer_id = {}'.format(c_id))
        fetch = cur.fetchall()
        w1 = fetch[0][1]
        w2 = fetch[0][2]
        w3 = fetch[0][3]
        conn.close()
        uri = '/ncc/adgroups'
        method = 'GET'
        r = requests.get(BASE_URL + uri, params={'customerId':c_id}, 
                         headers=get_header(method, uri, api_key, secret_key, c_id))
        opti_list = r.json()

        if choice == 'MAB':
            context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=f'현재 적용 알고리즘 : {choice}\n현재 예산 총합 : {max_daily_budget}\n예산최적화 적용여부 설정(클릭해서 변경)')
        else:
            context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=f'현재 적용 알고리즘 : {choice}\n현재 예산 총합 : {max_daily_budget}\n\n현재 가중치\n클릭당 비용 : {w1}\n1000회 노출당 비용 : {w2}\n클릭률 : {w3}\n\n예산최적화 적용여부 설정(클릭해서 변경)')
            
        for i in range(len(opti_list)):
            if opti_list[i]['nccAdgroupId'] in edit_list:
                bt1 = InlineKeyboardButton("{}".format(opti_list[i]['name'][:14]),callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_del']))
                bt2 = InlineKeyboardButton("O", callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_del']))
            else:
                bt1 = InlineKeyboardButton("{}".format(opti_list[i]['name'][:14]), callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_add']))
                bt2 = InlineKeyboardButton("X", callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_add']))
            keyboard.append([bt1,bt2])
        bt5 = InlineKeyboardButton("MAB로 변경", callback_data='|'.join(['MAB',c_id,'change_mab']))
        bt6 = InlineKeyboardButton("KPI로 변경", callback_data='|'.join(['KPI',c_id,'change_mab']))
        keyboard.append([bt5,bt6])
        bt3 = InlineKeyboardButton("최대 예산 설정", callback_data='set_opti_budget')
        bt4 = InlineKeyboardButton("가중치 설정", callback_data='|'.join([c_id, 'opti_kpi'])) ####
        keyboard.append([bt3,bt4])
        bt7 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, 'naver_cmp_list']))
        keyboard.append([bt7])
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)        
    
    
    if 'change_mab'in query.data:
        choice,c_id, dname = query.data.split('|')
        if choice == 'KPI':
            choice = 'kpi'
            conn = sqlite3.connect('/datadrive/Crontab/login.db')
            cur = conn.cursor()
            cur.execute("UPDATE now SET choice = ? WHERE telegram_id = ?",(choice,u_id))
            conn.commit()
            cur.execute('SELECT * FROM mab WHERE customer_id = {}'.format(c_id))
            fetch = cur.fetchall()
            max_daily_budget = fetch[0][1]
            edit_list = list(fetch[0][2].split(','))
            cur.execute('SELECT choice FROM now WHERE telegram_id = {}'.format(u_id))
            fet = cur.fetchone()
            choice = fet[0].upper()
            cur.execute('SELECT * FROM kpi WHERE customer_id = {}'.format(c_id))
            fetch = cur.fetchall()
            w1 = fetch[0][1]
            w2 = fetch[0][2]
            w3 = fetch[0][3]
            conn.close()
            uri = '/ncc/adgroups'
            method = 'GET'
            r = requests.get(BASE_URL + uri, params={'customerId':c_id}, 
                             headers=get_header(method, uri, api_key, secret_key, c_id))
            opti_list = r.json()

            if choice == 'MAB':
                context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=f'현재 적용 알고리즘 : {choice}\n현재 예산 총합 : {max_daily_budget}\n예산최적화 적용여부 설정(클릭해서 변경)')
            else:
                context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=f'현재 적용 알고리즘 : {choice}\n현재 예산 총합 : {max_daily_budget}\n\n현재 가중치\n클릭당 비용 : {w1}\n1000회 노출당 비용 : {w2}\n클릭률 : {w3}\n\n예산최적화 적용여부 설정(클릭해서 변경)')
         
            for i in range(len(opti_list)):
                if opti_list[i]['nccAdgroupId'] in edit_list:
                    bt1 = InlineKeyboardButton("{}".format(opti_list[i]['name'][:14]),callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_del']))
                    bt2 = InlineKeyboardButton("O", callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_del']))
                else:
                    bt1 = InlineKeyboardButton("{}".format(opti_list[i]['name'][:14]), callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_add']))
                    bt2 = InlineKeyboardButton("X", callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_add']))
                keyboard.append([bt1,bt2])
            bt5 = InlineKeyboardButton("MAB로 변경", callback_data='|'.join(['MAB',c_id,'change_mab']))
            bt6 = InlineKeyboardButton("KPI로 변경", callback_data='|'.join(['KPI',c_id,'change_mab']))
            keyboard.append([bt5,bt6])
            bt3 = InlineKeyboardButton("최대 예산 설정", callback_data='set_opti_budget')
            bt4 = InlineKeyboardButton("가중치 설정", callback_data='|'.join([c_id, 'opti_kpi']))
            keyboard.append([bt3,bt4])
            bt7 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, 'naver_cmp_list']))
            keyboard.append([bt7])
            reply_markup = InlineKeyboardMarkup(keyboard)
            context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup) 
                
            
        else:
            choice = 'mab'
            conn = sqlite3.connect('/datadrive/Crontab/login.db')
            cur = conn.cursor()
            cur.execute("UPDATE now SET choice = ? WHERE telegram_id = ?",(choice,u_id))
            conn.commit()
            cur.execute('SELECT * FROM mab WHERE customer_id = {}'.format(c_id))
            fetch = cur.fetchall()
            max_daily_budget = fetch[0][1]
            edit_list = list(fetch[0][2].split(','))
            cur.execute('SELECT choice FROM now WHERE telegram_id = {}'.format(u_id))
            fet = cur.fetchone()
            choice = fet[0].upper()
            cur.execute('SELECT * FROM kpi WHERE customer_id = {}'.format(c_id))
            fetch = cur.fetchall()
            w1 = fetch[0][1]
            w2 = fetch[0][2]
            w3 = fetch[0][3]
            conn.close()
            uri = '/ncc/adgroups'
            method = 'GET'
            r = requests.get(BASE_URL + uri, params={'customerId':c_id}, 
                             headers=get_header(method, uri, api_key, secret_key, c_id))
            opti_list = r.json()
        
            if choice == 'MAB':
                context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=f'현재 적용 알고리즘 : {choice}\n현재 예산 총합 : {max_daily_budget}\n예산최적화 적용여부 설정(클릭해서 변경)')
            else:
                context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=f'현재 적용 알고리즘 : {choice}\n현재 예산 총합 : {max_daily_budget}\n\n현재 가중치\n클릭당 비용 : {w1}\n1000회 노출당 비용 : {w2}\n클릭률 : {w3}\n\n예산최적화 적용여부 설정(클릭해서 변경)')
         
            for i in range(len(opti_list)):
                if opti_list[i]['nccAdgroupId'] in edit_list:
                    bt1 = InlineKeyboardButton("{}".format(opti_list[i]['name'][:14]),callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_del']))
                    bt2 = InlineKeyboardButton("O", callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_del']))
                else:
                    bt1 = InlineKeyboardButton("{}".format(opti_list[i]['name'][:14]), callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_add']))
                    bt2 = InlineKeyboardButton("X", callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_add']))
                keyboard.append([bt1,bt2])
            
            bt5 = InlineKeyboardButton("MAB로 변경", callback_data='|'.join(['MAB',c_id,'change_mab']))
            bt6 = InlineKeyboardButton("KPI로 변경", callback_data='|'.join(['KPI',c_id,'change_mab']))
            keyboard.append([bt5,bt6])
            bt3 = InlineKeyboardButton("최대 예산 설정", callback_data='set_opti_budget')
            bt4 = InlineKeyboardButton("가중치 설정", callback_data='|'.join([c_id, 'opti_kpi']))
            keyboard.append([bt3,bt4])
            bt7 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, 'naver_cmp_list']))
            keyboard.append([bt7])
            reply_markup = InlineKeyboardMarkup(keyboard)
            context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup) 
    
    
    
    if 'set_opti_budget'in query.data:
        context.bot.sendMessage(text='변경할 최적화 최대 예산를 입력하세요.\nex) max:10000',chat_id=query.message.chat_id)

    
    if 'opti_del' in query.data:
        c_id,grp_id, dname = query.data.split('|')
        conn = sqlite3.connect('/datadrive/Crontab/login.db')
        cur = conn.cursor()
        cur.execute('SELECT * FROM mab WHERE customer_id = {}'.format(c_id))
        fetch = cur.fetchall()
        max_daily_budget = fetch[0][1]
        edit_list = list(fetch[0][2].split(','))
        if grp_id in edit_list:
            edit_list.remove(grp_id)
        grp_str = ','.join(edit_list)
        cur.execute("UPDATE mab SET grp_id = ? WHERE grp_id = ?",(grp_str,fetch[0][2]))
        conn.commit()

        cur2 = conn.cursor()
        cur2.execute('SELECT * FROM mab WHERE customer_id = {}'.format(c_id))
        fetch = cur2.fetchall()
        edit_list = list(fetch[0][2].split(','))
        cur.execute('SELECT choice FROM now WHERE telegram_id = {}'.format(u_id))
        fet = cur.fetchone()
        choice = fet[0].upper()
        cur.execute('SELECT * FROM kpi WHERE customer_id = {}'.format(c_id))
        fetch = cur.fetchall()
        w1 = fetch[0][1]
        w2 = fetch[0][2]
        w3 = fetch[0][3]
        conn.close()
        uri = '/ncc/adgroups'
        method = 'GET'
        r = requests.get(BASE_URL + uri, params={'customerId':c_id}, 
                         headers=get_header(method, uri, api_key, secret_key, c_id))
        opti_list = r.json()        
        if choice == 'MAB':
            context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=f'현재 적용 알고리즘 : {choice}\n현재 예산 총합 : {max_daily_budget}\n예산최적화 적용여부 설정(클릭해서 변경)')
        else:
            context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=f'현재 적용 알고리즘 : {choice}\n현재 예산 총합 : {max_daily_budget}\n\n현재 가중치\n클릭당 비용 : {w1}\n1000회 노출당 비용 : {w2}\n클릭률 : {w3}\n\n예산최적화 적용여부 설정(클릭해서 변경)')
         
        for i in range(len(opti_list)):
            if opti_list[i]['nccAdgroupId'] in edit_list:
                bt1 = InlineKeyboardButton("{}".format(opti_list[i]['name'][:14]),callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_del']))
                bt2 = InlineKeyboardButton("O", callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_del']))
            else:
                bt1 = InlineKeyboardButton("{}".format(opti_list[i]['name'][:14]), callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_add']))
                bt2 = InlineKeyboardButton("X", callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_add']))
            keyboard.append([bt1,bt2])
        bt5 = InlineKeyboardButton("MAB로 변경", callback_data='|'.join(['MAB',c_id,'change_mab']))
        bt6 = InlineKeyboardButton("KPI로 변경", callback_data='|'.join(['KPI',c_id,'change_mab']))
        keyboard.append([bt5,bt6])
        bt3 = InlineKeyboardButton("최대 예산 설정", callback_data='set_opti_budget')
        bt4 = InlineKeyboardButton("가중치 설정", callback_data='|'.join([c_id, 'opti_kpi']))
        keyboard.append([bt3,bt4])
        bt7 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, 'naver_cmp_list']))
        keyboard.append([bt7])
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup) 
    
    
    if 'opti_add' in query.data:
        c_id,grp_id, dname = query.data.split('|')
        conn = sqlite3.connect('/datadrive/Crontab/login.db')
        cur = conn.cursor()
        cur.execute('SELECT * FROM mab WHERE customer_id = {}'.format(c_id))
        fetch = cur.fetchall()
        edit_list = list(fetch[0][2].split(','))
        if grp_id not in edit_list:
            edit_list.append(grp_id)
        grp_str = ','.join(edit_list)
        cur.execute("UPDATE mab SET grp_id = ? WHERE grp_id = ?",(grp_str,fetch[0][2]))
        conn.commit()
        cur2 = conn.cursor()
        cur2.execute('SELECT * FROM mab WHERE customer_id = {}'.format(c_id))
        fetch = cur2.fetchall()
        max_daily_budget = fetch[0][1]
        edit_list = list(fetch[0][2].split(','))
        cur.execute('SELECT choice FROM now WHERE telegram_id = {}'.format(u_id))
        fet = cur.fetchone()
        choice = fet[0].upper()
        cur.execute('SELECT * FROM kpi WHERE customer_id = {}'.format(c_id))
        fetch = cur.fetchall()
        w1 = fetch[0][1]
        w2 = fetch[0][2]
        w3 = fetch[0][3]
        conn.close()
        uri = '/ncc/adgroups'
        method = 'GET'
        r = requests.get(BASE_URL + uri, params={'customerId':c_id}, 
                         headers=get_header(method, uri, api_key, secret_key, c_id))
        opti_list = r.json()        
        if choice == 'MAB':
            context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=f'현재 적용 알고리즘 : {choice}\n현재 예산 총합 : {max_daily_budget}\n예산최적화 적용여부 설정(클릭해서 변경)')
        else:
            context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=f'현재 적용 알고리즘 : {choice}\n현재 예산 총합 : {max_daily_budget}\n\n현재 가중치\n클릭당 비용 : {w1}\n1000회 노출당 비용 : {w2}\n클릭률 : {w3}\n\n예산최적화 적용여부 설정(클릭해서 변경)')
         
        for i in range(len(opti_list)):
            if opti_list[i]['nccAdgroupId'] in edit_list:
                bt1 = InlineKeyboardButton("{}".format(opti_list[i]['name'][:14]),callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_del']))
                bt2 = InlineKeyboardButton("O", callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_del']))
            else:
                bt1 = InlineKeyboardButton("{}".format(opti_list[i]['name'][:14]), callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_add']))
                bt2 = InlineKeyboardButton("X", callback_data='|'.join([c_id,opti_list[i]['nccAdgroupId'],'opti_add']))
            keyboard.append([bt1,bt2])
        bt5 = InlineKeyboardButton("MAB로 변경", callback_data='|'.join(['MAB',c_id,'change_mab']))
        bt6 = InlineKeyboardButton("KPI로 변경", callback_data='|'.join(['KPI',c_id,'change_mab']))
        keyboard.append([bt5,bt6])
        bt3 = InlineKeyboardButton("최대 예산 설정", callback_data='set_opti_budget')
        bt4 = InlineKeyboardButton("가중치 설정", callback_data='|'.join([c_id, 'opti_kpi']))
        keyboard.append([bt3,bt4])
        bt7 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, 'naver_cmp_list']))
        keyboard.append([bt7])
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup) 
        

    if 'naver_cmp_list' in query.data:
        #print(query)
        c_id, dname = query.data.split('|')
#         global customer_id, api_key, secret_key
        conn = sqlite3.connect('/datadrive/Crontab/login.db')
        cur = conn.cursor()
        cur.execute('SELECT * FROM naver WHERE telegram_id = {}'.format(chat_id))
        fet = cur.fetchall()
        customer_id = fet[0][2]
        api_key = fet[0][0]
        secret_key = fet[0][1]
        conn.close()
        cmp_list = get_cmp_list(c_id)
        #print(cmp_list)
        keyboard = []
        conn.close()
        if cmp_list != 0:
            if len(cmp_list) > 0 :
                context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text='{}의 캠페인 목록'.format(c_id))
                for i in range(len(cmp_list)):
                    bt1 = InlineKeyboardButton("{}".format(cmp_list[i]['name']), callback_data='|'.join([str(i),c_id,cmp_list[i]['nccCampaignId'],'naver_grp_list']))
                    bt2 = InlineKeyboardButton("상태보기", callback_data='|'.join([c_id,str(i),cmp_list[i]['nccCampaignId'],'get_cmp_stat']))
                    bt3 = InlineKeyboardButton("지표보기  >>", callback_data='|'.join([c_id,str(i),cmp_list[i]['nccCampaignId'],'get_cmp_info']))
                    keyboard.append([bt1,bt2,bt3])
                    # reply_markup = InlineKeyboardMarkup(keyboard)
                bt4 =InlineKeyboardButton("<<  뒤로가기", callback_data='login')
                bt5 = InlineKeyboardButton("예산 최적화", callback_data='|'.join([c_id,'opti_shoose']))
                keyboard.append([bt4,bt5])
                reply_markup = InlineKeyboardMarkup(keyboard)
                context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)
            else:
                context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text='{}의 캠페인이 없습니다.'.format(c_id))
        else:
            context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text='{}의 캠페인이 없습니다.'.format(c_id))

    if 'naver_grp_list' in query.data:
        try:
            num,c_id, cmp_list, dname = query.data.split('|')
        except:
            c_id, cmp_list, dname = query.data.split('|')
        grp_list = get_grp_list(c_id,cmp_list)
        cmp_name = grp_list[0]['name']
        keyboard = []
        if len(grp_list) > 0:
            context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text='애드그룹 목록')
            for i in range(len(grp_list)):
                bt1 = InlineKeyboardButton("{} ".format(grp_list[i]['name']), callback_data='|'.join([str(i),c_id,grp_list[i]['nccAdgroupId'],'naver_nad_list']))
                bt2 = InlineKeyboardButton("상태보기 ", callback_data='|'.join([str(i),c_id,grp_list[i]['nccAdgroupId'],'get_grp_stat']))
                bt3 = InlineKeyboardButton("지표보기  >>", callback_data='|'.join([c_id,str(i),grp_list[i]['nccAdgroupId'],'get_grp_info']))
                keyboard.append([bt1,bt2,bt3])
            keyboard.append([InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, 'naver_cmp_list']))])
            reply_markup = InlineKeyboardMarkup(keyboard)
            context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)
        else:
            context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text='{}의 애드그룹이 없습니다.'.format(cmp_name))
            keyboard.append([InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, 'naver_cmp_list']))])
            reply_markup = InlineKeyboardMarkup(keyboard)
            context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)

    if 'naver_nad_list' in query.data:
        num, c_id, grp_list, dname = query.data.split('|')
        nad_list = get_nad_list(c_id, grp_list)
#         print(nad_list[1]['nccAdId'])
        grp_name, cmp_id = get_grp_info(num, c_id, grp_list)
        keyboard = []
        if len(nad_list) > 0:
            context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text='광고 목록')
            for i in range(len(nad_list)):
                bt1 = InlineKeyboardButton("{}".format(nad_list[i]['ad']['headline']), callback_data='3333')
                #get_nad_list(customer_id, grp_list):
                bt2 = InlineKeyboardButton("상태보기 ", callback_data='|'.join([str(i),c_id,nad_list[i]['nccAdId'],'naver_nad_stat']))
                bt3 = InlineKeyboardButton("지표보기  >>", callback_data='|'.join([grp_list,c_id,str(i),'get_nad_info']))
                keyboard.append([bt1,bt2,bt3])

            keyboard.append([InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, 'naver_cmp_list']))])
            reply_markup = InlineKeyboardMarkup(keyboard)
            context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)

        else:
            context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text='{}의 광고가 없습니다.'.format(grp_name))

    if 'get_nad_stats' in query.data:
        c_id, grp_list, num , dname = query.data.split('|')
        nad_list, nad_name = get_nad(c_id, grp_list,int(num))
        context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=nad_list.replace('{','',1).replace('}','',1))
        bt1 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([num,c_id,grp_list, 'naver_nad_list']))
        keyboard.append([bt1])
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)
        #keyboard.append([InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id,grp_list[0]['nccCampaignId'], 'naver_nad_list']))])
    
    if 'get_cmp_stat' in query.data:
        keyboard = []
        c_id,num, cmp_id , dname = query.data.split('|')
        stat = get_cmp_stat(c_id,num,cmp_id)
        row = ''
        for i in range(len(stat)):
            row += '{} : {}\n\n'.format(stat['info'].index[i],stat['info'].values[i])

        #text=''.format(stat['info'].index[0])
        context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=row)
        
        bt1 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, 'naver_cmp_list']))
        bt2 = InlineKeyboardButton("하루예산 수정", callback_data='|'.join([c_id,num, cmp_id,'edit_cmp']))
        keyboard.append([bt1,bt2])
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)
    
    if 'get_cmp_info' in query.data:
        c_id,num, cmp_id , dname = query.data.split('|')
        context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text='날짜 선택')
        bt1 = InlineKeyboardButton("오늘", callback_data='|'.join([c_id,cmp_id, 'today','get_stat']))
        bt2 = InlineKeyboardButton("최근7일", callback_data='|'.join([c_id,cmp_id,'last7days','get_stat']))
        keyboard.append([bt1,bt2])
        keyboard.append([InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, 'naver_cmp_list']))])
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)
    
    if 'get_grp_info' in query.data:
        c_id,num, grp_id , dname = query.data.split('|')
        num, stat,json = get_grp_list2(c_id,grp_id)
        cmp_list = json[int(num)]['nccCampaignId']
        context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text='날짜 선택')
        bt1 = InlineKeyboardButton("오늘", callback_data='|'.join([c_id,grp_id, 'today','get_stat']))
        bt2 = InlineKeyboardButton("최근7일", callback_data='|'.join([c_id,grp_id,'last7days','get_stat']))
        keyboard.append([bt1,bt2])
        keyboard.append([InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id,cmp_list,'naver_grp_list']))])
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)                                       

    if 'get_nad_info' in query.data:
        grp_list,c_id,num , dname = query.data.split('|')
        # grp_list
        json = get_nad_list(c_id,grp_list)
        nad_id = json[int(num)]['nccAdId']
        
        context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text='날짜 선택')
        bt1 = InlineKeyboardButton("오늘", callback_data='|'.join([c_id,nad_id, 'today','naver_stat_nad']))
        bt2 = InlineKeyboardButton("최근7일", callback_data='|'.join([c_id,nad_id,'last7days','naver_stat_nad']))
        keyboard.append([bt1,bt2])
        keyboard.append([InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([num,c_id,grp_list,'naver_nad_list']))])
        reply_markup = InlineKeyboardMarkup(keyboard)  
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)          
        
        
        
    if 'edit_cmp' in query.data:
        c_id,num, cmp_id , dname = query.data.split('|')
        edit_stat = get_cmp_stat(c_id,int(num),cmp_id)
        context.bot.sendMessage(text='하루 동안 이 캠페인에서 지불할 의사가 있는 최대 비용을 설정합니다.\n70원에서 10억원까지 입력 가능(10원 단위 입력)\nex) 하루예산:2000',chat_id=query.message.chat_id)
                                           
    if 'edit_grp' in query.data:
        num, grp_id , dname = query.data.split('|')
        edit_stat = get_grp_stat(c_id,int(num),grp_id)
        context.bot.sendMessage(text='네이버 통합검색 영역을 기준으로, 해당 광고 그룹에 속한 키워드의 한 번 클릭에 대해 지불할 의사가 있는 최대 비용입니다.\n입찰 금액: 최소 70원부터 최대 100,000원까지(VAT 제외 금액, 10원 단위로 입력)\nex) 입찰가:200',chat_id=query.message.chat_id)
        
    if 'grp_edit' in query.data:
        num, grp_id , dname = query.data.split('|')
        edit_stat = get_grp_stat(c_id,int(num),grp_id)
        context.bot.sendMessage(text=' 하루 동안 이 광고그룹에서 지불할 의사가 있는 최대 비용을 설정합니다.\n입찰 금액: 최소 70원부터 최대 10억원까지(10원 단위로 입력)\nex) 그룹예산:2000',chat_id=query.message.chat_id)
        
        
    if 'get_stat' in query.data:
        keyboard = []
        c_id, cmp_id, datePreset, dname = query.data.split('|')
        uri = '/stats'
        method = 'GET'
        r = requests.get(BASE_URL + uri, params={'ids': cmp_id, 
                                                 'fields': '["ctr", "clkCnt","impCnt","salesAmt", "cpc"]', 
                                                 'datePreset': '{}'.format(datePreset)}, 
                         headers=get_header(method, uri, api_key, secret_key, customer_id))
        info = r.json()['data']
        df = json_normalize(info)
        df['cpm'] = int(round(df['salesAmt'] / df['impCnt'] * 1000))
        df = df[['id', 'ctr', 'clkCnt', 'cpc', 'impCnt', 'cpm', 'salesAmt']].T
        df.rename({0:'info'}, axis=1, inplace=True)
        stat = df
        stat.rename(index={'ctr': '클릭률', "clkCnt": '클릭수', "impCnt":'노출수', "salesAmt":'총 비용', "cpm":'노출당 비용', 'id':'캠페인아이디', "cpc":'클릭당 비용'}, inplace=True)
        row = ''
        for i in range(len(stat)):
            row += '{} : {}\n\n'.format(stat['info'].index[i],stat['info'].values[i])

        #text=''.format(stat['info'].index[0])
        context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=row)
        
        bt1 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, 'naver_cmp_list']))
        bt2 = InlineKeyboardButton("시간대별로", callback_data='|'.join([c_id,cmp_id,datePreset,'get_stat_b']))
        keyboard.append([bt1,bt2])
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)
        
    if 'naver_stat_nad' in query.data:
        keyboard = []
        c_id, nad_id, datePreset, dname = query.data.split('|')
        uri = '/stats'
        method = 'GET'
        r = requests.get(BASE_URL + uri, params={'ids': nad_id, 
                                                 'fields': '["ctr", "clkCnt","impCnt","salesAmt", "cpc"]', 
                                                 'datePreset': '{}'.format(datePreset)}, 
                         headers=get_header(method, uri, api_key, secret_key, customer_id))
        info = r.json()['data']
        df = json_normalize(info)
        df['cpm'] = int(round(df['salesAmt'] / df['impCnt'] * 1000))
        df = df[['id', 'ctr', 'clkCnt', 'cpc', 'impCnt', 'cpm', 'salesAmt']].T
        df.rename({0:'info'}, axis=1, inplace=True)
        stat = df
        stat.rename(index={'ctr': '클릭률', "clkCnt": '클릭수', "impCnt":'노출수', "salesAmt":'총 비용', "cpm":'노출당 비용', 'id':'캠페인아이디', "cpc":'클릭당 비용'}, inplace=True)
        row = ''
        for i in range(len(stat)):
            row += '{} : {}\n\n'.format(stat['info'].index[i],stat['info'].values[i])

        #text=''.format(stat['info'].index[0])
        context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=row)
        
        bt1 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, 'naver_cmp_list']))
        bt2 = InlineKeyboardButton("시간대별로", callback_data='|'.join([c_id,nad_id,datePreset,'nad_stat_b']))
        keyboard.append([bt1,bt2])
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)
    

    if 'get_stat_b' in query.data:  
        keyboard = []
        c_id, cmp_id, datePreset, dname = query.data.split('|')
        uri = '/stats'
        method = 'GET'
        r = requests.get(BASE_URL + uri, params={'ids': cmp_id, 
                                                 'fields': '["ctr", "clkCnt","cpc","impCnt","salesAmt" ]', 
                                                 'datePreset': '{}'.format(datePreset),
                                                'breakdown' : 'hh24'}, 
                         headers=get_header(method, uri, api_key, secret_key, customer_id))
        stat = r.json()['data'][0]['breakdowns']
        
        row = ''
        for i in range(len(stat)):
            stat[i]['cpm'] = int(round(stat[i].get('salesAmt') / stat[i].get('impCnt') * 1000))
            stat[i]['시간'] = stat[i].pop('name')
            stat[i]['클릭률'] = stat[i].pop('ctr')
            stat[i]['클릭수'] = stat[i].pop('clkCnt')
            stat[i]['클릭당 비용'] = stat[i].pop('cpc')
            stat[i]['노출수'] = stat[i].pop('impCnt')
            stat[i]['노출당 비용'] = stat[i].pop('cpm')
            stat[i]['총 비용'] = stat[i].pop('salesAmt')
            
            items = stat[i].items()
            for k,v in items:
                row += '{} : {} \n'.format(k,v)
            row = row + '\n'

        #text=''.format(stat['info'].index[0])
        context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=row)
        
        bt1 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id,cmp_id,datePreset, 'get_stat']))
        keyboard.append([bt1])
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)

    if 'nad_stat_b' in query.data:  
        keyboard = []
        c_id, nad_id, datePreset, dname = query.data.split('|')
        uri = '/stats'
        method = 'GET'
        r = requests.get(BASE_URL + uri, params={'ids': nad_id, 
                                                 'fields': '["ctr", "clkCnt","cpc","impCnt","salesAmt" ]', 
                                                 'datePreset': '{}'.format(datePreset),
                                                'breakdown' : 'hh24'}, 
                         headers=get_header(method, uri, api_key, secret_key, customer_id))
        stat = r.json()['data'][0]['breakdowns']
        row = ''
        for i in range(len(stat)):
            stat[i]['cpm'] = int(round(stat[i].get('salesAmt') / stat[i].get('impCnt') * 1000))
            stat[i]['시간'] = stat[i].pop('name')
            stat[i]['클릭률'] = stat[i].pop('ctr')
            stat[i]['클릭수'] = stat[i].pop('clkCnt')
            stat[i]['클릭당 비용'] = stat[i].pop('cpc')
            stat[i]['노출수'] = stat[i].pop('impCnt')
            stat[i]['노출당 비용'] = stat[i].pop('cpm')
            stat[i]['총 비용'] = stat[i].pop('salesAmt')
            items = stat[i].items()
            for k,v in items:
                row += '{} : {} \n'.format(k,v)
            row = row + '\n'

        #text=''.format(stat['info'].index[0])
        context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=row)
        
        bt1 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id,nad_id,datePreset, 'naver_stat_nad']))
        keyboard.append([bt1])
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)        
        
    if 'naver_nad_stat' in query.data:    
        keyboard = []
        num,c_id, nad_id , dname = query.data.split('|')

        grp_list = get_nad_grp(c_id, nad_id)
 
        num, stat, json = get_nad_list2(c_id, grp_list,num)

        row = ''
        for i in range(len(stat)):
            row += '{} : {}\n\n'.format(stat['info'].index[i],stat['info'].values[i])

        #text=''.format(stat['info'].index[0])
        context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=row)
        
        bt1 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([num,c_id, grp_list,'naver_nad_list']))
        keyboard.append([bt1])
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)
        
    if 'get_grp_stat' in query.data:
        keyboard = []
        num,c_id, grp_id , dname = query.data.split('|')
        num, stat,json = get_grp_list2(c_id,grp_id)
        cmp_list = json[int(num)]['nccCampaignId']
        row = ''
        for i in range(len(stat)):
            row += '{} : {}\n\n'.format(stat['info'].index[i],stat['info'].values[i])

        #text=''.format(stat['info'].index[0])
        context.bot.edit_message_text(chat_id=chat_id ,message_id=query.message.message_id ,text=row)
        
        bt1 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, cmp_list,'naver_grp_list']))
        bt2 = InlineKeyboardButton("기본 입찰가 수정", callback_data='|'.join([str(num), grp_id,'edit_grp']))
        bt3 = InlineKeyboardButton("그룹예산 수정", callback_data='|'.join([str(num), grp_id,'grp_edit']))
        keyboard.append([bt1,bt2,bt3])
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.bot.edit_message_reply_markup(chat_id=chat_id ,message_id=query.message.message_id ,reply_markup=reply_markup)                                  
        
    if 'prepare' in query.data:
        context.bot.sendMessage(text='준비 중입니다.',chat_id=query.message.chat_id)
        #update.message.reply_text("준비중 입니다.")
        
def start(update, context):
    id = check_id(update, context)
    nickname = check_nickname(update, context)
    logger.info("사용자 id :{}  닉 : {} ".format(id, nickname))
    
    bt1 = InlineKeyboardButton("로그인", callback_data='login')
    bt2 = InlineKeyboardButton("계정등록", callback_data='register')
    reply_markup = InlineKeyboardMarkup([[bt1,bt2]])
    update.message.reply_text("Smartmind Lighthouse Bot", reply_markup=reply_markup)


def search(update, context):
    global bidAmt,dailyBudget,maxbudget #,ncccampaignid #, customer_id, api_key, secret_key
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    user_id = check_id(update, context)
    nickname = check_nickname(update, context)
    message = update.message.text.replace(' ','')
    cur = conn.cursor()
    cur.execute('SELECT customer_id FROM naver WHERE telegram_id = {}'.format(user_id))
    aa = cur.fetchall()
    conn.close() 
    if len(aa)>0:
        c_id = str(aa[0][0])
    else:
        c_id = str(0)
    keyboard = []
    
    
    if '가중치:'in message.lower():
        keyboard = []
        message = message.lower().replace('가중치:', '')
        w1,w2,w3 = message.split(',')
        try:
            w1 = float(w1)
            w2 = float(w2)
            w3 = float(w3)
        except:
            update.message.reply_text("잘못 입력 하셨습니다.\n클릭당 비용, 1000회노출당 비용, 클릭률 순서대로 쉼표로 구분해서 주고싶은 가중치를 입력하세요.\nex) 가중치:0.2,0.5,0.3",reply_markup=reply_markup)
        if type(w1) == float and type(w2) == float and type(w3) == float: ####
            conn = sqlite3.connect('/datadrive/Crontab/login.db')
            cur2 = conn.cursor()
            cur2.execute("UPDATE kpi SET w_cpc = ?, w_cpm = ?, w_ctr = ? WHERE customer_id = ?",(w1,w2,w3,c_id))
            conn.commit()
            conn.close()  
            keyboard.append([InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, 'opti_shoose']))])
            reply_markup = InlineKeyboardMarkup(keyboard)
            update.message.reply_text("클릭당 비용 : {}\n1000회노출당 비용 : {}\n클릭률 : {}\n수정 완료.".format(w1,w2,w3),reply_markup=reply_markup)
            

        
    elif 'max:'in message.lower():
        keyboard = []
        message = message.lower().replace('max:', '')
        maxbudget = message
        try:
            if type(int(maxbudget)) == int:
                conn = sqlite3.connect('/datadrive/Crontab/login.db')
                cur2 = conn.cursor()
                
                cur2.execute("UPDATE mab SET daily_budget = ? WHERE customer_id = ?",(maxbudget,c_id))
                conn.commit()
                conn.close()  
                keyboard.append([InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, 'naver_cmp_list']))])
                reply_markup = InlineKeyboardMarkup(keyboard)
                update.message.reply_text("{}원으로 수정 완료.".format(maxbudget),reply_markup=reply_markup)
                
        except:
            update.message.reply_text("잘못 입력하셨습니다.")
            
    elif '그룹예산:' in  message.lower():
        message = message.lower().replace('그룹예산:', '')
        dailyBudget = message
        update.message.reply_text("변경할 광고그룹 아이디를 입력하세요.\nex) 광고그룹 아이디:grp-0123-456")
            
    elif '하루예산:' in message.lower():
        message = message.lower().replace('하루예산:', '')
        dailyBudget = message
        update.message.reply_text("변경할 캠페인아이디를 입력하세요.\nex) 캠페인 아이디:cmp-0123-456")
    
    elif '입찰가:' in message.lower():
        message = message.lower().replace('입찰가:', '')
        bidAmt = message
        update.message.reply_text("변경할 광고그룹 아이디를 입력하세요.\nex) 그룹광고 아이디:grp-0123-456")
                                           
                                           
    elif '캠페인아이디:' in message.lower():
        keyboard = []
        message = message.lower().replace('캠페인아이디:', '')
        ncccampaignid = message
        api_key,secret_key,customer_id,num, edit_stat,json = get_cmp_stat2(c_id,ncccampaignid)
        json[int(num)]['dailyBudget'] = dailyBudget

        uri = '/ncc/campaigns/'
        method = 'PUT'
        r = requests.put(BASE_URL + uri,params={'nccCampaignId':ncccampaignid},json=[json[int(num)]],headers=get_header(method, uri, api_key, secret_key, customer_id))
        if r.status_code == 200:
            api_key,secret_key,customer_id,num, stat,json = get_cmp_stat2(c_id,ncccampaignid)
            row = ''
            for i in range(len(stat)):
                row += '{} : {}\n\n'.format(stat['info'].index[i],stat['info'].values[i])
                
            bt1 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, 'naver_cmp_list']))
            keyboard.append([bt1])
            reply_markup = InlineKeyboardMarkup(keyboard)
            context.bot.sendMessage(text=row,chat_id=user_id,reply_markup=reply_markup)
        else:
            context.bot.sendMessage(text='잘못 입력 하셨습니다.\n70원에서 10억원까지 입력 가능(10원 단위 입력)',chat_id=user_id)


    elif '광고그룹아이디:' in message.lower():
        keyboard = []
        message = message.lower().replace('광고그룹아이디:', '')
        nccAdgroupId = message
        api_key,secret_key,customer_id,num, edit_stat,json = get_grp_stat2(c_id,nccAdgroupId)

        json = json[int(num)]
        json['dailyBudget'] = dailyBudget
        cmp_list = json['nccCampaignId']
        uri = '/ncc/adgroups/' + nccAdgroupId
        method = 'PUT'
        r = requests.put(BASE_URL + uri,json=json,headers=get_header(method, uri, api_key, secret_key, customer_id))
        if r.status_code == 200:
            api_key,secret_key,customer_id,num, stat,json = get_grp_stat2(c_id,nccAdgroupId)
            row = ''
            for i in range(len(stat)):
                row += '{} : {}\n\n'.format(stat['info'].index[i],stat['info'].values[i])
            bt1 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, cmp_list,'naver_grp_list']))
            keyboard.append([bt1])
            reply_markup = InlineKeyboardMarkup(keyboard)
            context.bot.sendMessage(text=row,chat_id=user_id,reply_markup=reply_markup)
            #context.bot.edit_message_reply_markup(chat_id=user_id ,message_id=update.message.message_id ,reply_markup=reply_markup) 
        else:
            context.bot.sendMessage(text='잘못 입력 하셨습니다.\n최소 70원부터 최대 10억원까지(VAT 제외 금액, 10원 단위로 입력)\nex) 그룹 예산:2000',chat_id=user_id)                                       
           

            
    elif '그룹광고아이디:' in message.lower():
        keyboard = []
        message = message.lower().replace('그룹광고아이디:', '')
        nccAdgroupId = message
        api_key,secret_key,customer_id,num, edit_stat,json = get_grp_stat2(c_id,nccAdgroupId)

        json = json[int(num)]
        json['bidAmt'] = bidAmt
        cmp_list = json['nccCampaignId']
        uri = '/ncc/adgroups/' + nccAdgroupId
        method = 'PUT'
        r = requests.put(BASE_URL + uri,json=json,headers=get_header(method, uri, api_key, secret_key, customer_id))
        if r.status_code == 200:
            api_key,secret_key,customer_id,num, stat,json = get_grp_stat2(c_id,nccAdgroupId)
            row = ''
            for i in range(len(stat)):
                row += '{} : {}\n\n'.format(stat['info'].index[i],stat['info'].values[i])
            
            bt1 = InlineKeyboardButton("<<  뒤로가기", callback_data='|'.join([c_id, cmp_list,'naver_grp_list']))
            keyboard.append([bt1])
            reply_markup = InlineKeyboardMarkup(keyboard)
            context.bot.sendMessage(text=row,chat_id=user_id,reply_markup=reply_markup)
            #context.bot.edit_message_reply_markup(chat_id=user_id ,message_id=update.message.message_id ,reply_markup=reply_markup) 
        else:
            context.bot.sendMessage(text='잘못 입력 하셨습니다.\n최소 70원부터 최대 100,000원까지(VAT 제외 금액, 10원 단위로 입력)\nex) 기본 입찰가:200',chat_id=user_id)                                       
                                           
                                           
    elif len(message) > 120: 
        conn = sqlite3.connect('/datadrive/Crontab/login.db')
        cur = conn.cursor()
        customer_id, api_key, secret_key= message.split(',')
        cur = conn.cursor()
        cur.execute('SELECT customer_id, api_key, secret_key FROM naver WHERE telegram_id = "{}" AND api_key = "{}" AND secret_key = "{}" AND customer_id = "{}"'.format(user_id,api_key,secret_key,customer_id))
        fet = cur.fetchall()
        if len(fet) > 0:
            update.message.reply_text("이미 등록된 값입니다.")

        else:
            uri = '/ncc/campaigns'
            method = 'GET'
            try:
                r = requests.get(BASE_URL + uri, params={'Id':customer_id}, 
                         headers=get_header(method, uri, api_key, secret_key, customer_id))
                sc = r.status_code
            except:
                sc = 400
            if sc == 200:
                info = [api_key, secret_key, customer_id, user_id]
                ins_sql='insert into naver values(?,?,?,?)'
                cur.executemany(ins_sql,[info])
                conn.commit()
                keyboard.append([InlineKeyboardButton("로그인", callback_data='login')])
                reply_markup = InlineKeyboardMarkup(keyboard)
                update.message.reply_text("{} 등록 완료.".format(customer_id),reply_markup=reply_markup)
                
            else:
                update.message.reply_text("계정 정보를 잘못 입력 하셨습니다.")
        
    else:
        update.message.reply_text("잘못 입력 하셨습니다.")
        
        
        
# def grp_budget_update(update, context):
#     user_id = check_id(update, context)
#     nickname = check_nickname(update, context)
    
#     conn = sqlite3.connect('/datadrive/Crontab/login.db')
#     cur = conn.cursor()
#     cur.execute('SELECT api_key,secret_key,customer_id FROM naver WHERE telegram_id = {}'.format(user_id))
#     fetch = cur.fetchone()
#     api_key = fetch[0]
#     secret_key = fetch[1]
#     customer_id = fetch[2]
#     cur.execute('SELECT telegram_id FROM naver WHERE api_key = "{}" AND secret_key = "{}" AND customer_id = {} '.format(fetch[0],fetch[1],fetch[2]))
#     fetch = cur.fetchall()
#     send_list = []
#     for i in range(len(fetch)):
#         send_list.append(fetch[i][0])

#     driver = 'ODBC Driver 17 for SQL Server'
#     server = 'vodasql.database.windows.net'
#     database = 'GateWay-DSP'
#     username = 'smartmind'
#     password = '1!tmakxmakdlsem'
#     conn2 = pyodbc.connect(f'Driver={driver};'
#                           f'Server={server};'
#                           f'Database={database};'
#                           f'uid={username};'
#                           f'pwd={password}')
#     cursor = conn2.cursor()
#     today = date.today().strftime('%Y-%m-%d')
#     yesterday = (date.today() - timedelta(1)).strftime('%Y-%m-%d')
#     df = pd.read_sql_query(f'''SELECT * FROM dbo.aa_naver_ads WHERE date = '{yesterday}' order by 'date_hour'
#                             ''',conn2)
#     ctr_dict = {}
#     ctrList = []
#     grp_id_list = df['adgroup_id'].unique().tolist()
#     #daily_budget
#     for i in grp_id_list:
#         impressions = df.loc[df['adgroup_id'] == i, :]['impCnt'].astype('float').sum()
#         clicks = df.loc[df['adgroup_id'] == i, :]['clkCnt'].astype('float').sum()
#         ctr = math.ceil(clicks / impressions * 10000)/100
#         ctrList.append(ctr)
#         ctr_dict[i] = ctr
#     ads = grp_id_list
#     ACTUAL_CTR = dict(zip(ads, ctrList))
#     random_dict = RS(ads,ACTUAL_CTR)
#     epsilon_dict = EG(ads,ACTUAL_CTR)
#     thompson_dict = TS(ads,ACTUAL_CTR)
#     ucb1_dict = UCB(ads,ACTUAL_CTR)
#     TotalReward = [random_dict['reward'], epsilon_dict['reward'], thompson_dict['reward'], ucb1_dict['reward']]
#     Algorithms = ['RandomSelection', 'EpsilonGreedy', 'ThompsonSampling', 'Ucb1']
#     comparisonList = dict(zip(Algorithms, TotalReward))
#     for key, value in comparisonList.items():
#         if value == max(comparisonList.values()):
#             opt_algorithms = key
#     if opt_algorithms == 'RandomSelection':
#         algorithm = random_dict
#     elif opt_algorithms == 'EpsilonGreedy':
#         algorithm = epsilon_dict
#     elif opt_algorithms == 'ThompsonSampling':
#         algorithm = thompson_dict
#     else:
#         algorithm = ucb1_dict
# #     print(comparisonList)
# #     print(opt_algorithms)
# #     print(algorithm['ads_count'])
#     budget_list = {}
#     for i in ads:
#         budget = math.floor(round(algorithm['ads_count'][i] * 4000)/10)*10
#         if budget < 70:
#             budget = 70
#         budget_list[i] = budget
#     grp_ids = list(budget_list.keys())
#     for k in budget_list:
#         uri = '/ncc/adgroups/' + k
#         method = 'GET'
#         r = requests.get(BASE_URL + uri, headers=get_header(method, uri, api_key, secret_key, customer_id))
#         create = r.json()
#         method = 'PUT'
#         create['dailyBudget'] = budget_list.get(k)
#         r = requests.put(BASE_URL + uri, params={'nccAdgroupId': k}, json=create, headers=get_header(method, uri, api_key, secret_key, customer_id))
#         #print("{}. response status_code = {}".format(k,r.status_code))
#     print(opt_algorithms)
#     row = ''
#     for k in budget_list:
#         row += df.loc[df['adgroup_id'] == k,'adgroup_name'].unique()[0]+' : '+str(budget_list.get(k))+'원\n'
#     for id in send_list:
#         context.bot.sendMessage(text=f'{today} 예산분배\n'+'선택된 알고리즘 : '+opt_algorithms+'\n'+row,chat_id=id)
        
    

if __name__ == "__main__":
    nn = datetime.now().strftime("%y%m%d")
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s')
    file_handler = logging.FileHandler('./telegram_bot_{}.log'.format(nn),encoding='utf-8')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    BASE_URL = 'https://api.naver.com'

    my_token = '1741553288:AAFjvybYZ3n4mnQJcUVbSgMIm64WGiTK6qo' # @Smartmind_Lighthouse_Bot
    print('start Smartmind_Lighthouse_Bot chat bot')
    conn = sqlite3.connect('/datadrive/Crontab/login.db')
    updater = Updater(my_token,use_context=True)
    # message_handler = MessageHandler(Filters.text, search)
    # updater.dispatcher.add_handler(message_handler)
    #updater.dispatcher.add_error_handler(error)잘못 입력하셨습니다.
    updater.dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), search))
    #updater.dispatcher.add_handler(MessageHandler(Filters.command, unknown))
    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CommandHandler('login', login))
    updater.dispatcher.add_handler(CommandHandler('register', register))
    #updater.dispatcher.add_handler(CommandHandler('update', grp_budget_update))
    updater.dispatcher.add_handler(CallbackQueryHandler(button))
    help_handler = CommandHandler('help', help_command)
    updater.dispatcher.add_handler(help_handler)
    updater.dispatcher.add_handler(CommandHandler('search', partial(search, offset=0)))

    updater.start_polling(timeout=3, clean=True) #drop_pending_updates

    updater.idle()
