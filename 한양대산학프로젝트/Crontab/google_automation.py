#!/usr/bin/env python
# coding: utf-8

# # <구글 자동화>

# # DB에서 광고별 CTR 구하기

# In[1]:


# 네 개의 같은 형식의 배너광고
idList = ['sumin', 'nyw-01', 'jungwon', 'tony']
idList


# In[37]:


# DB에서 불러오기
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

df = pd.read_sql_query('''SELECT * FROM dbo.aa_google_ads WHERE Campaign IN ('tony', 'jungwon', 'sumin','nyw-01') AND Day = (SELECT CONVERT(VARCHAR(50), GETDATE(), 23) AS 'TODAY') 
                       ''',conn)
df


# In[38]:


ctrList = []

for i in idList:
    impressions = df.loc[df['Campaign'] == i, :]['Impressions'].astype('int').sum()
    clicks = df.loc[df['Campaign'] == i, :]['Clicks'].astype('int').sum()
    ctr = round((clicks / impressions) * 100, ndigits=2)
    ctrList.append(ctr)


# In[39]:


ctrList


# # MAB(Multi-armed bandits) 알고리즘 성능 비교

# In[10]:


import numpy as np
import pandas as pd
from scipy.stats import beta, bernoulli
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots
import random
import math


# In[11]:


def algorithm_performance(chosen_ads, total_reward, regret_list):
    """
    Function that will show the performance of each algorithm we will be using in this tutorial.
    """

    # calculate how many time each Ad has been choosen
    count_series = pd.Series(chosen_ads).value_counts(normalize=True) # 1000번 중 광고가 노출될 확률
    print('Ad A has been shown', count_series['A']*100, '% of the time.') # 김수민
    print('Ad B has been shown', count_series['B']*100, '% of the time.') # 노연우
    print('Ad C has been shown', count_series['C']*100, '% of the time.') # 방정원
    print('Ad D has been shown', count_series['D']*100, '% of the time.') # 성형훈

    print('Total Reward (Number of Clicks):', total_reward) # print total Reward

    x = np.arange(0, n, 1)

    # plot the calculated CTR for Ad A
    trace0 = go.Scatter(x=x,
                       y=ctr['A'],
                       name='Calculated CTR for Ad A',
                       line=dict(color=('rgba(10, 108, 94, .7)'),
                                 width=2))


    # plot the calculated CTR for Ad B
    trace1 = go.Scatter(x=x,
                       y=ctr['B'],
                       name='Calculated CTR for Ad B',
                       line=dict(color=('rgba(187, 121, 24, .7)'),
                                 width=2))

    # plot the line with actual CTR for Ad C
    trace2 = go.Scatter(x=x,
                       y=ctr['C'],
                       name='Calculated CTR for Ad C',
                       line = dict(color = ('rgba(14, 50, 100, .7)'),
                                   width = 2))    

    
    # plot the line with actual CTR for Ad D
    trace3 = go.Scatter(x=x,
                       y=ctr['D'],
                       name='Calculated CTR for Ad D',
                       line = dict(color = ('rgba(110, 15, 105, .7)'),
                                   width = 2))   

    # plot the Regret values as a function of trial number
    trace4 = go.Scatter(x=x,
                       y=regret_list,
                       name='Regret')

    fig = make_subplots(rows=2, cols=1, shared_yaxes=True, shared_xaxes=True)

    fig.add_trace(trace0, row=1, col=1)
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=1)
    fig.add_trace(trace3, row=1, col=1)
    fig.add_trace(trace4, row=2, col=1)

    fig.update_layout(
        title='Simulated CTR Values and Algorithm Regret',
        xaxis={'title': 'Trial Number'},
        yaxis1={'title': 'CTR value'},
        yaxis2={'title': 'Regret Value'}
    )
    
    return fig


# In[12]:


ads = ['A', 'B' ,'C', 'D']
ACTUAL_CTR = dict(zip(ads, ctrList))


# ## <a id='random'>Random Selection</a> 
# 
# * *0% - Exploration*
# * *100% - Exploitation*
# 
# Let's start with the most naïve approach - Random Selection. The Random Selection algorithm doesn't do any Exploration, it just chooses randomly the Ad to show. 
# 
# You can think of it as coin flip - if you get heads you show Ad A, if you get tails you show Ad B. So if you have 2 ads, each add will appear ~50% (=100%/2) of the time. I guess you can tell already what are the disadvantages of this model, but let's look on simulation.

# In[13]:


# For each alrorithm we will perform 1000 trials
n = 1000


# In[14]:


regret = 0 
total_reward = 0
regret_list = [] # list for collecting the regret values for each impression (trial)
ctr = {'A': [], 'B': [], 'C': [], 'D': []} # lists for collecting the calculated CTR 
chosen_ads = [] # list for collecting the number of randomly choosen Ad

# set the initial values for impressions and clicks 
impressions = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
clicks = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

for i in range(n):    
    
    random_ad = np.random.choice(ads, p=[1/4, 1/4, 1/4, 1/4]) # randomly choose the ad
    chosen_ads.append(random_ad) # add the value to list
    
    impressions[random_ad] += 1 # add 1 impression value for the choosen Ad
    did_click = bernoulli.rvs(ACTUAL_CTR[random_ad]) # simulate if the person clicked on the ad usind Actual CTR value
  
    if did_click:
        clicks[random_ad] += did_click # if person clicked add 1 click value for the choosen Ad
    
    # calculate the CTR values and add them to list
    if impressions['A'] == 0:
        ctr_0 = 0
    else:
        ctr_0 = clicks['A']/impressions['A']
        
    if impressions['B'] == 0:
        ctr_1 = 0
    else:
        ctr_1 = clicks['B']/impressions['B']
    
    if impressions['C'] == 0:
        ctr_2 = 0
    else:
        ctr_2 = clicks['C']/impressions['C']
    
    if impressions['D'] == 0:
        ctr_3 = 0
    else:
        ctr_3 = clicks['D']/impressions['D']
        
    ctr['A'].append(ctr_0)
    ctr['B'].append(ctr_1)
    ctr['C'].append(ctr_2)
    ctr['D'].append(ctr_3)
    
    # calculate the regret and reward
    regret += max(ACTUAL_CTR.values()) - ACTUAL_CTR[random_ad]
    regret_list.append(regret)
    total_reward += did_click


# In[15]:


ACTUAL_CTR


# In[16]:


# save the reward and regret values for future comparison
random_dict = {
    'reward':total_reward, 
    'regret_list':regret_list, 
    'ads_count':pd.Series(chosen_ads).value_counts(normalize=True)
}


# ## <a id='epsilon'>Epsilon Greedy</a> 
# 
# * *~15% - Exploration*
# * *~85% - Exploitation*
# 
# The next approach is Epsilon-Greedy algorithm. Its logic can be defined as follows:
# 1. Run the experiment for some initial number of times (**Exploration**).
# 2. Choose the winning variant with the highest score for initial period of exploration.
# 3. Set the Epsilon value, **$\epsilon$**.
# 4. Run experiment with choosing the winning variant for **$(1-\epsilon)\% $** of the time and other options for **$\epsilon\%$** of the time (**Exploitation**).
# 
# Let's take a look at this algorithm in practice:

# In[17]:


e = .15 # set the Epsilon value
n_init = 1000 # number of impressions to choose the winning Ad

# set the initial values for impressions and clicks 
impressions = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
clicks = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

for i in range(n_init):
    random_ad = np.random.choice(ads, p=[1/4, 1/4, 1/4, 1/4]) # randomly choose the ad
    
    impressions[random_ad] += 1
    did_click = bernoulli.rvs(ACTUAL_CTR[random_ad])
    if did_click:
        clicks[random_ad] += did_click
        
ctr_0 = clicks['A'] / impressions['A']
ctr_1 = clicks['B'] / impressions['B']
ctr_2 = clicks['C'] / impressions['C']
ctr_3 = clicks['D'] / impressions['D']
win_index = np.argmax([ctr_0, ctr_1, ctr_2, ctr_3]) # select the Ad number with the highest CTR


print('After', n_init, 'initial trials Ad', ads[win_index], 
      'got the highest CTR =', round(np.max([ctr_0, ctr_1, ctr_2, ctr_3]), 2), 
      '(Real CTR value is', ACTUAL_CTR[ads[win_index]], ').'
      '\nIt will be shown', (1-e)*100, '% of the time.')


# In[18]:


regret = 0 
total_reward = 0
regret_list = [] # list for collecting the regret values for each impression (trial)
ctr = {'A': [], 'B': [], 'C': [], 'D': []} # lists for collecting the calculated CTR 
chosen_ads = [] # list for collecting the number of randomly choosen Ad

# set the initial values for impressions and clicks 
impressions = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
clicks = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

# update probabilities
p = np.full(len(ads), 1/len(ads))
p[:] = e / (len(p) - 1)
p[win_index] = 1 - e

for i in range(n):    
    win_ad = np.random.choice(ads, p=p) # randomly choose the ad # p는 뽑힐 확률
    chosen_ads.append(win_ad) # add the value to list
    
    impressions[win_ad] += 1 # add 1 impression value for the choosen Ad
    did_click = bernoulli.rvs(ACTUAL_CTR[win_ad]) # simulate if the person clicked on the ad usind Actual CTR value
    
    if did_click:
        clicks[win_ad] += did_click # if person clicked add 1 click value for the choosen Ad
    
    # calculate the CTR values and add them to list
    if impressions['A'] == 0:
        ctr_0 = 0
    else:
        ctr_0 = clicks['A']/impressions['A']
        
    if impressions['B'] == 0:
        ctr_1 = 0
    else:
        ctr_1 = clicks['B']/impressions['B']

    if impressions['C'] == 0:
        ctr_2 = 0
    else:
        ctr_2 = clicks['C']/impressions['C']
    
    if impressions['D'] == 0:
        ctr_3 = 0
    else:
        ctr_3 = clicks['D']/impressions['D']
        
    ctr['A'].append(ctr_0)
    ctr['B'].append(ctr_1)
    ctr['C'].append(ctr_2)
    ctr['D'].append(ctr_3)
    
    # calculate the regret and reward
    regret += max(ACTUAL_CTR.values()) - ACTUAL_CTR[win_ad]
    regret_list.append(regret)
    total_reward += did_click


# In[19]:


epsilon_dict = {
    'reward':total_reward, 
    'regret_list':regret_list, 
    'ads_count':pd.Series(chosen_ads).value_counts(normalize=True)}


# ## <a id='ts'>Thompson Sampling</a> 
# 
# * *50% - Exploration*
# * *50% - Exploitation*
# 
# The Thompson Sampling exploration part is more sophisticated than e-Greedy algorithm. We have been using **Beta distribution**\* here, however Thompson Sampling can be generalized to sample from any distributions over parameters.
# 
# > \*In probability theory and statistics, the **beta distribution** is a family of continuous probability distributions defined on the interval [0, 1] parametrized by two positive shape parameters, denoted by $\alpha$ and $\beta$, that appear as exponents of the random variable and control the shape of the distribution. 
# 
# *If you want to know more about Beta distribution here is an [article](http://varianceexplained.org/statistics/beta_distribution_and_baseball/) I found extremely useful.*
# 
# Logic:
# 
# 1. Choose prior distributions for parameters $\alpha$ and $\beta$.
# 2. Calculate the $\alpha$ and $\beta$ values as: $\alpha = prior + hits$, $\beta = prior + misses$. * In our case hits = number of clicks, misses = number of impressions without a click. Priors are useful if you have some prior information about the actual CTR’s of your ads. Here, we do not, so we’ll use 1.0.*
# 3. Estimate actual CTR’s by sampling values of Beta distribution for each variant $B(\alpha_i, \beta_i)$ and choose the sample with the highest value (estimated CTR).
# 4. Repeat 2-3.

# In[20]:


regret = 0 
total_reward = 0
regret_list = [] 
ctr = {'A': [], 'B': [], 'C': [], 'D': []}
index_list = [] 
impressions = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
clicks = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
priors = {'A': 1, 'B': 1, 'C': 1, 'D': 1} # prior 정보 없어서 1로 사용

win_ad = np.random.choice(ads, p=[1/4, 1/4, 1/4, 1/4]) ## randomly choose the first shown Ad

for i in range(n):    
    
    impressions[win_ad] += 1
    did_click = bernoulli.rvs(ACTUAL_CTR[win_ad])
    if did_click:
        clicks[win_ad] += did_click
    
    ctr_0 = random.betavariate(priors['A']+clicks['A'], priors['B'] + impressions['A'] - clicks['A'])
    ctr_1 = random.betavariate(priors['A']+clicks['B'], priors['B'] + impressions['B'] - clicks['B'])
    
    ctr_2 = random.betavariate(priors['A']+clicks['C'], priors['C'] + impressions['C'] - clicks['C'])
    ctr_3 = random.betavariate(priors['A']+clicks['D'], priors['D'] + impressions['D'] - clicks['D'])
    
    win_ad = ads[np.argmax([ctr_0, ctr_1, ctr_2, ctr_3])]
    chosen_ads.append(win_ad)
    
    ctr['A'].append(ctr_0)
    ctr['B'].append(ctr_1)
    ctr['C'].append(ctr_2)
    ctr['D'].append(ctr_3)
    
    
    regret += max(ACTUAL_CTR.values()) - ACTUAL_CTR[win_ad]
    regret_list.append(regret)    
    total_reward += did_click


# In[21]:


thompson_dict = {
    'reward':total_reward, 
    'regret_list':regret_list, 
    'ads_count':pd.Series(chosen_ads).value_counts(normalize=True)}


# ## <a id='ucb'>Upper Confidence Bound (UCB1)</a> 
# 
# * *50% - Exploration*
# * *50% - Exploitation*
# 
# Unlike the Thompson Sampling algorithm, the Upper Confidence Bound cares more about the uncertainty (high variation) of each variant. The more uncertain we are about one variant, the more important it is to explore. 
# 
# Algorithm chooses the variant with the highest upper confidence bound value (UCB) which represents the highest reward guess for the variant. It is defind as follows:
# 
# $UCB = \bar x_i + \sqrt{\frac{2 \cdot \log{t}}{n}}$ ,
# 
# where $\bar x_i$ - the (CTR rate) for $i$-th step,
# 
# $t$ - total number of (impressions) for all variants,
# 
# $n$ - total number of (impressions) for choosen variant
# 
# The logic is rather straightforward:
# 
# 1. Calculate the UCB for all variants.
# 2. Choose the variant with the highest UCB.
# 3. Go to 1.

# In[22]:


regret = 0 
total_reward = 0
regret_list = [] 
ctr = {'A': [], 'B': [], 'C': [], 'D': []}
index_list = [] 
impressions = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
clicks = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

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


# In[23]:


ucb1_dict = {
    'reward':total_reward, 
    'regret_list':regret_list, 
    'ads_count':pd.Series(chosen_ads).value_counts(normalize=True)}


# In[24]:


TotalReward = [random_dict['reward'], epsilon_dict['reward'], thompson_dict['reward'], ucb1_dict['reward']]


# In[25]:


Algorithms = ['RandomSelection', 'EpsilonGreedy', 'ThompsonSampling', 'Ucb1']


# In[26]:


comparisonList = dict(zip(Algorithms, TotalReward))
comparisonList


# In[27]:


for key, value in comparisonList.items():
    if value == max(comparisonList.values()):
        BestAlgorithms = key
        print(key)    


# # 가장 성능 좋은 알고리즘으로 최적화하기

# In[28]:


assignment = pd.Series()


# In[29]:


if BestAlgorithms == 'RandomSelection':
    assignment = random_dict['ads_count']
    print(random_dict['ads_count'])
elif BestAlgorithms == 'EpsilonGreedy':
    assignment = epsilon_dict['ads_count']
    print(epsilon_dict['ads_count'])
elif BestAlgorithms == 'ThompsonSampling':
    assignment = thompson_dict['ads_count']
    print(thompson_dict['ads_count'])
elif BestAlgorithms == 'Ucb1':
    assignment = ucb1_dict['ads_count']
    print(ucb1_dict['ads_count'])


# # 예산 변경하기

# In[30]:


def main():
    # Initialize both campaign and budget services
    campaign_service = client.GetService('CampaignService', version='v201809')
    budget_service = client.GetService('BudgetService', version='v201809')

    # Create a budget ID with no name, specify that it is not a shared budget
    # Establish budget as a micro amount
    budget = {
        'name': None,
        'isExplicitlyShared': 'false',
        'amount': {
            'microAmount': microAmount[i]
            }
        }

    budget_operations = [{
        'operator': 'ADD',
        'operand': budget
        }]

    budget_id = budget_service.mutate(budget_operations)['value'][0]['budgetId']


    # Construct operations and update campaign.
    operations = [{
        'operator': 'SET',
        'operand': {
        'id': campaign_id,
        'budget': {
            'budgetId': budget_id
            }
        }
    }]

    campaigns = campaign_service.mutate(operations)


# In[31]:


assignment = assignment.sort_index()
assignment


# In[32]:


microAmount = []

for i in assignment:
    microAmount.append(int(i * 4000000000))


# In[33]:


microAmount


# In[34]:


campaignId = [12648918438, 12648575430, 12648482372, 12648942942]


# In[36]:


from googleads import adwords

client = adwords.AdWordsClient.LoadFromStorage('googleads.yaml')
for i in range(4):
    client = adwords.AdWordsClient.LoadFromStorage('googleads.yaml')
    client.SetClientCustomerId('274-559-3213')

    campaign_id = campaignId[i] 
    main()

