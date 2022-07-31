#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 특정 날짜를 포함시키고 싶으면 during에 start date, end date 입력하고 select에 Date넣기
# 날짜를 포함시키고 싶지 않으면 during 없애기

from googleads import adwords
import pandas as pd
import numpy as np
import io # 웹정보를 제공하는 정해진 알고리즘으로 불러와 사용자에게 필요한 정보로 변환시켜주는 서비스를 제공
pd.set_option("display.max_columns",500)
pd.set_option("display.max_rows",500)

# Define output as a string
output = io.StringIO() # 문자열을 텍스트 파일처럼 취급할 수 있게 해줌

# Initialize appropriate service.
adwords_client = adwords.AdWordsClient.LoadFromStorage('/datadrive/Crontab/googleads.yaml')

report_downloader = adwords_client.GetReportDownloader(version='v201809')

# Create report query.
report_query = ('''
select Date, HourOfDay,CampaignName,CampaignId,EngagementRate, TotalAmount, Amount,AdvertisingChannelType,Conversions,ConversionRate,CostPerConversion, AverageCpm, Interactions,Clicks,Cost,AverageCpe,Impressions,Ctr,AverageCpc,  AbsoluteTopImpressionPercentage, AccountCurrencyCode, AccountDescriptiveName, AccountTimeZone, ActiveViewCpm, ActiveViewCtr, ActiveViewImpressions, ActiveViewMeasurability, ActiveViewMeasurableCost, ActiveViewMeasurableImpressions, ActiveViewViewability,  AdvertisingChannelSubType, AdvertisingChannelType, AllConversions,  Amount, AverageCost, AverageCpc, AverageCpe, AverageCpm, AverageCpv, AveragePosition, BaseCampaignId, BiddingStrategyId, BiddingStrategyName, BiddingStrategyType, BudgetId, CampaignDesktopBidModifier, CampaignGroupId, CampaignId, CampaignMobileBidModifier, CampaignName, CampaignStatus, CampaignTabletBidModifier, CampaignTrialType, DayOfWeek, EndDate, Engagements, EnhancedCpcEnabled, ExternalCustomerId, FinalUrlSuffix, GmailForwards, GmailSaves, GmailSecondaryClicks, HasRecommendedBudget,  Impressions, InteractionRate, Interactions, InteractionTypes, IsBudgetExplicitlyShared, LabelIds, Labels, MaximizeConversionValueTargetRoas, Month, MonthOfYear, NumOfflineImpressions, NumOfflineInteractions, OfflineInteractionRate, Period, Quarter, RecommendedBudgetAmount, SearchAbsoluteTopImpressionShare, SearchBudgetLostAbsoluteTopImpressionShare, SearchBudgetLostImpressionShare, SearchBudgetLostTopImpressionShare, SearchClickShare, SearchExactMatchImpressionShare, SearchImpressionShare, SearchRankLostAbsoluteTopImpressionShare, SearchRankLostImpressionShare, SearchRankLostTopImpressionShare, SearchTopImpressionShare, ServingStatus, StartDate, TopImpressionPercentage, TrackingUrlTemplate, UrlCustomParameters,   ViewThroughConversions, Week, Year
from CAMPAIGN_PERFORMANCE_REPORT
where CampaignStatus = 'ENABLED'
during TODAY  ''')
#where CampaignStatus = 'ENABLED'
# Write query result to output file
report_downloader.DownloadReportWithAwql(
    report_query, 
    'CSV',
    output,
    client_customer_id='274-559-3213', # denotes which ad account to pull from
    skip_report_header=True, 
    skip_column_header=False,
    skip_report_summary=True,
    include_zero_impressions=True)


output.seek(0)

df_108 = pd.read_csv(output)

#df.sort_values(by='Day',inplace=True)
df_108


# In[3]:


data = df_108 
data.tail()


# In[4]:


data = data.reset_index().drop('index', axis=1)
data.head()


# In[5]:


data['Hour of day'] = data['Hour of day'].astype(str)


# In[6]:


for i in range(data.shape[0]):
    if len(data['Hour of day'][i]) == 2 :
        data['Hour of day'][i] = data['Hour of day'][i] + ":00:00"
    else:
        data['Hour of day'][i] = "0" + data['Hour of day'][i] + ":00:00"

data['date_hour'] = data['Day'].str.cat(data['Hour of day'], sep=' ')
data['date_hour'] = pd.to_datetime(data['date_hour'])


# In[7]:


data.tail(3)


# In[8]:


data['Budget'] = data['Budget']/1000000
data.head()


# In[9]:


data.loc[data['Campaign'] == 'jungwon', :].sort_values(by='date_hour')


# In[10]:


aa_google_ads = data
aa_google_ads


# In[11]:


# aa_google_ads.reset_index(inplace=True)


# In[12]:


aa_google_ads


# In[13]:


#aa_google_ads.drop(labels = 'level_0', axis =1)


# In[14]:


from sqlalchemy.types import NVARCHAR, Float, Date, DateTime, BigInteger 
import pyodbc
from sqlalchemy import create_engine


# In[15]:


server = 'vodasql.database.windows.net'
database = 'GateWay-DSP' 
username = 'smartmind'
password = '1!tmakxmakdlsem' 
engn = create_engine(f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server")
col ={"index" :NVARCHAR(length="30"),
"Day" :Date,
"Hour of day" :NVARCHAR(length="30"),
"Campaign" :NVARCHAR(length="30"),
"Campaign ID" :NVARCHAR(length="30"),
"Engagement rate" :NVARCHAR(length="30"),
"Total Budget amount" :NVARCHAR(length="30"),
"Budget" :NVARCHAR(length="30"),
"Advertising Channel" :NVARCHAR(length="30"),
"Conversions" :NVARCHAR(length="30"),
"Conv. rate" :NVARCHAR(length="30"),
"Cost / conv." :NVARCHAR(length="30"),
"Avg. CPM" :NVARCHAR(length="30"),
"Interactions" :NVARCHAR(length="30"),
"Clicks" :BigInteger,
"Cost" :NVARCHAR(length="30"),
"Avg. CPE" :NVARCHAR(length="30"),
"Impressions" :BigInteger,
"CTR" :NVARCHAR(length="30"),
"Avg. CPC" :NVARCHAR(length="30"),
"Impr. (Abs. Top) %" :NVARCHAR(length="30"),
"Currency" :NVARCHAR(length="30"),
"Account" :NVARCHAR(length="30"),
"Time zone" :NVARCHAR(length="30"),
"Active View avg. CPM" :NVARCHAR(length="30"),
"Active View viewable CTR" :NVARCHAR(length="30"),
"Active View viewable impressions" :NVARCHAR(length="30"),
"Active View measurable impr. / impr." :NVARCHAR(length="30"),
"Active View measurable cost" :NVARCHAR(length="30"),
"Active View measurable impr." :NVARCHAR(length="30"),
"Active View viewable impr. / measurable impr." :NVARCHAR(length="30"),
"Advertising Sub Channel"  :NVARCHAR(length="30"),
"All conv." :NVARCHAR(length="30"),
"Avg. Cost" :NVARCHAR(length="30"),
"Avg. CPV" :NVARCHAR(length="30"),
"Avg. position" :NVARCHAR(length="30"),
"Base Campaign ID" :NVARCHAR(length="30"),
"Bid Strategy ID":NVARCHAR(length="30"),
"Bid Strategy Name" :NVARCHAR(length="30"),
"Bid Strategy Type" :NVARCHAR(length="30"),
"Budget ID" :NVARCHAR(length="30"),
"Desktop bid adj." :NVARCHAR(length="30"),
"Campaign Group ID" :NVARCHAR(length="30"),
"Mobile bid adj." :NVARCHAR(length="30"),
"Campaign state" :NVARCHAR(length="30"),
"Tablet bid adj." :NVARCHAR(length="30"),
"Campaign Trial Type" :NVARCHAR(length="30"),
"Day of week" :NVARCHAR(length="30"),
"End date" :NVARCHAR(length="30"),
"Engagements" :NVARCHAR(length="30"),
"Enhanced CPC enabled" :NVARCHAR(length="30"),
"Customer ID" :NVARCHAR(length="30"),
"Final URL suffix" :NVARCHAR(length="30"),
"Gmail forwards" :NVARCHAR(length="30"),
"Gmail saves":NVARCHAR(length="30"),
"Gmail clicks to website" :NVARCHAR(length="30"),
"Has recommended Budget" :NVARCHAR(length="30"),
"Interaction Rate" :NVARCHAR(length="30"),
"Interaction Types" :NVARCHAR(length="30"),
"Budget explicitly shared" :NVARCHAR(length="30"),
"Label IDs" :NVARCHAR(length="30"),
"Labels" :NVARCHAR(length="30"),
"Target ROAS (Maximize Conversion Value)" :NVARCHAR(length="30"),
"Month" :NVARCHAR(length="30"),
"Month of Year" :NVARCHAR(length="30"),
"Phone impressions" :NVARCHAR(length="30"),
"Phone calls" :NVARCHAR(length="30"),
"PTR" :NVARCHAR(length="30"),
"Budget period" :NVARCHAR(length="30"),
"Quarter" :NVARCHAR(length="30"),
"Recommended Budget amount":NVARCHAR(length="30"),
"Search abs. top IS" :NVARCHAR(length="30"),
"Search lost abs. top IS (budget)" :NVARCHAR(length="30"),
"Search Lost IS (budget)" :NVARCHAR(length="30"),
"Search lost top IS (budget)" :NVARCHAR(length="30"),
"Click share" :NVARCHAR(length="30"),
"Search Exact match IS" :NVARCHAR(length="30"),
"Search Impr. share" :NVARCHAR(length="30"),
"Search lost abs. top IS (rank)" :NVARCHAR(length="30"),
"Search Lost IS (rank)" :NVARCHAR(length="30"),
"Search lost top IS (rank)" :NVARCHAR(length="30"),
"Search top IS" :NVARCHAR(length="30"),
"Campaign serving status" :NVARCHAR(length="30"),
"Start date" :NVARCHAR(length="30"),
"Impr. (Top) %" :NVARCHAR(length="30"),
"Tracking template" :NVARCHAR(length="30"),
"Custom parameter" :NVARCHAR(length="30"),
"Value / all conv." :NVARCHAR(length="30"),
"Value / conv." :NVARCHAR(length="30"),
"View-through conv." :NVARCHAR(length="30"),
"Week" :NVARCHAR(length="30"),
"Year" :NVARCHAR(length="30"),
"date_hour" :DateTime
      
     }



# In[17]:


import pandas as pd
from tqdm import tqdm

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def insert_with_progress(aa_google_ads):
    con = create_engine(f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server", fast_executemany = True)
    aa_google_ads.to_sql("aa_google_ads", con=con, if_exists="append", index=False, dtype = col )


insert_with_progress(aa_google_ads)




from sqlalchemy import create_engine, select, delete, text






