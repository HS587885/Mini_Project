#!/usr/bin/env python
# coding: utf-8

# In[7]:


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

df = pd.read_sql_query('''SELECT * FROM dbo.aa_google_ads 
      ''',conn)
df


# In[3]:


server = 'vodasql.database.windows.net'
database = 'GateWay-DSP' 
username = 'smartmind'
password = '1!tmakxmakdlsem' 
engn = create_engine(f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server")


# In[5]:


from sqlalchemy import create_engine, select, delete, text

with engn.begin() as conn:
    conn.execute(text('''DELETE FROM dbo.aa_google_ads
     WHERE Day = (SELECT CONVERT(VARCHAR(50), GETDATE(), 23) AS 'TODAY')
         '''))


# In[6]:


check1 = pd.read_sql_query('''SELECT Campaign, date_hour , count(*) AS Count FROM dbo.aa_google_ads GROUP BY Campaign, date_hour HAVING count(*) > 1 
                        ''',engn)
check1

