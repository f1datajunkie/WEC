# -*- coding: utf-8 -*-
# # Live Timing
#
# The Al Kamel WEC live timing screens are driven from regulalry updated JSON feeds:

# ## Preview Data
#
# Let's get a preview of the data...

# +
import requests
import pandas as pd
from pandas.io.json import json_normalize

#Things like timing tools
from utils import getTime
# -


url = 'https://storage.googleapis.com/fiawec-prod/assets/live/WEC/__data.json'

# We need to add a timestamp to the URL to ensure we don't keep hitting a cached element.

# +
import time

def getMs():
    return int(round(time.time() * 1000))

def getUrl():
    return '{}?_t={}'.format(url, getMs())

getMs(),getUrl()

# +
# Grab the data
d = requests.get(getUrl())

# Preview raw JSON
#d.json()
# -

# We can use the *pandas* `json_normalize` to tabularise the data:

json_normalize(d.json(), 'entries' ).head()

json_normalize(d.json(), 'driversResult' )#.tail()

json_normalize(d.json()).drop(['entries','driversResult'], axis=1).T

# +
# Would it be netter for this to return a dict of dataframes, 
# then we don't have to worry about getting the right number of arguments?
DRIVERS_RESULT = 'drivers_result'
ENTRIES = 'car_entries'
META = 'race_meta'

ENTRIES_TEAM = 'entries_team'
ENTRIES_DRIVER = 'entries_driver'
ENTRIES_LAPS = 'entries_laps'

def get_data(url):
    ''' Get current timing screen data. '''
    
    def annotate_driversResult(d):
        ''' Annotate the driversResult dataframe. '''
        d['lastLapInS']=d['lastLap'].apply(getTime)
        d['bestLapInS']=d['bestLap'].apply(getTime)
        return d

    d = requests.get(url)
    dj = d.json()
    
    meta = json_normalize(dj).drop(['entries','driversResult'], axis=1)
    timestamp = int(meta['params.timestamp'].iloc[0])
    
    entries = json_normalize(dj, 'entries' )
    entries['timestamp'] = timestamp
    
    driversResult = json_normalize(dj, 'driversResult' )
    driversResult = annotate_driversResult(driversResult)
    driversResult['timestamp'] = timestamp
    
    entries_laps_cols = ['ranking','number','driver_id','driver','lap','gap','gapPrev',
                         'team','category','classGap','classGapPrev','lastlap','lastLapDiff','speed',
                         'currentSector1','currentSector2','currentSector3']
    entries_team_cols = ['number','id','isWEC','category','nationality','team','tyre','car']
    entries_driver_cols = ['number','id','driver_id','category','team','driver','car']
        
    entries_laps = entries[entries_laps_cols]
    entries_team = entries[entries_team_cols]
    entries_driver = entries[entries_driver_cols]
    
    tables = {META: meta,
              ENTRIES: entries,
              DRIVERS_RESULT: driversResult,
              ENTRIES_LAPS: entries_laps,
              ENTRIES_TEAM: entries_team,
              ENTRIES_DRIVER: entries_driver,
             }
    
    return tables


# -

tables = get_data(url)
#meta = tables[META]
#entries = tables[ENTRIES]
#, driversResult, entries_laps, entries_meta,

tables[META]

tables[ENTRIES_LAPS]

# ## Try Live Lap Times...
#  Can we grab and plot laptimes?

tables = get_data(getUrl())
tables[ENTRIES].iloc[0]

# Define a database schema for the timing data.
#
# We can also create some derived columns, such as lap times in seconds.

# +
raceMeta_table = '''
CREATE TABLE IF NOT EXISTS  "{_table}" (
    "params.eventName" TEXT,
    "params.SessionId" TEXT,
    "params.timestamp" INTEGER,
    "params.elapsedTime" TEXT,
    "params.racestate" TEXT,
    "params.safetycar" TEXT,
    "params.weather" TEXT,
    "params.airTemp" FLOAT,
    "params.trackTemp" FLOAT,
    "params.humidity" FLOAT,
    "params.pressure" FLOAT,
    "params.windSpeed" FLOAT,
    "params.windDirection" FLOAT,
    "params.elapsed" INTEGER,
    "params.remaining" INTEGER,
    "params.svg" TEXT,

    PRIMARY KEY (`params.sessionId`, `params.timestamp`) ON CONFLICT IGNORE
);
'''


driversResult_table = '''
CREATE TABLE IF NOT EXISTS  "{_table}" (
    "driverID" INTEGER,
    "laps" INTEGER,
    "bestLap" TEXT,
    "lastLap" TEXT,
    "drivingTime" INTEGER,
    "percentDriving" FLOAT,
    "bestLapNumber" INTEGER,
    "pitstop" INTEGER,
    "lastLapDiff" TEXT,
    "categoryPosition" FLOAT,

    "bestLapInS" FLOAT,
    "lastLapInS" FLOAT,
    
    "timestamp" INTEGER,

    PRIMARY KEY (driverID, laps) ON CONFLICT IGNORE
);
'''

#Perhaps create this as raw and then have some other tables and views (eg current sectors etc)
#that are more processed?
#If we save this each time we will have a lot of redundanct info?
# So maybe better to grab some things on first appearance, then things
# that are mutable into a more regularly updated table?

entries_table = '''
CREATE TABLE IF NOT EXISTS  "{_table}" (
    "timestamp" INTEGER,

    "ranking" INTEGER,
    "number" INTEGER,
    "id" INTEGER,
    "driver_id" INTEGER,
    "state" TEXT,
    "isWEC" TEXT,
    "category" TEXT,
    "nationality" TEXT,
    "team" TEXT,
    "tyre" TEXT,
    "driver" TEXT,
    "car" TEXT,
    "lap" INTEGER,
    "gap" TEXT,
    "av_laps" TEXT ,
    "av_time" TEXT,
    "d1l1" TEXT,
    "d2l1" TEXT,
    "d3l1" TEXT,  
    "gapPrev" TEXT,
    "classGap" TEXT,
    "classGapPrev" TEXT,
    "lastlap" TEXT,
    "lastLapDiff" TEXT,
    "pitstop" INTEGER,
    "bestlap" TEXT,
    "speed" FLOAT,
    "bestSector1" TEXT,
    "bestSector2" TEXT, 
    "bestSector3" TEXT,
    "currentSector1" TEXT,
    "currentSector2" TEXT,
    "currentSector3" TEXT,
    "sector" INTEGER,
    "lastPassingTime" INTEGER,
    "categoryPosition" INTEGER,
    "position.percent" FLOAT,
    "position.sector" INTEGER,
    "position.timestamp" INTEGER,
    
    PRIMARY KEY (timestamp, number) ON CONFLICT IGNORE
);
'''

entries_laps_table = '''
CREATE TABLE IF NOT EXISTS  "{_table}" (
    "driver" TEXT,
    "lap" INTEGER,
    
    "ranking" INTEGER,
    "number" INTEGER,
    "driver_id" INTEGER,
    "gap" TEXT,
    "gapPrev" TEXT,
    "team" TEXT,
    "category" TEXT,
    "classGap" TEXT,
    "classGapPrev" TEXT,
    "lastlap" TEXT,
    "lastLapDiff" TEXT,
    "speed" FLOAT,
    "currentSector1" TEXT,
    "currentSector2" TEXT,
    "currentSector3" TEXT,
    
    PRIMARY KEY (driver_id, lap) ON CONFLICT IGNORE
);
'''

entries_team_table = '''
CREATE TABLE IF NOT EXISTS  "{_table}" (
    "number" INTEGER,
    "team" TEXT,
    "car" TEXT,
    "category" TEXT,
    "nationality" TEXT,
    "id" INTEGER,
    "isWEC" TEXT,
    "tyre" TEXT,
    
    PRIMARY KEY (number) ON CONFLICT IGNORE
);
'''


entries_driver_table = '''
CREATE TABLE IF NOT EXISTS  "{_table}" (
    "driver" TEXT,
    "driver_id" INTEGER,
    "team" TEXT,
    "car" TEXT,
    "category" TEXT,
    "number" INTEGER,
    "id" INTEGER,
    
    PRIMARY KEY (number, driver_id) ON CONFLICT IGNORE
);
'''
# -

# Initialise the database tables...

import sqlite3
from sqlite_utils import Database

dbname='newtest.db'

# +
# #!rm $dbname

# +
conn = sqlite3.connect(dbname, timeout=10)

DB = Database(conn)

# +
#Setup database tables
c = conn.cursor()


c.executescript(raceMeta_table.format(_table=META))
c.executescript(driversResult_table.format(_table=DRIVERS_RESULT))
c.executescript(entries_table.format(_table=ENTRIES))

c.executescript(entries_laps_table.format(_table=ENTRIES_LAPS))

c.executescript(entries_team_table.format(_table=ENTRIES_TEAM))
c.executescript(entries_driver_table.format(_table=ENTRIES_DRIVER))


#to do - other tables

# -

#The conflict ignore in the table definition means we can ignore collisions on the insert
def updateDB(tables):
    ''' Update database.
        tables: a dict of dataframes, keyed by table name.
    '''
    for table in tables:
        DB[table].insert_all(tables[table].to_dict(orient='records'))



tables = get_data(getUrl())
updateDB(tables)

for t in DB.table_names():
    display( pd.read_sql('SELECT * FROM {} LIMIT 2'.format(t), conn) )


# +
#Check what's in the db
def getDriverLatestInDb():
    sql = 'SELECT driverID, MAX(laps) AS laps FROM {} GROUP BY driverID'.format(DRIVERS_RESULT)
    currdb = pd.read_sql_query(sql, conn)
    return currdb

currdb = getDriverLatestInDb()
currdb.head()

# +
# Compare the current df to the db
tables = get_data(getUrl())

dr = tables[DRIVERS_RESULT][['driverID','laps']]

# Generate a set of tuples from the most recent data grab showing records not in the db
# We can use these to drive the chart, before adding them to the db
new_records = pd.DataFrame(set([tuple(x) for x in dr.values])-set([tuple(x) for x in currdb.values]),
                           columns =['driverID', 'lap']).sort_values('driverID')
new_records.head()
# -
# ## Database Update Loop
#
# Simple callback routine for automatically updating the db.

from tornado.ioloop import PeriodicCallback
from tornado import gen


# +
@gen.coroutine
def dbupdater():
    tables = get_data(getUrl())
    updateDB(tables)
    
    #if ENDOFSESSION:
    #    cbdf.stop()



# +
periodInS = 20

periodInMilliS = periodInS * 1000

cbdf = PeriodicCallback(dbupdater, periodInMilliS)
cbdf.start()
# -

# !ls -Al $dbname

pd.read_sql('SELECT * FROM {}'.format(META), conn).head()

cbdf.stop()

# ## Streaming Charts
#
# Let's see if we can get some streaming charts going using data polled from the Al Kamel live timing JSON feed...
#
# We'll crib from http://holoviews.org/user_guide/Streaming_Data.html to start.
#
# See *ipynb/streaming-chart-poc.ipynb*.

# +
# %matplotlib inline

import pandas as pd
import holoviews as hv
import streamz
from holoviews.streams import Pipe

hv.extension('bokeh')

# +
import numpy as np

from holoviews.streams import Buffer

# -

# Try the plot...

# ### By Driver
#
# Chart laptimes for each driver.
#
# TO DO - how do we plot different traces on the chart?
#
# At the moment, perhaps just filter on a single driver.
#
# TO DO: better to do it on a car...

# +
# The buffer length is the max width of the chart?
# Do we need a buffer for each trace?
ncols = 2
buffer = Buffer(np.zeros((0, ncols)), length=50)

#Callback period in milliseconds
period = 20000

@gen.coroutine
def f():
    currdb = getDriverLatestInDb()
    
    tables = get_data(url)
    
    dr = tables[DRIVERS_RESULT][['driverID','laps']]
    
    print('s',set([tuple(x) for x in currdb.values]), 'x',set([tuple(x) for x in dr.values]))

    new_records = pd.DataFrame(set([tuple(x) for x in dr.values])-set([tuple(x) for x in currdb.values]),
                               columns =['driverID', 'lap'])
    #print(new_records)
    new_drivers_records = tables[DRIVERS_RESULT][tables[DRIVERS_RESULT]['driverID'].isin(new_records['driverID'])]
    if not new_drivers_records.empty:
        driverRecord = new_drivers_records[new_drivers_records['driverID']==402][['laps','lastLapInS']].values.tolist()
        #print(driverRecord)
        if driverRecord != [[]]:
            buffer.send(np.array(driverRecord))
        #buffer.send(np.array([[count, np.random.rand()]]))

    updateDB(tables)
    
cb = PeriodicCallback(f, period)
cb.start()

#Do we need to do one of these for each trace?
hv.DynamicMap(hv.Curve, streams=[buffer]).opts(padding=0.1, width=800)

# -

# !ls -Al

#Stop the data stream
cb.stop()

# ### By Car
#
# Chart laptimes for each car.


