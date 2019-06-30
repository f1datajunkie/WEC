---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# Simple Demo

Timing info available as PDF, eg [here](https://assets.lemans.org/explorer/pdf/courses/2019/24-heures-du-mans/classification/race/24-heures-du-mans-2019-classification-after-24h.pdf).

CSV data from Al Kamel using links of form:

`http://fiawec.alkamelsystems.com/Results/08_2018-2019/07_SPA%20FRANCORCHAMPS/267_FIA%20WEC/201905041330_Race/Hour%206/23_Analysis_Race_Hour%206.CSV`


*Links don't seem to appear on e.g. [classification data](http://fiawec.alkamelsystems.com/)? So where else might they be found?*
<!-- #endregion -->

```python
%matplotlib inline
import pandas as pd
```

```python
#Add the parent dir to the import path
import sys
sys.path.append("..")

#Import contents of the utils.py package in the parent directory
from py.utils import *
```

```python
url = 'http://fiawec.alkamelsystems.com/Results/08_2018-2019/07_SPA%20FRANCORCHAMPS/267_FIA%20WEC/201905041330_Race/Hour%206/23_Analysis_Race_Hour%206.CSV'
```

```python
laptimes = pd.read_csv(url, sep=';').dropna(how='all', axis=1)
laptimes.columns = [c.strip() for c in laptimes.columns]

#Tidy the data a little... car and driver number are not numbers
laptimes[['NUMBER','DRIVER_NUMBER']] = laptimes[['NUMBER','DRIVER_NUMBER']].astype(str)

laptimes.head()
```

```python
laptimes.columns
```

The `DRIVER_NUMBER` is relative to a car. It may be useful to also have a unique driver number, `CAR_DRIVER`:

```python
laptimes['CAR_DRIVER'] = laptimes['NUMBER'] + '_' + laptimes['DRIVER_NUMBER']
laptimes[['NUMBER','DRIVER_NUMBER','CAR_DRIVER']].head()
```

## Quick demo chart

Some simple plots to show how we can use widgets etc.

```python
laptimes['LAP_TIME_S'] = laptimes['LAP_TIME'].apply(getTime)
laptimes[['LAP_TIME','LAP_TIME_S']].head()
```

```python
from ipywidgets import interact


@interact(number=laptimes['NUMBER'].unique().tolist(),)
def plotLapByNumber(number):
    laptimes[laptimes['NUMBER']==number].plot(x='LAP_NUMBER',y='LAP_TIME_S')
```

```python
@interact(number=laptimes['NUMBER'].unique().tolist(),)
def plotLapByNumberDriver(number):
    # We can pivot long to wide on driver number, then plot all cols against the lapnumber index
    laptimes[laptimes['NUMBER']==number].pivot(index='LAP_NUMBER',columns='DRIVER_NUMBER', values='LAP_TIME_S').plot()

```

```python
@interact(number=laptimes['NUMBER'].unique().tolist(),)
def plotLapByNumberDriverWithPit(number):
    # We can pivot long to wide on driver number, then plot all cols against the lapnumber index
    #Grap the matplotli axes so we can overplot onto them
    ax = laptimes[laptimes['NUMBER']==number].pivot(index='LAP_NUMBER',columns='DRIVER_NUMBER', values='LAP_TIME_S').plot()
    #Also add in pit laps
    laptimes[(laptimes['NUMBER']==number) & (laptimes['CROSSING_FINISH_LINE_IN_PIT']=='B')].plot.scatter(x='LAP_NUMBER',y='LAP_TIME_S', ax=ax)
    
```

## Stint Detection

Some simple heuristics for detecting stints:

- car stint: between each pit stop;
- driver session: session equates to continuous period in car;
- driver stint: relative to pit stops; this may be renumbered for each session?

```python
#Driver session

#Create a flag to identify when we enter the pit, aka an INLAP
laptimes['INLAP'] = laptimes['CROSSING_FINISH_LINE_IN_PIT'] == 'B'

#Make no assumptions about table order - so sort by lap number
laptimes = laptimes.sort_values(['NUMBER','LAP_NUMBER'])

# Identify a new stint for each car by sifting the pitting / INLAP flag within car tables
laptimes['OUTLAP'] = laptimes.groupby('NUMBER')['INLAP'].shift(fill_value=True)

laptimes[['DRIVER_NUMBER', 'INLAP','OUTLAP']].head()
```

```python
#This is a count of the number of times a driver is in a vehicle after a pit who wasn't in it before
#Also set overall lap = 1 to be a driver change
laptimes['driverchange'] = (~laptimes['DRIVER_NUMBER'].eq(laptimes['DRIVER_NUMBER'].shift())) | (laptimes['LAP_NUMBER']==1)

laptimes['DRIVER_SESSION'] = laptimes.groupby(['NUMBER', 'DRIVER_NUMBER'])['driverchange'].cumsum().astype(int)
laptimes[['DRIVER_NUMBER', 'driverchange','DRIVER_SESSION','LAP_NUMBER']][42:48]
```

```python
# Car stint
#Create a counter for each pit stop - the pit flag is entering pit at end of stint
#  so a new stint applies on the lap after a pit
#Find the car stint based on count of pit stops
laptimes['CAR_STINT'] = laptimes.groupby('NUMBER')['OUTLAP'].cumsum().astype(int)

laptimes[['CROSSING_FINISH_LINE_IN_PIT', 'INLAP', 'OUTLAP', 'CAR_STINT']].head()
```

```python
#Driver stint - a cumulative count for each driver of their stints
laptimes['DRIVER_STINT'] = laptimes.groupby('CAR_DRIVER')['OUTLAP'].cumsum().astype(int)

#Let's also derive another identifier - CAR_DRIVER_STINT
laptimes['CAR_DRIVER_STINT'] = laptimes['CAR_DRIVER'] + '_' + laptimes['DRIVER_STINT'].astype(str)

laptimes[['CAR_DRIVER', 'CROSSING_FINISH_LINE_IN_PIT', 'INLAP','CAR_STINT', 'DRIVER_STINT', 'CAR_DRIVER_STINT']].tail(20).head(10)

```

```python
#Driver session stint - a count for each driver of their stints within a particular driving session
laptimes['DRIVER_SESSION_STINT'] = laptimes.groupby(['CAR_DRIVER','DRIVER_SESSION'])['OUTLAP'].cumsum().astype(int)
laptimes[['CAR_DRIVER', 'CROSSING_FINISH_LINE_IN_PIT', 'INLAP','CAR_STINT', 'DRIVER_STINT', 'CAR_DRIVER_STINT', 'DRIVER_SESSION_STINT']].head()

```

## Lap Counts Within Stints

It may be convenient to keep track of lap counts within stints.

```python
# lap count by car stint - that is, between each pit stop
laptimes['LAPS_CAR_STINT'] = laptimes.groupby(['NUMBER','CAR_STINT']).cumcount()+1

#lap count by driver
laptimes['LAPS_DRIVER'] = laptimes.groupby('CAR_DRIVER').cumcount()+1

#lap count by driver session
laptimes['LAPS_DRIVER_SESSION'] = laptimes.groupby(['CAR_DRIVER','DRIVER_SESSION']).cumcount()+1

#lap count by driver stint
laptimes['LAPS_DRIVER_STINT'] = laptimes.groupby(['CAR_DRIVER','DRIVER_STINT']).cumcount()+1

laptimes[['LAPS_CAR_STINT', 'LAPS_DRIVER', 'LAPS_DRIVER_SESSION', 'LAPS_DRIVER_STINT']].tail()
```

## Basic Individal Driver Reports

Using those additional columns, we should be able to start creating reports by driver by facetting on individual drivers.

(Note: it might be interesting to do some datasette demos with particular facets, which make it easy to select teams, drivers, etc.)

```python
import qgrid
qgrid.show_grid(laptimes[['LAP_NUMBER', 'NUMBER', 'CAR_DRIVER',  'INLAP', 'CAR_STINT', 
                          'CAR_DRIVER_STINT', 'DRIVER_STINT', 'DRIVER_SESSION', 'DRIVER_SESSION_STINT']])
```

## Simple Stint Reports

Using the various stint details, we can pull together a simple set of widgets to allow us to explore times by car / driver.

```python
import ipywidgets as widgets
from ipywidgets import interact
```

```python
cars = widgets.Dropdown(
    options=laptimes['NUMBER'].unique(), # value='1',
    description='Car:', disabled=False )

drivers = widgets.Dropdown(
    options=laptimes[laptimes['NUMBER']==cars.value]['CAR_DRIVER'].unique(),
    description='Driver:', disabled=False)

driversessions = widgets.Dropdown(
    options=laptimes[laptimes['CAR_DRIVER']==drivers.value]['DRIVER_SESSION'].unique(),
    description='Session:', disabled=False)

driverstints = widgets.Dropdown(
    options=laptimes[laptimes['DRIVER_SESSION']==driversessions.value]['DRIVER_SESSION_STINT'].unique(),
    description='Stint:', disabled=False)

def update_drivers(*args):
    driverlist = laptimes[laptimes['NUMBER']==cars.value]['CAR_DRIVER'].unique()
    drivers.options = driverlist
    
def update_driver_session(*args):
    driversessionlist = laptimes[(laptimes['CAR_DRIVER']==drivers.value)]['DRIVER_SESSION'].unique()
    driversessions.options = driversessionlist
    
def update_driver_stint(*args):
    driverstintlist = laptimes[(laptimes['CAR_DRIVER']==drivers.value) &
                               (laptimes['DRIVER_SESSION']==driversessions.value)]['DRIVER_SESSION_STINT'].unique()
    driverstints.options = driverstintlist
    
cars.observe(update_drivers, 'value')
drivers.observe(update_driver_session,'value')
driversessions.observe(update_driver_stint,'value')

def laptime_table(car, driver, driversession, driverstint):
    #just basic for now...
    display(laptimes[(laptimes['CAR_DRIVER']==driver) &
                     (laptimes['DRIVER_SESSION']==driversession) &
                     (laptimes['DRIVER_SESSION_STINT']==driverstint) ][['CAR_DRIVER', 'DRIVER_SESSION',
                                                         'DRIVER_STINT', 'DRIVER_SESSION_STINT',
                                                         'LAP_NUMBER','LAP_TIME', 'LAP_TIME_S']])
    
interact(laptime_table, car=cars, driver=drivers, driversession=driversessions, driverstint=driverstints);

```

```python
def laptime_chart(car, driver, driversession, driverstint):
    tmp_df = laptimes[(laptimes['CAR_DRIVER']==driver) &
                     (laptimes['DRIVER_SESSION']==driversession) &
                     (laptimes['DRIVER_SESSION_STINT']==driverstint) ][['CAR_DRIVER', 'DRIVER_SESSION',
                                                         'DRIVER_STINT', 'DRIVER_SESSION_STINT',
                                                         'LAP_NUMBER','LAP_TIME', 'LAP_TIME_S']]['LAP_TIME_S'].reset_index(drop=True)
    if not tmp_df.empty:
        tmp_df.plot()
        
interact(laptime_chart, car=cars, driver=drivers, driversession=driversessions, driverstint=driverstints);

```

```python
#Also add check boxes to suppress inlap and outlap?
inlaps = widgets.Checkbox( value=True, description='Inlap',
                           disabled=False )

outlaps = widgets.Checkbox( value=True, description='Outlap',
                           disabled=False )


#Plot laptimes by stint for a specified driver
def laptime_charts(car, driver, driversession, inlap, outlap):
    tmp_df = laptimes
    
    if not inlap:
        tmp_df = tmp_df[~tmp_df['INLAP']]
    if not outlap:
        tmp_df = tmp_df[~tmp_df['OUTLAP']]
        
    tmp_df = tmp_df[(tmp_df['CAR_DRIVER']==driver) &
                     (tmp_df['DRIVER_SESSION']==driversession) ].pivot(index='LAPS_DRIVER_STINT',
                                                                       columns='DRIVER_SESSION_STINT', 
                                                                       values='LAP_TIME_S').reset_index(drop=True)
    
    if not tmp_df.empty:
        tmp_df.plot()



interact(laptime_charts, car=cars, driver=drivers, driversession=driversessions, inlap=inlaps, outlap=outlaps);


```

## Simple Position Calculations

Some simple demonstrations of calculating position data.

Naively, calculate position based on lap number and accumulated time (there may be complications based on whether the lead car records a laptine from pit entry...).

```python
#Find accumulated time in seconds
laptimes['ELAPSED_S']=laptimes['ELAPSED'].apply(getTime)


#Check
laptimes['CHECK_ELAPSED_S'] = laptimes.groupby('NUMBER')['LAP_TIME_S'].cumsum()

laptimes[['ELAPSED','ELAPSED_S','CHECK_ELAPSED_S']].tail()
```

We can use the position to identify the leader on each lap and from that a count of leadlap number for each car:

```python
#Find position based on accumulated laptime
laptimes = laptimes.sort_values('ELAPSED_S')
laptimes['POS'] = laptimes.groupby('LAP_NUMBER')['ELAPSED_S'].rank()

#Find leader naively
laptimes['leader'] = laptimes['POS']==1

#Find lead lap number
laptimes['LEAD_LAP_NUMBER'] = laptimes['leader'].cumsum()

laptimes[['LAP_NUMBER','LEAD_LAP_NUMBER']].tail()
```

## Simple Position Chart - Top 10 At End

Find last lap number, then get top 10 on that lap.

```python
LAST_LAP = laptimes['LEAD_LAP_NUMBER'].max()
LAST_LAP
```

```python
#Find top 10 at end
cols = ['NUMBER','TEAM', 'DRIVER_NAME', 'CLASS','LAP_NUMBER','ELAPSED']
laptimes[laptimes['LEAD_LAP_NUMBER']==LAST_LAP].sort_values(['LEAD_LAP_NUMBER', 'POS'])[cols].head(10)

```

```python
laptimes.columns
```

```python

```
