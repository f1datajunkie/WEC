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

#Create a flag to identify when we enter the pit
laptimes['pitting'] = laptimes['CROSSING_FINISH_LINE_IN_PIT'] == 'B'
# Identify a new stint for each car by sifting the pitting flag within car tables
laptimes['newstint'] = laptimes.groupby('NUMBER')['pitting'].shift(fill_value=True)

laptimes[['DRIVER_NUMBER', 'pitting','newstint']].head()
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
laptimes['CAR_STINT'] = laptimes.groupby('NUMBER')['newstint'].cumsum().astype(int)

laptimes[['CROSSING_FINISH_LINE_IN_PIT', 'pitting', 'newstint', 'CAR_STINT']].head()
```

```python
#Driver stint - a cumulative count for each driver of their stints
laptimes['DRIVER_STINT'] = laptimes.groupby('CAR_DRIVER')['newstint'].cumsum().astype(int)

#Let's also derive another identifier - CAR_DRIVER_STINT
laptimes['CAR_DRIVER_STINT'] = laptimes['CAR_DRIVER'] + '_' + laptimes['DRIVER_STINT'].astype(str)

laptimes[['CAR_DRIVER', 'CROSSING_FINISH_LINE_IN_PIT', 'pitting','CAR_STINT', 'DRIVER_STINT', 'CAR_DRIVER_STINT']].tail(20).head(10)

```

```python
#Driver session stint - a count for each driver of their stints within a particular driving session
laptimes['DRIVER_SESSION_STINT'] = laptimes.groupby(['CAR_DRIVER','DRIVER_SESSION'])['newstint'].cumsum().astype(int)
laptimes[['CAR_DRIVER', 'CROSSING_FINISH_LINE_IN_PIT', 'pitting','CAR_STINT', 'DRIVER_STINT', 'CAR_DRIVER_STINT', 'DRIVER_SESSION_STINT']].head()
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
qgrid.show_grid(laptimes[['LAP_NUMBER', 'NUMBER', 'CAR_DRIVER',  'pitting', 'CAR_STINT', 
                          'CAR_DRIVER_STINT', 'DRIVER_STINT', 'DRIVER_SESSION', 'DRIVER_SESSION_STINT']])
```

```python

```

```python

```
