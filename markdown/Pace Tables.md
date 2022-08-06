---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Pace Tables & Charts

The race history chart provides a useful way of reviewing the evolution of a race, but sometimes it can be hard to read off how much faster or slower one car is than another at any particular point in a race.

*Pace tables* and *pace charts* are an attempt to try to highlight pace differences for drivers on each lap of the race.


## Pace Tables

*Pace tables* use driver relative rebased laptime deltas to highlight the pace difference for a named driver relative to other drivers.

Pace may be given as a speed delta (in terms of s / km difference) or as a laptime delta.

The s / km pace is given as:

`laptime_delta / circuit length`

```python
%matplotlib inline

import pandas as pd
```

```python
url = 'http://fiawec.alkamelsystems.com/Results/08_2018-2019/07_SPA%20FRANCORCHAMPS/267_FIA%20WEC/201905041330_Race/Hour%206/23_Analysis_Race_Hour%206.CSV'

```

```python
laptimes = pd.read_csv(url, sep=';').dropna(how='all', axis=1)
laptimes.columns = [c.strip() for c in laptimes.columns]

#Tidy the data a little... car and driver number are not numbers
laptimes[['NUMBER','DRIVER_NUMBER']] = laptimes[['NUMBER','DRIVER_NUMBER']].astype(str)

# Enrichment
# Add the parent dir to the import path
import sys
sys.path.append("../py")

# Import contents of the utils.py package in the parent directory
from utils import *

# Get laptimes in seconds
laptimes['LAP_TIME_S'] = laptimes['LAP_TIME'].apply(getTime)

# Find accumulated time in seconds
laptimes['ELAPSED_S']=laptimes['ELAPSED'].apply(getTime)
laptimes['PIT_TIME_S']=laptimes['PIT_TIME'].apply(getTime)

laptimes['S1_S']=laptimes['S1'].apply(getTime)
laptimes['S2_S']=laptimes['S2'].apply(getTime)
laptimes['S3_S']=laptimes['S3'].apply(getTime)

# Find position based on accumulated laptime
laptimes = laptimes.sort_values('ELAPSED_S')
laptimes['POS'] = laptimes.groupby('LAP_NUMBER')['ELAPSED_S'].rank()

# Find leader naively
laptimes['leader'] = laptimes['POS']==1

# Find lead lap number
laptimes['LEAD_LAP_NUMBER'] = laptimes['leader'].cumsum()

laptimes.head()
```

Lap time deltas can be calculated for each lap as follows, rebasing the deltas relative to the laptimes for a particular car as given by the `rebase` car number:

```python
laptimes_wide = laptimes.pivot(index='NUMBER',
                                columns='LAP_NUMBER',
                                values='LAP_TIME_S')

rebase = "8" # Find lap time deltas to the corresponding lap time for this car

pace = (laptimes_wide - laptimes_wide.loc[rebase])
pace.head()
```

We can style the table to show how much time the rebased car made or lost on each lap compared to other cars.

```python
#In a live notebook, this produces a styled bar chart within each cell
pace.T[['3','11']].head(10).style.bar(align='zero', color=['#d65f5f', '#5fba7d'])
```

Note that we can change the direction of the colourway depending on who the anticipated reader of the chart is. For example, if we take the perspective of someone associated with the rebased car, they may want to know where they gained time (a Good Thing) compared to other cars, in which case we might colour slower (larger, positive delta) laptimes for the other cars as *green* and faster (lower, negative delta) laptimes as *red*.

However, if we want to see where other cars are *gaining* on the rebased car (a Good Thing, perhaps, under this reading) we may want to colour the faster (lower, negative delta) times as *green* and the slower (larger, positive delta) times as *red*.


An alternative but equivalent way of manipulating the data is to select on the basis of index value, and then transpose the dataframe.

```python
pace.loc[['8','3','11']].T.head()
```

Data in this wide format, indexed by lapnumber and with columns containing rebased delta times on each lap for each car, can be plotted from directly:

```python
pace.loc[['8','3','11']].T.plot();
```

We can also generate the accumulated delta times for each car:

```python
pace.loc[['8','3','11']].T.cumsum().head()
```

And then plot those:

```python
ax = pace.loc[['8','3','11']].T.cumsum().plot();

#Need to overplot each with pit events
inpitlaps = laptimes[(laptimes['NUMBER']==rebase) & ~(laptimes['PIT_TIME'].isnull())][:]
inpitlaps.loc[:,'y']=0
inpitlaps.plot.scatter(x='LAP_NUMBER',y='y', ax=ax);
```

### Plotting Using plotly Charts

We can also plot charts using the *plotly* charting library.

This provides us with a slightly different grammar for creating charts, albeit equally, if not more expressive, than the basic *pandas* plotting tools.

To create a simple time series chart, we need to put the data into a long format:

```python
long_pace = pace.loc[['8','3','11']].T.cumsum().reset_index().melt(id_vars='LAP_NUMBER',
                                                                   value_name='ACCUMULATED_DELTA')
long_pace.head()
```

We can then plot directly from this dataframe:

```python
import plotly.express as px

fig = px.line(long_pace, x="LAP_NUMBER", y="ACCUMULATED_DELTA", color='NUMBER')
fig.show()
```

We can overplot the chart through the addition of extra traces:

```python
import plotly.graph_objects as go
fig.add_trace(go.Scatter(x=inpitlaps['LAP_NUMBER'], y=inpitlaps['y'],
                         mode='markers', name='Pit stops'))
```

## Plotting Against Elapsed Time

One of the problems with this view is if cars are several laps apart, in which case to compare pace we really need to be comparing times relative to the lead lap or elapsed time, otherwise we may be comparing laptimes recorded at very different elapsed race times and possibly different conditions.

```python
laptimes_wide_elapsed = laptimes.pivot(index='NUMBER',
                                columns='LAP_NUMBER',
                                values='LAP_TIME_S')
rebase = "8"
pace = (laptimes_wide - laptimes_wide.loc[rebase])
```

```python
#merge the pace back in to laptimes, then plot against the elapsed time
#We can get the delats by NUMBER and LAP_NUMBER
pd.melt(pace.reset_index(), id_vars=['NUMBER']).head()
```

```python
#Merge this with the elapsed time
tmp = pd.melt(pace.reset_index(),
              id_vars=['NUMBER']).merge(laptimes[['NUMBER','LAP_NUMBER','ELAPSED_S']])
tmp.head()
```

```python
tmp['cumvalue'] = tmp.groupby('NUMBER')['value'].cumsum()
```

```python
fig = px.line(tmp[tmp['NUMBER'].isin(['11','3','8'])],
              x="ELAPSED_S", y="value", color='NUMBER')
fig.show()
```

```python
fig = px.line(tmp[tmp['NUMBER'].isin(['11','3','8'])],
              x="ELAPSED_S", y="cumvalue", color='NUMBER')
fig.show()
```

### Plotting Against Lead Lap

One of the difficulties associated with performing analysis against the elapsed race time is the continuous nature of the elapsed time variable.

A more convenient discrete time basis is the lead lap count, rather than the lap number an individual car is on.

Variously, a car may record zero, one or more laptimes for any given lead lap:

- *zero*: for example, when a car is lapped by the leader;
- *one*: a car is lapping at about the same rate as the leader;
- *two or more*: a car unlaps itself, for example if the leader has pitted or as part of an unwinding during a safety car period.

For convenience, where a car records more than one laptime for a given lead lap, we might allocate it a lap time according to the last lap it completed on the lead lap and then rebase from that.

Note that this approach may lead to a loss of laptime information which might make a nonsense of certain reports (for example, ones based on total accumulated laptime).

```python
leadlaptimes_last = laptimes.drop_duplicates(subset=['NUMBER', 'LEAD_LAP_NUMBER'],
                                        keep='last')

leadlaptimes_last_wide =  leadlaptimes_last.pivot(index='NUMBER',
                                columns='LEAD_LAP_NUMBER',
                                values='LAP_TIME_S')
leadlaptimes_last_wide.head()
```

```python
#This chart loses information where there is more one lap per lead lap
leadlap_last_pace = (leadlaptimes_last_wide - leadlaptimes_last_wide.loc[rebase])

ax = leadlap_last_pace.loc[['8','3','11']].T.cumsum().plot();
inpitlaps.plot.scatter(x='LEAD_LAP_NUMBER',y='y', ax=ax);
```

Alternatively, where a car records more than one laptime on a given lead lap, we might choose to set the corresponding laptime to the sum of the laptimes recorded on the lead lap. This approach is more likely to be useful if we are running accumulated laptime time calculations becuase there is no loss off laptime information:

```python
leadlaptimes_sum_wide = laptimes.groupby(['NUMBER','LEAD_LAP_NUMBER'])['LAP_TIME_S'].sum().reset_index().pivot(index='NUMBER',
                                columns='LEAD_LAP_NUMBER',
                                values='LAP_TIME_S')
leadlaptimes_sum_wide.head()

leadlap_sum_pace = (leadlaptimes_sum_wide - leadlaptimes_sum_wide.loc[rebase])

ax = leadlap_sum_pace.loc[['8','3','11']].T.cumsum().plot();
inpitlaps.plot.scatter(x='LEAD_LAP_NUMBER',y='y', ax=ax);
```

### Generating a pit time neutralistion mask

One thing we notice is that the pace table and charts are cluttered by the inlap and outlap times, particularly when running calculations against the lead lap.

It may be better to try to neutralise those (for each car including the rebased car )and deal with a pit analysis more specifically elsewhere, such as by comparing inlaps and outlap times,and perhaps also making comparisons relative to flying lap just before the inlap and just after the outlap.

The *pandas* [`DataFrame.mask()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mask.html) method lets us replace values in a dataframe if the corresponding mask element is true.

This means we can find the delta times on each lap and then neutralise the ones that are associated either with the inlap or outlap of the rebase car, or the inlap or outlap of each other car.

*Bear in mind that in general deltas measured relative to `LAP_NUMBER` may be recorded at very different elapsed race times, although  when comparing cars that are directly racing each other, they are likely to be on the same lap, or thereabouts, on any given lead lap.*

The resulting "masked" delta times table will not give a true summary of all the delta times, and the accumulated delta time will be incorrect, but we will get a cleaner view of the pace in the intervening periods.

Let's start by neutralising inlap and outlap times in the context of `LAP_NUMBER` calculations. (We will concern ourselves with neutralisation over lead lap calculations later.)
```python
#laptimes.pivot(index='NUMBER',columns='LAP_NUMBER', values='PIT_TIME_S')
laptimes['INLAP'] = (laptimes['CROSSING_FINISH_LINE_IN_PIT'] == 'B')
laptimes['OUTLAP'] = ~laptimes['PIT_TIME'].isnull()

laptimes['PITMASK'] = laptimes['INLAP'] | laptimes['OUTLAP']

#Go defensive making sure we have no NA values
pitmask = laptimes.pivot(index='NUMBER', columns='LAP_NUMBER',
                         values='PITMASK').fillna(False)
pitmask.head()
```
```python
#Mask the times for each car's inlaps and outlaps
masked_pace = pace.mask(pitmask, 0)
#We also need to and each row with the inlaps and outlaps for the rebased row
#pitmask.loc[rebase].T gives the mask for the rebase row
masked_pace = masked_pace.mul(~pitmask.loc[rebase], axis=1 )
masked_pace.head()
```

We can also blank the times by setting masked values to NA:

```python
from numpy import NaN
masked_pace = pace.mask(pitmask, NaN)
#We also need to and each row with the inlaps and outlaps for the rebased row
#pitmask.loc[rebase].T gives the mask for the rebase row
masked_pace = masked_pace.mul(pitmask.loc[rebase].map({True:NaN,False:1}), axis=1 )
masked_pace.head()
```

The NA values introduce breaks in the plotted lines:

```python
masked_pace.T[['8','3','11']].plot();
```

Using the accumulated deltas we see the breaks more clearly, although it should be recalled that these are introduced when *either* the rebased car *or* the reported on car goes through the pits.

```python
ax = masked_pace.T[['8','3','11']].cumsum().plot();
inpitlaps.plot.scatter(x='LAP_NUMBER',y='y', ax=ax);
```

Visual inspection of that charts suggests certain trends... so could we use the corresponding neutralised data points for the basis of linear pace models, perhaps over various sliding windows?

<!-- #region hideCode=false hidePrompt=false -->
Also, what do we need to do to neutralise inlap and outlap times when working with *leadlap* calculations?
<!-- #endregion -->

```python
# what are the issues here? Do we simply use the LEAD_LAP_NUMBER rather than LAP_NUMBER?
#If there are duplicate rows, we want to null everything we can?
# Or will the masks handle everything anyway...?
pitmask2 = laptimes.drop_duplicates(subset=['NUMBER', 'LEAD_LAP_NUMBER'],
                                        keep='last').pivot(index='NUMBER',
                                                           columns='LEAD_LAP_NUMBER',
                                                           values='PITMASK').fillna(False)

masked_leadlap_pace = leadlap_last_pace.mask(pitmask2, NaN)
#We also need to and each row with the inlaps and outlaps for the rebased row
#pitmask.loc[rebase].T gives the mask for the rebase row
masked_leadlap_pace = masked_leadlap_pace.mul(pitmask2.loc[rebase].map({True:NaN,False:1}), axis=1 )
masked_leadlap_pace.T[['8','3','11']].cumsum().plot();

```

### Rebasing According to Other Bases

As well as rebasing to a particular driver, we can rebase to the fastest time recorded for each lap:

```python
pace_fastest = (laptimes_wide - laptimes_wide.min())
pace_fastest.head()
```

Slightly more involved is to rebase relative to the car in first position (or indeed, any specified position) at the end of each lap:

```python
pos = 1 
pace_pos1 = (laptimes_wide - laptimes[laptimes['POS']==pos].set_index('LAP_NUMBER')['LAP_TIME_S'])
pace_pos1.head()

```

## Neutralising Accumulated Pit Times

The gaps in the pit stop neutralised pace charts represent a loss of information.

If we want to concentrate on *pace* rather than on time lost through being spent in the pits, for each car we could provide a naive dummy estimate of the inlap and outlap pace by setting them each to:

`((inlap_time + outlap_time) - pitstop_time) / 2`.

However, we note that we also have access to sector times, so we might be able to make a better estimate based using sector times and a proportionate subtraction of the pit stop time from the inlap and outlap times.

To start working towards this, it might make sense to start pulling out pit data specifically.

```python
pit_data = laptimes[laptimes['INLAP'] | laptimes['OUTLAP']][:]
pit_data['LAP_TYPE'] = pit_data['INLAP'].map({True:'INLAP', False:'OUTLAP'})
pit_data.head()
```

As a basis for comparison, let's see what the sector times were for the fastest 5 laps of the race:

```python
pit_sector_cols = ['NUMBER','S1_S','S2_S','S3_S','PIT_TIME_S','LAP_TIME_S', 'LAP_NUMBER']

laptimes.sort_values('LAP_TIME_S').head()[pit_sector_cols]
```

We can inspect the inlap and outlap sector times for a specific car to get a feel for how they behave:

```python
pit_sector_cols.append('LAP_TYPE')

pit_data[pit_data['NUMBER'].isin(['8'])][pit_sector_cols].sort_values(['NUMBER','LAP_NUMBER'])
```

```python
# remove outlier sector times
from scipy import stats
import numpy as np

pit_data.loc[np.abs(stats.zscore(pit_data['S1_S'])) > 3, 'S1_S'] = NaN
pit_data.loc[np.abs(stats.zscore(pit_data['S2_S'])) > 3, 'S2_S'] = NaN
pit_data.loc[np.abs(stats.zscore(pit_data['S3_S'])) > 3, 'S3_S'] = NaN
```

```python
pit_data.boxplot(column=['S1_S', 'S2_S', 'S2_S']);
```

```python
#Need a more aggressive cut-off...
#Need a more robust way of setting this...
pit_data.loc[pit_data['S1_S'] > 100, 'S1_S'] = NaN
pit_data.loc[pit_data['S2_S'] > 100, 'S2_S'] = NaN
pit_data.loc[pit_data['S3_S'] > 100, 'S3_S'] = NaN

pit_data.boxplot(column=['S1_S', 'S2_S', 'S2_S']);
```

We can also generate interactive *plotly* box plots from long format data.

For example, lets get a base line set of flying lap data based on laptimes lass than 200s:

```python
#Laptimes < 250s

lap_filter = ( (laptimes['LAP_TIME_S']<200) & ~laptimes['PITMASK'])

laps_under200s_lap_sectors = laptimes[lap_filter][['S1_S','S2_S','S3_S']].melt()
laps_under200s_lap_sectors.head()

```

```python
import plotly.express as px

fig = px.box(laps_under200s_lap_sectors, x="variable", y="value")
fig.show()
```

How does the distribution of sectos times on the pit related laps compare with the the lap sector times on flying laps?

On an inlap, wwe might expect at least the third sector time to show an elevated time due to pit loss and perhaps an element of the pit stop time.


How about the distribution on the outlap? We'd expect at least the first sector to have an elevated first sector time for a several reasons:

- it is likely to include a pit stop time and pit lane exit loss time;
- it make take time for the tyres to get up to speed.


Let's grab the sector times for inlaps and outlaps:

```python
inlap_sectors = pit_data[pit_data['LAP_TYPE']=='INLAP'][['LAP_TYPE','S1_S','S2_S','S3_S']].melt(id_vars='LAP_TYPE')
outlap_sectors = pit_data[pit_data['LAP_TYPE']=='OUTLAP'][['LAP_TYPE','S1_S','S2_S','S3_S']].melt(id_vars='LAP_TYPE')

outlap_sectors.head()
```

Now we can compare them to the flying laps in a single chart, grouping the box plots according to sector and lap type:

```python
import plotly.graph_objects as go

fig = go.Figure()


fig.add_trace(go.Box(
    y = laps_under200s_lap_sectors['value'],
    x = laps_under200s_lap_sectors['variable'],
    name = 'Laps under 250s',
    marker_color='#00851B'
))

fig.add_trace(go.Box(
    y = inlap_sectors['value'],
    x = inlap_sectors['variable'],
    name = 'Inlap',
    marker_color='#FF851B'
))

fig.add_trace(go.Box(
    y = outlap_sectors['value'],
    x = outlap_sectors['variable'],
    name ='Outlap',
    marker_color = '#FF4136'
))


fig.update_layout(
    yaxis_title='Sector time',
    boxmode='group' # group together boxes of the different traces for each value of x
)
fig.show()
```

```python
#Which cars produced the 50 fastest laps?
laptimes.sort_values('LAP_TIME_S').head(50)['NUMBER'].unique()
```

## Per Cent Based Pace Comparisons

In *Formula One*, the 107% regulation specifies that in the first round of qualifying, cars must qualify with a laptime within 107% of the laptime of the car with the fastest laptime in order to guarantee a race place, although in recent years stewards have not tended to exclude cars on this basis.

Using *per cent* based pace comparisons has several advantages:

- if the race pace changes due to weather conditions, if all drivers are affected equally by changes in race conditions, each getting "X%" slower, for example, the same lap time deltas unde wet and dry conditions would correspond to different relative pace deltas;
- using *per cent* based comparisons allows us to more fairly compare pace on circuits with different lap times, or more fairly compare pace over different length sectors with different typical sector times on particular circuit.

<!-- #region -->
So how might we go about making *per cent* based pace calculations?

Recall the original calculations:

```python
pace = (laptimes_wide - laptimes_wide.loc[rebase])
pace.head()
```

To convert these to percentages, we need to divide through each laptime by the corresponding laptime for the driver against which we are rebasing rather than finding the delta.
<!-- #endregion -->

```python
pace_pc = (laptimes_wide / laptimes_wide.loc[rebase])
pace_pc.head()
```

Alternatively, we might want to express the time delta as a percentage of the rebased car's laptime, which means that we can more easily colour the table based on whether the sign of the percentage difference corrsponds to a slower (negative) or faster (positive) laptime:

```python
pace_pcr = pace_pc - 1
pace_pcr.T[['3','11']].head(10).style.bar(align='zero', color=['#5fba7d', '#d65f5f'])
```

## Sector Pace

As well as making pace judgements around laptimes, we can take a finer grained view and make comparisons around sector times.


## Intra-Team Pace Comparisons

One of the comparisons we are likely to want to make is across laptimes and sector times for each of the drivers within the same team.

For a fair comparison, we need to make pace assessments against some sort of "fixed" basis. If we make pace assessments based on laptime deltas for any given lap to the fastest lap time recorded on the associated leadlap, we can make pace just against the best laptime recorded  on each lap. (Alternatively, we could use the ultimate lap for each lead lap, based on the sum of the best sector times recorded on each lead lap.)

```python

```