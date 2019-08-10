---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Pace Tables

The race history chart provides a useful way of reviewing the evolution of a race, but some times it an be hard to read off how much faster or slower one car is than another at any particular point in a race.

*Pace tables* and *pace charts* are an attempt to try to highlight pace differences for drivers on each lap (as given by lead lap) of the race.


## Pace Tables

*Pace tables* use driver relative rebased laptime deltas to show the pace difference for a named driver relative to other drivers.

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

# Find position based on accumulated laptime
laptimes = laptimes.sort_values('ELAPSED_S')
laptimes['POS'] = laptimes.groupby('LAP_NUMBER')['ELAPSED_S'].rank()

# Find leader naively
laptimes['leader'] = laptimes['POS']==1

# Find lead lap number
laptimes['LEAD_LAP_NUMBER'] = laptimes['leader'].cumsum()

laptimes.head()
```

Lap time deltas can be calculated for each lap as follows:

```python
laptimes_wide = laptimes.pivot(index='NUMBER',
                                columns='LAP_NUMBER',
                                values='LAP_TIME_S')
rebase = "8"
pace = (laptimes_wide - laptimes_wide.loc[rebase])
pace.head()
```

We can style the table to show how much time the rebased car made or lost on each lap compared to other cars.

```python
pace.T[['3','11']].head(20).style.bar(align='zero', color=['#d65f5f', '#5fba7d'])
```

We can also select out data by index vaue, and then transpose.

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
inpitlaps = laptimes[(laptimes['NUMBER']==rebase) & ~(laptimes['PIT_TIME'].isnull())]
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
#merge the pace back in to laptimes, then plot out the elapsed
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

One thing we notice is that the pace table and charts are cluttered by the inlap and outlap times...

It may be better to try to neutralise those (for each car including the rebased car )and deal with a pit analysis more specifically elsewhere, such as by comparing inlaps and outlap times,and perhaps also making comparisons relative to flying lap just before the inlap and just after the outlap.


### Generating a pit time neutralistion mask

The *pandas* [`DataFrame.mask()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mask.html) method lets us replace values in a dataframe if the corresponding mask element is true.

This means we can find the delta times on each lap and then neutralise the ones that are associated either with the inlap or outlap of the rebase car, or the inlap or outlap of each other car.

*Bear in mind that in general deltas measured relative to `LAP_NUMBER` may be recorded at very different elapsed race times, although  when comparing cars that are directly racing each other, they are likely to be on the same lap, or thereabouts, on any given lead lap.*

The resulting "masked" delta times table will not give a true summary of all the delta times, and the accumulated delta time will be incorrect, but we will get a cleaner view of the pace in the intervening periods.

```python
#laptimes.pivot(index='NUMBER',columns='LAP_NUMBER', values='PIT_TIME_S')
laptimes['INLAP'] = (laptimes['CROSSING_FINISH_LINE_IN_PIT'] == 'B')
laptimes['OUTLAP'] = ~laptimes['PIT_TIME'].isnull()

laptimes['PITMASK'] = laptimes['INLAP'] | laptimes['OUTLAP']

#Go defensive making sure we have no NA values
pitmask = laptimes.pivot(index='NUMBER', columns='LAP_NUMBER', values='PITMASK').fillna(False)
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

Using the accumulated deltas we see the breaks more clearly, altough it should be recalled that these are introduced when *either* the rebased car *or* the reported on car goes through the pits.

```python
ax = masked_pace.T[['8','3','11']].cumsum().plot();
inpitlaps.plot.scatter(x='LAP_NUMBER',y='y', ax=ax);
```

```python
laptimes.groupby('NUMBER')['PIT_TIME_S'].sum()[['8','3','11']]
```

```python

```
