# Race Events

Simple sketches around identifying race related events.

```python
# using notebook rather than inline enables 3D matplotlib plots
%matplotlib inline

import pandas as pd
```

```python
url = 'http://fiawec.alkamelsystems.com/Results/08_2018-2019/07_SPA%20FRANCORCHAMPS/267_FIA%20WEC/201905041330_Race/Hour%206/23_Analysis_Race_Hour%206.CSV'

```

```python
laptimes = pd.read_csv(url, sep=';').dropna(how='all', axis=1)
laptimes.columns = [c.strip() for c in laptimes.columns]

# Tidy the data a little... car and driver number are not numbers
laptimes[['NUMBER','DRIVER_NUMBER']] = laptimes[['NUMBER','DRIVER_NUMBER']].astype(str)
```

## Core Enrichment

Really need to move the df enrichment into a utils fn before the tech debt gets too much!


```python
# Add the parent dir to the import path
import sys
sys.path.append("../py")

# Import contents of the utils.py package in the parent directory
from utils import *

# Get laptimes in seconds
laptimes['LAP_TIME_S'] = laptimes['LAP_TIME'].apply(getTime)

# Find accumulated time in seconds
laptimes['ELAPSED_S']=laptimes['ELAPSED'].apply(getTime)

# Find pit time in seconds
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

```python
# TO DO - check laps where a car is lapped or unlaps to see we have the correct lap, lead lap and time data

# A car may miss a lead lap number if it is lapped. So in the race history what happens?
```

```python

```

## LapTime Distribution

Eyeball the laptime distribution just to get a feel for it.

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(20,10)})

ax = sns.boxplot(x="LEAD_LAP_NUMBER", y="LAP_TIME_S", data=laptimes[laptimes['LAP_TIME_S']<500])
plt.xticks(rotation=90);
```

Can we get signal from the differerces between fastest lap time per lead lap?

```python
laptimes.groupby(['LEAD_LAP_NUMBER'])['LAP_TIME_S'].min().diff()[1:].plot()
```

```python
#WHy find the min here? TO DO
ax2 = laptimes.groupby(['LEAD_LAP_NUMBER'])['LAP_TIME_S'].min().diff().fillna(0).plot(kind='bar')
sns.boxplot(x="LEAD_LAP_NUMBER", y="LAP_TIME_S", data=laptimes[laptimes['LAP_TIME_S']<500], ax=ax2)
sns.set(rc={'figure.figsize':(20,10)})
ax2;
```

## Slow Laps

Slow laps may be a safety car, may be yellow flags, may be the weather...

Start by identifying median and standard deviation of laptimes on a per lead lap basis.

```python
#pandas 0,25?
#laptimes_summary_stats = laptimes.groupby(['LEAD_LAP_NUMBER']).agg(median_lap_time_s=('LAP_TIME_S', 'median'),
#                                                                   sd_lap_time_s=('LAP_TIME_S', 'std'))

laptimes_summary_stats = laptimes.groupby(['LEAD_LAP_NUMBER']).agg({'LAP_TIME_S':['mean', 'median','std', 'max']})

laptimes_summary_stats.columns = laptimes_summary_stats.columns.get_level_values(1)

laptimes_summary_stats.head()
```

```python
laptimes_summary_stats.plot()
```

Okay, so we have at least one outlier.

Let's look for others... say, laptimes with a z-score of more than 3, which is to say, more than 3 standard deviations away from the mean.

```python
from scipy import stats
import numpy as np

laptimes[(np.abs(stats.zscore(laptimes['LAP_TIME_S'])) > 3)]
```

We don't really want `NaN` values in the laptimes, but for now let's create a set of "clean" laptimes that set the outliers to a `NaN` value.

```python
from numpy import NaN

laptimes['CLEAN_LAP_TIME_S'] = laptimes['LAP_TIME_S']

#If the zscore on a laptime is greater than three, set the CLEAN_LAP_TIME_S value to NaN
# else use the original value
laptimes.loc[np.abs(stats.zscore(laptimes['LAP_TIME_S'])) > 3, 'CLEAN_LAP_TIME_S'] = NaN
```

Now let's create the summary stats over the cleaned data.

```python
laptimes_summary_stats = laptimes.groupby(['LEAD_LAP_NUMBER']).agg({'CLEAN_LAP_TIME_S':['mean', 'median','std', 'max']})

laptimes_summary_stats.columns = laptimes_summary_stats.columns.get_level_values(1)

```

```python
laptimes_summary_stats.reset_index().plot(kind='scatter', x='LEAD_LAP_NUMBER', y='median')
```

```python
laptimes_summary_stats.reset_index().plot(kind='scatter', x='LEAD_LAP_NUMBER', y='std')
```

```python
#laptimes_summary_stats.plot(kind='scatter', x='median', y='std')

%matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D


scatter3d = plt.figure().gca(projection='3d')
scatter3d.scatter(laptimes_summary_stats.index, laptimes_summary_stats['std'], laptimes_summary_stats['median'])
```

## Pit Events Per Lap

Is there any signal to be had from pit events per lap?

```python
%matplotlib inline
sns.set(rc={'figure.figsize':(20,10)})

laptimes['INLAP'] = laptimes['CROSSING_FINISH_LINE_IN_PIT']== 'B'
laptimes.groupby('LEAD_LAP_NUMBER')['INLAP'].apply(lambda x: x.sum()).plot(kind='bar');
```

```python
ax = sns.boxplot(x="LEAD_LAP_NUMBER", y="LAP_TIME_S", data=laptimes[laptimes['LAP_TIME_S']<500])
laptimes.groupby('LEAD_LAP_NUMBER')['INLAP'].apply(lambda x: x.sum()).plot(kind='bar', ax=ax);
```

### Slow Laps from Top Speed

If we have a full lap yellow or safety car, this should be reflected in the top speed recorded for the lap.

Trivially exploring this for an arbitrary car:

```python
laptimes[laptimes['NUMBER']=='1'].plot(x='LAP_NUMBER',y='TOP_SPEED');
```

or as a set of median or Nth quantile speeds:

```python
ax = laptimes.groupby('LAP_NUMBER')['TOP_SPEED'].quantile(1).plot();
laptimes.groupby('LAP_NUMBER')['TOP_SPEED'].quantile(.85).plot(ax=ax);
```

## Simple Classifier

Can we build a simple K-means classifier with K=2 to identify slow laps? 

```python
from sklearn.cluster import KMeans
```

Let's just do something really simple and see if we can classify based on the median and standard deviation of the laptimes.

```python
kmeans = KMeans(n_clusters=2, random_state=0).fit( laptimes_summary_stats[['median', 'std']] )
```

```python
import numpy as np

laptimes_summary_stats['CLUSTER_GROUP'] = kmeans.labels_

#colours = np.where(laptimes_summary_stats['CLUSTER_GROUP'], 'red', 'green')
#laptimes_summary_stats.reset_index().plot(kind='scatter', x='LEAD_LAP_NUMBER', y='median', color=colours)
ax = sns.scatterplot(x="LEAD_LAP_NUMBER", y="median", hue = 'CLUSTER_GROUP',
                     data=laptimes_summary_stats.reset_index())

```

*(The classification is worsned, if anything, if we add the leader laptime deltas in.)*


How about over all the laptimes? Though we need to find a way of handling `NaN`s and maybe doing something to normalise pit stop times to something closer to the median lap time on a given lead lap? Or do we want to keep that extra signal in?

```python
tmp = laptimes[['LEAD_LAP_NUMBER','NUMBER','CLEAN_LAP_TIME_S','INLAP']]

#Let's try voiding times that are pit times
tmp.loc[tmp['INLAP'], 'CLEAN_LAP_TIME_S'] = NaN


# Now create a table of car laptimes by leadlap

#Some cars may have multiple laptimes recorded on one lead lap
#in this case, we need to reduce the multiple times to a single time, eg min, or mean
car_by_lap = tmp.pivot_table(index='LEAD_LAP_NUMBER', columns='NUMBER', values='CLEAN_LAP_TIME_S', aggfunc='min')
car_by_lap.head()
```

```python
#Fill na with row mean, though it requires a hack as per ??
#SHould we use mean, median, or min?

car_by_lap_clean = car_by_lap.T.fillna(car_by_lap.mean(axis=1)).T
car_by_lap_clean.head()
```

```python
kmeans = KMeans(n_clusters=2, random_state=0).fit( car_by_lap_clean )
```

Now let's highlight clustered laps and see if we've picked out the slow ones...

```python
laptimes_summary_stats['CLUSTER_GROUP'] = kmeans.labels_

colours = np.where(laptimes_summary_stats['CLUSTER_GROUP'], 'red', 'green')

#ax = sns.scatterplot(x="LEAD_LAP_NUMBER", y="median", hue = 'CLUSTER_GROUP' data=tips)
laptimes_summary_stats.reset_index().plot(kind='scatter', x='LEAD_LAP_NUMBER', y='median', color=colours)
```

Seems to be pretty good, though around lap 25 a couple of misclassifications, perhaps?

*(Using actual INLAP times, rather than non-pitting lead lap average times for INLAPS, doesn't seem to affect the classification?)*

This also differs from the simple classification based on median/std by identifying lead lap 24 as an atypical laptime event?

```python
laptimes_summary_stats['unit']=1
laptimes_summary_stats.reset_index().plot(kind='scatter', x='LEAD_LAP_NUMBER', y='unit', color=colours)
```

*(It doesn't make any difference if we add the summary stats in too...)*


### Streak Detection

There are several receipes out there for streak detection. It may be useful to try to collect them and then come up with a best practice way of calculating streaks?

```python
colours_df = pd.DataFrame({'event':[c!='red' for c in colours]})
#Set the index to be lead lap number - indexed on 1 rather than 0
colours_df.index += 1

colours_df.head()
```

```python
#via https://stackoverflow.com/a/51626783/454773

def streak(dfc):
    ''' Streak calculation: take a dataframe column containing a list of Boolean values
        and return a list of paired values containing index values of the start and end of each streak
        for the True boolean value. '''
    return ((~dfc).cumsum()[dfc]
            .reset_index()
            .groupby(['event'])['index']
            .agg(['first','last'])
            .values
            .tolist())

streak(colours_df['event']), streak(~colours_df['event'])
```

```python
## Better streak / stint code in Le Mans notebook.
```

```python
def streak_len(streak_list, lap_index = 1):
    ''' Return a dataframe showing streak lap start, end, length. '''
    tmp_df = pd.DataFrame(streak_list, columns=['Start', 'Stop'])
    tmp_df['Length'] = tmp_df['Stop'] - tmp_df['Start'] + 1
    #Align index to first lap number
    tmp_df.index += lap_index
    return tmp_df

streak_len( streak( ~colours_df['event'] ) )
```

## Simple Race History Chart

One of the ways we can use the atypical laptime indicator is to neutralise affected areas of the race history chart, making it (arguably), easier to read.

For example, a raw race history chart, which shows how the lap times for each driver compare over the course of a race by comparison to the winner's mean laptime, might look like the following:

```python
#Get the number of the winning car
#which is to say, the one in first position at the end of the race
LAST_LAP = laptimes['LEAD_LAP_NUMBER'].max()

winner = laptimes[(laptimes['LAP_NUMBER']==LAST_LAP) & 
                  (laptimes['POS']==1)]['NUMBER'].iloc[0]
winner
```

```python
#Get the mean laptime for the winner
winner_mean_laptime_s = laptimes[laptimes['NUMBER']==winner]['LAP_TIME_S'].mean().round(decimals=3)
winner_mean_laptime_s
```

```python
#Calculate the "race history" laptimes
#?? laptimes['LEAD_LAP_NUMBER'] gives track position? laptimes['LAP_NUMBER'] give race history
laptimes['RACE_HISTORY_ELAPSED_LAP_TIME_S'] = (winner_mean_laptime_s * laptimes['LAP_NUMBER']) - laptimes['ELAPSED_S']

```

```python
#Let's filter, for now, by just the top 10 cars
top10 = laptimes[laptimes['LEAD_LAP_NUMBER']==LAST_LAP].sort_values(['LEAD_LAP_NUMBER', 'POS'])[['NUMBER']].reset_index(drop=True).head(10)
top10.index +=1
```

We can add highlighted areas to the chart to identify atypical laps.

```python
sns.set(style="ticks")
plt.style.use("dark_background")

#Setting the palette size is a hack: https://github.com/mwaskom/seaborn/issues/1515#issuecomment-482189290
data = laptimes[laptimes['NUMBER'].isin(top10['NUMBER'])]

ax = sns.lineplot(x="LEAD_LAP_NUMBER", y="RACE_HISTORY_ELAPSED_LAP_TIME_S",
                  units = 'NUMBER', hue = 'NUMBER', palette=sns.color_palette("Set1", len(data['NUMBER'].unique())),
                  estimator=None, lw=1,
                  data=data)

#Let's add in some indicators showing the atypical laps

def atypical_lap_band(row, ax):
    ax.axvspan(row['Start']-0.5, row['Stop']+0.5, alpha=0.5, color='lightyellow')
    
streak_len( streak( ~colours_df['event'] ) ).apply(lambda x: atypical_lap_band(x, ax), axis=1);
```

TO DO - if a car is lapped it may have more that on lap time per lead lap - we need to plot the one with largest elapsed time. ??Are we losing a row somewhere where a car is lapped?

```python
# Race history chart - if we plot the lead lap, we get the track position?
```

```python
data = laptimes[laptimes['NUMBER'].isin(top10['NUMBER'])].groupby(['LEAD_LAP_NUMBER','NUMBER'])['RACE_HISTORY_ELAPSED_LAP_TIME_S'].max().reset_index()

ax = sns.lineplot(x="LEAD_LAP_NUMBER", y="RACE_HISTORY_ELAPSED_LAP_TIME_S",
                  units = 'NUMBER', hue = 'NUMBER', palette=sns.color_palette("Set1", len(data['NUMBER'].unique())),
                  estimator=None, lw=1,
                  data=data)

streak_len( streak( ~colours_df['event'] ) ).apply(lambda x: atypical_lap_band(x, ax), axis=1);
```

One thing we notice about the race history chart is that safety car periods can result in an increase in typical laptimes which can dramatically affect the appearance of the chart.

However, if we identify laps with atypical laptimes across the field, we can 'neutralise" those laps by replacing the recorded times with dummy laptimes.

One model for generating dummy laptimes is to identify a "laptime corrective" in the form of a subtractive term:

`leader_neutralised_lap_time = leader_laptime_on_slow_lap - corrective`

So what should that `leader_neutralised_lap_time` be?

<!-- #region -->
If we want to "neutralise" the safety car lap times, then we might expect the race history chart to show a flat line for the cars travelling at a neutralised pace.

The neutralised pace should thus be set to the winner's mean laptime over the non-neutralised laps (the neutralised laps should not otherwise change the mean laptime basis that defines the race history chart).

But whose pace should we use for the basis of neutralisation?

Ideally, we want to use the pace that defines the pace of the field. If a safety car is controlling race pace, then it makes sense to use the laptime of the leader on a given lap as the race defining pace and define a corrective based on this pace.

That is:

```python
leaders_neutralised_lap_time = winners_non_neutralised_laps_mean_lap_time

```

We can create a corrective for each laptime as follows:

```python

leaders_neutralised_lap_time = leaders_lap_time_on_neutralisation_lap - corrective_on_neutralisation_lap

```

which gives:

```python
corrective_on_neutralisation_lap = leaders_lap_time_on_neutralisation_lap - winners_non_neutralised_laps_mean_lap_time
```

and more generally:

```python
neutralised_lap_time = lap_time_on_neutralisation_lap - corrective_on_neutralisation_lap

```

*Bounds get fiddly if atypical laps appear at a fence post (first lap, last lap), so for now let's use the fact the first lap is clear to set the neutralised lap time during an atypical stint to be the lap time of the leader on the lap before the atypical run.*
<!-- #endregion -->

Let's start by getting hold of the lap numbers for the laps we want to neutralise, and the lap number of the laps at the start of neutralisation periods:

```python
#Get the lap numbers for the atypical laps (that is, the laps we want to neutralise)
NEUTRALISED_LAPS = laptimes_summary_stats[laptimes_summary_stats['CLUSTER_GROUP']==1].index

#As a convenience, we may wish to get the first lap in a neutralisation period
NEUTRALISED_LAP_STARTS = streak_len( streak( ~colours_df['event'] ) )['Start']
```

The corrective is the difference between the lap leader's time during a neutralisation period and the neutralised lap target time.

The neutralised lap target time is the winner's mean lap time over the non-neutralised laps.

```python
##Set the target time to the mean of the winner on just the typical laps?
winners_mean_typical = laptimes[(laptimes['NUMBER']==winner) & ~(laptimes['LEAD_LAP_NUMBER'].isin(NEUTRALISED_LAPS))]['LAP_TIME_S'].mean().round(decimals=3)
winners_mean_typical
```

Set up a dummy column, `desired`, to be the lap time we want the leader to have on laps in the neutralisation period, then generate the corrective as the difference between their actual lap time and this desired lap time for the neutralisation laps. Set the dummy `desired` column to the dummy value 0 otherwise (that is, on racing laps).

```python
colours_df['desired'] = 0
colours_df.loc[colours_df.index.isin(NEUTRALISED_LAPS), 'desired'] = winners_mean_typical
colours_df['desired'].fillna(method='ffill', inplace=True)

#The corrective basis is then the lap time of the lead lap leader
colours_df['corrective_basis'] = laptimes[laptimes['POS']==1].set_index('LEAD_LAP_NUMBER')['LAP_TIME_S']

colours_df['corrective'] = 0
colours_df.loc[colours_df.index.isin(NEUTRALISED_LAPS), 'corrective'] = colours_df['corrective_basis'] - colours_df['desired']

colours_df.head()
```

```python
#We now need to apply the lap based correctives to each lap in the laptimes dataset
laptimes = pd.merge(laptimes, colours_df['corrective'], left_on='LEAD_LAP_NUMBER', right_index=True)
```

```python
#Repeat the race history calculations using the neutralised lap times
laptimes['NEUTRALISED_LAP_TIME_S'] = laptimes['LAP_TIME_S'] - laptimes['corrective']
laptimes['ELAPSED_NEUTRALISED_LAP_TIME_S'] = laptimes.groupby('NUMBER')['NEUTRALISED_LAP_TIME_S'].cumsum()
```

```python
neutralised_winner_mean_laptime_s = laptimes[laptimes['NUMBER']==winner]['NEUTRALISED_LAP_TIME_S'].mean().round(decimals=3)
neutralised_winner_mean_laptime_s

```

```python
#Calculate the "neutralised race history" laptimes
#If we multiply by lead_lap_number we can get track position? TO DO
laptimes['NEUTRALISED_RACE_HISTORY_ELAPSED_LAP_TIME_S'] = (neutralised_winner_mean_laptime_s * laptimes['LAP_NUMBER']) - laptimes['ELAPSED_NEUTRALISED_LAP_TIME_S']

```

```python
#Should make a function for this
data = laptimes[laptimes['NUMBER'].isin(top10['NUMBER'])].groupby(['LEAD_LAP_NUMBER','NUMBER'])['NEUTRALISED_RACE_HISTORY_ELAPSED_LAP_TIME_S'].max().reset_index()

ax = sns.lineplot(x="LEAD_LAP_NUMBER", y="NEUTRALISED_RACE_HISTORY_ELAPSED_LAP_TIME_S",
                  units = 'NUMBER', hue = 'NUMBER', palette=sns.color_palette("Set1", len(data['NUMBER'].unique())),
                  estimator=None, lw=1,
                  data=data)

#Let's add in some indicators showing the atypical laps

def atypical_lap_band(row, ax):
    ax.axvspan(row['Start']-0.5, row['Stop']+0.5, alpha=0.5, color='lightyellow')
    
streak_len( streak( ~colours_df['event'] ) ).apply(lambda x: atypical_lap_band(x, ax), axis=1);
```

```python
top10
```

```python
laptimes[(laptimes['NUMBER']=='31') & (laptimes['LAP_NUMBER']>70)& (laptimes['LAP_NUMBER']<90) ][['LAP_NUMBER','LEAD_LAP_NUMBER']]
```
### Rebased Race History Charts

Race history charts are typically calculated relative to the race winner (or for a live race history chart, the race leader). However, we can also rebase race history charts to other cars, although this may make the chart harder to interpret. For example, we could rebase the times to:

- a particular car;
- the class leader;
- the car with the lowest lap time on each lead lap.





```python

```
