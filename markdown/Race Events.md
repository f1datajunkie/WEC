# Race Events

Simple attempts at identifying race related events.

```python
#using notebook rather than inline enables 3D matplotlib plots
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
```

## Core Enrichment

Really need to move the df enrichment into a utils fn before the tech debt gets too much!


```python
#Add the parent dir to the import path
import sys
sys.path.append("..")

#Import contents of the utils.py package in the parent directory
from py.utils import *

#Get laptimes in seconds
laptimes['LAP_TIME_S'] = laptimes['LAP_TIME'].apply(getTime)

#Find accumulated time in seconds
laptimes['ELAPSED_S']=laptimes['ELAPSED'].apply(getTime)

#Find position based on accumulated laptime
laptimes = laptimes.sort_values('ELAPSED_S')
laptimes['POS'] = laptimes.groupby('LAP_NUMBER')['ELAPSED_S'].rank()

#Find leader naively
laptimes['leader'] = laptimes['POS']==1

#Find lead lap number
laptimes['LEAD_LAP_NUMBER'] = laptimes['leader'].cumsum()


laptimes.head()
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
laptimes[(np.abs(stats.zscore(laptimes['LAP_TIME_S'])) > 3)]
```

We don't really want `NaN` values in the laptimes, but for now lets create a set of "clean" laptimes that do `NaN` the outliers.

```python
from scipy import stats
import numpy as np
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
laptimes['RACE_HISTORY_LAP_TIME_S'] = (winner_mean_laptime_s * laptimes['LEAD_LAP_NUMBER']) - laptimes['ELAPSED_S']
```

```python
#Lat's filter, for now, by just the top 10 cars
top10 = laptimes[laptimes['LEAD_LAP_NUMBER']==LAST_LAP].sort_values(['LEAD_LAP_NUMBER', 'POS'])[['NUMBER']].reset_index(drop=True).head(10)
top10.index +=1
```

```python
sns.set(style="ticks")
plt.style.use("dark_background")

#Setting the palette size is a hack: https://github.com/mwaskom/seaborn/issues/1515#issuecomment-482189290
data = laptimes[laptimes['NUMBER'].isin(top10['NUMBER'])]

ax = sns.lineplot(x="LEAD_LAP_NUMBER", y="RACE_HISTORY_LAP_TIME_S",
                  units = 'NUMBER', hue = 'NUMBER', palette=sns.color_palette("Set1", len(data['NUMBER'].unique())),
                  estimator=None, lw=1,
                  data=data)

#Let's add in some indicators showing the atypical laps

def atypical_lap_band(row, ax):
    ax.axvspan(row['Start']-0.5, row['Stop']+0.5, alpha=0.5, color='lightyellow')
    
streak_len( streak( ~colours_df['event'] ) ).apply(lambda x: atypical_lap_band(x, ax), axis=1);
```

Safety car periods result in an increase in typical laptimes which can dramatically affect the appearance of the chart.

However, if we identify laps with atypical laptimes across the field, we can 'neutralise" those laps by replacing the recorded times with dummy laptimes.

One model for generating dummy laptimes is to identify a "laptime corrective" in the form of a subtractive term:

`corrective = leader_laptime_on_slow_lap - leader_mean_lap_time_across_none_slow_laps`

and then subtract that corrective value from all cars during the slow laps.

We can then further highlight the chart to identify the laps where we neutralalised the lap times. 

```python
winnerlaps = laptimes[laptimes['NUMBER']==winner].reset_index(drop=True)
winnerlaps.index += 1

#CLUSTER_GROUP 0 are the normal laps (We can't guarantee that? Need some sort of check)
```

<!-- #region -->
A crude corrective is the simple mean laptime for the winner.

We could also calculate the mean over just the "racing" laps, or just the atypical laps

*CLUSTER_GROUP 0 are the normal laps, 1 the atypical laps (tho can't guarantee that coding? Need some sort of check?)*

```python
corrective = winnerlaps[laptimes_summary_stats['CLUSTER_GROUP']==0]['LAP_TIME_S'].mean().round(decimals=3)
```

Or we could set a corrective based on the winner's fastest lap:
```python
corrective = laptimes[laptimes['NUMBER']==winner]['LAP_TIME_S'].min()
```

If the corrective is a single, constant value, we can apply it simply. For example, if the lap is atypical, set the value to the corrective offset, else use the original lap time.

```python
laptimes['RACE_HISTORY_CORRECTIVE'] = 0

#Apply the corrective to atypical lap laptimes
laptimes.loc[laptimes['LEAD_LAP_NUMBER'].isin(NEUTRALISED_LAPS), 'RACE_HISTORY_CORRECTIVE'] = corrective
```
<!-- #endregion -->

However, a more sensible corrective would be relative to the laptimes just before and just after the atypical laps, such as the mean time of the laps immediately before and after the atypical stint

*Bounds get fiddly if atypical laps appear at a fence post (first lap, last lap), so for now let's use the fact the first lap is clear to set the neutralised lap time during an atypical stint to be the lap time of the leader on the lap before the atypical run.*

```python
#Get the lap numbers for the atypical laps (that is, the laps we want to neutralise)
NEUTRALISED_LAPS = laptimes_summary_stats[laptimes_summary_stats['CLUSTER_GROUP']==1].index

#Get the first lap in atypical run
NEUTRALISED_LAP_STARTS = streak_len( streak( ~colours_df['event'] ) )['Start']

#Note we may get a fence post error if lap 1 is to be neutralised
NEUTRALISED_LAP_DATA = laptimes[(laptimes['POS']==1) & (laptimes['LEAD_LAP_NUMBER'].isin(NEUTRALISED_LAP_STARTS-1))][['LAP_NUMBER','LAP_TIME_S']].set_index('LAP_NUMBER')

#Or should we use the fastest lap time on the lead lap?
#NEUTRALISED_LAP_DATA = laptimes[laptimes['LAP_NUMBER'].isin(NEUTRALISED_LAP_STARTS-1)].groupby('LEAD_LAP_NUMBER')['LAP_TIME_S'].min()#.set_index('LAP_NUMBER')

#Nudge the neutralised time from the lap prior to the atypical lap run start to the atypical lap run start
NEUTRALISED_LAP_DATA.index += 1
NEUTRALISED_LAP_DATA.rename(columns={"LAP_TIME_S": "NEUTRAL_LAP_TIME_S"}, inplace=True)

NEUTRALISED_LAP_DATA
```

The corrective is now the difference between the lap leader's time during an atypical run and the neutralised lap target time.

```python
NEUTRALISED_LAP_DATA['ACTUAL'] = laptimes[(laptimes['POS']==1) & (laptimes['LAP_NUMBER'].isin(NEUTRALISED_LAP_STARTS))][['LAP_NUMBER','LAP_TIME_S']].set_index('LAP_NUMBER')
NEUTRALISED_LAP_DATA

```

```python
colours_df['desired'] = 0

#Use a desired of 0 for typical laps, desired time (or NA) for atypical laps
colours_df.loc[colours_df.index.isin(NEUTRALISED_LAPS), 'desired'] = NEUTRALISED_LAP_DATA['NEUTRAL_LAP_TIME_S']
#Then fill the desired time down
colours_df['desired'].fillna(method='ffill', inplace=True)

##?? maybe we should just set the target time to the mean of the winner on just the typical laps?
winners_mean_typical = laptimes[(laptimes['NUMBER']==winner) & ~(laptimes['LEAD_LAP_NUMBER'].isin(NEUTRALISED_LAPS))]['LAP_TIME_S'].mean().round(decimals=3)
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
laptimes['NEUTRALISED_RACE_HISTORY_LAP_TIME_S'] = (neutralised_winner_mean_laptime_s * laptimes['LEAD_LAP_NUMBER']) - laptimes['ELAPSED_NEUTRALISED_LAP_TIME_S']

```

```python
#Should make a function for this
data = laptimes[laptimes['NUMBER'].isin(top10['NUMBER'])]

ax = sns.lineplot(x="LEAD_LAP_NUMBER", y="NEUTRALISED_RACE_HISTORY_LAP_TIME_S",
                  units = 'NUMBER', hue = 'NUMBER', palette=sns.color_palette("Set1", len(data['NUMBER'].unique())),
                  estimator=None, lw=1,
                  data=data)

#Let's add in some indicators showing the atypical laps

def atypical_lap_band(row, ax):
    ax.axvspan(row['Start']-0.5, row['Stop']+0.5, alpha=0.5, color='lightyellow')
    
streak_len( streak( ~colours_df['event'] ) ).apply(lambda x: atypical_lap_band(x, ax), axis=1);
```

```python
laptimes
```

```python

```
