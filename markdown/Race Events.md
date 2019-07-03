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
ax = sns.scatterplot(x="LEAD_LAP_NUMBER", y="median", hue = 'CLUSTER_GROUP', data=laptimes_summary_stats.reset_index())

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
car_by_lap = tmp.pivot_table(index='LEAD_LAP_NUMBER',columns='NUMBER', values='CLEAN_LAP_TIME_S', aggfunc='min')
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

There are several receipes out there for streak detection. It may be useful to try to collect them and then come up with a best practice way of calcualting streaks?

```python
colours_df = pd.DataFrame({'event':[c!='red' for c in colours]})
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

```python

```
