<!-- #region -->
# Le Mans 24 Hours, WEC, 2019

Recreating charts in Racecar Engineering.


*Series: Coding for Racecar Engineers.*
<!-- #endregion -->

### Coding for Racecar Engineers

Many engineers are accustomed to fashioning their own tools from whatever happens to be at hand whenever the need arises. 

In this series of articles, we will look at how "coding", which is to say computer programming, can also be appropriated by engineers on as-and-when basis for helping them *get stuff done* with data and mathematical models.

Unlike the limitations of working with the physical properties of physical materials  will also see how code provides great flexibility as general purpose tool for building other tools. also automation

No programming or coding knowledge is assumed, nor is access to anything other than a computer browser and an internet connection.

So let's get started...

---

```python
%matplotlib inline
import pandas as pd
```

```python
url = 'http://fiawec.alkamelsystems.com/Results/08_2018-2019/08_LE%20MANS/276_FIA%20WEC/201906151500_Race/Hour%2024/23_Analysis_Race_Hour%2024.CSV'
```

```python
laptimes = pd.read_csv(url, sep=';').dropna(how='all', axis=1)
laptimes.columns = [c.strip() for c in laptimes.columns]
laptimes.head()
```

```python
#Add the parent dir to the import path
import sys
sys.path.append("..")

#Import contents of the utils.py package in the parent directory
from py.utils import *

#Get laptimes in seconds
for i in ['S1', 'S2', 'S3', 'LAP_TIME']:
    laptimes['{}_S'.format(i)] = laptimes[i].apply(getTime)

#Tidy the data a little... car and driver number are not numbers
laptimes[['NUMBER','DRIVER_NUMBER']] = laptimes[['NUMBER','DRIVER_NUMBER']].astype(str)

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

```python
laptimes.columns
```

## Mean laptime over 20 best laps


```python
#selct out inlaps
#~laptimes['CROSSING_FINISH_LINE_IN_PIT'].isnull()
```

```python
laptimes.groupby('NUMBER')['LAP_TIME_S'].nsmallest(20).groupby('NUMBER').mean().round(decimals=3).to_frame().head(10)

```

```python
def fastest20pc(carlaps):
    num = int(carlaps.count() * 0.2)
    return carlaps.nsmallest(num).mean().round(decimals=3)

laptimes.groupby('NUMBER')['LAP_TIME_S'].apply(fastest20pc).to_frame().head()
```

```python
laptimes.groupby('NUMBER')['LAP_TIME_S'].min().sort_values().to_frame().head()
```

## Rising Average

```python
laptimes[laptimes['NUMBER']=='8']['LAP_TIME_S'].sort_values().expanding(min_periods=1).mean().reset_index(drop=True).plot();

```

```python
LAST_LAP = laptimes['LAP_NUMBER'].max()
cols = ['NUMBER','TEAM', 'DRIVER_NAME', 'CLASS','LAP_NUMBER','ELAPSED']
top5 = laptimes[laptimes['LEAD_LAP_NUMBER']==LAST_LAP].sort_values(['LEAD_LAP_NUMBER', 'POS'])[cols].head(5).reset_index(drop=True)
top5
```

```python
SELECTED = top5['NUMBER'].to_list() + ['17']
```

```python
laptimes['CAR_LAP_RANK'] = laptimes.sort_values(['NUMBER','LAP_TIME_S']).groupby('NUMBER').cumcount()
```

```python
laptimes['CAR_LAP_RANKED_CUMSUM'] = laptimes.sort_values(['NUMBER','LAP_TIME_S']).groupby('NUMBER')['LAP_TIME_S'].cumsum()

```

```python
#The rank count starts at 0
laptimes['CAR_LAP_RANKED_CUMAV'] = (laptimes['CAR_LAP_RANKED_CUMSUM'] / (laptimes['CAR_LAP_RANK']+1)).round(decimals=3)


```

```python
laptimes[laptimes['NUMBER'].isin(SELECTED)].pivot(index='CAR_LAP_RANK',
                                                         columns='NUMBER', 
                                                         values='CAR_LAP_RANKED_CUMAV').plot();
```

### Advanced Technique

The *pandas* package contains a lot of powerful tools for working with tabular datasets, including predefined ways of calculating cumuluative averages.

```python
tmp =  laptimes[laptimes['NUMBER'].isin(SELECTED)].pivot(index='CAR_LAP_RANK',
                                                         columns='NUMBER', 
                                                         values='LAP_TIME_S')
tmp.expanding(min_periods=1).mean().reset_index(drop=True).plot();
```

```python
tmp.tail()
```

```python
tmp.expanding(min_periods=1).mean().tail()
```

The *pandas* dataframe `mask()` method allows us to test a particular condition for each cell; if the condition evaluates as `False`, we retain the original value, but if it evaulates as `True` we replace the cell value with a  corresponding value also passed to the `mask()` method.

If we pass a dataframe into the mask method with the same structure (that is, the same number of rows and columns) as the dataframe we are applying the mask to, we can essentially define a test for each cell, as well as a potential replacement value for each cell.

TO DO

```python
tmp.expanding(min_periods=1).mean().mask(tmp.isnull(),tmp).plot();
```

```python
laptimes['CAR_LAP_RANK_DRIVER'] = laptimes.sort_values(['NUMBER','DRIVER_NAME','LAP_TIME_S']).groupby(['NUMBER','DRIVER_NAME']).cumcount()


tmp2 =  laptimes[laptimes['NUMBER']=='7'].pivot(index='CAR_LAP_RANK_DRIVER',
                                                         columns='DRIVER_NAME', 
                                                         values='LAP_TIME_S')


tmp2.expanding(min_periods=1).mean().mask(tmp2.isnull(),tmp2).plot();
```

```python
### Stints
```

```python
#https://stackoverflow.com/a/26914036/454773
#https://stackoverflow.com/a/38064349/454773 generalised shift
df = laptimes[laptimes['NUMBER']=='7'][:]
df['CAR_LEG'] =  ( df['DRIVER_NAME']!= df['DRIVER_NAME'].shift()).astype('int').cumsum()
df2 =pd.DataFrame({'LAP_START' : df.groupby('CAR_LEG')['LAP_NUMBER'].first(), 
              'LAP_END' : df.groupby('CAR_LEG')['LAP_NUMBER'].last(),
              'CONSECUTIVE' : df.groupby('CAR_LEG')['LAP_NUMBER'].size(), 
              'NAME' : df.groupby('CAR_LEG')['DRIVER_NAME'].first()}).reset_index(drop=True)
df2
```

```python
df = laptimes[laptimes['NUMBER']=='7'][:]
df['CAR_STINT'] =  ( df['CROSSING_FINISH_LINE_IN_PIT'].shift()=='B').astype('int').cumsum()
df2 =pd.DataFrame({'LAP_START' : df.groupby('CAR_STINT')['LAP_NUMBER'].first(), 
              'LAP_END' : df.groupby('CAR_STINT')['LAP_NUMBER'].last(),
              'CONSECUTIVE' : df.groupby('CAR_STINT')['LAP_NUMBER'].size(), 
              'NAME' : df.groupby('CAR_STINT')['DRIVER_NAME'].first()}).reset_index(drop=True)
df2.head()
```

```python
df2['CONSECUTIVE'].max(), df2.groupby('NAME')['CONSECUTIVE'].max()
```

```python
laptimes.groupby('NUMBER')[['TOP_SPEED','LAP_NUMBER']].max().head()
```

```python
laptimes.groupby('NUMBER')[['S1_S','S2_S','S3_S']].min().head()
```

```python

```

```python

```

```python

```
