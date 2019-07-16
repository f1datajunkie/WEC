<!-- #region -->
# Le Mans 24 Hours, WEC, 2019

Recreating charts in Racecar Engineering.

The August, 2019 (v29(8)) issue of Racecar Engineering provided the annual review of the final race of the FIA WEC championship calender, the Le Mans 24 Hours, including analysis, tables and charts summarising the race.

But how exactly were the data tables and visualisations produced? And can we find a way of automating their production so we can produce similar analyses for other race events?

In this article, we see how modern, code based data analysis techniques based on ideas from the emerging field of "reproducible research", compared to more traditional spreadsheet based approaches, can be used to create perform data analyses with just one or two lines of code.

This article won't teach you how to be professional programmer, but it will hopefully show you how, with a few simpl formulations, you can start to *get stuff done, a line of code at a time*.


*Series: Coding for Racecar Engineers.*
<!-- #endregion -->

### Coding for Racecar Engineers

Many engineers are accustomed to fashioning their own tools from whatever happens to be at hand whenever the need arises. 

In this series of articles, we will look at how "coding", which is to say computer programming, can also be appropriated by engineers on as-and-when basis for helping them *get stuff done* with data and mathematical models.

Unlike the limitations of working with the physical properties of physical materials  will also see how code provides great flexibility as general purpose tool for building other tools.  ?? also automation ?? Infinte toolbox - which can have its downsides!

No programming or coding knowledge is assumed, nor is access to anything other than a computer browser and an internet connection.

So let's get started...

---


The programming language we're going to use is popular in data analysis and web application development. As with many programming languages, the basic language only contains a few commands that perform simple operations, such as performing a numeric calculation, iterating over a list of items and performing an operation on each in turn, or testing a logical condition and taking one of two possible actions depending on whether the tested condition evaluated as true or false.

However, from these basic code instructions, we can also build up far more powerful instructions capable of performing ever more complex operations.

In this article, we'll use several such packages:

- *pandas*, which provides commands that allow us to work easily with tabular datasets (you might think of this as a package that makes it easy to perform spreadsheet like operations on a data table);
- *plotly*, a package for generating a wide range of interactive charts; and
- *cufflinks*, which extends the *pandas* package so that we can easily create *plotly* charts from data managed using *pandas*.


We can start by importing the *pandas* package. The package contains lots of routines that will call by reference to the package itself. By convention, and to make things quicker and easier to type, we often refer to the *pandas* package by the alias `pd`.

```python
import pandas as pd
```

The *pandas* package provides a wide range of tools for loading in data from different data files, including Excel spreadsheets, CSV files (with all sorts of different delimiters), as well as directly from connected databases.

Files can be loaded in from a local computer, or accessed directly from a web address / URL.


### FIA WEC Laptime Data

Official results and classification data, as well event timing data, is available as data files, as well as PDF documents, on the Al Kamel Systems website.

From the results page for the 2018-19 season Le Mans 24 Hours, we can find the URL for the full set of laptimes recorded in a text based CSV document as follows:

???? screenshot ???

???useful to read URL - can also get laptimes by lap?

```python
url = 'http://fiawec.alkamelsystems.com/Results/08_2018-2019/08_LE%20MANS/276_FIA%20WEC/201906151500_Race/Hour%2024/23_Analysis_Race_Hour%2024.CSV'
```

## Downloading and Saving Files Automatically

One of the powerful ways of using code is for automating common tasks. If you want to download a large number of files from the web, you *could* go to each URL in turn, manually, and save the file; or you could let a machine do it.

Of course, there are the inevitable set up concerns: if you want to save a file to disk, *where* do you want to save it?

```python
#Suppose I want to save the data in a directory data_dir in the parent directory to this notebook
data_dir = '../data'

#The directory may or may not already exist - so if it doesn't, then create it

#The os package lets us work with calls to the operating system
import os

# Create data directory if doesn't exist
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
```

To download a file, we can use the `requests` package. This will retrieve a file from a web location into memory and then we can save it to disk.

*For downloading large files that are too big to fit into memory, we can also use requests to stream the file to disk in a series of manageably sized chunks.*

Of course, if we want to save the file to disk, we need to give the file a name, and specify where we want to put it.

For our laptime data, we can get the filename from the URL:

```python
#The path decodes the URL string into several parts; we want the last part - the filename
fname = os.path.basename(url)
fname
```

We also need to find a way to construct the path that says where we want the file with this name to be saved: 

```python
os.path.join(data_dir, fname)
```

Now we can download the file:

```python
import requests

#Download file into memory
r = requests.get(url)

#And save it
with open(os.path.join(data_dir, fname), 'wb') as f:
    f.write(r.content)
```

We can check the file has been saved by viewing the contents of the data directory, which will be returned as a list of filenames:

```python
os.listdir( data_dir )
```

The data file can be automatically downloaded from the URL into memory (i.e. *not* saved to disk as a file) and converted into a *pandas* dataframe using a *pandas* command of the form:

`pd.read_csv(url)`

In reality, however, things are never quite that simple. (This is one reason why much coding activity can be frustrating!)

Here's how I actually loaded the data in and previewed the first few rows:

```python
laptimes = pd.read_csv(url, sep=';').dropna(how='all', axis=1)

#Alternatively, we could load from the downloaded file:
#laptimes = pd.read_csv(os.path.join(data_dir, fname), sep=';').dropna(how='all', axis=1)

laptimes.head()
```

So how does that work?

On its own, the `pd.read_csv()` command assumes that the data file to be loaded is, by default, a *comma separated file*, where each line of the data file represents one row of data, and the data in each column are separated using commas.

The Al Kamel CSV data file uses a semi-colon, rather than a comma, separator, so we actually need to load the file using  a command of the form:

`pd.read_csv(url, sep=';')`

When you do load in the data, there appears to be at least one column that is just filled with blank "not assigned" `NA`, or *not a number*, `NaN`, values; that is, `all` the values are null values in an empty column. We can tell *pandas* to drop a a completely empty column (`axis=1` in a *pandas* data frame; rows are identified as `axis=0`) as follows:

`pd.read_csv(url, sep=';').dropna(how='all', axis=1)`

We could also add a further `.dropna(how='all',axis=0)` step to remove any and all completely blank rows:

`pd.read_csv(url, sep=';').dropna(how='all', axis=1).dropna(how='all', axis=0)`

One other thing to note: when we do download the data, we want to be able to refer to it somehow. We do this by assigning the loaded data to a variable called `laptimes` using the assignment operator `=`.

Note that the equals sign on its own is *not* a way of testing if one thing equals another; instead, we can read it as *"is assigned the value"*:

`variable = value`

Which is to say: *the variable called `variable` is assigned the value of `value`*.

We can preview the first few rows of the dataframe using the command `dataframe_variable.head()` or `dataframe_variable.head(N)` where `N` is an integer to display the top *N* rows. 

*As well as `.head()` you can also `.tail()` to see rows from the bottom of the table.*


### Viewing Column Names
It can often be useful to preview all the column names in the table. Adding `.columns` to the name of a dataframe variable will display all the column names:

```python
laptimes.columns
```

<!-- #region -->
Each column name is represented as a *text string*. All programming languages have a notion of different "types" of things, such as integers, floating point numbers (eg `12.345`), booleans (`True` / `False`) and strings (sequences of alphanumeric and punctuation characters).

The strings are identified as strings because they appear in quotes. The `[` and `]` characters show that we actually have a *list* of comma separated strings:

`['a', 'list', 'of' 'strings']`


Sometimes, we may find that a column name includes a space character, which is a form of *whitespace*, along with tabs and end of line characters. Trying to remember what whitespace is where around a column name is an unnecessary distraction if we want to refer to a column by name, so let's fix that before we go any further.

*A lot of coding activity is spent setting things up so that you can then do the things you actually want to do...*

We can use some voodoo magic known as a Python *list comprehension* to `strip()` whitespace from the beginning and end of a character string.

The list comprehension creates a list, `[...]`, by iterating through each column name (`for c in laptimes.columns`) and stripping any whitespace wrapping each column name in turn (`c.strip()`). The list of stripped ("cleaned") column names is then assigned back to the list of column names, effectively rewriting them as the clean, whitespace removed, names. 
<!-- #endregion -->

```python
laptimes.columns = [c.strip() for c in laptimes.columns]
laptimes.columns
```

## Referencing the Data In a Particular Column or Set of Columns
One of the powerful features of *pandas* is that it lets us refer to one or more columns of values by referencing the corresponding column name.

For example, here are the first few values in the `TEAM` column:

```python
laptimes['TEAM'].head()
```

And here you see a dirty little secret of many coding languages: the meaning of a particular symbol may be dependent on context. In this case, the `[..]` characters in the phrase `laptimes['TEAM']` identify an *index value* on the `laptimes` dataframe, *not* a list. 


Perhaps confusingly though, we can pass a list of values into an index selector to access values from multiple columns at the same time:

```python
laptimes[ ['TEAM', 'MANUFACTURER'] ].head()
```

One other handy thing we can do with columns is look to see what unique values are contained within the column.

For example, what teams are there?

```python
laptimes['TEAM'].unique()
```

We can also find unique values in one column associated with unique values in another column using the notion of groups.

If we want to find the unique combimations of manufacturer and team, we can view just the `MANUFACTURER` and `TEAM` columns and then drop all duplicated combinations.

To make the table easier to read, we can then sort the resulting table across on or more columns.

```python
laptimes[['MANUFACTURER','TEAM']].drop_duplicates().sort_values(['MANUFACTURER', 'TEAM'])
```

As well as filtering the table to only show specified *columns*, we can also filter the table to only display particular *rows*. One way of achieving this is to create a dummy column of boolean values that say, for each row, whether a particular condition is `True` of `False`. This dummy column can then be used to select just the rows for which the condition evaluates as `True`.

For example, let's pull out values for a particular manufacturer. We test for equality using the `==` operator; for quality, we will bracket the quality test although we do not strictly need to: the `==` operator has precedence over the assignement operator (`=`).

```python
IS_PORSCHE = (laptimes['MANUFACTURER'] == 'Porsche')
IS_PORSCHE.head()
```

We can now use this as a filter term:

```python
laptimes[IS_PORSCHE][['MANUFACTURER','TEAM']].drop_duplicates().sort_values(['MANUFACTURER', 'TEAM'])
```

We can test against multiple teams using the `.isin()` method that takes a *list* of teams as an argument:

```python
IS_IN = laptimes['MANUFACTURER'].isin( ['Porsche', 'BMW'] )
laptimes[IS_IN][['MANUFACTURER','TEAM']].drop_duplicates().sort_values(['MANUFACTURER', 'TEAM'])
```
## Getting the Data Straight

As well as the dirty little secrets of coding, there are also the dirty little secrets of data analysis and visualisation is that creating analyses and visualisations is often one of the quickest parts of the task, *if* the data is in the right form.

But it generally isn't: *cleaning* the data (stripping whitespace, tidying up badly formed data, working out what to do with null or missing values, reconciling things that are supposed to be the same but aren't) describes the activity of getting the data into a fit state where we can begin to work with it.

In many data analysis tasks, cleaning the data can take more time than the time required to actually do anything useful with it.

Cleaning may seem like an overhead, but it can save a lot of time later on if you know you can trust the data when you find an odd result or if something doesn't appear to work correctly: there's no point trying to debug your code if it's actually the data that is broken.

When we loaded the data, we took care to drop empty columns, and then took a defensive step of tidying up the column names by removing any whitespace around them. But we can also do a bit more...

For example, on loading the data, *pandas* made an educated guess about what *type* of data was in each column (*strings* are generalised as `object`):

```python
#Show the datatype (dtype) of each column in the laptimes dataframe
laptimes.dtypes
```

One of the things we notice is that the car number (`NUMBER`) and driver number (`DRIVER NUMBER`) are represented as integers. This makes some sort of sense (the numbers are *numbers*, after all) but it wouldn't really make sense to add car numbers `1` and `7` to give car `8`; however, it could make sense if we could "add" them to give `1 and 7` for example.

So rather than treating the car and driver numbers as integers, let's turn them into strings to ensure we don't actually try to add them, or plot them, as numbers.

```python
#Tidy the data a little... car and driver number are not numbers
string_columns = ['NUMBER','DRIVER_NUMBER']
laptimes[ string_columns ] = laptimes[ string_columns ].astype(str)
```

*Normalising* the data can refer to a range of activities that include getting clean data into a format that we can actually start to analyse.


If we look at the Le Mans laptime data, we notice, for example, that the laptimes are given in the form `3:22.215`. This is fine for human readers, but the computer just sees it as a string, rather than as a representation of a time value in the form `minutes:seconds`.

To make it useful, we need to cast those times either to an object that represents time as a numerical quantity with certain attributes (60 minute types in an hour type, for example) or that represents it in a simpler, meaningful form, such as the equivalent number of seconds).

```python
#Add the parent dir to the import path
import sys
sys.path.append("..")

#Import contents of the utils.py package in the parent directory
from py.utils import *

#Get laptimes in seconds
for i in ['S1', 'S2', 'S3', 'LAP_TIME']:
    laptimes['{}_S'.format(i)] = laptimes[i].apply(getTime)

#Find accumulated time in seconds
laptimes['ELAPSED_S']=laptimes['ELAPSED'].apply(getTime)
```

```python
#Find position based on accumulated laptime
laptimes = laptimes.sort_values('ELAPSED_S')
laptimes['POS'] = laptimes.groupby('LAP_NUMBER')['ELAPSED_S'].rank()

#Find leader naively
laptimes['leader'] = laptimes['POS']==1

#Find lead lap number
laptimes['LEAD_LAP_NUMBER'] = laptimes['leader'].cumsum()

laptimes.head()
```

## Mean laptime over 20 best laps


```python
#select out inlaps
#~laptimes['CROSSING_FINISH_LINE_IN_PIT'].isnull()
```

```python
laptimes.groupby('NUMBER')['LAP_TIME_S'].nsmallest(20).groupby('NUMBER').mean().round(decimals=3).to_frame().head(10)

```

```python
def fastest20pc(carlaps, pc=0.2):
    ''' Return the mean of the twenty percent fastest laptimes from a pandas Series of laptimes. '''
    
    #How many laps make up the 20% (default) of all laps?
    num = int(carlaps.count() * pc)
    
    #Return the mean of the fastest 20 per cent of laptimes, rounded to 3dp
    return carlaps.nsmallest(num).mean().round(decimals=3)

laptimes.groupby('NUMBER')['LAP_TIME_S'].apply(fastest20pc).to_frame().head()
```

```python
laptimes.groupby('NUMBER')['LAP_TIME_S'].min().sort_values().to_frame().head()
```

## Rising Average

The *rising average* laptime chart shows how the average (mean) laptime for a car increases for increasing laptime.

To create the chart, we need to sort the laptimes to show, for each car, its laptimes in increasing laptime order.

```python
#!pip install plotly cufflinks
#cufflinks provides plotly bindings for pandas
import cufflinks
cufflinks.go_offline(connected=False)
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
                                                         values='CAR_LAP_RANKED_CUMAV').iplot();
```

It can often be instructive to construct a chart in this way, by clearly thinking through some baby steps that build up the data items that you want to plot.

There is, however, often a quick way, using some of the tools that are provided as part of the *pandas* package. (Knowledge of the tools available within the package come with time, as does the expert knowledge that allows you to identify when a particular tool is the one that will help solve your current problem.)

```python
laptimes[laptimes['NUMBER']=='8']['LAP_TIME_S'].sort_values().expanding(min_periods=1).mean().reset_index(drop=True).iplot();

```

### Advanced Technique

The *pandas* package contains a lot of powerful tools for working with tabular datasets, including predefined ways of calculating cumulative averages.

```python
tmp =  laptimes[laptimes['NUMBER'].isin(SELECTED)].pivot(index='CAR_LAP_RANK',
                                                         columns='NUMBER', 
                                                         values='LAP_TIME_S')
tmp.expanding(min_periods=1).mean().reset_index(drop=True).iplot();
```

We get the horizontal line at the right where the NA values were - TO DO

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
tmp.expanding(min_periods=1).mean().mask(tmp.isnull(),tmp).iplot();
```

We can use a similar trick to show the rising average laptime for each driver in a particular car.

```python
laptimes['CAR_LAP_RANK_DRIVER'] = laptimes.sort_values(['NUMBER','DRIVER_NAME','LAP_TIME_S']).groupby(['NUMBER','DRIVER_NAME']).cumcount()


tmp2 =  laptimes[laptimes['NUMBER']=='7'].pivot(index='CAR_LAP_RANK_DRIVER',
                                                         columns='DRIVER_NAME', 
                                                         values='LAP_TIME_S')


tmp2.expanding(min_periods=1).mean().mask(tmp2.isnull(),tmp2).iplot();
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

## GTE Pro

```python
#It's not the av over 20 best laps...
laptimes[laptimes['NUMBER']=='51'].sort_values('LAP_TIME_S')['LAP_TIME_S'].nsmallest(20).mean().round(decimals=3)
```

```python
#It's over best 20 per cent of laps
laptimes.groupby('NUMBER')['LAP_TIME_S'].apply(fastest20pc).to_frame().loc['51']
```

# Porsche Best etc  - Sector times



```python
laptimes.columns
```

```python
laptimes['MANUFACTURER'].unique()
```

The Porsche Bends and Ford Chicane sector times are contained in a separate data file, unfortunately only published as a PDF file rather than a data file.

But that needn't stop us...

There are several tools out there that allow us to grab tabular data out of a PDF document, although sometimes we may need to do quite a bit of cleaning to tidy it up and get it into a state that we can properly work with.

?? screenshot and breakout box of how to get data out using tabula ??

```python

#apt-get update ** apt-get install -y ghostscript && apt-get clean

# pip install opencv-python camelot-py sortedcontainers
```

```python
# Best Specific Sectors
sector_times_url = 'http://fiawec.alkamelsystems.com/Results/08_2018-2019/08_LE%20MANS/276_FIA%20WEC/201906151500_Race/Hour%2024/13_BestSpecificSectors_Race_Hour%2024.PDF'

```

```python
#Another way of grabbing the data file - dump this for the requests approach
!wget $sector_times_url
```

```python
#tabula - we could install this in the container used to demo the notebook?
# java, tabula-java, tabula-py ?
```

I have downloaded the sectors PDF and extracted the data from it using Tabula. This is what the raw data I extracted looks like.

```python
#T6, T7, Porsche, Ford, T6+T7
s = pd.read_csv('../data/tabula-13_BestSpecificSectors_Race_Hour 24.csv').dropna( subset=['Kph']).dropna(how='all', axis=1)
s.head()
```

The actual column names, which have not been extracted, are ... TO DO


If you look closely at the columns, you'll notice that some of them actually contain two sets of data: the Tabula extractor didn't manage to identify them as separate columns.

So back again we go to tidying things up so that we *can* start to work with them.

One of the things we notice from the column names is that the table appears to have a repeated structure. In fact, it looks like the table has a position column followed by five subtables arranged side by side; each sub-table contains two columns, a *Driver* column and a compound *Time / Kph* column. Apart from the subtable on the far right, where the *Time* and *Kph* columns have been separately identified.

So lets regularise everything, getting the column names to conform to the same pattern and making the last subtable look like the others.

```python
#Make the rightmost table look like the first by joining the Time and Kph columns into a single column
s['Time Kph.4'] = s['Time'].astype(str)+' '+s['Kph'].astype(str)

#Drop the original Time and Kph columns from the rightmost subtable
s.drop(columns=['Time', 'Kph'], inplace=True)

#Rename the column names for the leftmost subtable so they have the same form as the other subtable column names
s.rename(index=str, columns={"Driver": "Driver.0", "Time Kph": "Time Kph.0"}, inplace=True)
s.head()
```

The table in this format is known as a *wide* format. We can convert it to a *long* format by concatenating each of the subtables into one long table.

The `stubnames` argument tells the `wide_to_long` function to look for column names starting with the stub pattern as the basis for the transformation.

The `i` parameter... whilst the `j` parameter...

```python
s2 = pd.wide_to_long(s, stubnames=['Driver', 'Time Kph'], sep='.', i='Pos', j='SECTOR')
s2.head()
```

```python
s2[['NUMBER','DRIVER_NAME']] = s2['Driver'].str.strip().str.extract(r'(\d+\s?)(.*)$',expand=True).astype(str)
s2['NUMBER'] = s2['NUMBER'].str.strip()
s2['DRIVER_NAME'] = s2['DRIVER_NAME'].str.strip()
s2.head()
```

```python
s2[['TIME','KPH']] = s2['Time Kph'].str.strip().str.extract(r'([\d\.]+\s?)([\d\.]*)$',expand=True).astype(float).round(decimals=3)
s2.head()
```

```python
s2.reset_index(inplace=True)
s2['SECTOR'] = s2['SECTOR'].map({0:'T6', 1:'T7', 2:'Porsche', 3:'Ford', 4:'T6_T7'})
s2.drop(columns=['Driver', 'Time Kph'], inplace=True)
s2.head()
```

## Gap Graph

The article includes an example "gap graph" showing the gap between a target car and two other cars. The gap is simple the difference in elapsed time at each lap.

The *rebasing* notion allows us to generate differences in time relative to a specified driver.

```python
def _rebaseTimes(times, car=None):
    ''' Rebase times relative to specified car. '''
    
    if car is None:
        return times
    
    return times - times.loc[car]
```

```python
wide_laptimes = laptimes.pivot(index='NUMBER', columns='LAP_NUMBER', values='ELAPSED_S')
wide_laptimes.head()
```

```python
_rebaseTimes(wide_laptimes, car='11').head()
```

We can limit the rows are displayed by filtering based on the dataframe index, which is to say, the car number.

```python
_rebaseTimes(wide_laptimes, car='8').loc[['8','11','7']]
```

The data frame has the lap number as the column heading and the car number as the index. To plot the data against lap number on the x-axis, we need to *transpose* the table (that is, swap rows and columns). The `.T` method does this straightforwardly:

```python
_rebaseTimes(wide_laptimes, car='8').loc[['8','11','7']].T.head()
```

The data frame looks like something we can work with, so now we can simply plot the gap times:

```python
_rebaseTimes(wide_laptimes, car='8').loc[['8','11','7']].T.iplot()
```

```python
_rebaseTimes(wide_laptimes, car='8').loc[['8','11','7']].T.iplot()
```

```python

```
