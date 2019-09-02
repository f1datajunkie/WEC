# # First Practice Report
#
# *Proof of concept*

# +
import pandas as pd

from utils import getTime
# -

url_classification = 'http://fiawec.alkamelsystems.com/Results/09_2019-2020/01_SILVERSTONE/285_FIA%20WEC/201908301140_Free%20Practice%201/03_Classification_Free%20Practice%201.CSV'
url_analysis = 'http://fiawec.alkamelsystems.com/Results/09_2019-2020/01_SILVERSTONE/285_FIA%20WEC/201908301140_Free%20Practice%201/23_Analysis_Free%20Practice%201.CSV'


df_classification = pd.read_csv(url_classification, sep=';').dropna(how='all', axis=1)
df_classification.columns = [c.strip() for c in df_classification.columns]
df_classification.head()

df_classification.columns

# ### Parsing Out Drivers Info
#
# The classification data table could do with some normalisation. Let's start by pulling out the driver details.
#
# We can start by making the data wide on driver number and long on driver details.

drivercols = [c for c in df_classification.columns if c.startswith('DRIVER')]
drivercols

tmp = pd.wide_to_long(df_classification[[ 'NUMBER']+drivercols],
                ['DRIVER{}_'.format(i) for i in range(1,7)],
                i=['NUMBER'], j="Property", suffix='.*').dropna(how='all', axis=1).reset_index()
tmp

# We can then melt this from wide on drivers to long on everything:

tmp2 = pd.wide_to_long(tmp, ['DRIVER'],
                i=['NUMBER', 'Property'],j="DriverNumber", suffix='.*').reset_index()
tmp2

# Then pivot back to give us wide on driver details.
#
# There looks to be something fishy going on with the pivot, so I've hacked it from whatever form it was in to a dict then back to a dataframe.

# +
tmp2['_idx'] = tmp2['DriverNumber']+tmp2['NUMBER'].astype(str)
tmp3 = tmp2.pivot(index='_idx', columns='Property', values='DRIVER').dropna(how='all').reset_index()
tmp3[['DriverNum','NUMBER']] = tmp3['_idx'].str.split('_',expand=True)
tmp3 = pd.DataFrame(tmp3.to_dict()).drop(columns=['_idx'])[['NUMBER','DriverNum','SECONDNAME','FIRSTNAME',
                                                           'COUNTRY','ECM Country Id','ECM Driver Id','LICENSE']]
df_classification_drivers = tmp3.set_index(['NUMBER', 'DriverNum']).sort_index()

#Tidy up on memory
tmp = tmp2 = tmp3 = None

# The ECM Driver Id should be an int but there's at least one nan...
# I guess we could cast it to an integer in string space?
df_classification_drivers

# +
df_classification_core = df_classification.drop(columns=drivercols)
df_classification_core['timeInS'] = df_classification_core['TIME'].apply(getTime)
df_classification_core['gapFirstInS'] = df_classification_core['GAP_FIRST'].apply(getTime)
df_classification_core['gapPreviousInS'] = df_classification_core['GAP_PREVIOUS'].apply(getTime)

df_classification_core.head()
# -

# ## Laptime Analysis

df_analysis = pd.read_csv(url_analysis, sep=';').dropna(how='all', axis=1)
df_analysis.columns = [c.strip() for c in df_analysis.columns]
df_analysis.head()

df_analysis.columns


