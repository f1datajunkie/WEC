```python
import numpy as np
```

```python
import pandas as pd
import holoviews as hv
import streamz
from holoviews.streams import Pipe
```

```python
#hv.extension('bokeh')
hv.extension('plotly')
```

## Line Chart - Buffer

```python
from holoviews.streams import Buffer

from tornado.ioloop import PeriodicCallback
from tornado import gen
```

```python
##This recipe seems to provide us with a way of updating a streaming chart 
import numpy as np
```

```python
from holoviews.streams import Buffer
```

```python
from tornado.ioloop import PeriodicCallback
from tornado import gen
```

```python
count = 0
buffer = Buffer(np.zeros((0, 2)), length=5)
```

```python
@gen.coroutine
def f():
    global count
    count += 1
    buffer.send(np.array([[count, np.random.rand()]]))
```

```python
cb = PeriodicCallback(f, 10000)
cb.start()
```

```python
hv.DynamicMap(hv.Curve, streams=[buffer]).opts(padding=0.1, width=600, color = 'red',)
```

```python
buffer.data
#So we could look at the contents of the buffer
# and then send items we haven't yet added to the buffer...
```

```python
cb.stop()
```

## Example Stream From DataFrame

Can we get an example chart going from a dataframe?

```python
import pandas as pd

import sys
sys.path.append("../py")
from utils import getTime
```

```python
url_analysis = 'http://fiawec.alkamelsystems.com/Results/09_2019-2020/01_SILVERSTONE/285_FIA%20WEC/201908301140_Free%20Practice%201/23_Analysis_Free%20Practice%201.CSV'

```

```python
df = pd.read_csv(url_analysis, sep=';')
df.columns = [c.strip() for c in df.columns]
df['LAP_TIME_IN_S'] = df['LAP_TIME'].apply(getTime)
df.head()
```

```python
df[df['NUMBER']==1][['LAP_NUMBER','LAP_TIME_IN_S']].iloc[1].values.tolist()
```

```python
rowcount = 0

subdf = df[df['NUMBER']==1]
maxrows = len(subdf)

dfbuffer = Buffer(np.zeros((0, 2)), length=50)

@gen.coroutine
def g():
    global rowcount
    item = subdf[['LAP_NUMBER','LAP_TIME_IN_S']].iloc[rowcount].values.tolist()
    dfbuffer.send(np.array([item]))
    rowcount += 1
    
    if rowcount>=maxrows:
        cbdf.stop()


```

```python
#How can we get the thing to stop?

cbdf = PeriodicCallback(g, 200)
cbdf.start()
hv.DynamicMap(hv.Curve, streams=[dfbuffer]).opts(padding=0.1, width=600, color = 'green',)
```

```python
cbdf.stop()
```

```python

```

```python

```
