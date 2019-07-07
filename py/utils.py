# This is a utility function that should really be:
# - written properly
# - documented
# - tested
# - packaged
# Or similar functions may exist in other packages, and I just haven't found them yet!

from datetime import timedelta
import pandas as pd

def getTime(ts, ms=False, astime=False):
    ''' Decode a delta time string and return time as numeric or timedelta type'''
    def formatTime(t):
        return float("%.3f" % t)
    
    def retNull(astime=False):
        if astime:
            return pd.to_datetime('')
        else:
            return None #What about np.nan?
    
    ts=str(ts).strip()
    if not ts:
        return retNull(astime)
    
    t=ts.split(':')
    if len(t)==3:
        tt = timedelta(hours=int(t[0]), minutes=int(t[1]), seconds=float(t[2]))
    elif len(t)==2:
        tt=timedelta(minutes=int(t[0]), seconds=float(t[1]))
    else:
        tt = pd.to_numeric(t[0], errors='coerce')
        if pd.isnull(tt):
            return retNull(astime)
        tt=timedelta(seconds=tt)
    
    if astime:
        return tt
    elif ms:
        return 1000*formatTime(tt.total_seconds())

    return formatTime(tt.total_seconds())
