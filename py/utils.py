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


#Via https://gist.github.com/wy193777/0e2a4932e81afc6aa4c8f7a2984f34e2
import requests
#from tqdm import tqdm

def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """

    dst_path = os.path.dirname(dst)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    file_size = int(requests.head(url).headers["Content-Length"])

    #?? won't this append if the file already exists?
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size
