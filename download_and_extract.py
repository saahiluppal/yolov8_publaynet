from pathlib import Path
from typing import Optional

import json
import os

import tarfile

from tqdm import tqdm
import requests


DATA = "extracted"

LABELS_URL = "https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/labels.tar.gz"
TRAIN_URLS = [
    "https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-0.tar.gz",
    "https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-1.tar.gz",
    "https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-2.tar.gz",
    "https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-3.tar.gz",
    "https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-4.tar.gz",
    "https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-5.tar.gz",
    "https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-6.tar.gz",
]

# https://stackoverflow.com/a/62113293 - with minor updates
def download(url: str, fname: Optional[str] = None):
    if not fname:
        fname = os.path.basename(url)

    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))

    if os.path.exists(fname) and os.path.getsize(fname) == total:
        print("Caching...")
        return

    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

# extract the tarfile
def untar(fname: str, destination: Optional[str] = DATA):
    tar = tarfile.open(fname)
    tar.extractall(destination)
    tar.close()


if __name__ == '__main__':
    
    for train_url in TRAIN_URLS: download(train_url)
    download(LABELS_URL)
    untar(os.path.basename(LABELS_URL))
    for train_url in TRAIN_URLS: untar(os.path.basename(train_url))