#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import h5py

def load_phenotypes(path, a_id="SUB_ID"):
    pheno = pd.read_csv(path, encoding='latin1')
    pheno[a_id] = pheno[a_id].astype(str)
    return pheno

def run_progress(fn, collection, message=None, jobs=1):
    if not message:
        message = "Processing {current} of {total}"

    def wrapper(i, el):
        pbar.set_description(message.format(current=i+1, total=total))
        pbar.update()
        return fn(el)

    total = len(collection)
    pbar = tqdm(total=total, leave=False)

    if jobs == 1:
        result = [wrapper(i, el) for i, el in enumerate(collection)]
    else:
        with Pool(jobs) as p:
            result = p.starmap(wrapper, enumerate(collection))

    pbar.close()
    return result

def hdf5_handler(path, mode='r'):
    return h5py.File(path, mode)
