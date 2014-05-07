#!/usr/bin/env python

from __future__ import division
import pandas as pd
import hddm
from myhddm import sdt, defmod, vis, sims

def test_dfs():
	
	df=pd.read_csv("/usr/local/lib/python2.7/site-packages/myhddm/test_dfs/allsx_ewma.csv")
	simdf=pd.read_csv("/usr/local/lib/python2.7/site-packages/myhddm/test_dfs/simdf.csv")
	
	return df, simdf
	
	