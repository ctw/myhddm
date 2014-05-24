#!/usr/bin/env python

from __future__ import division
import hddm
import pandas as pd
import numpy as np
from patsy import dmatrix
import os
from mydata.munge import find_path

pth=find_path()
data=pd.read_csv(pth+"/beh_hddm/allsx_feat.csv")

def z_link_func(x, data=data):
    stim = (np.asarray(dmatrix('0 + C(s,[[1],[-1]])', {'s':data.stimulus.ix[x.index]})))
    return 1 / (1 + np.exp(-(x * stim)))

def v_link_func(x, data=data):
    stim = (np.asarray(dmatrix('0 + C(s,[[1],[-1]])', {'s':data.stimulus.ix[x.index]})))
    return x * stim


def define_model(mname, project='imaging', regress=False):
	
	check_model(mname)
	
	data=find_data(mname, project)
	
	if project=='imaging':
		intercept="b50N"
	else:
		intercept="c50N"
	
	if regress:
		if mname=='msm':
			msm_vreg = {'model': 'v ~ 1 + C(cue, Treatment('+intercept+'))', 'link_func': v_link_func}
			m=hddm.HDDMRegressor(data, msm_vreg, depends_on={'v':'stim', 'z':'cue'},  bias=True, informative=False, include=['v', 'z', 't', 'a'])
		elif mname=='pbm':
			m=hddm.HDDM(data, depends_on={'v':'stim', 'z':'cue'}, informative=False, bias=True, include=['v', 'z', 't', 'a'])
		elif mname=='dbm':
			dbm_vreg = {'model': 'v ~ 1 + C(cue, Treatment('+intercept+'))', 'link_func': v_link_func(data=data)}
			m = hddm.HDDMRegressor(data, dbm_vreg, depends_on={'v':'stim'},  bias=False, informative=False, include=['v', 't', 'a'])
		elif mname=='dbmz':
			dbmz_vreg = {'model': 'v ~ 1 + C(cue, Treatment('+intercept+'))', 'link_func': v_link_func}
			m=hddm.HDDMRegressor(data, dbmz_vreg, depends_on={'v':'stim'},  bias=True, informative=False, include=['v', 'z', 't', 'a'])
		
	else:
		if mname=='msmt':
			m=hddm.HDDM(data, depends_on={'v':['stim', 'cue'], 'z':'cue', 't':['stim', 'cue']}, bias=True, informative=False, include=['v', 'z', 't', 'a', 'sv', 'sz', 'st'])
		elif mname=='msm':
			m=hddm.HDDM(data, depends_on={'v':['stim', 'cue'], 'z':'cue'},  bias=True, informative=False, include=['v', 'z', 't', 'a', 'sv', 'sz', 'st'])
		elif mname=='pbm':
			m=hddm.HDDM(data, depends_on={'v':'stim', 'z':'cue'}, bias=True, informative=False, include=['v', 'z', 't', 'a', 'sv', 'sz', 'st'])
		elif mname=='dbm':
			m=hddm.HDDM(data, depends_on={'v':['stim', 'cue']}, bias=False, informative=False, include=['v', 'z', 't', 'a', 'sv', 'sz', 'st'])
		elif mname=='dbmz':
			m=hddm.HDDM(data, depends_on={'v':['stim', 'cue']}, bias=True, informative=False, include=['v', 'z', 't', 'a', 'sv', 'sz', 'st'])
		
	return m

def define_sxbayes(mname, data, project='imaging', regress=False):
	
	m=define_single(mname, data, project='imaging', regress=False)
	return m

def define_single(mname, data, project='imaging', regress=False):

	check_model(mname)

	if project=='imaging':
		vreg = {'model': 'v ~ 1 + C(cue, Treatment("b50N"))', 'link_func': v_link_func}
	else:
		vreg = {'model': 'v ~ 1 + C(cue, Treatment("c50N"))', 'link_func': v_link_func}

	if regress:
		if mname=='msm':
			m=hddm.HDDMRegressor(data, vreg, depends_on={'v':'stim', 'z':'cue'},  bias=True, informative=False, include=['v', 'z', 't', 'a'])
		elif mname=='pbm':
			m=hddm.HDDM(data, depends_on={'v':'stim', 'z':'cue'}, informative=False, bias=True, include=['v', 'z', 't', 'a'])
		elif mname=='dbm':
			dbm = hddm.HDDMRegressor(data, vreg, depends_on={'v':'stim'},  bias=False, informative=False, include=['v', 't', 'a'])
		elif mname=='dbmz':
			m=hddm.HDDMRegressor(data, vreg, depends_on={'v':'stim'},  bias=True, informative=False, include=['v', 'z', 't', 'a'])

	else:
		if mname=='msmt':
			m=hddm.HDDM(data, depends_on={'v':['stim', 'cue'], 'z':'cue', 't':['stim', 'cue']}, informative=False, bias=True, include=['v', 'z', 't', 'a'])
		elif mname=='msm':
			m=hddm.HDDM(data, depends_on={'v':['stim', 'cue'], 'z':'cue'}, informative=False, bias=True, include=['v', 'z', 't', 'a'])
		elif mname=='pbm':
			m=hddm.HDDM(data, depends_on={'v':'stim', 'z':'cue'}, informative=False, bias=True, include=['v', 'z', 't', 'a'])
		elif mname=='dbm':
			m=hddm.HDDM(data, depends_on={'v':['stim', 'cue']}, informative=False, bias=False, include=['v', 't', 'a'])
		elif mname=='dbmz':
			m=hddm.HDDM(data, depends_on={'v':['stim', 'cue']}, informative=False, bias=True, include=['v', 'z', 't', 'a'])

	return m

def build_model(mname, project='imaging'):

	m=define_model(mname, project)
	
	m=load_traces(m, mname, project)
			
	return m

def check_model(mname):
	
	mname_list=['msmt', 'msm', 'pbm', 'dbm', 'dbmz']
	
	if mname not in mname_list:
		print "mname not recognized: must be 'msmt', 'msm', 'pbm', or 'dbm'"
		exit()
	else:
		print "building ", mname
	
def find_data(mname, project='imaging'):
	
	pth=find_path()
	
	if project=='behav':
		data=pd.read_csv(pth+"beh_hddm/allsx_feat.csv")
		
	else:
		data=pd.read_csv(pth+"img_hddm/allsx_ewma.csv")
		
	return data

def load_traces(m, mname, project='imaging'):
	
	pth=find_path()
	
	if project=='behav':		
		m.load_db(pth+"beh_hddm/"+mname+"/"+mname+"_traces.db", db='pickle')
		
	else:	
		m.load_db(pth+"img_hddm/"+mname+"/"+mname+"_traces.db", db='pickle')
	
	return m


def build_avgm(m, project='imaging'):
	
	avgm=m.get_average_mname()
	avgm=load_traces(avgm, project)
	
	return avgm
	
if __name__=="__main__":
	main()