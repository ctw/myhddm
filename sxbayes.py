#!/usr/bin/env python

from __future__ import division
import hddm, os
import numpy as np
import pandas as pd
from myhddm import defmod, parse, vis
from mydata.munge import find_path
from patsy import dmatrix

#data=pd.read_csv("/Users/kyle/Desktop/beh_hddm/allsx_feat.csv")


def z_link_func(x, data):
    stim = (np.asarray(dmatrix('0 + C(s,[[1],[-1]])', {'s':data.stimulus.ix[x.index]})))
    return 1 / (1 + np.exp(-(x * stim)))

def v_link_func(x, data):
    stim = (np.asarray(dmatrix('0 + C(s,[[1],[-1]])', {'s':data.stimulus.ix[x.index]})))
    return x * stim

def run_models(mname, project, regress=False):
	
	#bayes fit all subject
	allsx_df=fit_sx(mname, project=project, regress=regress)
	
	#parse model output
	subdf=parse_allsx(allsx_df)
	pdict=subdf_to_pdict(subdf)
	
	#simulate and compare with observed data
	data=defmod.find_data(mname, project=project)
	simdf=vis.predict(pdict, data, ntrials=160, nsims=100, save=True, RTname="SimRT_EvT.jpeg", ACCname="SimACC_EvT.jpeg")
	simdf.to_csv("simdf_sxbayes.csv")
	
	#save pdict; can be reloaded and transformed back into
	#the original pdict format by the following commands 
	#1.  pdict=pd.read_csv("sxbayes_pdict.csv")
	#2.  pdict=pdict.to_dict()
	params=pd.DataFrame(pdict)
	params.to_csv("sxbayes_pdict.csv", index=False)

def aic(model):
	k = len(model.get_stochastics())
	logp = sum([x.logp for x in model.get_observeds()['node']])
	return 2 * k - 2 * logp

def bic(model):
	k = len(model.get_stochastics())
	n = len(model.data)
	logp = sum([x.logp for x in model.get_observeds()['node']])
	return -2 * logp + k * np.log(n)

def dic(model):
	return model.dic
	
def fit_sx(mname, project='behav', regress=False):
	
	pth=find_path()

	data=defmod.find_data(mname, project)
	
	grp_dict={}; subj_params=[]; aic_list=[]; bic_list=[]; dic_list=[]; ic_dict={}

	for subj_idx, subj_data in data.groupby('subj_idx'):

		m_sx=defmod.define_sxbayes(mname, subj_data, project=project, regress=regress)
		m_sx.sample(1000, burn=500, dbname=str(subj_idx)+"_"+mname+'_traces.db', db='pickle')
		
		sx_df=parse.stats_df(m_sx)
		sx_df=sx_df.drop("sub", axis=1)
		sx_df['sub']=[subj_idx]*len(sx_df)
		
		subj_params.append(sx_df)
		aic_list.append(aic(m_sx)); bic_list.append(bic(m_sx)); dic_list.append(m_sx.dic)

	allsx_df=pd.concat(subj_params)
	allsx_df.to_csv(mname+"_SxStats.csv", index=False)
	
	ic_dict={'aic':aic_list, 'bic':bic_list, 'dic':dic_list}
	ic_df=pd.DataFrame(ic_dict)
	ic_df.to_csv(mname+"_IC_Rank.csv")
	
	return allsx_df

def parse_allsx(allsx_df):
	
	stims=[]; stim_list=['face', 'house']
	cues=[]; cue_list=['a90H', 'b70H', 'c50N', 'd70F', 'e90F']
	params=[]

	for p in allsx_df['param']:
		if ")" in list(p):
			params.append(p[0])
			cond=p[2:-1]
			if cond in stim_list:
				stims.append(cond)
			else: 
				stims.append("constant")
			if cond in cue_list:
				cues.append(cond)
			else:
				cues.append("constant")
		else:
			params.append(p)
			stims.append("constant")
			cues.append("constant")
		
	allsx_df['stim']=stims
	allsx_df['cue']=cues
	allsx_df['parameter']=params
	allsx_df['noise']=['constant']*len(allsx_df)
	
	subdf=allsx_df[['sub', 'param', 'mean', 'parameter', 'cue', 'stim', 'noise']]
	
	subdf=parse.txtparse(subdf, 'sub')
	subdf.index=range(len(subdf))	
	
	return subdf


def subdf_to_pdict(mname, subdf):
	
	cond_list=['a90H_face', 'b70H_face', 'c50N_face', 'd70F_face', 'e90F_face', 'a90H_house', 'b70H_house', 'c50N_house', 'd70F_house', 'e90F_house']
	
	allsx={}
	conditions={}
	params={}
	
	for sx, sxdata in subdf.groupby('sub'):
		conditions={}
		for cond in cond_list:
			cond_cue=cond.split("_")[0]
			cond_stim=cond.split("_")[1]
			conditions[cond]={'a':sxdata.ix[sxdata['parameter']=='a', 'mean'].unique()[0], 
				't':sxdata.ix[sxdata['parameter']=='t', 'mean'].unique()[0], 
				'v':sxdata.ix[(sxdata['parameter']=='v')&(sxdata['stim']==cond_stim), 'mean'].unique()[0], 
				'z':sxdata.ix[(sxdata['parameter']=='z')&(sxdata['cue']==cond_cue), 'mean'].unique()[0]}

			for i in sxdata.param:
			
				if "_" not in list(i) or i.split("_")[1][0]=='I':
					continue
			
				if i.split("_")[1][0]=='C':
					cue=i.split(']')[0][-4:]
				if cue==cond_cue:
					conditions[cond]['v']=conditions[cond]['v']+sxdata.ix[sxdata['param']==i, 'mean'].unique()[0]
	
		allsx={str(sx):conditions}
	
	return allsx

def simform(subdf, sc=None):
	"""
	RETURNS: 1

		*condsdf (pandas DataFrame):	one column for each experimental cue
										columns for sub_id and params as well 
										(is used to make pdict (which is used for simulating)							

	"""
	
	groupdf=False

	nparams=len(subdf.parameter.unique())

	nsubs=len(subdf['sub'].unique())
	nrows=nsubs*nparams

	if nrows==nparams:
		groupdf=True

	if len(subdf.cue.unique())<5:
		condsdf=pd.DataFrame(np.zeros(nrows*6).reshape((nrows, 6)), columns=['a80H_face', 'b50N_face', 'c80F_face', 
			'a80H_house', 'b50N_house', 'c80F_house'])
	else:
		condsdf=pd.DataFrame(np.zeros(nrows*10).reshape((nrows, 10)), columns=['a90H_face', 'b70H_face', 'c50N_face', 
			'd70F_face', 'e90F_face', 'a90H_house', 'b70H_house', 'c50N_house', 'd70F_house', 'e90F_house'])

	counter=1
	for cond in condsdf.columns:
		cue_n=cond.split('_')[0]
		img_n=cond.split('_')[1]

		if counter==1:
			cdf=subdf.ix[subdf['stim'].isin([img_n, 'constant']) & subdf['cue'].isin([cue_n, 'constant']), ['sub', 'parameter', 'mean']]
			cdf.index=range(len(cdf))
			if not groupdf:
				condsdf['sub']=cdf['sub'].values
			condsdf['param']=cdf['parameter'].values
		else:
			cdf=subdf.ix[subdf['stim'].isin([img_n, 'constant']) & subdf['cue'].isin([cue_n, 'constant']), ['mean']]
			cdf.index=range(len(cdf))
		condsdf[cond]=cdf['mean'].values

		counter+=1

	if sc is not None:
		for i in condsdf.columns:
			if '_' in i: 
				isplit=i.split('_')
				if 'face' in isplit and sc=='v':
					condsdf.ix[(condsdf['param']==sc), i]=abs(condsdf.ix[(condsdf['param']==sc), i])
				elif 'face' in isplit and sc=='z':
					condsdf.ix[(condsdf['param']==sc), i]=1-condsdf.ix[(condsdf['param']==sc), i]

	return condsdf

def create_pdict(condsdf, grp_dict=None):
	"""
	Arguments: condsdf (pandas dataframe)

	Returns: 
		*pdict (dict):		dict for all subs with parameter names and values
							estimated for each exp. cue included in the 
							original model.

		 					is used to loop through when simulating 
							data with hddm.generate.gen_rand_data()

							structure: 

								{subID{cond{param : param_value}}}
	"""
	add_z=False

	if 'z' not in condsdf.param.unique():
		add_z=True

	condsdf.index=condsdf.param

	pdict=dict()
	for subj, group in condsdf.groupby('sub'):
	    sdict=dict()
	    for cond in group:
			if cond == 'sub':
				continue
			elif cond == 'param':
				continue
			sdict[cond]=dict(group[cond])

	    pdict[subj]=sdict

	if hasattr(grp_dict, "keys"):
		for sub in pdict:
			for cond in pdict[sub]:
				pdict[sub][cond]['sv']=grp_dict['sv']
				pdict[sub][cond]['st']=grp_dict['st']
				pdict[sub][cond]['sz']=grp_dict['sz']
				if add_z:
					pdict[sub][cond]['z']=0.5
	return pdict


				
def get_sxmodel_dic():				
	
	#models=['v', 'z', 'vz']
	models=['msm', 'dbm',  'dbmz', 'pbm']
	dic_list=[]
	dic_dict={}
	#skip_sx=[0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 14, 25, 28]
	skip_sx=[0, 5, 19]
	globalp="/Users/DunovanK/Desktop/beh_hddm/EWMA5/subj_bayes/MCMC10K_NoVar/"
	#pth=find_path()
	#globalp=pth+"img_hddm/subj_bayes/"
	for m in models:
		dic_list=[]
		os.chdir(globalp+m)
		for sx in range(26):

			if sx in skip_sx:
				continue

			subj=pd.read_table(str(sx)+"_params.txt", delim_whitespace=True, header=0, index_col=0)['mean']
			dic_list.append(subj.ix['DIC:'])

		dic_dict[m]=dic_list

	mdic_df=pd.DataFrame(dic_dict)
	return mdic_df		