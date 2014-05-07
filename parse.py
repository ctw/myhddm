#!/usr/bin/env python
"""
Includes Functions For:

	::Creating dataframe (pandas) stim for hddm.model.gen_stats() 

	::Doing heavy parsing and reformatting
		**Creates extentions of stats dataframe including columns for:
				*parameter name
				*subj id
				*cue
				*cue 
				*stim 
				*etc...
		**Writes reformatted dataframe out to .csv in working dir

	::Transforming stats output for more convenient access to subj parameters by cue:
		**Flexible to take AllPriors or HNL coded data
		**Also able to accomodate different model configurations (i.e. bias_hyp=vz) 
				*set up for z, v, v+z, and vz
		**Columns for each cue (e.g. a90H_face, b70H_face, ...e90F_house)
		**Column indexing parameter name (e.g. 'v', 'a', 'st', etc..)
		**Column for subj_id

	::Creating hierarchical dictionary from condsdf:
		**{subj_x{cue_y{param:param_value}}}
		**can be sampled from to generate data 
		   for each subject/cue using 
		   sim_subs() function: 
		   		uses hddm.generate.gen_rand_data() 
				to create a full dataset containing 
				ntrials per sub, per cue.
	
	::Doing basic data aggregation for empirical and simulated data
	 	**including average over subs RT or accuracy for each cue
		**calculate SE for RT or acc. for each cue

			
Main Functions:
			
	1. parse_stats(model)
			*does all necessary formatting
			 in order to plot emp v. sim data

"""	
from __future__ import division
import hddm

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def stats_df(model, save=False):
	"""
	RETURNS: 1
		*model_stats (pandas DataFrame):	same as hddm.HDDM.gen_stats() with 
											column added for parameter names
											(usually call this "fulldf")
	"""
	if not hasattr(model, 'columns'):
		model_stats=model.gen_stats()
		model_stats['param']=model_stats.index
	else:
		model_stats=model

	slist=list()
	for i in model_stats['param']:
		x=i.split('.')
		if x[-1].isdigit():
			sint=int(x[-1])
			slist.append(sint)	
		else: slist.append("GRP")
	model_stats['sub']=slist
	
	if save:
		model_stats.to_csv('fulldf.csv', index=False)
	
	return model_stats

def parse_stats(minput, varlvl='grp', input_isdf=False, sc=None):
	"""
	Arguments:
		
		minput (HDDM model 						(1)	hddm model complete with MCMC
		OR pd.DataFrame):							traces, stats, etc...
												(2) pandas dataframe of hddm.gen_stats() 

	RETURNS 1: parsed_list=[subdf, condsdf, pdict]

		*subdf (pandas DataFrame): 	 			dataframe containing separate columns 
												for cue, stim, cue+stim, params, etc...
												(is written to a csv file in wd)		

		*condsdf (pandas DataFrame):			one column for each experimental cue
												columns for sub_id and params as well 
												(is used to make pdict (which is used for simulating)

		*pdict (dict):							dictionary created from condsdf, used 
												for simulating data for each sub/cue
												using hddm.generate.gen_rand_data()
	"""
	grp_dict=None

	if input_isdf:
		fulldf=minput
		
		slist=list()
		for i in fulldf['param']:
			x=i.split('.')
			if x[-1].isdigit():
				sint=int(x[-1])
				slist.append(sint)	
			else: slist.append("GRP")
		fulldf['sub']=slist
		
		fulldf.to_csv("fulldf.csv")
		
	else:
		fulldf=stats_df(model=minput)
		
	subdf=get_subdf(fulldf=fulldf)
	
	if hasattr(minput, 'split_param'):
		sc=minput.split_param
	else:
		sc=sc
		
	grpdf=get_grpdf(fulldf=fulldf)
	
	if varlvl=='grp':
		intervar=['sv', 'sz', 'st']
		grp_dict=dict()
		for i in fulldf['param']:
			if i in intervar:
				grp_dict[i]=fulldf.ix[(fulldf['param']==i), 'mean'].values[0]
			
	if len(subdf.noise.unique())>1:
		pdict=[]; nlist=['68', '69']; cdf_list=[]
		for n in nlist:
			
			subdf_n=subdf[subdf['noise'].isin([n, 'constant'])]
			subdf_n.index=range(len(subdf_n))
			condsdf_n=simform(subdf=subdf_n, sc=sc)
			condsdf_n.index=condsdf_n['param']

			pdict_n=create_pdict(condsdf_n, grp_dict)
			
			condsdf_n['noise']=[n]*len(condsdf_n)
			subdf_n.to_csv("subdf"+str(n)+".csv", index=False)
			condsdf_n.to_csv("condsdf"+str(n)+".csv", index=False)
			
			cdf_list.append(condsdf_n)
				
			pdict.append(pdict_n)
		
		condsdf=cdf_list[0].append(cdf_list[1])
		condsdf.to_csv("condsdf.csv", index=False)
				
	else:
		condsdf=simform(subdf=subdf, sc=sc)
		condsdf.index=condsdf['param']
		pdict=create_pdict(condsdf=condsdf, grp_dict=grp_dict)
		condsdf.to_csv("condsdf.csv")

	parsed_list=[subdf, condsdf, pdict]

	return parsed_list


def get_subdf(fulldf):
	"""
	Arguments:
		
		fulldf (pd.DataFrame):			pandas dataframe of hddm.gen_stats() output
										(fulldf as in full set of individual and group stats)
	
	RETURNS: 1
		
		*subdf (pandas DataFrame): 	 	dataframe containing separate columns 
										for cue, stim, cue+stim, params, etc...
										(is written to a csv file in wd)
	"""

	subdf=fulldf.ix[(fulldf['sub']!='GRP'), ['sub', 'param', 'mean']]
	subdf.index=range(len(subdf))
	
	#Make column for parameter
	plist=list()
	for i in subdf.param:
		p=i.split('_')[0]
		plist.append(p)
	subdf['parameter']=plist

	subdf=txtparse(subdf, 'sub')
	subdf.index=range(len(subdf))	
	
	return subdf

def get_grpdf(fulldf):
	"""
	Arguments:
		
		fulldf (pd.DataFrame):			pandas dataframe of hddm.gen_stats() output
										(fulldf as in full set of individual and group stats)
	
	RETURNS: 1
		
		grpdf (pandas DataFrame): 	 	dataframe containing separate columns 
										for cue, stim, cue+stim, params, etc...
										(is written to a csv file in wd)
	"""

	grpdf=fulldf.ix[(fulldf['sub']=='GRP'), ['param', 'mean']]
	#Make column for parameter
	plist=list()
	for i in grpdf.param:
		if '.' in i:
			p=i.split('(')[0]
		else: p=i
		plist.append(p)
	grpdf['parameter']=plist

	grpdf=txtparse(grpdf, 'group')
	grpdf.index=range(len(grpdf))
	
	return grpdf
	
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
	
def txtparse(dataframe, lvl):
	"""
	Parses stats into a dataframe for all subjects or at the group level, depnding on "lvl"
	
	""" 
	
	#make column for cue
	condlist=list()
	for i in dataframe.param:
		if '(' in i:
			cond_name=i.split('(')[1].split(')')[0]
		else: 
			cond_name='constant'

		condlist.append(cond_name)
	
	dataframe['cue']=condlist

	allnoise=['68', '69']
	allcues=['90H', '70H', 'neutral', '70F', '90F', 
				'50N','a90H', 'b70H', 'c50N', 'd70F', 
				'e90F', 'a80H', 'b50N', 'c80F']
	allimgs=['face', 'house', 'Face', 'House']

	cuelist=[]; noiselist=[]; stimlist=[]

	listd={'cue':[allcues, cuelist], 'noise':[allnoise, noiselist], 'stim':[allimgs, stimlist]}

	for i in condlist:
		i=str(i)
		for k in listd.keys():
			kval=[kval for kval in listd[k][0] if kval in i.split('.') or kval==i]
		
			if kval:
				listd[k][1].append(kval[0])
			else:
				listd[k][1].append('constant')
	
	dataframe['stim']=listd['stim'][1]
	dataframe['cue']=listd['cue'][1]
	dataframe['noise']=listd['noise'][1]
	
	dataframe=change_cue(data=dataframe)
	
	if lvl=='sub':
		dataframe.to_csv("subdf.csv", index=False)
	else:
		dataframe.to_csv("grpdf.csv", index=False)
	
	return dataframe


def change_cue(data):

	if len(data['cue'].unique())>=5:
	
		if '50N' in data['cue'].unique() or '50N.face' in data['cue'].unique():
			data.cue.replace('50N', 'c50N', inplace=True)
			data.cue.replace('90H', 'a90H', inplace=True)
			data.cue.replace('70H', 'b70H', inplace=True)
			data.cue.replace('90F', 'e90F', inplace=True)
			data.cue.replace('70F', 'd70F', inplace=True)
			
		if 'neutral' in data['cue'].unique() or 'neutral.face' in data['cue'].unique():
			data.cue.replace('neutral', 'c50N', inplace=True)
			data.cue.replace('90H', 'a90H', inplace=True)
			data.cue.replace('70H', 'b70H', inplace=True)
			data.cue.replace('90F', 'e90F', inplace=True)
			data.cue.replace('70F', 'd70F', inplace=True)

			
		elif '50/50' in data['cue'].unique() or '50/50.face' in data['cue'].unique():
			data.cue.replace('50/50', 'c50N', inplace=True)
			data.cue.replace('90H', 'a90H', inplace=True)
			data.cue.replace('70H', 'b70H', inplace=True)
			data.cue.replace('90F', 'e90F', inplace=True)
			data.cue.replace('70F', 'd70F', inplace=True)
			

	
	return data

def get_empirical_means(data, code_type):
	"""
	Gets empirical accuracy and rt means from dataframe 
	
	RETURNS: 4
		*face_emp_acc (np.array):	empirical accuracy means for  
									face responses across all cues
		*house_emp_acc (np.array):	empirical accuracy means for  
									house responses across all cues
		*face_emp_rts (np.array):	empirical response time means for  
									correct face responses across all cues
		*house_emp_rts (np.array):	empirical response time means for  
									correct house responses across cues
	"""
	
	data['rt']=abs(data['rt'])
	
	data=change_cue(data)
	
	if 'acc' not in data.columns:
		#add accuracy column to simdf
		data['acc']=data['response'].values
		data.ix[(data['stim']=='house') & (data['acc']==0), 'acc']=2
		data.ix[(data['stim']=='house') & (data['acc']==1), 'acc']=0
		data.ix[(data['stim']=='house') & (data['acc']==2), 'acc']=1

	
	accdf=data[['subj_idx', 'cue', 'stim', 'acc']]
	acc_pivot=pd.pivot_table(accdf, rows='subj_idx', cols=['stim', 'cue'], values='acc', aggfunc=np.average)
	
	allcor=data[data['acc'].isin([1])]
	cor_pivot=pd.pivot_table(allcor, values='rt', cols=['stim', 'cue'], rows=['subj_idx'], aggfunc=np.average)
	
	
	for i in acc_pivot.mean(0):
		if code_type=='HNL':
			face_emp_acc=np.array(acc_pivot.mean(0)[:3].values)
			house_emp_acc=np.array(acc_pivot.mean(0)[3:].values)
			face_emp_rts=np.array(cor_pivot.mean(0)[:3].values)
			house_emp_rts=np.array(cor_pivot.mean(0)[3:].values)
		else:
			face_emp_acc=np.array(acc_pivot.mean(0)[:5].values)
			house_emp_acc=np.array(acc_pivot.mean(0)[5:].values)
			face_emp_rts=np.array(cor_pivot.mean(0)[:5].values)
			house_emp_rts=np.array(cor_pivot.mean(0)[5:].values)
	
	return face_emp_acc, house_emp_acc, face_emp_rts, house_emp_rts
	
def get_emp_error_rt(data):

	data['rt']=abs(data['rt'])
	data=change_cue(data)
	
	allerr=data[data['acc'].isin([0])]
	err_pivot=pd.pivot_table(allerr, values='rt', cols=['stim', 'cue'], rows=['subj_idx'], aggfunc=np.average)
		
	for i in err_pivot.mean(0):
		
		face_err=np.array(err_pivot.mean(0)[:len(data.cue.unique())].values)
		house_err=np.array(err_pivot.mean(0)[len(data.cue.unique()):].values)
		
	return face_err, house_err

def get_theo_error_rt(simdf):
	
	allerr=simdf[simdf['acc'].isin([0])]
	err_pivot=pd.pivot_table(allerr, values='rt', cols=['stim', 'cue'], rows=['subj_idx'], aggfunc=np.average)
	
	for i in err_pivot.mean(0):

		fsim_err=np.array(err_pivot.mean(0)[:len(simdf.cue.unique())].values)
		hsim_err=np.array(err_pivot.mean(0)[len(simdf.cue.unique()):].values)
	
	return fsim_err, hsim_err
	
	
def get_theo_rt(simdf, code_type):
	"""
	Calculates and returns the average RT for each
	simulated cue (averaged over simulated subject means)
		
	RETURNS: 2
		
		*face_theo_rts (numpy array):	array of predicted rt means for 
		 								correct face responses across all
										prob. cues
		
		*house_theo_rts (numpy array):	array of predicted rt means for 
	 									correct house responses across all
										prob. cues

	
	"""
	#GET THEORETICAL RT MEANS
	from scipy.stats import stats
	allcor=simdf[simdf['acc'].isin([1])]
	cor_pivot=pd.pivot_table(allcor, values='rt', cols=['stim', 'cue'], rows=['subj_idx'], aggfunc=np.average)

	for i in cor_pivot.mean(0):
		if code_type=='HNL':
			face_theo_rts=np.array(cor_pivot.mean(0)[:3].values)
			house_theo_rts=np.array(cor_pivot.mean(0)[3:].values)
		else:
			face_theo_rts=np.array(cor_pivot.mean(0)[:5].values)
			house_theo_rts=np.array(cor_pivot.mean(0)[5:].values)


	return face_theo_rts, house_theo_rts

def get_theo_acc(simdf, code_type):
	"""
	Calculates and returns the average accuracy for each
	simulated condition (averaged over simulated subject means)

	RETURNS: 2
		
		*face_theo_acc (numpy array):	array of predicted accuracy 
										means for face responses across 
										all prob. cues
		
		*house_theo_acc (numpy array): 	array of predicted accuracy 
										means for house responses across 
										all prob. cues

	"""
	from scipy.stats import stats
	
	accdf=simdf[['subj_idx', 'cue', 'stim', 'acc']]
	acc_pivot=pd.pivot_table(accdf, rows='subj_idx', cols=['stim', 'cue'], values='acc', aggfunc=np.average)

	for i in acc_pivot.mean(0):
		if code_type=='HNL':
			face_theo_acc=np.array(acc_pivot.mean(0)[:3].values)
			house_theo_acc=np.array(acc_pivot.mean(0)[3:].values)
		else:
			face_theo_acc=np.array(acc_pivot.mean(0)[:5].values)
			house_theo_acc=np.array(acc_pivot.mean(0)[5:].values)

	return face_theo_acc, house_theo_acc

def get_emp_SE(data, code_type):	

	from scipy.stats import stats
	
	
	allcor=data[data['acc'].isin([1])]
	cor_pivot=pd.pivot_table(allcor, values='rt', cols=['stim', 'cue'], rows=['subj_idx'], aggfunc=np.average)
	acc_pivot=pd.pivot_table(data, values='acc', cols=['stim', 'cue'], rows=['subj_idx'], aggfunc=np.average)
	#Get theoretical RT S.E.M's
	sem_rt=[]
	for img, cue in cor_pivot.columns:
		x=stats.sem(cor_pivot[img][cue])
		sem_rt.append(x)

	#Get theoretical ACCURACY S.E.M's
	sem_acc=[]
	for img, cue in acc_pivot.columns:
		x=stats.sem(acc_pivot[img][cue])
		sem_acc.append(x)

	if code_type=='HNL':
		face_emp_acc_SE=sem_acc[:3]
		house_emp_acc_SE=sem_acc[3:]
		face_emp_rts_SE=sem_rt[:3]
		house_emp_rts_SE=sem_rt[3:]
	else:
		face_emp_acc_SE=sem_acc[:5]
		house_emp_acc_SE=sem_acc[5:]
		face_emp_rts_SE=sem_rt[:5]
		house_emp_rts_SE=sem_rt[5:]
	
	sem_list=[face_emp_acc_SE, house_emp_acc_SE, face_emp_rts_SE, house_emp_rts_SE]

	return sem_list

	
if __name__ == "__main__":
	main()	
	
	
	
	
	
	