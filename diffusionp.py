#!/usr/bin/env python

"""
Takes conds_df file and makes a pandas dataframe
for each parameter
		rows: subs
		cols: cues

TODO: easy access to sdt subj parameters 
 
	1. 	put sdt .csv files for dp_bayes, c_bayes, dp_analytic, 
		and c_analytic in one file for AllPrior models 
		("~/HDDM/SDTmodels/allp_estimates/") and another
		for all HNL models ("~/HDDM/SDTmodels/hnl_estimates/")
	
	2.	read in with pandas using a global path
		
			example: cbayes=pd.read_csv("~/HDDM/SDTmodels/allp_estimates/c_bayes.csv")
	
	3. 	Then probably should just concat the two df's want to correlate (axis=0) 
		and run df.corr() then write to .csv... only need to do scipy.stats.stats.pearsonr() 
		if the correlations look good enough
"""

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def get_p(condsdf, par=str(), noise=False):
	#z estimates same for face and
	#house conds so only get from face columns
	
	if 'noise' in condsdf.columns and len(condsdf['noise'].unique())>1:
		condsdf=condsdf[condsdf['noise']!=69]
	
	subp=condsdf.ix[(condsdf['param']==par)]
	for col in subp.columns:
		if 'face' not in col:
			del subp[col]

	if len(condsdf.columns)>=10:
		subp.columns=['a90H', 'b70H', 'c50N', 'd70F', 'e90F']
	else:
		subp.columns=['a80H','b50N', 'c80F']

	subp.index=np.arange(len(condsdf['sub'].unique()))

	return subp

def get_dimg(condsdf, par=str(), noise=False):	
	
	if 'noise' in condsdf.columns and len(condsdf['noise'].unique())>1:
		condsdf=condsdf[condsdf['noise']!=69]
	
	
	#create two df's for subj. estimates of 
	#param(face) and param(house) then subtract 
	#each column to create df of difference b/w 
	#param values across stim type for that 
	#parameter (i.e. vface-vhouse)	
	subpf=condsdf.ix[(condsdf['param']==par)]
	for col in subpf.columns:
		if 'face' not in col:
			del subpf[col]
	
	subph=condsdf.ix[(condsdf['param']==par)]
	for col in subph.columns:
		if 'house' not in col:
			del subph[col]

	#give vf and vh common col id's 
	#and put subvf-subvh in subdv
	if len(condsdf.columns)>=10:
		subpf.columns=['a90H', 'b70H', 'c50N', 'd70F', 'e90F']
		subph.columns=['a90H', 'b70H', 'c50N', 'd70F', 'e90F']
	else:
		subpf.columns=['a80H','b50N', 'c80F']
		subph.columns=['a80H','b50N', 'c80F']
	
	diff_img=subpf-subph
	diff_img.index=np.arange(len(condsdf['sub'].unique()))
	
	return diff_img


def get_dnoise(condsdf, par=str(), noise=False):	

	#create two df's for subject estimates of 
	#param(68) and param(69) then subtract 
	#each column to create df of difference b/w 
	#param values across stim type for that 
	#parameter (i.e. v68-v69)
	
	subp68=condsdf.ix[(condsdf['param']==par)&(condsdf['noise']==68)]
	subp69=condsdf.ix[(condsdf['param']==par)&(condsdf['noise']==69)]
	
	nlist=[subp68, subp69]
	
	for ndf in nlist:
		for col in ndf.columns:
			if 'face' not in col:
				del ndf[col]
		ndf.index=np.arange(len(ndf))

	if len(condsdf.columns)>=10:
		subp68.columns=['a90H', 'b70H', 'c50N', 'd70F', 'e90F']
		subp69.columns=['a90H', 'b70H', 'c50N', 'd70F', 'e90F']
	else:
		subp68.columns=['a80H','b50N', 'c80F']
		subp69.columns=['a80H','b50N', 'c80F']
	
	
	diff_noise=subp69-subp68

	return diff_noise



def get_diffusion(condsdf, depends_on=dict(), save=False):

	"""
	Args:

		condsdf (pandas dataframe):		column id's are cue names 
										(also sub, param columns but these 
										are removed by a function in this script)
										rows are subj parameter estimates

		depends_on (dict):				dictionary of all diffusion parameters used
										in model which was used to produce condsdf.

										example:
											depends_on={'a':'constant', 'v':'stim', 't':'cue', 'z':'cue'}
	::Returns::

		diffusion_params (dict)			dictionary of pandas dataframes, each containing
										diffusion model parameter values across cues (column names) 
										for each parameter included in depends_on.

	"""



	diffusion_params=dict()
	for param in depends_on.keys():
		if 'stim' in depends_on[param]:
			par=get_dimg(condsdf, par=param)
			tag='dimg_'+param
			diffusion_params[tag]=par
		if 'noise' in depends_on[param]:
			par=get_dnoise(condsdf, par=param)
			tag='dnoise_'+param
			diffusion_params[tag]=par
		if 'cue' in depends_on[param]:
			par=get_p(condsdf, par=param)
			tag='cue_'+param
			diffusion_params[tag]=par

	return diffusion_params

			
if __name__ == "__main__":
	main()	


		
		
		
		
		
		
		
		
		