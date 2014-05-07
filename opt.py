#!/usr/bin/env python
"""
Includes functions for parsing parameter output from HDDM.optimize() routines

"""


from __future__ import division
import hddm

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def get_pdict(params):

	flatdf=get_flatdf(params)
	
	if len(flatdf.noise.unique())>1:
		pdict=[]
		flat68=(flatdf[flatdf['noise'].isin(['68', 'constant'])])
		flat69=(flatdf[flatdf['noise'].isin(['69', 'constant'])])
		c68=flat_simform(flat68)
		c69=flat_simform(flat69)
		pdict.append(flat_pdict(c68))
		pdict.append(flat_pdict(c69))		

	else:
		condsdf=flat_simform(flatdf)
		pdict=flat_pdict(condsdf)
	
	return pdict
	
def get_flatdf(group_pdict, save=False):
	"""
	Parses stats into a dataframe for chi-square estimated group params"

	""" 

	dataframe=pd.DataFrame(columns=['param', 'mean'], index=group_pdict.keys())
	dataframe.param=group_pdict.keys()
	dataframe.mean=group_pdict.values()	


	condlist=list()
	for i in dataframe.param:
		if '(' in i:
			cond_name=i.split('(')[1].split(')')[0]
		else: 
			cond_name='constant'

		condlist.append(cond_name)

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

	plist=list()
	for i in group_pdict.keys():
		p=i.split('(')[0]
		plist.append(p)
	dataframe['param']=plist
	
	if save:
		dataframe.to_csv("flatdf.csv", index=False)

	return dataframe



def flat_simform(flatdf):
	"""
	RETURNS: 1

		*flatdf (pandas DataFrame):		group-level df with one column for each experimental cue/stim combo
										(is used to make pdict (which is used for simulating)							

	"""

	nrows=len(flatdf.param.unique())

	if len(flatdf['cue'].unique())<5:
		condsdf=pd.DataFrame(np.zeros(nrows*6).reshape((nrows, 6)), columns=['a80H_face', 'b50N_face', 'c80F_face', 
			'a80H_house', 'b50N_house', 'c80F_house'])
	else:
		condsdf=pd.DataFrame(np.zeros(nrows*10).reshape((nrows, 10)), columns=['a90H_face', 'b70H_face', 'c50N_face', 
			'd70F_face', 'e90F_face', 'a90H_house', 'b70H_house', 'c50N_house', 'd70F_house', 'e90F_house'])


	for cond in condsdf.columns:
		cue_n=cond.split('_')[0]
		img_n=cond.split('_')[1]
		cdf=flatdf.ix[flatdf['stim'].isin([img_n, 'constant']) & flatdf['cue'].isin([cue_n, 'constant']), ['param', 'mean']]

		cdf=cdf.sort('param')
		cdf.index=range(len(cdf))
	
		condsdf['param']=cdf['param'].values
		condsdf[cond]=cdf['mean'].values
		
	return condsdf


def flat_pdict(df, addz=False):

	df.index=df.param
	if 'z' not in df.index:
		addz=True
	
	pdict=dict()
	for cond in df:
		if cond == 'param':
			continue
		
		pdict[cond]=dict(df[cond])
		if addz:
			pdict[cond]['z']=0.5
		
	return pdict


if __name__ == "__main__":
	main()	

