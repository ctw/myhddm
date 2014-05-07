#!/usr/bin/env python

from __future__ import division
import hddm
import pandas as pd
import numpy as np

def sim_exp(pdict, ntrials=500, p_outlier=None, pfast=0, pslow=0, nsims_per_sub=1):
	"""
	Simulates a dataset for each subject(i), for every condition(j) in pdict of 
	size n=ntrials*exp_proportion.  <-- Makes simulated trial count for each condition
	proportional to the number of empirical observations (unbalanced face/house obs across cues).
	Also, does light parsing and reformatting on output dataframe

	RETURNS: 2

		*sim_df (pandas DataFrame): columns for condition name, sub id, cue, stim, 
			 						response (1: upperbound, 0: lowerbound), 
									accuracy (1: cor, 0: err), and response time

		*param_dict (dict):			dataframe of parameters used 
									(only needed if noise/outliers 
									simulated)

	"""
	param_dict=dict()
	for i, sub in enumerate(pdict):
		for num, cond in enumerate(pdict[sub].keys()):
			#check if cue predicts image (i.e. 90F --> face)
			if cond.split('_')[0][-1].lower()==cond.split('_')[1][0].lower():
				#make total trials for this simulated condition
				#proportional to number of experimental
				#trials in this condition
				perc=int(cond[1:3])*.01
			else:
				perc=1-(int(cond[1:3])*.01)
			exptrials=perc*ntrials
		
			if 'p' in pdict[sub][cond].keys():
				pfast=pdict[sub][cond]['p']
				pslow=pdict[sub][cond]['p']
		
			nfast=int((exptrials/2)*pfast)
			nslow=int((exptrials/2)*pslow)

			data, parameters = hddm.generate.gen_rand_data(params={cond:pdict[sub][cond]}, subjs=nsims_per_sub, 
				n_fast_outliers=nfast, n_slow_outliers=nslow, size=exptrials)
			data.subj_idx[:]=sub

			if i==0 and num==0:
				simdf=data
			else:
				simdf=pd.concat([simdf, data], ignore_index=True)
			param_dict[i]=parameters	
	
	simdf=ref_simdf(simdf)

	return simdf, param_dict

def sim_and_concat(params, nsims=25, ntrials=100):
	
	simdf_list=[]
	
	for i in range(nsims):	
		simdf, params_used=sim_exp(pdict=params, ntrials=ntrials)
		simdf['sim_num']=simdf['subj_idx'].copy()
		simdf.sim_num[:]=i
		simdf_list.append(simdf)
	
	all_simdfs=pd.concat(simdf_list)
	
	return all_simdfs

def sim_exp_subj(pdict, ntrials=500, p_outlier=None, pfast=0, pslow=0, nsims_per_sub=1):
	"""
	Simulates a dataset for a single subject, for every condition in pdict of 
	size n=ntrials*exp_proportion.  <-- Makes simulated trial count for each condition
	proportional to the number of empirical observations (unbalanced face/house obs across cues).
	
	Also, does light parsing and reformatting on output dataframe.  

	RETURNS: 2

		*sim_df (pandas DataFrame): columns for condition name, sub id, cue, stim, 
			 						response (1: upperbound, 0: lowerbound), 
									accuracy (1: cor, 0: err), and response time

		*param_dict (dict):			dataframe of parameters used 
									(only needed if noise/outliers 
									simulated)

	"""
	param_dict=dict()
	for i, cond in enumerate(pdict.keys()):

		#check if cue predicts image (i.e. 90F --> face)
		if cond.split('_')[0][-1].lower()==cond.split('_')[1][0].lower():
			#make total trials for this simulated condition
			#proportional to number of experimental
			#trials in this condition
			perc=int(cond[1:3])*.01
		else:
			perc=1-(int(cond[1:3])*.01)
		exptrials=perc*ntrials

		if 'p' in pdict[cond].keys():
			pfast=pdict[cond]['p']
			pslow=pdict[cond]['p']

		nfast=int((exptrials/2)*pfast)
		nslow=int((exptrials/2)*pslow)

		data, parameters = hddm.generate.gen_rand_data(params={cond:pdict[cond]}, subjs=nsims_per_sub, 
			n_fast_outliers=nfast, n_slow_outliers=nslow, size=exptrials)


		if i==0:
			simdf=data
		else:
			simdf=pd.concat([simdf, data], ignore_index=True)
		param_dict[i]=parameters	

	simdf=ref_simdf(simdf)

	return simdf, param_dict



def sim_subs(pdict, ntrials=500, p_outlier=None, pfast=0, pslow=0, nsims_per_sub=1):
	
	param_dict=dict()
	nfast=int((ntrials)*pfast)
	nslow=int((ntrials)*pslow)
	for i, x in enumerate(pdict):

		data, parameters = hddm.generate.gen_rand_data(params=pdict[x], subjs=nsims_per_sub, 
						n_fast_outliers=nfast, n_slow_outliers=nslow, size=ntrials)
		data.subj_idx[:]=x

		if i==0:
			simdf=data
		else:
			simdf=pd.concat([simdf, data], ignore_index=True)
		param_dict[i]=parameters	

	simdf=ref_simdf(simdf)
	
	return simdf, param_dict

def sim_noise_sep(pdict, ntrials=100, nsims=10, simfx=sim_exp, pfast=0, pslow=0, nsims_per_sub=1):
	for i in range(nsims):
		p68=pdict[0]
		p69=pdict[1]
		simdf68, params_used=simfx(pdict=p68, ntrials=ntrials, pfast=pfast, pslow=pslow, nsims_per_sub=nsims_per_sub)
		simdf69, params_used=simfx(pdict=p69, ntrials=ntrials, pfast=pfast, pslow=pslow, nsims_per_sub=nsims_per_sub)
		
		simdf68['noise']=['68']*len(simdf68)
		simdf69['noise']=['69']*len(simdf69)
		
		if i==0:
			simdf=pd.concat([simdf68, simdf69], ignore_index=True)
		else:
			simdf=pd.concat([simdf, simdf68], ignore_index=True)
			simdf=pd.concat([simdf, simdf69], ignore_index=True)
	
	return simdf


def sim_grp(pdict, ntrials=5000, pfast=0.00, pslow=0.00, nsims_per_sub=25, subj_noise=0.1):

	param_dict=dict()
	nfast=int((ntrials)*pfast)
	nslow=int((ntrials)*pslow)

	simdf, parameters = hddm.generate.gen_rand_data(params=pdict, subjs=nsims_per_sub, 
					n_fast_outliers=nfast, n_slow_outliers=nslow, size=ntrials, subj_noise=subj_noise)

	simdf=ref_simdf(simdf)

	return simdf, parameters




def sim_fastF_slowH(pdict, ntrials=500, p_outlier=None, pfast=0, pslow=0, nsims_per_sub=1):
	"""
	Simulates a dataset for each subject(i), for every condition(j) in pdict of size n=ntrials
	Also, does light parsing and reformatting on output dataframe

	RETURNS: 2

		*sim_df (pandas DataFrame): columns for condition name, sub id, cue, stim, 
			 						response (1: upperbound, 0: lowerbound), 
									accuracy (1: cor, 0: err), and response time

		*param_dict (dict):			dataframe of parameters used 
									(only needed if noise/outliers 
									simulated)

	"""

	param_dict=dict()
	for i, sub in enumerate(pdict):
		for num, cond in enumerate(pdict[sub].keys()):
			if '90F_face' in cond:
				exptrials=int(.9*ntrials)
				nfast=int((pfast*0.3)*exptrials)
				nslow=int(0.01*exptrials)
			elif '70F_face' in cond:
				exptrials=int(.7*ntrials)
				nfast=int((pfast*0.5)*exptrials)
				nslow=int(0.02*exptrials)
			elif '50N_face' in cond:
				exptrials=int(.5*ntrials)
				nfast=int(pfast*exptrials)
				nslow=0
			elif '70H_face' in cond:
				exptrials=int(.3*ntrials)
				nfast=int(pfast*exptrials)
				nslow=0
			elif '90H_face' in cond:
				exptrials=int(.1*ntrials)
				nfast=int(pfast*exptrials)
				nslow=0
			elif '90H_house' in cond:
				exptrials=int(.9*ntrials)
				nfast=0
				nslow=int(pslow*exptrials)
			elif '70H_house' in cond:
				exptrials=int(.7*ntrials)
				nfast=0
				nslow=int(pslow*exptrials)
			elif '50N_house' in cond:
				exptrials=int(.5*ntrials)
				nfast=0
				nslow=int(pslow*exptrials)
			elif '70F_house' in cond:
				exptrials=int(.3*ntrials)
				nfast=0
				nslow=int(pslow*exptrials)
			elif '90F_house' in cond:
				exptrials=int(.1*ntrials)
				nfast=0
				nslow=int(pslow*exptrials)
			elif '80H_face' in cond:
				exptrials=int(.2*ntrials)
				nfast=int(pfast*exptrials)
				nslow=0
			elif '80H_house' in cond:
				exptrials=int(.8*ntrials)
				nfast=0
				nslow=int(pslow*exptrials)
			elif '50N_face' in cond:
				exptrials=int(.5*ntrials)	
				nfast=int(pfast*exptrials)
				nslow=0			
			elif '50N_house' in cond:
				exptrials=int(.5*ntrials)
				nfast=0
				nslow=int(pslow*exptrials)
			elif '80F_house' in cond:
				exptrials=int(.2*ntrials)
				nfast=0
				nslow=int(pslow*exptrials)
			elif '80F_face' in cond:
				exptrials=int(.8*ntrials)
				nfast=int(pfast*exptrials)
				nslow=0
			data, parameters = hddm.generate.gen_rand_data(params={cond:pdict[sub][cond]}, subjs=nsims_per_sub, 
				n_fast_outliers=nfast, n_slow_outliers=nslow, size=exptrials)
			data.subj_idx[:]=sub

			if i==0 and num==0:
				simdf=data
			else:
				simdf=pd.concat([simdf, data], ignore_index=True)
			param_dict[i]=parameters	

	simdf=ref_simdf(simdf)

	return simdf, param_dict



def ref_simdf(simdf):
	#add separate cols for 
	#stim and cue names
	sim_cue=list()
	sim_img=list()
	for cond in simdf['condition']:
		if '_' in cond:
			img=cond.split('_')[1]
			cue=cond.split('_')[0]
		sim_cue.append(cue)
		sim_img.append(img)
	simdf['stim']=sim_img
	simdf['cue']=sim_cue

	#add accuracy column to simdf
	simdf['acc']=simdf['response'].values
	simdf.ix[(simdf['stim']=='house') & (simdf['response']==0), 'acc']=1
	simdf.ix[(simdf['stim']=='house') & (simdf['response']==1), 'acc']=0


	return simdf



if __name__ == "__main__":
	main()	


