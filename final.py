#!/usr/bin/env python

from __future__ import division
import hddm
import kabuki
import pandas as pd
import os
from mydata.munge import find_path
from myhddm import defmod, parse, opt, vis, sims

def analyze_models(nsims=100, ntrials=100):
	
	mnames=['msm', 'dbm', 'dbmz', 'pbm']
	#mnames=['pbm']
	bias=True
	data=pd.read_csv("/Users/kyle/Desktop/beh_hddm/allsx_feat.csv")
	
	for m in mnames:
		
		if m=='dbm':
			bias=False
		
		model=defmod.define_model(m, project='behav')
		
		mpath="/Users/kyle/Desktop/beh_hddm/revised_models/"+m
		os.chdir(mpath)
		
		m0=model; m1=model; m2=model
		mlist=[m0.load_db(m+"_traces0.db", db='pickle'), m1.load_db(m+"_traces1.db", db='pickle'), m2.load_db(m+"_traces2.db", db='pickle')]
		allmodels=kabuki.utils.concat_models(mlist)
		allmodels.print_stats(m+"_stats_all.txt")
		
		vis.plot_neutral_traces(allmodels)
		for node in ['z', 'vf', 'vh']:
			vis.plot_posterior_nodes(allmodels, node)
			
		gparams={}; subj_params=[]
		
		msingle=defmod.define_single(m, project='behav')
		
		for subj_idx, subj_data in data.groupby('subj_idx'):
			m_subj=hddm.HDDM(subj_data, depends_on=msingle.depends_on, bias=bias, include=msingle.include)
			sx_params=m_subj.optimize('ML')
			pdict=opt.get_pdict(sx_params)
			subj_params.append(sx_params)
			gparams[subj_idx]=pdict
		
		#write gparams to .txt file for reloading later
		f=open(m+'mle_gparams.txt', 'w')
		f.write('gparams=' + repr(gparams) + '\n')
		f.close()
		
		simdf_list=[]
		for i in range(nsims):
			simdf, params_used=sims.sim_exp(pdict=gparams, ntrials=ntrials, pfast=0.0, pslow=0.0, nsims_per_sub=1)
			simdf['sim_n']=[i]*len(simdf.index)
			simdf_list.append(simdf)

		simdf=pd.concat(simdf_list)

		params = pd.DataFrame(subj_params)
		simdf.to_csv(m+"_simdf.csv")
		params.to_csv(m+"_sparams.csv", index=False)