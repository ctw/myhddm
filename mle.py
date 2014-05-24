#!/usr/bin/env python

from __future__ import division
import hddm
from myhddm import defmod, vis, sdt, opt
import pandas as pd
import numpy as np

def aic(model):
    k = len(model.get_stochastics())
    logp = sum([x.logp for x in model.get_observeds()['node']])
    return 2 * k - logp

def bic(model):
    k = len(model.get_stochastics())
    n = len(model.data)
    logp = sum([x.logp for x in model.get_observeds()['node']])
    return -2 * logp + k * np.log(n)

def dic(model):
	return model.dic

def optimize_sx(mname, project='imaging'):

	m=defmod.define_model(mname, project=project)
	data=m.data
	if 'z' in m.depends_on.keys():
		bias=True
	else:
		bias=False

	grp_dict={}; subj_params=[]; aic_list=[]; bic_list=[]; dic_list=[]; ic_dict={}

	for subj_idx, subj_data in data.groupby('subj_idx'):

		m_subj=hddm.HDDM(subj_data, depends_on=m.depends_on, bias=bias, include=m.include)

		sx_params=m_subj.optimize('ML')

		pdict=opt.get_pdict(sx_params)
		subj_params.append(sx_params)
		aic_list.append(aic(m_subj)); bic_list.append(bic(m_subj)); #dic_list.append(m_subj.dic)

		grp_dict[subj_idx]=pdict

	ic_dict={'aic':aic_list, 'bic':bic_list}
	ic_df=pd.DataFrame(ic_dict)
	ic_df.to_csv(mname+"_IC_Rank.csv")
	#write grp_dict to .txt file for reloading later
	f=open('mle_params.txt', 'w')
	f.write('grp_dict=' + repr(grp_dict) + '\n')
	f.close()

	params = pd.DataFrame(subj_params)
	simdf=vis.predict(grp_dict, data, ntrials=100, nsims=100, save=True, RTname="dbmz_RT.jpeg", ACCname="dbmz_ACC.jpeg")
	#simdf=vis.predict(grp_dict, df, ntrials=160, nsims=100, save=True, RTname="SimRT_EvT.jpeg", ACCname="SimACC_EvT.jpeg")
	simdf.to_csv("simdf_opt.csv")
	params.to_csv("subj_params_opt.csv", index=False)
	sdt.plot_rho_sdt(data, simdf)
	empvsim=sdt.rho_sdt(data, simdf)

	return grp_dict, ic_df
