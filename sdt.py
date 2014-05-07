"""
add up Hit, Miss, FA, and CR counts 
for each subject in a simulated
dataset

First run proc.parse_stats(model) to get subdf, condsdf, and pdict
then run sim_subs(pdict) to get dataframe, param_dict

calc SDT counts from dataframe
"""


from __future__ import division
from scipy.stats import norm
from myhddm import diffusionp, defmod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def counts(dataframe, target='H', num=None):
	"""
	returns a dictionary of dataframes, one for each cue

	cols are sdt measures: 	[CR, FA, H, M]
	rows are subj. counts:  [sub(1)...sub(n)]
	
	"""
	if 'subj_idx' in dataframe.columns:
		s='subj_idx'
	else:
		s='sub'
	
	img='stim'
	c='cue'

	sdt_dict={}
	counts_list=[]
	for cue in dataframe[c].unique():
		counts={}
		if target=='H':
			counts['H']=[len(cols.ix[(cols[img]=='house')&(cols[c]==cue)&(cols['acc']==1)]) for sub, cols in dataframe.groupby(s)]
			counts['FA']=[len(cols.ix[(cols[img]=='face')&(cols[c]==cue)&(cols['acc']==0)]) for sub, cols in dataframe.groupby(s)]
			counts['M']=[len(cols.ix[(cols[img]=='house')&(cols[c]==cue)&(cols['acc']==0)]) for sub, cols in dataframe.groupby(s)]
			counts['CR']=[len(cols.ix[(cols[img]=='face')&(cols[c]==cue)&(cols['acc']==1)]) for sub, cols in dataframe.groupby(s)]			
		elif target=='F':
			counts['H']=[len(cols.ix[(cols[img]=='face')&(cols[c]==cue)&(cols['acc']==1)]) for sub, cols in dataframe.groupby(s)]
			counts['FA']=[len(cols.ix[(cols[img]=='house')&(cols[c]==cue)&(cols['acc']==0)]) for sub, cols in dataframe.groupby(s)]
			counts['M']=[len(cols.ix[(cols[img]=='face')&(cols[c]==cue)&(cols['acc']==0)]) for sub, cols in dataframe.groupby(s)]
			counts['CR']=[len(cols.ix[(cols[img]=='house')&(cols[c]==cue)&(cols['acc']==1)]) for sub, cols in dataframe.groupby(s)]
			
		countsdf=pd.DataFrame(counts)
		#countsdf.to_csv(cue+"_counts.csv", index=False)
		sdt_dict[cue]=countsdf	
	
	return sdt_dict

def calc_sdt(sdtdf):

	sdtdf['HR']=sdtdf['H']/(sdtdf['H']+sdtdf['M'])
	sdtdf['HR'].replace(1.0, .999, inplace=True)
	sdtdf['HR'].replace(0.0, .001, inplace=True)	
	
	sdtdf['zH']=norm.ppf(sdtdf['HR'])

	sdtdf['FAR']=sdtdf['FA']/(sdtdf['FA']+sdtdf['CR'])
	sdtdf['FAR'].replace(1.0, .999, inplace=True)
	sdtdf['FAR'].replace(0.0, .001, inplace=True)
	
	sdtdf['zFA']=norm.ppf(sdtdf['FAR'])

	sdtdf['dp']=sdtdf['zH']-sdtdf['zFA']
	sdtdf['c']=-0.5*(sdtdf['zH']+sdtdf['zFA'])

	return sdtdf

def get_hit_fa(sdt_dict):
	
	for df in sdt_dict:
		df=calc_sdt(sdt_dict[df])

	nsubs=len(sdt_dict[sdt_dict.keys()[0]])

	if len(sdt_dict)==5:
		cols=['a90H', 'b70H', 'c50N', 'd70F', 'e90F']
	else:
		cols=['a80H', 'b50N', 'c80F']

	hr_df=pd.DataFrame(np.zeros(nsubs*len(sdt_dict)).reshape(nsubs, len(sdt_dict)), columns=cols)
	fa_df=pd.DataFrame(np.zeros(nsubs*len(sdt_dict)).reshape(nsubs, len(sdt_dict)), columns=cols)

	for k in sdt_dict.keys():
		hr_df[k]=sdt_dict[k]['HR']
		fa_df[k]=sdt_dict[k]['FAR']
		
	return hr_df, fa_df


def get_hr_fa_info(dataframe, target='H', save=False):
	
	sdt_dict=counts(dataframe, target=target)
	hr_df, fa_df=get_hit_fa(sdt_dict)
	
	if target=='H':
		tag="house"
	elif target=='F':
		tag="face"
		
	hr_mean=hr_df.mean() 
	fa_mean=fa_df.mean()
	hr_std=hr_df.std() 
	fa_std=fa_df.std()
	hr_stderr=hr_std/np.sqrt(len(hr_df))
	fa_stderr=fa_std/np.sqrt(len(fa_df))
	
	hr_info=pd.DataFrame({'HR_Mean':hr_mean, 'HR_SE':hr_stderr})
	fa_info=pd.DataFrame({'FA_Mean':fa_mean, 'FA_SE':fa_stderr})
	
	if save:
		hr_df.to_csv(tag+"_target_hr_all.csv", index=False)
		fa_df.to_csv(tag+"_target_fa_all.csv", index=False)
		hr_info.to_csv(tag+"_target_hr_agg.csv", index=False)
		fa_info.to_csv(tag+"_target_fa_agg.csv", index=False)
	
	return hr_info, fa_info
	
	
def plot_rates(dataframe, target='H'):
	
	sdt_dict=counts(dataframe, target=target)
	hr_df, fa_df=get_hit_fa(sdt_dict)
	
	hr_mean=hr_df.mean(); fa_mean=fa_df.mean()
	hr_std=hr_df.std(); fa_std=fa_df.std()
	
	if target=='H':
		xlist=np.array(['50/50', '70H', '90H'])
		hr_ylist=np.array([hr_mean[2], hr_mean[1], hr_mean[0]])
		fa_ylist=np.array([fa_mean[2], fa_mean[1], fa_mean[0]])
		hr_err_ylist=np.array([hr_std[2], hr_std[1], hr_std[0]])
		fa_err_ylist=np.array([fa_std[2], fa_std[1], fa_std[0]])
		title_target='House'
	elif target=='F':
		xlist=np.array(['50/50', '70F', '90F'])
		hr_ylist=np.array([hr_mean[2], hr_mean[3], hr_mean[4]])
		fa_ylist=np.array([fa_mean[2], fa_mean[3], fa_mean[4]])
		hr_err_ylist=np.array([hr_std[2], hr_std[3], hr_std[4]])
		fa_err_ylist=np.array([fa_std[2], fa_std[3], fa_std[4]])
		title_target='Face'
	
	hr_err_ylist=hr_err_ylist/np.sqrt(len(hr_df))	
	fa_err_ylist=fa_err_ylist/np.sqrt(len(fa_df))	
	
	fig, axes=plt.subplots(2, figsize=(7, 10))
	fig.subplots_adjust(top=0.92, hspace=0.30, left=0.15, right=0.96, bottom=0.12)
	hr_ax=axes[0]
	fa_ax=axes[1]
	xnum=np.array([1, 2, 3])
	
	hr_ax.errorbar(xnum, hr_ylist, yerr=hr_err_ylist, lw=7.0, elinewidth=4.5, capsize=0, color='LimeGreen', ecolor='k')
	fa_ax.errorbar(xnum, fa_ylist, yerr=fa_err_ylist, lw=7.0, elinewidth=4.5, capsize=0, color='Red', ecolor='k')
	ax_list=[hr_ax, fa_ax]
	#hr_ax.set_title(title_target, fontsize=40) 
	hr_ax.title.set_y(1.03)
	hr_ax.set_ylim(0.7, 1.0)
	fa_ax.set_ylim(0.0, 0.4)
	hr_ax.set_yticks(np.array([0.7, 0.8, 0.9, 1.0]))
	fa_ax.set_yticks(np.arange(0, 0.5, .10))
	hr_ax.set_ylabel('Hit Rate', fontsize=30, labelpad=.5)
	fa_ax.set_ylabel('FA Rate', fontsize=30, labelpad=.5)
	fa_ax.set_xlabel('Prior Probability Cue', fontsize=30, labelpad=.5)
	
	for ax in ax_list:
		ax.set_xlim(0.5, 3.5)
		ax.set_xticks(np.array([1, 2, 3]))
		ax.set_xticklabels(xlist, fontsize=30)
		#ax.set_yticks(np.linspace(0.6, 1.10, .10)
		for tick in ax.yaxis.get_major_ticks():
		                tick.label.set_fontsize(22)
	
	
	plt.savefig(title_target+'_Hit_FA.jpeg', format='jpeg', dpi=900)
	
	
	#hr_ax.set_title(title_target, fontsize=40) 
	#hr_ax.title.set_y(1.03)
	#hr_ax.set_ylim(0.6, 1.0)
	#fa_ax.set_ylim(0.0, 0.4)
	##hr_ax.set_ylabel('Hit Rate', fontsize=28, labelpad=1)
	##fa_ax.set_ylabel('FA Rate', fontsize=28, labelpad=1)
	#fa_ax.set_xlabel('Prior Probability Cue', fontsize=35, labelpad=.45)
	#
	#for ax in ax_list:
	#	ax.set_xlim(0.5, 3.5)
	#	ax.set_xticks(np.array([1, 2, 3]))
	#	ax.set_xticklabels(xlist, fontsize=30)
	#	for tick in ax.yaxis.get_major_ticks():
	#	                tick.label.set_fontsize(22)
	#	
	#plt.savefig(title_target+'_Hit_FA.jpeg', format='jpeg', dpi=300)
	
def sim_rates_fill(rate_type='hit', ax=None, x=None, y=None, ind=0, last=np.zeros([3])):
	
	if rate_type=='hit':
		cline='#D6F5D6'
		cfill='#D6F5D6'
	else:
		cline='#FFB2B2'
		cfill='#FFB2B2'
	theo=ax.plot(x, y, '-', color=cline, lw=0.6, alpha=0.2)
	
	#'-', color='RoyalBlue', lw=0.6, alpha=0.2)
	#'-', color='FireBrick', lw=0.6, alpha=0.2)
	
	if ind!=0:
		y2=last
		fill=ax.fill_between(x, y, y-(y-y2), facecolor=cfill, alpha=0.05, lw=0)
	
	return y
	
	
def predict_rates(data, simdfs, mname='MSM', target='H'):
		
	sdt_dict=counts(data, target=target)
	hr_df, fa_df=get_hit_fa(sdt_dict)
	
	hr_mean=hr_df.mean(); fa_mean=fa_df.mean()
	hr_std=hr_df.std(); fa_std=fa_df.std()
	
	if target=='H':
		xlist=np.array(['50/50', '70H', '90H'])
		hr_ylist=np.array([hr_mean[2], hr_mean[1], hr_mean[0]])
		fa_ylist=np.array([fa_mean[2], fa_mean[1], fa_mean[0]])
		hr_err_ylist=np.array([hr_std[2], hr_std[1], hr_std[0]])
		fa_err_ylist=np.array([fa_std[2], fa_std[1], fa_std[0]])
		title_target='House'
	elif target=='F':
		xlist=np.array(['50/50', '70F', '90F'])
		hr_ylist=np.array([hr_mean[2], hr_mean[3], hr_mean[4]])
		fa_ylist=np.array([fa_mean[2], fa_mean[3], fa_mean[4]])
		hr_err_ylist=np.array([hr_std[2], hr_std[3], hr_std[4]])
		fa_err_ylist=np.array([fa_std[2], fa_std[3], fa_std[4]])
		title_target='Face'
	
	hr_err_ylist=hr_err_ylist/np.sqrt(len(hr_df))	
	fa_err_ylist=fa_err_ylist/np.sqrt(len(fa_df))
			
	fig, axes=plt.subplots(2, figsize=(7, 10))
	fig.set_tight_layout(True)	
	hr_ax=axes[0]
	fa_ax=axes[1]
	
	x=np.array([1, 2, 3])
	last_hr=np.zeros([3])
	last_fa=np.zeros([3])
	
	for simn, rest in simdfs.groupby('sim_num'):
		
		sim_dict=counts(rest, target=target)
		hr_sim_df, fa_sim_df=get_hit_fa(sim_dict)
		hr_sim_mean=hr_sim_df.mean(); fa_sim_mean=fa_sim_df.mean()
		
		if target=='H':
			hr_sim_ylist=np.array([hr_sim_mean[2], hr_sim_mean[1], hr_sim_mean[0]])
			fa_sim_ylist=np.array([fa_sim_mean[2], fa_sim_mean[1], fa_sim_mean[0]])
		elif target=='F':
			hr_sim_ylist=np.array([hr_sim_mean[2], hr_sim_mean[3], hr_sim_mean[4]])
			fa_sim_ylist=np.array([fa_sim_mean[2], fa_sim_mean[3], fa_sim_mean[4]])
			
		last_hr = sim_rates_fill(rate_type='hit', ax=hr_ax, x=x, y=hr_sim_ylist, ind=simn, last=last_hr)
		last_fa = sim_rates_fill(rate_type='fa', ax=fa_ax, x=x, y=fa_sim_ylist, ind=simn, last=last_fa)
		
	hr_ax.errorbar(x, hr_ylist, yerr=hr_err_ylist, lw=7.0, elinewidth=4.5, capsize=0, color='LimeGreen', ecolor='k')
	fa_ax.errorbar(x, fa_ylist, yerr=fa_err_ylist, lw=7.0, elinewidth=4.5, capsize=0, color='Red', ecolor='k')
	ax_list=[hr_ax, fa_ax]
	
	hr_ax.set_title(title_target, fontsize=38) 
	hr_ax.set_ylabel('Hit Rate', fontsize=35)
	fa_ax.set_ylabel('FA Rate', fontsize=35)
	hr_ax.set_ylim(0.7, 1.0)
	fa_ax.set_ylim(0.0, 0.4)
	hr_ax.set_yticks(np.array([0.7, 0.8, 0.9, 1.0]))
	fa_ax.set_yticks(np.arange(0, 0.5, .10))
	fa_ax.set_xlabel('Prior Probability Cue', fontsize=38)
	
	for ax in ax_list:
		ax.set_xlim(0.5, 3.5)
		ax.set_xticks(np.array([1, 2, 3]))
		ax.set_xticklabels(xlist, fontsize=35)
		for tick in ax.yaxis.get_major_ticks():
		                tick.label.set_fontsize(30)
		
	#plt.savefig(target+"_"+mname+'.jpeg', format='jpeg', dpi=900)
	plt.savefig(target+"_"+mname+'.png', format='png', dpi=500)
	
def get_params(sdt_dict):
	
	for df in sdt_dict:
		df=calc_sdt(sdt_dict[df])
	
	nsubs=len(sdt_dict[sdt_dict.keys()[0]])
	
	if len(sdt_dict)==5:
		cols=['a90H', 'b70H', 'c50N', 'd70F', 'e90F']
	else:
		cols=['a80H', 'b50N', 'c80F']
	
	cdf=pd.DataFrame(np.zeros(nsubs*len(sdt_dict)).reshape(nsubs, len(sdt_dict)), columns=cols)
	dpdf=pd.DataFrame(np.zeros(nsubs*len(sdt_dict)).reshape(nsubs, len(sdt_dict)), columns=cols)

	for k in sdt_dict.keys():
		cdf[k]=sdt_dict[k]['c']
		dpdf[k]=sdt_dict[k]['dp']

	return cdf, dpdf

def plot_params(data):
	
	from scipy.stats import stats
	
	countsdf=counts(data)
	c, d=get_params(countsdf)
	
	sem_crit=[]
	sem_dp=[]	
	for cond in c.columns:
		se_criterion=stats.sem(c[cond])
		se_dprime=stats.sem(d[cond])
	
		sem_crit.append(se_criterion)
		sem_dp.append(se_dprime)
		
		cmeans=c.describe().ix['mean', :].values
		dmeans=d.describe().ix['mean', :].values
	
	x=np.array([1,2,3])
	fig_c, ax_c=plt.subplots(1)
	fig_d, ax_d=plt.subplots(1)
	
	plotc=ax_c.errorbar(x, cmeans, yerr=sem_crit, elinewidth=2.5, ecolor='r', color='k', lw=4.0)
	plotd=ax_d.errorbar(x, dmeans, yerr=sem_dp, elinewidth=2.5, ecolor='r', color='k', lw=4.0)
	
	ax_list=[ax_c, ax_d]
	for a in ax_list:
		a.set_xlim(0.5, 3.5)
		a.set_xticks([1,2,3])
		a.set_xticklabels(['80H', '50N', '80F'], fontsize=16)
		a.set_xlabel("Prior Probability Cue", fontsize=20)
	
	ax_c.set_ylabel("Criterion (c)", fontsize=20)
	ax_d.set_ylabel("Discriminability (d')", fontsize=20)	
	
	
	#fig_c.savefig("criterion.jpeg", format='jpeg', dpi=400)
	#fig_d.savefig("dprime.jpeg", format='jpeg', dpi=400)
	fig_c.savefig("criterion.png", format='png', dpi=500)
	fig_d.savefig("dprime.png", format='png', dpi=500)	
	
def rho_models(dataframe, depends_on, condsdf):
	"""
	returns correlation matrix (pd.DataFrame) for sdt params and
	all diffusion parameters included in depends_on dict
	
	"""
	countsdt=counts(dataframe)
	crit, dprime=get_params(countsdt)
	
	diffusion_params=diffusionp.get_diffusion(condsdf, depends_on=depends_on)
	
	diff_sdt_rho=dict()

	for p in diffusion_params.keys():
		c=diffusion_params[p].corrwith(crit)
		dp=diffusion_params[p].corrwith(dprime)
		
		ctag=p+'_c'
		dptag=p+'_dp'
		
		diff_sdt_rho[ctag]=c
		diff_sdt_rho[dptag]=dp
		
	rho_mat=pd.DataFrame(diff_sdt_rho)
		
	return rho_mat

def rho_sdt(dataframe, simdf):
	"""
	returns correlation matrix (pd.DataFrame) for empirical c, d' against diffusion simulated c, d'
	
	"""
	emp_v_sim={}

	emp_sdt=counts(dataframe)
	sim_sdt=counts(simdf)

	#SAVE SIM SDT COUNTS
	for df in sim_sdt.keys():
		if df=='e90F':
			sim_sdt[df].to_csv("highF.csv", index=False)
		elif df=='c80F':
			sim_sdt[df].to_csv("highF.csv", index=False)
		elif df=='d70F':
			sim_sdt[df].to_csv("medF.csv", index=False)
		elif df=='c50N':
			sim_sdt[df].to_csv("neut.csv", index=False)
		elif df=='b50N':
			sim_sdt[df].to_csv("neut.csv", index=False)
		elif df=='b70H':
			sim_sdt[df].to_csv("medH.csv", index=False)
		elif df=='a90H':
			sim_sdt[df].to_csv("highH.csv", index=False)
		elif df=='a80H':
			sim_sdt[df].to_csv("highH.csv", index=False)


	simc, simdp=get_params(sim_sdt)
	empc, empdp=get_params(emp_sdt)
	
	ccorr=empc.corrwith(simc)
	dpcorr=empdp.corrwith(simdp)

	emp_v_sim['c']=ccorr
	emp_v_sim['dprime']=dpcorr
	
	empvsim=pd.DataFrame(emp_v_sim)
	#empvsim.to_csv("sdt_rho_matrix.csv")
	
	return empvsim

def plot_rho_heatmap():
	
	#data=pd.read_csv("/Users/DunovanK/Desktop/beh_hddm/AllP_dEWMA5.csv")
	
	emp_c=pd.read_csv('/Users/DunovanK/Desktop/beh_hddm/SDTModels/Empirical/emp_c.csv')
	emp_d=pd.read_csv('/Users/DunovanK/Desktop/beh_hddm/SDTModels/Empirical/emp_d.csv')
	msm_c=pd.read_csv('/Users/DunovanK/Desktop/beh_hddm/SDTModels/Theoretical/msm_sims/msm_c.csv')
	msm_d=pd.read_csv('/Users/DunovanK/Desktop/beh_hddm/SDTModels/Theoretical/msm_sims/msm_d.csv')
	pbm_c=pd.read_csv('/Users/DunovanK/Desktop/beh_hddm/SDTModels/Theoretical/pbm_sims/pbm_c.csv')
	pbm_d=pd.read_csv('/Users/DunovanK/Desktop/beh_hddm/SDTModels/Theoretical/pbm_sims/pbm_d.csv')
	dbm_c=pd.read_csv('/Users/DunovanK/Desktop/beh_hddm/SDTModels/Theoretical/dbm_sims/dbm_c.csv')
	dbm_d=pd.read_csv('/Users/DunovanK/Desktop/beh_hddm/SDTModels/Theoretical/dbm_sims/dbm_d.csv')

	#print emp_c
	#print emp_d
	
	
	vz_c=emp_c.corrwith(msm_c)
	vz_d=emp_d.corrwith(msm_d)
	z_c=emp_c.corrwith(pbm_c)
	z_d=emp_d.corrwith(pbm_d)
	v_c=emp_c.corrwith(dbm_c)
	v_d=emp_d.corrwith(dbm_d)	

	#print "z_d", z_d	
	#print "v_d", v_d
	#print "vz_d", vz_d
	#
	#print "z_c", z_c 
	#print "v_c", v_c
	#print "vz_c", vz_c
		
	criterion_corr=np.array([vz_c, v_c, z_c])
	dprime_corr=np.array([vz_d, v_d, z_d])
	
	#print criterion_corr
	#print dprime_corr
	
	fig=plt.figure(figsize=(10,14))
	fig.set_tight_layout(True)	
	#fig.suptitle("Correlation of Empirical and Theoretical SDT Parameters", fontsize=25)
	axc=fig.add_subplot(211)
	axd=fig.add_subplot(212)
	fig.subplots_adjust(top=.95, hspace=.1, left=0.10, right=.9, bottom=0.1)

	axc.set_ylim(-0.5, 2.5)
	axc.set_yticks([0, 1, 2])
	axc.set_yticklabels(['MSM', 'DBM', 'PBM'], fontsize=34)
	plt.setp(axc.get_yticklabels(), rotation=90)
	axc.set_xlim(-0.5, 4.5)
	axc.set_xticks([0, 1, 2, 3, 4])
	axc.set_xticklabels(['90H', '70H', '50/50', '70F', '90F'], fontsize=34)
	axc.set_title("Criterion", fontsize=36)
	axc.set_xlabel("Prior Probability Cue", fontsize=36)

	axd.set_ylim(-0.5, 2.5)
	axd.set_yticks([0, 1, 2])
	axd.set_yticklabels(['MSM', 'DBM', 'PBM'], fontsize=34)
	plt.setp(axd.get_yticklabels(), rotation=90)
	axd.set_xlim(-0.5, 4.5)
	axd.set_xticks([0, 1, 2, 3, 4])
	axd.set_xticklabels(['90H', '70H', '50/50', '70F', '90F'], fontsize=34)
	axd.set_title("Discriminability", fontsize=36)
	axd.set_xlabel("Prior Probability Cue", fontsize=36)


	axc_map=axc.imshow(criterion_corr, interpolation='nearest', cmap='Reds', origin='lower', vmin=0.5, vmax=1)
	plt.colorbar(axc_map, ax=axc, shrink=0.66)

	axd_map=axd.imshow(dprime_corr, interpolation='nearest', cmap='Reds', origin='lower', vmin=0.5, vmax=1)
	plt.colorbar(axd_map, ax=axd, shrink=0.66)
	
	#plt.savefig('SDT_Corr.eps', format='eps', dpi=400)
	#plt.savefig('SDT_Corr.jpeg', dpi=900)
	plt.savefig('SDT_Corr.png', format='png', dpi=500)

def plot_evs_corr(df, simdf):
	
	e=counts(df)
	s=counts(simdf)
	ec, edp=get_params(e)
	sc, sdp=get_params(s)
	nconds=len(ec.columns)

	fig=plt.figure(figsize=(10, 6))
	fig.suptitle("Correlation of Empirical and Theoretical SDT Parameters", fontsize=25)
	ax=fig.add_subplot(111)
	fig.subplots_adjust(top=.85)

	ax.set_ylim(-0.5, 1.5)
	ax.set_yticks([0,1])
	ax.set_yticklabels(['c', "d'"], fontsize=20)
	ax.set_xlim([-0.5, nconds-0.5])
	ax.set_xticks(np.arange(nconds))
	ax.set_xticklabels(ec.columns, fontsize=16)

	corr_c=ec.corrwith(sc)
	corr_dp=edp.corrwith(sdp)

	#for i in np.arange(nconds):
	ax_map=ax.imshow([corr_c, corr_dp], interpolation='nearest', cmap='Reds', origin='lower', vmin=0.1, vmax=1)
	plt.colorbar(ax_map, ax=ax, shrink=0.94)
	for i, cond in enumerate(corr_c):
		ax.text(i, 0, "r="+str(corr_c[i])[:4], ha='center', va='center', fontsize=16)	
	for i, cond in enumerate(corr_dp):
		ax.text(i, 1, "r="+str(corr_dp[i])[:4], ha='center', va='center', fontsize=16)	

def plot_rho_sdt(dataframe, simdf):
	
	emp_sdt=counts(dataframe)
	sim_sdt=counts(simdf)

	simc, simdp=get_params(sim_sdt)
	empc, empdp=get_params(emp_sdt)

	
	figc=plt.figure(figsize=(10, 5))
	figc.suptitle('Correlation of Empirical and HDDM Simulated SDT Criterion (c)' , fontsize=12)
	figc.subplots_adjust(top=0.85, wspace=0.1)
	counter=1
	for sim, emp in zip(simc.columns, empc.columns):
		ax=figc.add_subplot(1,len(simdf.cue.unique()),counter)
		x=empc[emp]
		y=simc[sim]
		m, b=np.polyfit(x, y, 1)
		ax.plot(x, y, 'ko', x, m*x+b, 'r-', lw=3.0, ms=3.5)
		ax.set_title(str(sim))
		#ax.set_xlabel('Empirical Criterion (c)')
		#ax.set_ylabel('HDDM Simulated SDT Criterion (c)')
		#ax.xaxis.set_major_locator(MaxNLocator(nbins = 4))
		#ax.yaxis.set_major_locator(MaxNLocator(nbins = 6))
		counter+=1
		plt.locator_params(axis='x', nbins=4)
		
		for tick in ax.xaxis.get_major_ticks():
		                tick.label.set_fontsize(10)
		
		if ax.is_first_col():
			ax.set_ylabel("Simulated Criterion (c)", fontsize=16)
			for tick in ax.yaxis.get_major_ticks():
			                tick.label.set_fontsize(10)
		else:
			plt.setp(ax.get_yticklabels(), visible=False)
	figc.text(0.4, 0.009, 'Empirical Criterion (c)', fontsize=16)
	figc.savefig('corr_sdt_c.jpeg', format='jpeg', dpi=300)
	
	figdp=plt.figure(figsize=(10, 5))
	figdp.suptitle("Correlation of Empirical and HDDM Simulated SDT Sensitivity (d')" , fontsize=12)
	figdp.subplots_adjust(top=0.85, wspace=0.1)
	counter=1
	for sim, emp in zip(simdp.columns, empdp.columns):
		ax=figdp.add_subplot(1,len(simdf.cue.unique()),counter)
		x=empdp[emp]
		y=simdp[sim]
		m, b=np.polyfit(x, y, 1)
		ax.plot(x, y, 'ko', x, m*x+b, 'r-', lw=3.0, ms=3.5)
		ax.set_title(str(sim))
		#ax.set_xlabel("Empirical SDT Sensitivity (d')")
		#ax.set_ylabel("HDDM Simulated SDT Sensitivity (d')")
		#ax.xaxis.set_major_locator(MaxNLocator(nbins = 6))
		#ax.yaxis.set_major_locator(MaxNLocator(nbins = 6))
		counter+=1
		
		for tick in ax.xaxis.get_major_ticks():
		                tick.label.set_fontsize(10)

		if ax.is_first_col():
			ax.set_ylabel("Simulated Sensitivity (d')", fontsize=16)
			for tick in ax.yaxis.get_major_ticks():
			                tick.label.set_fontsize(10)
		else:
			plt.setp(ax.get_yticklabels(), visible=False)
	figdp.text(0.4, 0.009, "Empirical Sensitivity (d')", fontsize=16)
	figdp.savefig('corr_sdt_dp.jpeg', format='jpeg', dpi=300)
	x=rho_sdt(dataframe, simdf)
	x.to_csv("sdt_corr.csv")
		
if __name__ == "__main__":
	main()	

