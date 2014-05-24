#!/usr/bin/env python
from __future__ import division
import sys
import os
import hddm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from myhddm import sims, sdt, parse, vis, defmod, mle


def rna(mname):

	m=defmod.define_model(mname)

	m.sample(5000, burn=1000, dbname=mname+'_traces.db', db='pickle')

	postproc(m, mname=mname)

def postproc(m, mname='model', varlvl='grp', traces_info=None):

	if not os.path.isdir("avg_pdf"):
		os.makedirs("avg_pdf")
	if not os.path.isdir("avg_quant"):
		os.makedirs("avg_quant")
	if not os.path.isdir("posteriors"):
		os.makedirs("posteriors")
	if not os.path.isdir("correlations"):
		os.makedirs("correlations")
	if not os.path.isdir("simulations"):
		os.makedirs("simulations")

	os.chdir("posteriors")

	m.plot_posteriors(save=True)
	plt.close('all')

	os.chdir("../")

	m.print_stats(fname=mname+'_stats.txt')
	parsed = parse.parse_stats(minput=m, varlvl=varlvl, input_isdf=False)
	subdf=parsed[0]
	condsdf=parsed[1]
	pdict=parsed[2]

	os.chdir('simulations')

	#pdict=mle.optimize_sx(mname)

	print "simulating from mle optimized params"
	simdf=vis.predict(params=pdict, data=m.data, simfx=sims.sim_exp, ntrials=240, pslow=0.0, pfast=0.0, nsims=100, save=True, RTname=mname+'_RT_simexp', ACCname=mname+'_ACC_simexp')
	simdf.to_csv("simdf.csv", index=False)
	simdf=pd.read_csv("simdf.csv")
	vis.predict_from_simdfs(m.data, simdf)
	plt.close('all')

	os.chdir('../')

	simdf.to_csv("simdf_opt.csv")

	os.chdir('correlations')

	print "estimating emp-theo sdt correlation, saving..."
	sdt_corr=sdt.rho_sdt(m.data, simdf)

	print "plotting emp-theo sdt correlation, saving..."
	sdt.plot_evs_corr(m.data, simdf)
	sdt_corr.to_csv("sdt_corr.csv")
	plt.close('all')

	os.chdir('../')

	#mname, traces_name=defmod.find_traces_imaging(m)


	avgm=m.get_average_model()
	#avgm.load_db(dbname=traces_name, db='pickle')
	avgm.load_db(dbname="msmIO_traces.db", db='pickle')

	os.chdir("./avg_pdf")

	avgm.plot_posterior_predictive(save=True, value_range=np.linspace(-6, 6, 100), figsize=(12, 10))

	os.chdir("../avg_quant")

	avgm.plot_posterior_quantiles(save=True, samples=250, figsize=(12, 10))

	plt.close('all')


if __name__=="__main__":
	main()
