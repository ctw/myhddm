#!/usr/bin/env python

from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from myhddm import sims, parse
import hddm
import kabuki
from kabuki.utils import interpolate_trace
from scipy.stats.mstats import mquantiles
from matplotlib import rc
import os
import seaborn as sns

def get_nodes(model, nodes, project='behav'):

	if project=='behav':

		if nodes=='z':
			z90H, z70H, z50N, z70F, z90F = model.nodes_db.node[['z(a90H)', 'z(b70H)', 'z(c50N)', 'z(d70F)', 'z(e90F)']]
			zlist=[z90H, z70H, z50N, z70F, z90F]
			return zlist

		elif nodes=='vf':
			v90Hface, v70Hface, v50Nface, v70Fface, v90Fface=model.nodes_db.node[['v(a90H.face)', 'v(b70H.face)', 'v(c50N.face)', 'v(d70F.face)', 'v(e90F.face)']]
			vflist=[v90Hface, v70Hface, v50Nface, v70Fface, v90Fface]
			return vflist

	 	elif nodes=='vh':
			v90Hhouse, v70Hhouse, v50Nhouse, v70Fhouse, v90Fhouse=model.nodes_db.node[['v(a90H.house)', 'v(b70H.house)', 'v(c50N.house)', 'v(d70F.house)', 'v(e90F.house)']]
			vhlist=[v90Hhouse, v70Hhouse, v50Nhouse, v70Fhouse, v90Fhouse]
			return vhlist

	else:
		if nodes=='z':
			z80H, z50N, z80F = model.nodes_db.node[['z(a80H)','z(b50N)', 'z(c80F)']]
			zlist=[z80H, z50N, z80F]
			return zlist

		elif nodes=='vf':
			v80Hface, v50Nface, v80Fface=model.nodes_db.node[['v(a80H.face)', 'v(b50N.face)', 'v(c80F.face)']]
			vflist=[v80Hface, v50Nface, v80Fface]
			return vflist

	 	elif nodes=='vh':
			v80Hhouse, v50Nhouse, v80Fhouse=model.nodes_db.node[['v(a80H.house)', 'v(b50N.house)', 'v(c80F.house)']]
			vhlist=[v80Hhouse, v50Nhouse, v80Fhouse]
			return vhlist


def plot_posterior_nodes(model, param_nodes, bins=100, lb=None, ub=None):
	
	sns.set_style('white')
	sns.despine()
	#title='Generic Title'
	if param_nodes=='z':
		nodes=get_nodes(model, 'z')
		xlabel='Mean Starting-Point' + r'$\/(\mu_{z})$'
		lb=.52
		ub=.67
	elif param_nodes=='vf':
		nodes=get_nodes(model, 'vf')
		xlabel='Mean Face Drift-Rate' + r'$\/(\mu_{vF})$'
		lb=.35
		ub=1.2
	elif param_nodes=='vh':
		nodes=get_nodes(model, 'vh')
		xlabel='Mean House Drift-Rate' + r'$\/(\mu_{vH})$'
		lb=-1.15
		ub=-.25
	else:
		print "Must provide argument: 'z', 'vf', or 'vh'"
	
	fig=plt.figure()
	fig.subplots_adjust(top=0.95, wspace=0.12, left=0.12, right=0.88, bottom=0.16)
	sns.despine()
	#fig.suptitle(title, fontsize=20)
	#fig.suptitle(title, fontsize=40)

	if lb is None:
		lb = min([min(node.trace()[:]) for node in nodes])
	if ub is None:
		ub = max([max(node.trace()[:]) for node in nodes])

	x_data = np.linspace(lb, ub, 600)
	#colors=['Green', 'LimeGreen', 'Black', 'Cyan', 'Blue']
	colors=['Red','Magenta', 'Black', 'Cyan', 'Blue']
	color_i=0
	for node in nodes:
		trace = node.trace()[:]
		#hist = interpolate_trace(x_data, trace, range=(trace.min(), trace.max()), bins=bins)
		hist = interpolate_trace(x_data, trace, range=(lb, ub), bins=bins)
		plt.plot(x_data, hist, label=node.__name__, lw=2., color=colors[color_i])
		plt.fill_between(x_data, hist, 0, label=node.__name__, color=colors[color_i], alpha=0.3)
		ax=plt.gca()
		ax.set_xlim(lb, ub)
		plt.setp(ax.get_yticklabels(), visible=False)
		sns.despine()
		ax.set_ylabel('Probability Mass', fontsize=22, labelpad=12)
		ax.set_xlabel(xlabel, fontsize=22, labelpad=13)
		plt.setp(ax.get_xticklabels(), fontsize=18)
		plt.locator_params(axis='x', nbins=10)
		color_i+=1
	#leg = plt.legend(loc='best', fancybox=True)
	#leg.get_frame().set_alpha(0.5)
	plt.ylim(ymin=0)
	plt.savefig(str(param_nodes)+'_posterior_nodes.png', dpi=600)
	#plt.savefig(str(param_nodes)+'_posterior_nodes.pdf', format='pdf')


def diff_traces(model, output='all', project='behav'):
	"""
	change output to 'neut' if just want difference
	bw face and house drift at neutral condition
	"""
	if project=='behav':
		v90Hface, v70Hface, v50Nface, v70Fface, v90Fface=model.nodes_db.node[['v(a90H.face)', 'v(b70H.face)', 'v(c50N.face)', 'v(d70F.face)', 'v(e90F.face)']]
		v90Hhouse, v70Hhouse, v50Nhouse, v70Fhouse, v90Fhouse=model.nodes_db.node[['v(a90H.house)', 'v(b70H.house)', 'v(c50N.house)', 'v(d70F.house)', 'v(e90F.house)']]
		z90H, z70H, z50N, z70F, z90F = model.nodes_db.node[['z(a90H)', 'z(b70H)', 'z(c50N)', 'z(d70F)', 'z(e90F)']]

		xtrace=abs(v50Nhouse.trace()) - abs(v50Nface.trace())
		vf1_trace=v90Hface.trace() - v50Nface.trace()
		vf2_trace=v70Hface.trace() - v50Nface.trace()
		vf3_trace=v70Fface.trace() - v50Nface.trace()
		vf4_trace=v90Fface.trace() - v50Nface.trace()
		vf_list=[vf1_trace, vf2_trace, vf3_trace, vf4_trace]

		vh1_trace=v90Hhouse.trace() - v50Nhouse.trace()
		vh2_trace=v70Hhouse.trace() - v50Nhouse.trace()
		vh3_trace=v70Fhouse.trace() - v50Nhouse.trace()
		vh4_trace=v90Fhouse.trace() - v50Nhouse.trace()
		vh_list=[vh1_trace, vh2_trace, vh3_trace, vh4_trace]

		z1_trace=z90H.trace() - z50N.trace()
		z2_trace=z70H.trace() - z50N.trace()
		z3_trace=z70F.trace() - z50N.trace()
		z4_trace=z90F.trace() - z50N.trace()
		z_list=[z1_trace, z2_trace, z3_trace, z4_trace]


	else:
		z80H, z50N, z80F = model.nodes_db.node[['z(a80H)','z(b50N)', 'z(c80F)']]
		v80Hhouse, v50Nhouse, v80Fhouse=model.nodes_db.node[['v(a80H.house)', 'v(b50N.house)', 'v(c80F.house)']]
		v80Hface, v50Nface, v80Fface=model.nodes_db.node[['v(a80H.face)', 'v(b50N.face)', 'v(c80F.face)']]

		xtrace=abs(v50Nhouse.trace()) - abs(v50Nface.trace())
		vf1_trace=v80Hface.trace() - v50Nface.trace()
		vf2_trace=v80Fface.trace() - v50Nface.trace()

		vf_list=[vf1_trace, vf2_trace]

		vh1_trace=v80Hhouse.trace() - v50Nhouse.trace()
		vh2_trace=v80Fhouse.trace() - v50Nhouse.trace()
		vh_list=[vh1_trace, vh2_trace]

		z1_trace=z80H.trace() - z50N.trace()
		z2_trace=z80F.trace() - z50N.trace()
		z_list=[z1_trace, z2_trace]

	if output=='all':
		return vf_list, vh_list, z_list
	else:
		return xtrace

def plot_neutral_traces(model):
	
	sns.set_style('white')
	
	c=diff_traces(model, output='neut')

	save_fig=True
	x_axis_list=r'$\mu_{vH}\/-\/\mu_{vF}$'

	
	fig=plt.figure()
	fig.subplots_adjust(top=0.95, wspace=0.12, left=0.12, right=0.88, bottom=0.16)
	sns.despine()
	
	ax=fig.add_subplot(111)
	sns.despine()
	
	ax.hist(c, bins=20, facecolor='DarkBlue', alpha=0.4)
	sns.despine()
	c_quantiles=mquantiles(c, prob=[0.025, 0.975])
	ax.axvline(c_quantiles[0], lw=2.0, ls='--', color='Black', alpha=0.4)
	ax.axvline(c_quantiles[1], lw=2.0, ls='--', color='Black', alpha=0.4)
	c_mean=str(c.mean())[:4]
	c_lower=str(c_quantiles[0])[:5]
	c_upper=str(c_quantiles[1])[:4]

	ax.text(0.5, .92, r'$\mu_\Delta=%s;\/\/95%sCI[%s, %s]$' % (c_mean, "\%", c_lower, c_upper), fontsize=22, va='center', ha='center', transform=ax.transAxes)

	pos_float=(c>0).mean()*100
	neg_float=(c<0).mean()*100
	pos_str=str(pos_float)
	neg_str=str(neg_float)
	pos=pos_str[:5]
	neg=neg_str[:4]

	ax.text(0.5, .82, r'$%s%s\/<\/0\/<\/%s%s$' % (neg, "\%", pos, "\%"), fontsize=22, va='center', ha='center', transform=ax.transAxes)
	plt.setp(ax.get_xticklabels(), fontsize=18)
	plt.setp(ax.get_yticklabels(), visible=False)
	plt.xlim(-0.1, 0.4)
	plt.ylim(0, 3000)

	plt.locator_params(axis='x', nbins=6)
	
	ax.set_xlabel(r'$\mu_{vH}\/-\/\mu_{vF}$', fontsize=24, labelpad=13)
	ax.set_ylabel("Probability Mass", fontsize=22, labelpad=12)

	plt.savefig("face_house_neutral_drift.png", format="png", dpi=600)
	#plt.savefig("face_house_neutral_drift.jpeg", format="jpeg", dpi=900)



def plot_diff_traces(model):

	vf_list, vh_list, z_list=diff_traces(model)

	param_list=[vf_list, vh_list, z_list]

	x_axis_list=[r'$\mu_{90H}\/-\/\mu_{50N}$', r'$\mu_{70H}\/-\/\mu_{50N}$', r'$\mu_{70F}\/-\/\mu_{50N}$', r'$\mu_{90F}\/-\/\mu_{50N}$']

	fig=plt.figure(figsize=(38, 5))
	fig.subplots_adjust(top=0.95, wspace=0.15, left=0.02, right=0.98, bottom=0.20)
	#fig.suptitle('Mean Face Drift-Rate' + r'$\/(\mu_{vF})$', fontsize=35)
	i=1
	for c in vf_list:
		ax=fig.add_subplot(1, 4, i)
		#ax.hist(c, bins=15, facecolor='SlateBlue', alpha=0.8)
		ax.hist(c, bins=15, facecolor='Blue', alpha=0.4)
		c_quantiles=mquantiles(c, prob=[0.025, 0.975])
		ax.axvline(c_quantiles[0], lw=3.0, ls='--', color='DarkGray', alpha=0.5)
		ax.axvline(c_quantiles[1], lw=3.0, ls='--', color='DarkGray', alpha=0.5)

		c_mean=str(c.mean())[:5]
		c_lower=str(c_quantiles[0])[:5]
		c_upper=str(c_quantiles[1])[:5]

		ax.text(0.5, .92, r'$\mu_\Delta=%s;\/95%sCI[%s, %s]}$' % (c_mean, "\%", c_lower, c_upper), fontsize=16, va='center', ha='center', transform=ax.transAxes)

		pos_float=(c>0).mean()*100
		neg_float=(c<0).mean()*100
		pos_str=str(pos_float)[:4]
		neg_str=str(neg_float)[:4]
		ax.text(0.5, .82, r'$%s%s\/<\/0\/<\/%s%s$' % (neg_str, "\%", pos_str, "\%"), fontsize=17, va='center', ha='center', transform=ax.transAxes)

		ax.set_xlabel(x_axis_list[i-1], fontsize=25, labelpad=12)

		plt.ylim(0, 3000)

		plt.locator_params(axis='x', nbins=6)
		plt.setp(ax.get_xticklabels(), fontsize=16)
		plt.setp(ax.get_yticklabels(), visible=False)
		#if ax.is_first_col():
		#	ax.set_ylabel("Probability Mass", fontsize=30, labelpad=10)
		i+=1

	plt.savefig("face_drift_comparisons.png", format="png", dpi=600)
	#plt.savefig("face_drift_comparisons.jpeg", format="jpeg", dpi=900)


	fig=plt.figure(figsize=(38, 5))
	fig.subplots_adjust(top=0.95, wspace=0.15, left=0.02, right=0.98, bottom=0.20)
	#fig.suptitle('Mean House Drift-Rate' + r'$\/(\mu_{vH})$', fontsize=35)
	i=1
	for c in vh_list:
		ax=fig.add_subplot(1, 4, i)
		#ax.hist(c, bins=15, facecolor='SlateBlue', alpha=0.8)
		ax.hist(c, bins=15, facecolor='Red', alpha=0.4)
		c_quantiles=mquantiles(c, prob=[0.025, 0.975])
		ax.axvline(c_quantiles[0], lw=3.0, ls='--', color='DarkGray', alpha=0.5)
		ax.axvline(c_quantiles[1], lw=3.0, ls='--', color='DarkGray', alpha=0.5)

		c_mean=str(c.mean())[:5]
		c_lower=str(c_quantiles[0])[:5]
		c_upper=str(c_quantiles[1])[:5]

		ax.text(0.5, .92, r'$\mu_\Delta = %s;\/95%s CI[%s, %s]}$' % (c_mean, "\%", c_lower, c_upper), fontsize=16, va='center', ha='center', transform=ax.transAxes)

		pos_float=(c>0).mean()*100
		neg_float=(c<0).mean()*100
		pos_str=str(pos_float)[:4]
		neg_str=str(neg_float)[:4]
		ax.text(0.5, .82, r'$%s%s\/<\/0\/<\/%s%s$' % (neg_str, "\%", pos_str, "\%"), fontsize=17, va='center', ha='center', transform=ax.transAxes)

		ax.set_xlabel(x_axis_list[i-1], fontsize=25, labelpad=10)

		plt.ylim(0, 3000)

		plt.locator_params(axis='x', nbins=6)
		plt.setp(ax.get_xticklabels(), fontsize=16)

		plt.setp(ax.get_yticklabels(), visible=False)
		#if ax.is_first_col():
		#	ax.set_ylabel("Probability Mass", fontsize=30, labelpad=10)

		i+=1
	plt.savefig("house_drift_comparisons.png", format="png", dpi=600)
	#plt.savefig("house_drift_comparisons.pdf", format="pdf")
	#plt.savefig("house_drift_comparisons.jpeg", format="jpeg", dpi=900)


	fig=plt.figure(figsize=(38, 5))
	fig.subplots_adjust(top=0.95, wspace=0.15, left=0.02, right=0.98, bottom=0.20)
	#fig.suptitle('Mean Starting-Point' + r'$\/(\mu_{z})$', fontsize=35)
	i=1
	for c in z_list:
		ax=fig.add_subplot(1, 4, i)
		ax.hist(c, bins=15, facecolor='SlateBlue', alpha=0.4)

		c_quantiles=mquantiles(c, prob=[0.025, 0.975])
		ax.axvline(c_quantiles[0], lw=3.0, ls='--', color='DarkGray', alpha=0.5)
		ax.axvline(c_quantiles[1], lw=3.0, ls='--', color='DarkGray', alpha=0.5)

		c_mean=str(c.mean())[:5]
		c_lower=str(c_quantiles[0])[:5]
		c_upper=str(c_quantiles[1])[:5]

		ax.text(0.5, .92, r'$\mu_\Delta = %s;\/95%s CI[%s, %s]}$' % (c_mean, "\%", c_lower, c_upper), fontsize=16, va='center', ha='center', transform=ax.transAxes)

		pos_float=(c>0).mean()*100
		neg_float=(c<0).mean()*100
		pos_str=str(pos_float)[:4]
		neg_str=str(neg_float)[:4]
		ax.text(0.5, .82, r'$%s%s\/<\/0\/<\/%s%s$' % (neg_str, "\%", pos_str, "\%"), fontsize=17, va='center', ha='center', transform=ax.transAxes)

		ax.set_xlabel(x_axis_list[i-1], fontsize=25, labelpad=12)

		plt.ylim(0, 1000)

		plt.locator_params(axis='x', nbins=6)
		plt.setp(ax.get_xticklabels(), fontsize=16)

		plt.setp(ax.get_yticklabels(), visible=False)

		#if ax.is_first_col():
		#	ax.set_ylabel("Probability Mass", fontsize=30, labelpad=10)

		i+=1

	plt.savefig("starting_point_comparisons.png", format="png", dpi=600)
	#plt.savefig("starting_point_comparisons.pdf", format="pdf")
	#plt.savefig("starting_point_comparisons.jpeg", format="jpeg", dpi=900)
	#plt.savefig("starting_point_comparisons.tiff", format="tiff", dpi=900)

def _plot_posterior_quantiles_node(node, axis, quantiles=(.1, .3, .5, .7, .9),
                                   samples=100, alpha=.5, hexbin=False,
                                   value_range=(0, 6),
                                   data_plot_kwargs=None, predictive_plot_kwargs=None):
	"""Plot posterior quantiles for a single node.

	:Arguments:

	node : pymc.Node
	    Must be observable.

	axis : matplotlib.axis handle
	    Axis to plot into.

	:Optional:

	value_range : numpy.ndarray
	    Range over which to evaluate the CDF.

	samples : int (default=10)
	    Number of posterior samples to use.

	alpha : float (default=.75)
	   Alpha (transparency) of posterior quantiles.

	hexbin : bool (default=False)
	   Whether to plot posterior quantile density
	   using hexbin.

	data_plot_kwargs : dict (default=None)
	   Forwarded to data plotting function call.

	predictive_plot_kwargs : dict (default=None)
	   Forwareded to predictive plotting function call.

	"""
	quantiles = np.asarray(quantiles)
	axis.set_xlim(value_range)
	axis.set_ylim((0, 1))

	sq_lower = np.empty((len(quantiles), samples))
	sq_upper = sq_lower.copy()
	sp_upper = np.empty(samples)
	for i_sample in range(samples):
	    kabuki.analyze._parents_to_random_posterior_sample(node)
	    sample_values = node.random()
	    sq_lower[:, i_sample], sq_upper[:, i_sample], sp_upper[i_sample] = data_quantiles(sample_values)

	y_lower = np.dot(np.atleast_2d(quantiles).T, np.atleast_2d(1 - sp_upper))
	y_upper = np.dot(np.atleast_2d(quantiles).T, np.atleast_2d(sp_upper))
	if hexbin:
	    if predictive_plot_kwargs is None:
	        predictive_plot_kwargs = {'gridsize': 85, 'bins': 'log', 'extent': (value_range[0], value_range[1], 0, 1)}
	    x = np.concatenate((sq_lower, sq_upper))
	    y = np.concatenate((y_lower, y_upper))
	    axis.hexbin(x.flatten(), y.flatten(), label='post pred lb', **predictive_plot_kwargs)
	else:
	    if predictive_plot_kwargs is None:
	        predictive_plot_kwargs = {'alpha': .3}
	    axis.plot(sq_lower, y_lower, 'o', label='post pred lb', color='LimeGreen', markersize=10, markeredgecolor=None, markeredgewidth=0.0, **predictive_plot_kwargs)
	    axis.plot(sq_upper, y_upper, 'o', label='post pred ub', color='RoyalBlue', markersize=10, markeredgecolor=None, markeredgewidth=0.0, **predictive_plot_kwargs)

	# Plot data
	data = node.value
	color = 'w' if hexbin else 'k'
	if data_plot_kwargs is None:
	    data_plot_kwargs = {'color': color, 'lw': 2., 'marker': 'o', 'markersize': 10}

	if len(data) != 0:
	    q_lower, q_upper, p_upper = data_quantiles(data)

	    axis.plot(q_lower, quantiles*(1-p_upper), **data_plot_kwargs)
	    axis.plot(q_upper, quantiles*p_upper, **data_plot_kwargs)

	#axis.set_xlabel('RT')
	#axis.set_ylabel('Prob respond')
	axis.set_ylim(bottom=0) # Likelihood and histogram can only be positive



def _plot_posterior_pdf_node(bottom_node, axis, value_range=None, samples=10, bins=100):
	"""Calculate posterior predictive for a certain bottom node.

	:Arguments:
		bottom_node : pymc.stochastic
			Bottom node to compute posterior over.

		axis : matplotlib.axis
			Axis to plot into.

		value_range : numpy.ndarray
			Range over which to evaluate the likelihood.

	:Optional:
		samples : int (default=10)
			Number of posterior samples to use.

		bins : int (default=100)
			Number of bins to compute histogram over.

	"""

	if value_range is None:
		# Infer from data by finding the min and max from the nodes
		raise NotImplementedError, "value_range keyword argument must be supplied."

	like = np.empty((samples, len(value_range)), dtype=np.float32)
	for sample in range(samples):
		_parents_to_random_posterior_sample(bottom_node)
		# Generate likelihood for parents parameters
		like[sample,:] = bottom_node.pdf(value_range)

	y = like.mean(axis=0)
	try:
		y_std = like.std(axis=0)
	except FloatingPointError:
		print "WARNING! %s threw FloatingPointError over std computation. Setting to 0 and continuing." % bottom_node.__name__
		y_std = np.zeros_like(y)

	# Plot pp
	#axis.plot(value_range, y, label='post pred', color='b')
	#axis.fill_between(value_range, y-y_std, y+y_std, color='b', alpha=.6)

	# Plot data
	if len(bottom_node.value) != 0:
		axis.hist(bottom_node.value.values, normed=True, color='Green',
				  range=(value_range[0], value_range[-1]), label='data',
				  bins=bins, histtype='stepfilled', lw=2., alpha=0.4)

	# Plot pp
	axis.plot(value_range, y, label='post pred', color='Blue')
	axis.fill_between(value_range, y-y_std, y+y_std, color='Blue', alpha=.6)
	axis.set_ylim(bottom=0) # Likelihood and histogram can only be positive
	axis.grid=True




def plot_avgm_predictive(model, plot_func=None, required_method='pdf', columns=None, save=True, path=None, figsize=(12,11), format='jpeg', **kwargs):

	from hddm import utils

	if 'value_range' is None:
		rt = np.abs(model.data['rt']); kwargs['value_range'] = (np.min(rt.min()-.2, 0), rt.max())

	if plot_func is None:
		pltmethod='_quant'
		plot_func = utils._plot_posterior_quantiles_node
	else:
		pltmethod='_quant'
	observeds = model.get_observeds()
	fig = plt.figure(figsize=figsize)
	i=1
	for tag, nodes in observeds.groupby('tag'):
		#fig = plt.figure(figsize=figsize)
		#fig.suptitle(tag, fontsize=18)
		fig.subplots_adjust(top=0.95, hspace=0, wspace=0)
		ax = fig.add_subplot(5, 2, i)
		#fig.text(0.04, 0.5, "Probability of Response", va="center", rotation="vertical", fontsize=30, family='sans-serif')
		ax.text(0.9, 0.7, tag[0][-3:], fontsize=32)
		# Plot individual subjects (if present)
		for node_name, bottom_node in nodes.iterrows():
			if not hasattr(bottom_node['node'], required_method):
				continue # skip nodes that do not define the required_method

			print str(node_name)
			print str(tag)
			plot_func(bottom_node['node'], ax, **kwargs)
			i=i+1

	for ax in fig.axes:
		ax.set_xlim(0.5, 5.5)
		ax.set_xticks([1, 2, 3, 4, 5])
		if ax.is_last_row():
			ax.set_xlabel("Response Time (s)", fontsize=30)
			for tick in ax.xaxis.get_major_ticks():
			                tick.label.set_fontsize(25)

			ax.set_xticklabels([1, 2, 3, 4, 5])

		else:
			plt.setp(ax.get_xticklabels(), visible=False)

		if ax.is_first_row():
			if ax.is_first_col():
				ax.set_title("Face", fontsize=40)
			else:
				ax.set_title("House", fontsize=40)

		if ax.is_first_col():
			#ax.set_ylabel("Probability", fontsize=16)
			ax.set_yticks([.1, .3, .5, .7, .9])
			ax.set_yticklabels([.1, .3, .5, .7, .9])
			for tick in ax.yaxis.get_major_ticks():
			                tick.label.set_fontsize(23)
		else:
			plt.setp(ax.get_yticklabels(), visible=False)

	fig.text(0.01, 0.5, "Probability of Response", va="center", rotation="vertical", fontsize=30, family='sans-serif')

	if save:
		print "Double Okay"
		fname = "AvgmQP"
		if path is None:
			path = '.'
		if isinstance(format, str):
			format = [format]
		[fig.savefig('%s.%s' % (os.path.join(path, fname), x), format=x, dpi=300) for x in format]


def plot_posterior_predictive(model, plot_func=None, required_method='pdf', columns=None, save=False, path=None,
							  figsize=(8,6), format='png', **kwargs):

	if plot_func is None:
		pltmethod='_pdf'
		plot_func = _plot_posterior_pdf_node
	else:
		pltmethod='_quant'
	observeds = model.get_observeds()

	if columns is None:
		# If there are less than 3 items to plot per figure,
		# only use as many columns as there are items.
		max_items = max([len(i[1]) for i in
						 observeds.groupby('tag').groups.iteritems()])
		columns = min(3, max_items)

	# Plot different conditions (new figure for each)
	for tag, nodes in observeds.groupby('tag'):
		fig = plt.figure(figsize=figsize)
		fig.suptitle(tag, fontsize=18)
		fig.subplots_adjust(top=0.95, hspace=0.10, wspace=0.10)

		# Plot individual subjects (if present)
		for subj_i, (node_name, bottom_node) in enumerate(nodes.iterrows()):
			if not hasattr(bottom_node['node'], required_method):
				continue # skip nodes that do not define the required_method

			ax = fig.add_subplot(np.ceil(len(nodes)/columns), columns, subj_i+1)

			if 'subj_idx' in bottom_node:
				#ax.set_title(str(bottom_node['subj_idx']))
				ax.text(0.05, 0.95, (str(bottom_node['subj_idx'])),
					va='top', transform=ax.transAxes,
					fontsize=16, fontweight='bold')

			plot_func(bottom_node['node'], ax, **kwargs)

		for ax in fig.axes:
			if ax.is_last_row():
				ax.set_xlabel("Response Time", fontsize=16)
				for tick in ax.xaxis.get_major_ticks():
				                tick.label.set_fontsize(14)
			else:
				plt.setp(ax.get_xticklabels(), visible=False)
			if ax.is_first_col():
				ax.set_ylabel("Probability", fontsize=16)
				for tick in ax.yaxis.get_major_ticks():
				                tick.label.set_fontsize(14)
			else:
				plt.setp(ax.get_yticklabels(), visible=False)

		# Save figure if necessary

		if save:
			fname = str(tag) + pltmethod
			if path is None:
				path = '.'
			if isinstance(format, str):
				format = [format]
			[fig.savefig('%s.%s' % (os.path.join(path, fname), x), format=x) for x in format]



def sub_dists(data, nbins=40, save=True):

	for i, rest in data.groupby('subj_idx'):
		fdata=rest[rest['stim']=='face']
		hdata=rest[rest['stim']=='house']

		face=hddm.utils.flip_errors(fdata)
		house=hddm.utils.flip_errors(hdata)

		face_rts=face.rt
		house_rts=house.rt

		subj_fig=plt.figure(figsize=(14, 8), dpi=150)
		axF = subj_fig.add_subplot(211, xlabel='RT', ylabel='count', title='FACE RT distributions')
		axH = subj_fig.add_subplot(212, xlabel='RT', ylabel='count', title='HOUSE RT distributions')

		axF.hist(face_rts, color='DodgerBlue', lw=1.5, bins=nbins, histtype='stepfilled', alpha=0.6)
		axH.hist(house_rts, color='LimeGreen', lw=1.5, bins=nbins, histtype='stepfilled', alpha=0.6)
		axF.grid()
		axH.grid()
		axF.set_xlim(-6, 6)
		axH.set_xlim(-6, 6)
		if save:
			subj_fig.savefig('Subj'+str(i)+'_RTDist'+'.jpeg', dpi=300)
		else:
			subj_fig.show()


def all_dists(data, nbins=40, save=False):


		fdata=hddm.utils.flip_errors(data[(data['stim']=='face')])
		hdata=hddm.utils.flip_errors(data[(data['stim']=='house')])

		fig=plt.figure(figsize=(14, 8))
		axF = fig.add_subplot(211, xlabel='RT', ylabel='count', title='FACE RT distributions')
		axH = fig.add_subplot(212, xlabel='RT', ylabel='count', title='HOUSE RT distributions')

		for i, facerts in fdata.groupby('subj_idx'):
		    axF.hist(facerts.rt, bins=nbins, label=str(i), histtype='stepfilled', alpha=0.3)
		for i, houserts in hdata.groupby('subj_idx'):
		    axH.hist(houserts.rt, bins=nbins, label=str(i), histtype='stepfilled', alpha=0.3)

		axF.grid()
		axH.grid()
		axF.set_xlim(-6, 6)
		axH.set_xlim(-6, 6)

		handles, labels = axF.get_legend_handles_labels()
		fig.legend(handles[:], labels[:], loc=7)

		if save:
			plt.savefig('AllSubj_RTDists.png')
		else:
			plt.show()


def pred_rtPLOT(code_type, rt_ax=None, xrt=None, yrtFace=None, yrtHouse=None, mname='EvT', ind=0, flast_rt=np.zeros([5]), hlast_rt=np.zeros([5])):
	"""
	Plotting function:
		*plots average empirical and theoretical accuracy
		 for each condition (averaged over subjects)
		*the matplotlib fill function covers the area between
		 all simulated datasets -- Better when running multiple
		 simulations.  If only running 1 sim, use the between_PLOTs
		*called by plot_predictive_behav()
	"""
	
	f_theo=rt_ax.plot(xrt, yrtFace, '-', color='RoyalBlue', lw=0.6, alpha=0.2)
	h_theo=rt_ax.plot(xrt, yrtHouse,'-', color='FireBrick', lw=0.6, alpha=0.2)

	if ind!=0:
		yrtF2=flast_rt
		yrtH2=hlast_rt
		face_fill=rt_ax.fill_between(xrt, yrtFace, yrtFace-(yrtFace-yrtF2), facecolor='RoyalBlue', alpha=0.05)
		house_fill=rt_ax.fill_between(xrt, yrtHouse, yrtHouse-(yrtHouse-yrtH2), facecolor='FireBrick', alpha=0.05)

	if code_type=='HNL':
		rt_ax.set_ylim(1.5, 3.5)
		rt_ax.set_xticks([1, 2, 3])
		rt_ax.set_xlim(0.5, 3.5)
		rt_ax.set_xticklabels(['80H', '50N', '80F'], fontsize=32)
		rt_ax.set_yticks(np.arange(1.5, 4.0, 0.5))
		rt_ax.set_yticklabels(np.arange(1.5, 4.0, 0.5), fontsize=25)
		rt_ax.set_ylabel('Response Time (s)', fontsize=35, labelpad=14)
		rt_ax.set_xlabel('Prior Probability Cue', fontsize=35, labelpad=10)
	else:
		rt_ax.set_ylim(1.6, 3.5)
		rt_ax.set_xticks([1, 2, 3, 4, 5])
		rt_ax.set_xlim(0.6, 5.5)
		rt_ax.set_xticklabels(['90H', '70H', '50/50', '70F', '90F'], fontsize=18)
		rt_ax.set_yticks(np.arange(1.5, 4.0, 0.5))
		rt_ax.set_yticklabels(np.arange(1.5, 4.0, 0.5), fontsize=18)
		
		if mname=='pbm':
			rt_ax.set_ylabel('Response Time (s)', fontsize=22, labelpad=14)
		
		rt_ax.set_xlabel('Prior Probability Cue', fontsize=22, labelpad=10)


	return yrtFace, yrtHouse

def pred_accPLOT(code_type, acc_ax=None, xacc=None, yaccFace=None, yaccHouse=None, mname='EvT', ind=0, flast_acc=np.zeros([5]), hlast_acc=np.zeros([5])):
	"""

	Plotting function:
		*plots average empirical and theoretical accuracy
		 for each condition (averaged over subjects)
		*the matplotlib fill function covers the area between
		 all simulated datasets -- Better when running multiple
		 simulations.  If only running 1 sim, use the between_PLOTs
		*called by plot_predictive_behav()

	"""
	
	f_theo=acc_ax.plot(xacc, yaccFace, '-', color='RoyalBlue', lw=0.6, alpha=0.2)
	h_theo=acc_ax.plot(xacc, yaccHouse, '-', color='FireBrick', lw=0.6, alpha=0.2)

	if ind!=0:
		yaccF2=flast_acc
		yaccH2=hlast_acc
		face_fill=acc_ax.fill_between(xacc, yaccFace, yaccFace-(yaccFace-yaccF2), facecolor='RoyalBlue', alpha=0.05)
		house_fill=acc_ax.fill_between(xacc, yaccHouse, yaccHouse-(yaccHouse-yaccH2), facecolor='FireBrick', alpha=0.05)

	if code_type=='HNL':
		acc_ax.set_ylim(0.6, 1.0)
		acc_ax.set_xticks([1, 2, 3])
		acc_ax.set_xlim(0.5, 3.5)
		acc_ax.set_xticklabels(['80H', '50N', '80F'], fontsize=32)
		acc_ax.set_yticks(np.arange(0.6, 1.05, .05))
		acc_ax.set_yticklabels(np.arange(0.6, 1.05, .05), fontsize=25)
		acc_ax.set_ylabel('Proportion Correct', fontsize=35, labelpad=14)
		acc_ax.set_xlabel('Prior Probability Cue', fontsize=35, labelpad=10)

	else:

		acc_ax.set_ylim(0.6, 1.0)
		acc_ax.set_xticks([1, 2, 3, 4, 5])
		acc_ax.set_xlim(0.5, 5.5)
		acc_ax.set_xticklabels(['90H', '70H', '50/50', '70F', '90F'], fontsize=18)
		acc_ax.set_yticks(np.arange(0.6, 1.05, .05))
		acc_ax.set_yticklabels(np.arange(0.6, 1.05, .05), fontsize=18)
		
		if mname=='pbm':
			acc_ax.set_ylabel('Proportion Correct', fontsize=22, labelpad=14)
			#acc_ax.set_xlabel('Prior Probability Cue', fontsize=22, labelpad=10)

	return yaccFace, yaccHouse
	
	
def predict_from_simdfs(data, simdfs, save=True, mname='EvT'):
	"""
	Arguments:

		data (pandas df):		pandas dataframe with the empirical data used
								to fit the model to generate the simulation parameters

		simdfs (pandas df):     pandas dataframe with multiple simulated datasets


	*plot behavioral data against model predictions from multiple simulations

	*If save=True, will save RT and ACC plots to working dir
	"""

	sns.set_style("white")

	if len(data.cue.unique())==3:
		x=np.array([1,2,3])
		code_type='HNL'
	else:
		x=np.array([1, 2, 3, 4, 5])
		code_type='AllP'

	face_acc, house_acc, face_rt, house_rt=parse.get_empirical_means(data=data, code_type=code_type)
	sem_list=parse.get_emp_SE(data, code_type)

	#init sep figure, axes for RT & ACC data
	fig_rt, ax_rt=plt.subplots(1)

	sns.despine()

	fig_acc, ax_acc=plt.subplots(1)

	sns.despine()

	fig_acc.subplots_adjust(top=0.9, left=0.15, right=0.88, bottom=0.15)
	fig_rt.subplots_adjust(top=0.9, left=0.15, right=0.88, bottom=0.15)

	flast_rt=np.zeros([5])
	hlast_rt=np.zeros([5])
	flast_acc=np.zeros([5])
	hlast_acc=np.zeros([5])

	for simn, rest in simdfs.groupby('sim_num'):

		Ftheo_acc, Htheo_acc = parse.get_theo_acc(simdf=rest, code_type=code_type)
		flast_acc, hlast_acc = pred_accPLOT(code_type=code_type, acc_ax=ax_acc, xacc=x, yaccFace=Ftheo_acc, yaccHouse=Htheo_acc, ind=simn, flast_acc=flast_acc, hlast_acc=hlast_acc, mname=mname)

		Ftheo_rt, Htheo_rt=parse.get_theo_rt(simdf=rest, code_type=code_type)
		flast_rt, hlast_rt = pred_rtPLOT(code_type=code_type, rt_ax=ax_rt, xrt=x, yrtFace=Ftheo_rt, yrtHouse=Htheo_rt, ind=simn, flast_rt=flast_rt, hlast_rt=hlast_rt, mname=mname)

	#plot empirical ACC
	f_emp_acc=ax_acc.errorbar(x, face_acc, yerr=sem_list[0], elinewidth=2.5, ecolor='k', color='Blue', lw=4.0)
	h_emp_acc=ax_acc.errorbar(x, house_acc, yerr=sem_list[1], elinewidth=2.5, ecolor='k', color='Red', lw=4.0)
	
	if mname=='pbm':
		ax_acc.legend((ax_acc.lines[-4], ax_acc.lines[-1], ax_acc.lines[0], ax_acc.lines[1]), ('Face Data', 'House Data', 'Face Model', 'House Model'), loc=0, fontsize=18)

	sns.despine()

	#plot empirical RT
	f_emp_rt=ax_rt.errorbar(x, face_rt, yerr=sem_list[2], elinewidth=2.5, ecolor='k', color='Blue', lw=4.0)
	h_emp_rt=ax_rt.errorbar(x, house_rt, yerr=sem_list[3], elinewidth=2.5, ecolor='k', color='Red', lw=4.0)
	
	if mname=='pbm':
		ax_rt.legend((ax_rt.lines[-4], ax_rt.lines[-1], ax_rt.lines[0], ax_rt.lines[1]), ('Face Data', 'House Data', 'Face Model', 'House Model'), loc=0, fontsize=18)

	sns.despine()

	flist=[fig_rt, fig_acc]

	if save:
		fig_rt.savefig(mname+'_rt.png', dpi=400)
		fig_acc.savefig(mname+'_acc.png', dpi=400)

def predict(params, data, simfx=sims.sim_exp, ntrials=160, pslow=0.0, pfast=0.0, nsims=100, nsims_per_sub=1, errors=False, save=False, RTname='RT_simexp', ACCname='Acc_simexp'):
	"""
	Arguments:

		params (dict):			hierarchical dictionary created with
								either parse_stats() or reformat_sims_input()

		data (pandas df):		pandas dataframe with the empirical data used
								to fit the model to generate the simulation parameters


	*Simulates subject-wise data using parameter estimates for each subj/condition
	 and calls rt and acc plotting functions to plot behavioral data against model predictions

	*If save=True, will save RT and ACC plots to working dir
	"""
	from myhddm import parse

	simdf_list=[]

	if len(data.cue.unique())==3:
		x=np.array([1,2,3])
		code_type='HNL'
	else:
		x=np.array([1, 2, 3, 4, 5])
		code_type='AllP'
		
	if errors:
		face_rt, house_rt=parse.get_emp_error_rt(data=data)
		face_acc, house_acc, cf, ch=parse.get_empirical_means(data=data, code_type=code_type)
	
	else:
		face_acc, house_acc, face_rt, house_rt=parse.get_empirical_means(data=data, code_type=code_type)
	
	face_rt_error, house_rt_error=parse.get_emp_error_rt(data=data)
	
	sem_list=parse.get_emp_SE(data, code_type)

	#init sep figure, axes for RT & ACC data
	fig_rt, ax_rt=plt.subplots(1)
	fig_acc, ax_acc=plt.subplots(1)

	fig_acc.subplots_adjust(top=0.9, left=0.15, right=0.88, bottom=0.15)
	fig_rt.subplots_adjust(top=0.9, left=0.15, right=0.88, bottom=0.15)

	flast_rt=np.zeros([5])
	hlast_rt=np.zeros([5])
	flast_acc=np.zeros([5])
	hlast_acc=np.zeros([5])

	for i in range(nsims):

		simdf, params_used=simfx(pdict=params, ntrials=ntrials, pfast=pfast, pslow=pslow, nsims_per_sub=nsims_per_sub)
		simdf['sim_n']=[i]*len(simdf.index)
		simdf_list.append(simdf)
		Ftheo_acc, Htheo_acc = parse.get_theo_acc(simdf=simdf, code_type=code_type)
		flast_acc, hlast_acc = pred_accPLOT(code_type=code_type, acc_ax=ax_acc, xacc=x, yaccFace=Ftheo_acc, yaccHouse=Htheo_acc, ind=i, flast_acc=flast_acc, hlast_acc=hlast_acc)
		
		if errors:
			Ftheo_rt, Htheo_rt=parse.get_theo_error_rt(simdf=simdf)
		else:
			Ftheo_rt, Htheo_rt=parse.get_theo_rt(simdf=simdf, code_type=code_type)
		
		flast_rt, hlast_rt = pred_rtPLOT(code_type=code_type, rt_ax=ax_rt, xrt=x, yrtFace=Ftheo_rt, yrtHouse=Htheo_rt, ind=i, flast_rt=flast_rt, hlast_rt=hlast_rt)

	simdf_concat=pd.concat(simdf_list)
	#plot empirical ACC
	#ax_acc.grid()
	#f_emp_acc=ax_acc.errorbar(x, face_acc, yerr=sem_list[0], elinewidth=3.5, ecolor='k', color='Blue', lw=6.0)
	#h_emp_acc=ax_acc.errorbar(x, house_acc, yerr=sem_list[1], elinewidth=3.5, ecolor='k', color='Green', lw=6.0)
	f_emp_acc=ax_acc.errorbar(x, face_acc, yerr=sem_list[0], elinewidth=3.5, ecolor='k', color='Blue', lw=6.0)
	h_emp_acc=ax_acc.errorbar(x, house_acc, yerr=sem_list[1], elinewidth=3.5, ecolor='k', color='Red', lw=6.0)
	#ax_acc.legend((ax_acc.lines[-4], ax_acc.lines[-1], ax_acc.lines[0], ax_acc.lines[1]), ('Face Data', 'House Data', 'Face Model', 'House Model'), loc=0, fontsize=18)

	#plot empirical RT
	#ax_rt.grid()
	#f_emp_rt=ax_rt.errorbar(x, face_rt, yerr=sem_list[2], elinewidth=3.5, ecolor='k', color='Blue', lw=6.0)
	#h_emp_rt=ax_rt.errorbar(x, house_rt, yerr=sem_list[3], elinewidth=3.5, ecolor='k', color='Green', lw=6.0)
	f_emp_rt=ax_rt.errorbar(x, face_rt, yerr=sem_list[2], elinewidth=3.5, ecolor='k', color='Blue', lw=6.0)
	h_emp_rt=ax_rt.errorbar(x, house_rt, yerr=sem_list[3], elinewidth=3.5, ecolor='k', color='Red', lw=6.0)
	#ax_rt.legend((ax_rt.lines[-4], ax_rt.lines[-1], ax_rt.lines[0], ax_rt.lines[1]), ('Face Data', 'House Data', 'Face Model', 'House Model'), loc=0, fontsize=18)

	simdf_concat.to_csv("simdf.csv")
	if save:
		fig_rt.savefig(RTname+'.jpeg', dpi=900)
		fig_acc.savefig(ACCname+'.jpeg', dpi=900)
		#fig_rt.savefig(RTname+'.png', format='png', dpi=500)
		#fig_acc.savefig(ACCname+'.png', format='png', dpi=500)
		#fig_rt.savefig(RTname+'.tif', format='tif', dpi=500)
		#fig_acc.savefig(ACCname+'.tif', fortmat='tif', dpi=500)
	return simdf_concat


def plot_data(data, save=True, RTname='RT_Data', ACCname='Acc_Data'):
	"""
	Arguments:

		params (dict):			hierarchical dictionary created with
								either parse_stats() or reformat_sims_input()

		data (pandas df):		pandas dataframe with the empirical data used
								to fit the model to generate the simulation parameters


	*Simulates subject-wise data using parameter estimates for each subj/condition
	 and calls rt and acc plotting functions to plot behavioral data against model predictions

	*If save=True, will save RT and ACC plots to working dir
	"""

	sns.set_style("white")
	#sns.despine()

	if len(data.cue.unique())==3:
		x=np.array([1,2,3])
		code_type='HNL'
	else:
		x=np.array([1, 2, 3, 4, 5])
		code_type='AllP'

	face_acc, house_acc, face_rt, house_rt=parse.get_empirical_means(data=data, code_type=code_type)
	sem_list=parse.get_emp_SE(data, code_type)

	#init sep figure, axes for RT & ACC data
	fig_rt, ax_rt=plt.subplots(1)
	sns.despine()
	fig_acc, ax_acc=plt.subplots(1)
	sns.despine()
	fig_acc.subplots_adjust(top=0.9, left=0.15, right=0.88, bottom=0.15)
	fig_rt.subplots_adjust(top=0.9, left=0.15, right=0.88, bottom=0.15)
	#fat
	#fig_rt.subplots_adjust(top=0.7, left=0.15, right=0.88, bottom=0.15)
	#plot empirical ACC
	f_emp_acc=ax_acc.errorbar(x, face_acc, yerr=sem_list[0], elinewidth=2.5, ecolor='k', color='Blue', lw=4.0)
	h_emp_acc=ax_acc.errorbar(x, house_acc, yerr=sem_list[1], elinewidth=2.5, ecolor='k', color='Red', lw=4.0)
	#ax_acc.set_title("Accuracy")
	ax_acc.legend((ax_acc.lines[-4], ax_acc.lines[-1]), ('Face', 'House'), loc=0, fontsize=18)

	#plot empirical RT
	f_emp_rt=ax_rt.errorbar(x, face_rt, yerr=sem_list[2], elinewidth=2.5, ecolor='k', color='Blue', lw=4.0)
	h_emp_rt=ax_rt.errorbar(x, house_rt, yerr=sem_list[3], elinewidth=2.5, ecolor='k', color='Red', lw=4.0)
	#ax_rt.set_title("Response-Time")
	ax_rt.legend((ax_rt.lines[-4], ax_rt.lines[-1]), ('Face', 'House'), loc=0, fontsize=18)

	if code_type=='HNL':
		ax_rt.set_ylim(1.5, 3.5)
		ax_rt.set_xticks([1, 2, 3])
		ax_rt.set_xlim(0.5, 3.5)
		ax_rt.set_xticklabels(['80H', '50N', '80F'], fontsize=32)
		ax_rt.set_ylabel('Response Time (s)', fontsize=35, labelpad=14)
		ax_rt.set_xlabel('Prior Probability Cue', fontsize=35, labelpad=10)
		ax_rt.set_yticklabels(np.arange(1.5, 4, 0.5), fontsize=25)
		ax_acc.set_ylim(0.7, 1.0)
		ax_acc.set_xticks([1, 2, 3])
		ax_acc.set_xlim(0.5, 3.5)
		ax_acc.set_xticklabels(['80H', '50N', '80F'], fontsize=32)
		ax_acc.set_ylabel('Proportion Correct', fontsize=35, labelpad=14)
		ax_acc.set_xlabel('Prior Probability Cue', fontsize=35, labelpad=10)
		ax_acc.set_yticks(np.arange(0.6, 1.05, .05))
		ax_acc.set_yticklabels(np.arange(0.6, 1.05, .05), fontsize=25)
	else:
		ax_rt.set_ylim(1.6, 3.5)
		ax_rt.set_xticks([1, 2, 3, 4, 5])
		ax_rt.set_xlim(0.6, 5.5)
		ax_rt.set_xticklabels(['90H', '70H', '50/50', '70F', '90F'], fontsize=18)
		ax_rt.set_yticks(np.arange(1.5, 4.0, 0.5))
		ax_rt.set_yticklabels(np.arange(1.5, 4.0, 0.5), fontsize=18)
		ax_rt.set_ylabel('Response Time (s)', fontsize=22, labelpad=14)
		ax_rt.set_xlabel('Prior Probability Cue', fontsize=22, labelpad=10)

		ax_acc.set_ylim(0.6, 1.0)
		ax_acc.set_xticks([1, 2, 3, 4, 5])
		ax_acc.set_xlim(0.5, 5.5)
		ax_acc.set_xticklabels(['90H', '70H', '50/50', '70F', '90F'], fontsize=18)
		ax_acc.set_yticks(np.arange(0.6, 1.05, .05))
		ax_acc.set_yticklabels(np.arange(0.6, 1.05, .05), fontsize=18)
		ax_acc.set_ylabel('Proportion Correct', fontsize=22, labelpad=14)
		ax_acc.set_xlabel('Prior Probability Cue', fontsize=22, labelpad=10)

	#save figures
	if save:
		fig_rt.savefig(RTname+'.png', dpi=600)
		fig_acc.savefig(ACCname+'.png', dpi=600)

if __name__ == "__main__":
	main()
