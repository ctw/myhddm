from scipy import linalg
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from sklearn.lda import LDA
from sklearn.qda import QDA
import pandas as pd
import seaborn as sns

# colormap
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'blue': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'red': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

def fh_lda():

	y=[];yf=[];yh=[];X=[];XF=[];XH=[]
	fdf=pd.read_csv("mds_face_coords.csv")
	hdf=pd.read_csv("mds_house_coords.csv")

	for i in fdf.index:
	    XF.append([fdf.ix[i, 0], fdf.ix[i, 1]])
	yf=[0]*len(XF)

	for i in hdf.index:
	    XH.append([hdf.ix[i, 0], hdf.ix[i, 1]])
	yh=[1]*len(XH)

	for i in XH:
	    XF.append(i)
	for i in yh:
	    yf.append(i)

	y=np.array(yf)
	X=np.array(XF)

	return X, y


# plot functions
def plot_LDA(lda, X, y, y_pred):
	#splot = plt.subplot(111)
	fig=plt.figure(figsize=(5, 6))
	ax=fig.add_subplot(111)
	#plt.title('Linear Discriminant Analysis')

	tp = (y == y_pred)  # True Positive
	tp0, tp1 = tp[y == 0], tp[y == 1]
	X0, X1 = X[y == 0], X[y == 1]
	X0_tp, X0_fp = X0[tp0], X0[~tp0]
	X1_tp, X1_fp = X1[tp1], X1[~tp1]
	xmin, xmax = X[:, 0].min(), X[:, 0].max()
	ymin, ymax = X[:, 1].min(), X[:, 1].max()

	# class 0: dots
	plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', ms=10, color='DarkBlue', alpha=0.6)
	plt.plot(X0_fp[:, 0], X0_fp[:, 1], 'o', ms=10, color='RoyalBlue', alpha=0.6)

	# class 1: dots
	plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', ms=10, color='FireBrick', alpha=0.6)
	plt.plot(X1_fp[:, 0], X1_fp[:, 1], 'o', ms=10, color='Crimson', alpha=0.6)

        plt.plot()

	ax=plt.gca()
	ax.set_xlim([-15000, 18000]); ax.set_ylim([-15000, 15000])

	# class 0 and 1 : areas
	nx, ny = 80, 40 #was 200, 100
	x_min, x_max = plt.xlim()
	y_min, y_max = plt.ylim()
	xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
	                     np.linspace(y_min, y_max, ny))
	zz=np.c_[xx.ravel(), yy.ravel()]
	Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
	Z = Z[:, 1].reshape(xx.shape)
	plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
	              norm=colors.Normalize(0., 1.))
	plt.contour(xx, yy, Z, [0.5], linewidths=3.5, colors='k')

	# means
	#print "cat1 mean coordinates: %s, %s" % (str(lda.means_[0][0]), str(lda.means_[0][1]))
	#print "cat2 mean coordinates: %s, %s" % (str(lda.means_[1][0]), str(lda.means_[1][1]))
	#plt.plot(lda.means_[0][0], lda.means_[0][1],
	#        'o', color='black', markersize=10)
	#plt.plot(lda.means_[1][0], lda.means_[1][1],
	#        'o', color='black', markersize=10)

        plt.plot(-7831.69871092267,-763.116931264117,
                'o', color='black', markersize=10, mec='Blue', mew=2)

        plt.plot(2296.02745742291, -306.329358115368,
                'o', color='black', markersize=10, mec='Red', mew=2)

	#ax=plt.gca()
	#ax.set_xlim([-15000, 18000]); ax.set_ylim([-15000, 15000])
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xlabel("Dimension 1", fontsize=16, labelpad=8)
	ax.set_ylabel("Dimension 2", fontsize=16, labelpad=10)
	plt.savefig("lda_keep.png", format='png', dpi=900)

	eigen_dist=pd.Series(lda.decision_function(X))
	eigen_dist.to_csv("eigen_distance_tobound_keep.csv", index=False)

	return ax


def main_lda():
	X,y=fh_lda()

	lda=LDA()
	lda.fit(X,y)

	splot=plot_LDA(lda, X, y, lda.fit(X,y).predict(X))
	return splot


def plot_correl(ax=None, figname="correlation_plot"):

	sns.set_style("white")
	sns.set_style("white", {"legend.scatterpoints": 1, "legend.frameon":Tru	
	#if ax:
	#    ax=sns.regplot(data.ix[:,0], data.ix[:,1], color='Red', scatter=True, ci=None, scatter_kws={'s':18}, ax=ax)
	#else:
	#    ax=sns.regplot(data.ix[:,0], data.ix[:,1], color='Blue', scatter=True, ci=None, scatter_kws={'s':1	
	dataf=pd.read_csv("FaceEigen_RT_keep.csv")
	datah=pd.read_csv("HouseEigen_RT_keep.csv")
	data_all=pd.read_csv("StimEigen_RT_keep.csv")	
	fig=plt.figure(figsize=(5, 6))
	ax=fig.add_subplot(1	
	axx=sns.regplot(data_all.ix[:,0], data_all.ix[:,1], color='Black', fit_reg=True, robust=True, label='All, r=.326**', scatter=True, ci=None, scatter_kws={'s':2}, ax=ax)
	axx=sns.regplot(datah.ix[:,0], datah.ix[:,1], color='Red', fit_reg=True, robust=True, scatter=True, ci=None, scatter_kws={'s':35}, ax=ax)
	axx=sns.regplot(dataf.ix[:,0], dataf.ix[:,1], color='Blue', fit_reg=True, robust=True, scatter=True, ci=None, scatter_kws={'s':35}, ax=ax)
	axx=sns.regplot(datah.ix[:,0], datah.ix[:,1], color='Red', fit_reg=True, robust=True, scatter=True, ci=None, scatter_kws={'s':35}, ax=ax)
	axx=sns.regplot(dataf.ix[:,0], dataf.ix[:,1], color='Blue', fit_reg=True, robust=True, scatter=True, ci=None, scatter_kws={'s':35}, ax=ax)
	axx=sns.regplot(dataf.ix[:,0], dataf.ix[:,1], color='Blue', fit_reg=True, robust=True, label='Face, r=.320*', scatter=True, ci=None, scatter_kws={'s':35}, ax=ax)
	axx=sns.regplot(datah.ix[:,0], datah.ix[:,1], color='Red', fit_reg=True, robust=True, label='House, r=.333*', scatter=True, ci=None, scatter_kws={'s':35}, ax=	
	fig.set_tight_layout(True)
	fig.subplots_adjust(left=.22, bottom=.14, top=.95, right=.7)
	ax.set_ylim([-1,1])
	ax.set_xlim([2,14])
	#ax.set_xticklabels(np.arange(2, 16, 2), fontsize=16)
	ax.set_xticklabels(np.arange(2, 16, 2), fontsize=10)
	ax.set_xlabel("Distance to Category Boundary", fontsize=12, labelpad	
	leg = ax.legend(loc='best', fancybox=True, fontsize=10)
	leg.get_frame().set_alpha(0.	
	#ax.legend(loc=0, fontsize=14)
	#plt.tight_layou	
	ax.set_ylabel("Response Time (s)", fontsize=12, labelpad=5)
	ax.set_yticklabels(np.arange(-1, 1.5, 0.5), fontsize=10)
	sns.despine()
	#plt.tight_layout(pad=2)
	#plt.subplots_adjust(left=.22, bottom=.14, top=.95, right=.7)
	plt.savefig(figname+".png", format='png', dpi=6	
	return fig, ax

def plot_correl_bycue(ax=None, figname="correlbycue_plot"):

	sns.set_style("white")
	sns.set_style("white", {"legend.scatterpoints": 1, "legend.frameon":True})

	df=pd.read_csv("/Users/kyle/Desktop/beh_hddm/MDS_Analysis/dist_RTxCue_allcor.csv")

	dataf=df[df['stim']=='face']
	datah=df[df['stim']=='house']

	fig=plt.figure(figsize=(10, 12))
	axf=fig.add_subplot(121)
	axh=fig.add_subplot(122)

	axx=sns.regplot(dataf['distance'], dataf['hcRT'], color='Red', fit_reg=True, robust=True, label='House Cue, r=-.19', scatter=True, ci=None, scatter_kws={'s':35}, ax=axf)
	axx=sns.regplot(dataf['distance'], dataf['ncRT'], color='Black', fit_reg=True, robust=False, label='Neutral Cue, r=-.15', scatter=True, ci=None, scatter_kws={'s':35}, ax=axf)
	axx=sns.regplot(dataf['distance'], dataf['fcRT'], color='Blue', fit_reg=True, robust=True, label='Face Cue, r=-.320*', scatter=True, ci=None, scatter_kws={'s':35}, ax=axf)

	axx=sns.regplot(datah['distance'], datah['hcRT'], color='Red', fit_reg=True, robust=True, label='House Cue, r=-.330*', scatter=True, ci=None, scatter_kws={'s':35}, ax=axh)
	axx=sns.regplot(datah['distance'], datah['ncRT'], color='Black', fit_reg=True, robust=True, label='Neutral Cue, r=-.18', scatter=True, ci=None, scatter_kws={'s':35}, ax=axh)
	axx=sns.regplot(datah['distance'], datah['fcRT'], color='Blue', fit_reg=True, robust=True, label='face Cue, r=-.09', scatter=True, ci=None, scatter_kws={'s':35}, ax=axh)

	#fig.set_tight_layout(True)
	#fig.subplots_adjust(left=.22, bottom=.14, top=.95, right=.7)
	for ax in fig.axes:
		ax.set_ylim([-1.2,1.2])
		ax.set_xlim([-5,18])
		#ax.set_xticklabels(np.arange(2, 16, 2), fontsize=16)
		#axf.set_xticklabels(np.arange(2, 16, 2), fontsize=10)
		ax.set_xlabel("Distance to Category Boundary", fontsize=12, labelpad=5)
	
		leg = ax.legend(loc='best', fancybox=True, fontsize=10)
		leg.get_frame().set_alpha(0.95)
	
		#ax.legend(loc=0, fontsize=14)
		#plt.tight_layout()
	
		ax.set_ylabel("Response Time (s)", fontsize=12, labelpad=5)
		#ax.set_yticklabels(np.arange(-1, 1.5, 0.5), fontsize=10)
		sns.despine()
		#plt.tight_layout(pad=2)
		#plt.subplots_adjust(left=.22, bottom=.14, top=.95, right=.7)
	
	plt.savefig(figname+".png", format='png', dpi=600)

	return fig


def plot_rho_heatmap():
	
	sns.set_style("white")
	pal=sns.blend_palette(['Darkred', 'Pink'], as_cmap=True)
	
	df=pd.read_csv("/Users/kyle/Desktop/beh_hddm/MDS_Analysis/dist_RTxCue_allcor.csv")
	
	dataf=df[df['stim']=='face']
	datah=df[df['stim']=='house']

	fhc=dataf['distance'].corr(dataf['hcRT'], method='spearman')
	fnc=dataf['distance'].corr(dataf['ncRT'], method='spearman')
	ffc=dataf['distance'].corr(dataf['fcRT'], method='spearman')
	hhc=datah['distance'].corr(datah['hcRT'], method='spearman')
	hnc=datah['distance'].corr(datah['ncRT'], method='spearman')
	hfc=datah['distance'].corr(datah['fcRT'], method='spearman')
	
	fcorr=np.array([fhc, fnc, ffc])
	hcorr=np.array([hhc, hnc, hfc])
	
	corr_matrix=np.array([fcorr, hcorr])
	
	fig=plt.figure(figsize=(10,8))
	fig.set_tight_layout(True)	
	
	ax=fig.add_subplot(111)
	
	fig.subplots_adjust(top=.95, hspace=.1, left=0.10, right=.9, bottom=0.1)

	ax.set_ylim(-0.5, 1.5)
	ax.set_yticks([0, 1])
	ax.set_yticklabels(['Face', 'House'], fontsize=24)
	plt.setp(ax.get_yticklabels(), rotation=90)
	ax.set_ylabel("Stimulus", fontsize=28, labelpad=8)
	ax.set_xlim(-0.5, 2.5)
	ax.set_xticks([0, 1, 2])
	ax.set_xticklabels(['House', 'Neutral', 'Face'], fontsize=24)
	ax.set_xlabel("Cue Type", fontsize=28, labelpad=8)
	ax_map=ax.imshow(corr_matrix, interpolation='nearest', cmap=pal, origin='lower', vmin=-0.40, vmax=0)
	plt.colorbar(ax_map, ax=ax, shrink=0.65)
	
	for i, cond in enumerate(corr_matrix):
		x=0
		for xval in cond:
			if -.35<xval<=-.30:
				ax.text(x, i, "r="+str(xval)[:5]+"*", ha='center', va='center', fontsize=29)
			elif xval<-.35:
				ax.text(x, i, "r="+str(xval)[:5]+"**", ha='center', va='center', fontsize=29)
			else:
				ax.text(x, i, "r="+str(xval)[:5], ha='center', va='center', fontsize=22)
			x+=1
	
	plt.savefig('corr.png', format='png', dpi=600)		