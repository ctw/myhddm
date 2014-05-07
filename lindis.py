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
	fdf=pd.read_csv("mds_face_coords_keep.csv")
	hdf=pd.read_csv("mds_house_coords_keep.csv")

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
	print "cat1 mean coordinates: %s, %s" % (str(lda.means_[0][0]), str(lda.means_[0][1]))
	print "cat2 mean coordinates: %s, %s" % (str(lda.means_[1][0]), str(lda.means_[1][1]))
	plt.plot(lda.means_[0][0], lda.means_[0][1],
	        'o', color='black', markersize=10)
	plt.plot(lda.means_[1][0], lda.means_[1][1],
	        'o', color='black', markersize=10)
	
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
	
	
	#if ax:
	#	ax=sns.regplot(data.ix[:,0], data.ix[:,1], color='Red', scatter=True, ci=None, scatter_kws={'s':18}, ax=ax)
	#else:
	#	ax=sns.regplot(data.ix[:,0], data.ix[:,1], color='Blue', scatter=True, ci=None, scatter_kws={'s':18})
	dataf=pd.read_csv("FaceEigen_RT_keep.csv")
	datah=pd.read_csv("HouseEigen_RT_keep.csv")
	data_all=pd.read_csv("StimEigen_RT_keep.csv")
	fig=plt.figure(figsize=(5, 6))
	ax=fig.add_subplot(111)
	ax=sns.regplot(data_all.ix[:,0], data_all.ix[:,1], color='Black', fit_reg=True, robust=True, scatter=True, ci=None, scatter_kws={'s':2}, ax=ax)
	ax=sns.regplot(datah.ix[:,0], datah.ix[:,1], color='Red', fit_reg=True, robust=True, scatter=True, ci=None, scatter_kws={'s':45}, ax=ax)
	ax=sns.regplot(dataf.ix[:,0], dataf.ix[:,1], color='Blue', fit_reg=True, robust=True, scatter=True, ci=None, scatter_kws={'s':45}, ax=ax)
	ax=sns.regplot(datah.ix[:,0], datah.ix[:,1], color='Red', fit_reg=True, robust=True, scatter=True, ci=None, scatter_kws={'s':45}, ax=ax)
	ax=sns.regplot(dataf.ix[:,0], dataf.ix[:,1], color='Blue', fit_reg=True, robust=True, scatter=True, ci=None, scatter_kws={'s':45}, ax=ax)
	ax=sns.regplot(datah.ix[:,0], datah.ix[:,1], color='Red', fit_reg=True, robust=True, scatter=True, ci=None, scatter_kws={'s':45}, ax=ax)
	ax=sns.regplot(dataf.ix[:,0], dataf.ix[:,1], color='Blue', fit_reg=True, robust=True, scatter=True, ci=None, scatter_kws={'s':45}, ax=ax)
	fig.set_tight_layout(True)
	fig.subplots_adjust(left=.22, bottom=.14, top=.95, right=.7)
	ax.set_ylim([-1,1])
	ax.set_xlim([2,14])
	ax.set_xticklabels(np.arange(2, 16, 2), fontsize=16)
	ax.set_xticklabels(np.arange(2, 16, 2), fontsize=12)
	ax.set_xlabel("Eigendistance to LDA Boundary", fontsize=16, labelpad=8)
	#plt.tight_layout()
	ax.set_ylabel("Normalized Response Time (s)", fontsize=16, labelpad=9)
	ax.set_yticklabels(np.arange(-1, 1.5, 0.5), fontsize=12)
	sns.despine()
	#plt.tight_layout(pad=2)
	#plt.subplots_adjust(left=.22, bottom=.14, top=.95, right=.7)
	plt.savefig(figname+".png", format='png', dpi=600)
	return fig,ax

