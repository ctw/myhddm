#!/usr/bin/env python

#TODO: Now apply subj-estimated EWMA cutoffs to eliminate fast outliers
from __future__ import division
import pandas as pd
import numpy as np

def cutoff_sd(data, sd=2.0):
	"""
	"Removes trials where RT is higher than 2SD above the mean for that subject/cue"
	
	::Arguments::
		
		data (pandas df)		pandas df containing typical HDDM 
								input format for hierarchical model
								NOTE: data input needs accuracy column
		
		sd (float)				stdev cutoff value, default is 2.0 (keep ~95%)
		 						NOTE: Ratcliff advises stdev cutoff at 1, 1.5, or 2
								
	::Returns::
	
		cleandf (pandas df)		pandas df with slow outlier removed removed
	
	"""
	
	grpdf=data.groupby(['subj_idx', 'cue', 'stim', 'acc'])
	
	#counter to make sure that 
	#cleandf is only initialized once, 
	#for sub1's first cue
	i=1
	
	for x, rest in grpdf:
		cutoff=rest['rt'].std()*sd + (rest['rt'].mean())
		
		if x[0]==1 and i==1:
			cleandf=data.ix[ (data['subj_idx']==x[0]) & (data['cue']==x[1]) & (data['stim']==x[2]) & (data['acc']==x[3]) & ( data['rt']<cutoff) ]
		
		else:
			#just call it something else
			#and append it to cleandf
			othersubs=data.ix[ (data['subj_idx']==x[0]) & (data['cue']==x[1]) & (data['stim']==x[2]) & (data['acc']==x[3]) & ( data['rt']<cutoff) ]
			
			cleandf=pd.concat([cleandf, othersubs], ignore_index=True)
		
		i+=1
	
	return cleandf
	
	
	
