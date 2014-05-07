#!/usr/bin/env python
from scipy.stats.mstats import mquantiles
from my_plots import diff_traces



def get_nodes(model, type='vz'):
	
	if type=='vz':
		v90Hface, v70Hface, v50Nface, v70Fface, v90Fface=model.nodes_db.node[['v(a90H.face)', 'v(b70H.face)', 'v(c50N.face)', 'v(d70F.face)', 'v(e90F.face)']] 
		v90Hhouse, v70Hhouse, v50Nhouse, v70Fhouse, v90Fhouse=model.nodes_db.node[['v(a90H.house)', 'v(b70H.house)', 'v(c50N.house)', 'v(d70F.house)', 'v(e90F.house)']] 
		z90H, z70H, z50N, z70F, z90F = model.nodes_db.node[['z(a90H)', 'z(b70H)', 'z(c50N)', 'z(d70F)', 'z(e90F)']]
		vFace=[v90Hface, v70Hface, v50Nface, v70Fface, v90Fface]
		vHouse=[v90Hhouse, v70Hhouse, v50Nhouse, v70Fhouse, v90Fhouse]
		z=[z90H, z70H, z50N, z70F, z90F]
		vz_nodes={'vF':vFace, 'vH':vHouse, 'z':z}
		return vz_nodes
	elif type=='v':
		v90Hface, v70Hface, v50Nface, v70Fface, v90Fface=model.nodes_db.node[['v(a90H.face)', 'v(b70H.face)', 'v(c50N.face)', 'v(d70F.face)', 'v(e90F.face)']] 
		v90Hhouse, v70Hhouse, v50Nhouse, v70Fhouse, v90Fhouse=model.nodes_db.node[['v(a90H.house)', 'v(b70H.house)', 'v(c50N.house)', 'v(d70F.house)', 'v(e90F.house)']] 
		vFace=[v90Hface, v70Hface, v50Nface, v70Fface, v90Fface]
		vHouse=[v90Hhouse, v70Hhouse, v50Nhouse, v70Fhouse, v90Fhouse]
		v_nodes={'vFace':vFace, 'vHouse':vHouse}
		return v_nodes
	elif type=='z':
		z90H, z70H, z50N, z70F, z90F = model.nodes_db.node[['z(a90H)', 'z(b70H)', 'z(c50N)', 'z(d70F)', 'z(e90F)']]
		z=[z90H, z70H, z50N, z70F, z90F]
		return z
	else:
		print 'Did not recognize type (v+z not supported, just use "v")'
		

def vz_credible(model):
	"""Prints node summary out to screen"""
	
	v90Hface, v70Hface, v50Nface, v70Fface, v90Fface=model.nodes_db.node[['v(a90H.face)', 'v(b70H.face)', 'v(c50N.face)', 'v(d70F.face)', 'v(e90F.face)']] 
	v90Hhouse, v70Hhouse, v50Nhouse, v70Fhouse, v90Fhouse=model.nodes_db.node[['v(a90H.house)', 'v(b70H.house)', 'v(c50N.house)', 'v(d70F.house)', 'v(e90F.house)']] 
	z90H, z70H, z50N, z70F, z90F = model.nodes_db.node[['z(a90H)', 'z(b70H)', 'z(c50N)', 'z(d70F)', 'z(e90F)']]
	
	#vF_list=[v90Hface, v70Hface, v50Nface, v70Fface, v90Fface]
	#vH_list=[v90Hhouse, v70Hhouse, v50Nhouse, v70Fhouse, v90Fhouse]
	#z_list=[z90H, z70H, z50N, z70F, z90F]
	
	all_list=[v90Hface, v70Hface, v50Nface, v70Fface, v90Fface, v90Hhouse, v70Hhouse, v50Nhouse, v70Fhouse, v90Fhouse, z90H, z70H, z50N, z70F, z90F]

	for node in all_list:
		#n_quant=mquantiles(node, prob=[0.025, 0.975])
		node.summary()
	
		
	
def allp_vz_htest(model):

	v90Hface, v70Hface, v50Nface, v70Fface, v90Fface=model.nodes_db.node[['v(a90H.face)', 'v(b70H.face)', 'v(c50N.face)', 'v(d70F.face)', 'v(e90F.face)']] 
	v90Hhouse, v70Hhouse, v50Nhouse, v70Fhouse, v90Fhouse=model.nodes_db.node[['v(a90H.house)', 'v(b70H.house)', 'v(c50N.house)', 'v(d70F.house)', 'v(e90F.house)']] 
	z90H, z70H, z50N, z70F, z90F = model.nodes_db.node[['z(a90H)', 'z(b70H)', 'z(c50N)', 'z(d70F)', 'z(e90F)']]

	fout=open('hypothesis_tests.txt', 'w')

	print>>fout, "Comparing drift-rates (v) for FACE trials across probability cues:"
	print>>fout, "\n"	
	print>>fout, "p(v90H.face < v90F.face) = %s" % str((v90Hface.trace() < v90Fface.trace()).mean())
	print>>fout, "p(v70F.face < v90F.face) = %s" % str((v70Fface.trace() < v90Fface.trace()).mean())
	print>>fout, "p(v50N.face < v90F.face) = %s" % str((v50Nface.trace() < v90Fface.trace()).mean())
	print>>fout, "p(v90H.face < v70F.face) = %s" % str((v90Hface.trace() < v70Fface.trace()).mean())
	print>>fout, "p(v70H.face < v70F.face) = %s" % str((v70Hface.trace() < v70Fface.trace()).mean())
	print>>fout, "p(v50N.face < v70F.face) = %s" % str((v50Nface.trace() < v70Fface.trace()).mean())
	print>>fout, "p(v70H.face < v50N.face) = %s" % str((v70Hface.trace() < v50Nface.trace()).mean())
	print>>fout, "p(v90H.face < v50N.face) = %s" % str((v90Hface.trace() < v50Nface.trace()).mean())
	print>>fout, "p(v90H.face < v70H.face) = %s" % str((v90Hface.trace() < v70Hface.trace()).mean())
	print>>fout, "\n\n\n"
	
	print>>fout, "Comparing drift-rates (v) for HOUSE trials across probability cues:/n/n"
	print>>fout, "\n"
	#probability that left side of inequality is greater in magnitude
	#all values will be negative so the actual inequality test is opposite than
	#what is printed.  Just remember, that ">" is referring to absolute magnitude
	#when comparing drift rates for house trials across cue conditions
	print>>fout, "p(v90H.house > v90F.house) = %s" % str((v90Hhouse.trace() < v90Fhouse.trace()).mean())
	print>>fout, "p(v70F.house > v90F.house) = %s" % str((v70Fhouse.trace() < v90Fhouse.trace()).mean())
	print>>fout, "p(v50N.house > v90F.house) = %s" % str((v50Nhouse.trace() < v90Fhouse.trace()).mean())
	print>>fout, "p(v90H.house > v70F.house) = %s" % str((v90Hhouse.trace() < v70Fhouse.trace()).mean())
	print>>fout, "p(v70H.house > v70F.house) = %s" % str((v70Hhouse.trace() < v70Fhouse.trace()).mean())
	print>>fout, "p(v50N.house > v70F.house) = %s" % str((v50Nhouse.trace() < v70Fhouse.trace()).mean())
	print>>fout, "p(v70H.house > v50N.house) = %s" % str((v70Hhouse.trace() < v50Nhouse.trace()).mean())
	print>>fout, "p(v90H.house > v50N.house) = %s" % str((v90Hhouse.trace() < v50Nhouse.trace()).mean())
	print>>fout, "p(v90H.house > v70H.house) = %s" % str((v90Hhouse.trace() < v70Hhouse.trace()).mean())	
	print>>fout, "\n\n\n"
	
	print>>fout, "Comparing starting-points (z0) across probability cues:"
	print>>fout, "\n"
	print>>fout, "p(z90H < z90F) = %s" % str((z90H.trace() < z90F.trace()).mean())
	print>>fout, "p(z70F < z90F) = %s" % str((z70F.trace() < z90F.trace()).mean())
	print>>fout, "p(z50N < z90F) = %s" % str((z50N.trace() < z90F.trace()).mean())
	print>>fout, "p(z90H < z70F) = %s" % str((z90H.trace() < z70F.trace()).mean())
	print>>fout, "p(z70H < z70F) = %s" % str((z70H.trace() < z70F.trace()).mean())
	print>>fout, "p(z50N < z70F) = %s" % str((z50N.trace() < z70F.trace()).mean())
	print>>fout, "p(z70H < z50N) = %s" % str((z70H.trace() < z50N.trace()).mean())
	print>>fout, "p(z90H < z50N) = %s" % str((z90H.trace() < z50N.trace()).mean())
	print>>fout, "p(z90H < z70H) = %s" % str((z90H.trace() < z70H.trace()).mean())	
	print>>fout, "\n\n\n"
	
	print>>fout, "2.5 and 97.5 '%' QUANTILES for difference between drift-rates (v) for FACE trials across probability cues:"
	print>>fout, "\n"	
	print>>fout, "v90H.face - v90F.face = %s" % str(mquantiles((v90Hface.trace() - v90Fface.trace()), prob=[0.025, 0.975]))
	print>>fout, "v70F.face - v90F.face = %s" % str(mquantiles((v70Fface.trace() - v90Fface.trace()), prob=[0.025, 0.975]))
	print>>fout, "v50N.face - v90F.face = %s" % str(mquantiles((v50Nface.trace() - v90Fface.trace()), prob=[0.025, 0.975]))
	print>>fout, "v90H.face - v70F.face = %s" % str(mquantiles((v90Hface.trace() - v70Fface.trace()), prob=[0.025, 0.975]))
	print>>fout, "v70H.face - v70F.face = %s" % str(mquantiles((v70Hface.trace() - v70Fface.trace()), prob=[0.025, 0.975]))
	print>>fout, "v50N.face - v70F.face = %s" % str(mquantiles((v50Nface.trace() - v70Fface.trace()), prob=[0.025, 0.975]))
	print>>fout, "v70H.face - v50N.face = %s" % str(mquantiles((v70Hface.trace() - v50Nface.trace()), prob=[0.025, 0.975]))
	print>>fout, "v90H.face - v50N.face = %s" % str(mquantiles((v90Hface.trace() - v50Nface.trace()), prob=[0.025, 0.975]))
	print>>fout, "v90H.face - v70H.face = %s" % str(mquantiles((v90Hface.trace() - v70Hface.trace()), prob=[0.025, 0.975]))
	print>>fout, "\n\n\n"
	
	
	print>>fout, "2.5 and 97.5 '%' QUANTILES for difference between drift-rates (v) for HOUSE trials across probability cues:/n/n"
	print>>fout, "\n"
	print>>fout, "v90H.house - v90F.house = %s" % str(mquantiles((v90Hhouse.trace() - v90Fhouse.trace()), prob=[0.025, 0.975]))
	print>>fout, "v70F.house - v90F.house = %s" % str(mquantiles((v70Fhouse.trace() - v90Fhouse.trace()), prob=[0.025, 0.975]))
	print>>fout, "v50N.house - v90F.house = %s" % str(mquantiles((v50Nhouse.trace() - v90Fhouse.trace()), prob=[0.025, 0.975]))
	print>>fout, "v90H.house - v70F.house = %s" % str(mquantiles((v90Hhouse.trace() - v70Fhouse.trace()), prob=[0.025, 0.975]))
	print>>fout, "v70H.house - v70F.house = %s" % str(mquantiles((v70Hhouse.trace() - v70Fhouse.trace()), prob=[0.025, 0.975]))
	print>>fout, "v50N.house - v70F.house = %s" % str(mquantiles((v50Nhouse.trace() - v70Fhouse.trace()), prob=[0.025, 0.975]))
	print>>fout, "v70H.house - v50N.house = %s" % str(mquantiles((v70Hhouse.trace() - v50Nhouse.trace()), prob=[0.025, 0.975]))
	print>>fout, "v90H.house - v50N.house = %s" % str(mquantiles((v90Hhouse.trace() - v50Nhouse.trace()), prob=[0.025, 0.975]))
	print>>fout, "v90H.house - v70H.house = %s" % str(mquantiles((v90Hhouse.trace() - v70Hhouse.trace()), prob=[0.025, 0.975]))
	print>>fout, "\n\n\n"
	
	print>>fout, "2.5 and 97.5 '%' QUANTILES for difference between starting-points (z0) across probability cues:"
	print>>fout, "\n"
	print>>fout, "z90H - z90F = %s" % str(mquantiles((z90H.trace() - z90F.trace()), prob=[0.025, 0.975]))
	print>>fout, "z70F - z90F = %s" % str(mquantiles((z70F.trace() - z90F.trace()), prob=[0.025, 0.975]))
	print>>fout, "z50N - z90F = %s" % str(mquantiles((z50N.trace() - z90F.trace()), prob=[0.025, 0.975]))
	print>>fout, "z90H - z70F = %s" % str(mquantiles((z90H.trace() - z70F.trace()), prob=[0.025, 0.975]))
	print>>fout, "z70H - z70F = %s" % str(mquantiles((z70H.trace() - z70F.trace()), prob=[0.025, 0.975]))
	print>>fout, "z50N - z70F = %s" % str(mquantiles((z50N.trace() - z70F.trace()), prob=[0.025, 0.975]))
	print>>fout, "z70H - z50N = %s" % str(mquantiles((z70H.trace() - z50N.trace()), prob=[0.025, 0.975]))
	print>>fout, "z90H - z50N = %s" % str(mquantiles((z90H.trace() - z50N.trace()), prob=[0.025, 0.975]))
	print>>fout, "z90H - z70H = %s" % str(mquantiles((z90H.trace() - z70H.trace()), prob=[0.025, 0.975]))
	print>>fout, "\n\n\n"
	
	vF, vH, z=diff_traces(model)
	print>>fout, "Face Drift Rates"
	print>>fout, "\n"
	for dtrace in vF:
		print>>fout,"p(node_difference > 0)=%s" % str((dtrace>0).mean())
		print>>fout,"p(node_difference < 0)=%s" % str((dtrace<0).mean())
		print>>fout, "\n\n"
	
	print>>fout, "House Drift Rates"	
	print>>fout, "\n"	
	for dtrace in vH:
		print>>fout,"p(node_difference > 0)=%s" % str((dtrace>0).mean())
		print>>fout,"p(node_difference < 0)=%s" % str((dtrace<0).mean())
		print>>fout, "\n\n"
	
	print>>fout, "Starting-Points"	
	print>>fout, "\n"	
	for dtrace in z:
		print>>fout,"p(node_difference > 0)=%s" % str((dtrace>0).mean())
		print>>fout,"p(node_difference < 0)=%s" % str((dtrace<0).mean())
		print>>fout, "\n\n"
	
	fout.close()



def hnl_vz_htest(model):
	
	v80Hface, v50Nface, v80Fface=model.nodes_db.node[['v(a80H.face)', 'v(b50N.face)','v(c80F.face)']] 
	v80Hhouse, v50Nhouse, v80Fhouse=model.nodes_db.node[['v(a80H.house)', 'v(b50N.house)', 'v(c80F.house)']]
	z80H, z50N, z80F = model.nodes_db.node[['z(a80H)', 'z(b50N)', 'z(c80F)']]
	
	fout=open('hypothesis_tests.txt', 'w')
	
	print>>fout, "Comparing drift-rates (v) for FACE trials across probability cues:"
	print>>fout, "\n"
	print>>fout, "p(v80H.face < v80F.face) = %s" % str((v80Hface.trace() < v80Fface.trace()).mean())
	print>>fout, "p(v50N.face < v80F.face) = %s" % str((v50Nface.trace() < v80Fface.trace()).mean())
	print>>fout, "p(v80H.face < v50N.face) = %s" % str((v80Hface.trace() < v50Nface.trace()).mean())
	print>>fout, "\n\n\n"
	
	print>>fout, "Comparing drift-rates (v) for HOUSE trials across probability cues:"
	print>>fout, "\n"
	#probability that left side of inequality is greater in magnitude
	#all values will be negative so the actual inequality test is opposite than
	#what is printed.  Just remember, that ">" is referring to absolute magnitude
	#when comparing drift rates for house trials across cue conditions
	print>>fout, "p(v80H.house > v80F.house) = %s" % str((v80Hhouse.trace() < v80Fhouse.trace()).mean())
	print>>fout, "p(v50N.house > v80F.house) = %s" % str((v50Nhouse.trace() < v80Fhouse.trace()).mean())
	print>>fout, "p(v80H.house > v50N.house) = %s" % str((v80Hhouse.trace() < v50Nhouse.trace()).mean())
	print>>fout, "\n\n\n"
	
	print>>fout, "Comparing starting-points (z0) across probability cues:"
	print>>fout, "\n"
	print>>fout, "p(z80H < z80F) = %s" % str((z80H.trace() < z80F.trace()).mean())
	print>>fout, "p(z50N < z80F) = %s" % str((z50N.trace() < z80F.trace()).mean())
	print>>fout, "p(z80H < z50N) = %s" % str((z80H.trace() < z50N.trace()).mean())
	print>>fout, "\n\n\n"

	fout.close()

def allp_v_htest(model):


	v90Hface, v70Hface, v50Nface, v70Fface, v90Fface=model.nodes_db.node[['v(a90H.face)', 'v(b70H.face)', 'v(c50N.face)', 'v(d70F.face)', 'v(e90F.face)']] 
	v90Hhouse, v70Hhouse, v50Nhouse, v70Fhouse, v90Fhouse=model.nodes_db.node[['v(a90H.house)', 'v(b70H.house)', 'v(c50N.house)', 'v(d70F.house)', 'v(e90F.house)']] 

	fout=open('hypothesis_tests.txt', 'w')

	print>>fout, "Comparing drift-rates (v) for FACE trials across probability cues:"
	print>>fout, "\n"	
	print>>fout, "p(v90H.face < v90F.face) = %s" % str((v90Hface.trace() < v90Fface.trace()).mean())
	print>>fout, "p(v70F.face < v90F.face) = %s" % str((v70Fface.trace() < v90Fface.trace()).mean())
	print>>fout, "p(v50N.face < v90F.face) = %s" % str((v50Nface.trace() < v90Fface.trace()).mean())
	print>>fout, "p(v90H.face < v70F.face) = %s" % str((v90Hface.trace() < v70Fface.trace()).mean())
	print>>fout, "p(v70H.face < v70F.face) = %s" % str((v70Hface.trace() < v70Fface.trace()).mean())
	print>>fout, "p(v50N.face < v70F.face) = %s" % str((v50Nface.trace() < v70Fface.trace()).mean())
	print>>fout, "p(v70H.face < v50N.face) = %s" % str((v70Hface.trace() < v50Nface.trace()).mean())
	print>>fout, "p(v90H.face < v50N.face) = %s" % str((v90Hface.trace() < v50Nface.trace()).mean())
	print>>fout, "p(v90H.face < v70H.face) = %s" % str((v90Hface.trace() < v70Hface.trace()).mean())
	print>>fout, "\n\n\n"
	
	print>>fout, "Comparing drift-rates (v) for HOUSE trials across probability cues:/n/n"
	print>>fout, "\n"
	#probability that left side of inequality is greater in magnitude
	#all values will be negative so the actual inequality test is opposite than
	#what is printed.  Just remember, that ">" is referring to absolute magnitude
	#when comparing drift rates for house trials across cue conditions
	print>>fout, "p(v90H.house > v90F.house) = %s" % str((v90Hhouse.trace() < v90Fhouse.trace()).mean())
	print>>fout, "p(v70F.house > v90F.house) = %s" % str((v70Fhouse.trace() < v90Fhouse.trace()).mean())
	print>>fout, "p(v50N.house > v90F.house) = %s" % str((v50Nhouse.trace() < v90Fhouse.trace()).mean())
	print>>fout, "p(v90H.house > v70F.house) = %s" % str((v90Hhouse.trace() < v70Fhouse.trace()).mean())
	print>>fout, "p(v70H.house > v70F.house) = %s" % str((v70Hhouse.trace() < v70Fhouse.trace()).mean())
	print>>fout, "p(v50N.house > v70F.house) = %s" % str((v50Nhouse.trace() < v70Fhouse.trace()).mean())
	print>>fout, "p(v70H.house > v50N.house) = %s" % str((v70Hhouse.trace() < v50Nhouse.trace()).mean())
	print>>fout, "p(v90H.house > v50N.house) = %s" % str((v90Hhouse.trace() < v50Nhouse.trace()).mean())
	print>>fout, "p(v90H.house > v70H.house) = %s" % str((v90Hhouse.trace() < v70Hhouse.trace()).mean())	
	print>>fout, "\n\n\n"

	fout.close()

def hnl_v_htest(model):

	v80Hface, v50Nface, v80Fface=model.nodes_db.node[['v(a80H.face)', 'v(b50N.face)','v(c80F.face)']] 
	v80Hhouse, v50Nhouse, v80Fhouse=model.nodes_db.node[['v(a80H.house)', 'v(b50N.house)', 'v(c80F.house)']]

	fout=open('hypothesis_tests.txt', 'w')

	print>>fout, "Comparing drift-rates (v) for FACE trials across probability cues:"
	print>>fout, "\n"	
	print>>fout, "p(v80H.face < v80F.face) = %s" % str((v80Hface.trace() < v80Fface.trace()).mean())
	print>>fout, "p(v50N.face < v80F.face) = %s" % str((v50Nface.trace() < v80Fface.trace()).mean())
	print>>fout, "p(v80H.face < v50N.face) = %s" % str((v80Hface.trace() < v50Nface.trace()).mean())
	print>>fout, "\n\n\n"	
	
	#probability that left side of inequality is greater in magnitude
	#all values will be negative so the actual inequality test is opposite than
	#what is printed.  Just remember, that ">" is referring to absolute magnitude
	#when comparing drift rates for house trials across cue conditions
	print>>fout, "Comparing drift-rates (v) for HOUSE trials across probability cues:"
	print>>fout, "\n"
	print>>fout, "p(v80H.house > v80F.house) = %s" % str((v80Hhouse.trace() < v80Fhouse.trace()).mean())
	print>>fout, "p(v50N.house > v80F.house) = %s" % str((v50Nhouse.trace() < v80Fhouse.trace()).mean())
	print>>fout, "p(v80H.house > v50N.house) = %s" % str((v80Hhouse.trace() < v50Nhouse.trace()).mean())
	print>>fout, "\n\n\n"
	
	fout.close()


def allp_z_htest(model):

	z90H, z70H, z50N, z70F, z90F = model.nodes_db.node[['z(a90H)', 'z(b70H)', 'z(c50N)', 'z(d70F)', 'z(e90F)']]

	fout=open('hypothesis_tests.txt', 'w')

	print>>fout, "Comparing starting-points (z0) across probability cues:"
	print>>fout, "\n"
	print>>fout, "p(z90H < z90F) = %s" % str((z90H.trace() < z90F.trace()).mean())
	print>>fout, "p(z70F < z90F) = %s" % str((z70F.trace() < z90F.trace()).mean())
	print>>fout, "p(z50N < z90F) = %s" % str((z50N.trace() < z90F.trace()).mean())
	print>>fout, "p(z90H < z70F) = %s" % str((z90H.trace() < z70F.trace()).mean())
	print>>fout, "p(z70H < z70F) = %s" % str((z70H.trace() < z70F.trace()).mean())
	print>>fout, "p(z50N < z70F) = %s" % str((z50N.trace() < z70F.trace()).mean())
	print>>fout, "p(z70H < z50N) = %s" % str((z70H.trace() < z50N.trace()).mean())
	print>>fout, "p(z90H < z50N) = %s" % str((z90H.trace() < z50N.trace()).mean())
	print>>fout, "p(z90H < z70H) = %s" % str((z90H.trace() < z70H.trace()).mean())	
	print>>fout, "\n\n\n"

	fout.close()

def hnl_z_htest(model):

	z80H, z50N, z80F = model.nodes_db.node[['z(a80H)', 'z(b50N)', 'z(c80F)']]

	fout=open('hypothesis_tests.txt', 'w')

	print>>fout, "Comparing starting-points (z0) across probability cues:"
	print>>fout, "\n"	
	print>>fout, "p(z80H < z80F) = %s" % str((z80H.trace() < z80F.trace()).mean())
	print>>fout, "p(z50N < z80F) = %s" % str((z50N.trace() < z80F.trace()).mean())
	print>>fout, "p(z80H < z50N) = %s" % str((z80H.trace() < z50N.trace()).mean())
	print>>fout, "\n\n\n"
	

	fout.close()

def TEST_FUNCTION(model):

	v90Hface, v70Hface, v50Nface, v70Fface, v90Fface=model.nodes_db.node[['v(90H.face)', 'v(70H.face)', 'v(50N.face)', 'v(70F.face)', 'v(90F.face)']] 
	v90Hhouse, v70Hhouse, v50Nhouse, v70Fhouse, v90Fhouse=model.nodes_db.node[['v(90H.house)', 'v(70H.house)', 'v(50N.house)', 'v(70F.house)', 'v(90F.house)']] 
	z90H, z70H, z50N, z70F, z90F = model.nodes_db.node[['z(90H)', 'z(70H)', 'z(50N)', 'z(70F)', 'z(90F)']]

	fout=open('hypothesis_tests.txt', 'w')

	print>>fout, "Comparing drift-rates (v) for FACE trials across probability cues:"
	print>>fout, "\n"	
	print>>fout, "p(v90H.face < v90F.face) = %s" % str((v90Hface.trace() < v90Fface.trace()).mean())
	print>>fout, "p(v70F.face < v90F.face) = %s" % str((v70Fface.trace() < v90Fface.trace()).mean())
	print>>fout, "p(v50N.face < v90F.face) = %s" % str((v50Nface.trace() < v90Fface.trace()).mean())
	print>>fout, "p(v90H.face < v70F.face) = %s" % str((v90Hface.trace() < v70Fface.trace()).mean())
	print>>fout, "p(v70H.face < v70F.face) = %s" % str((v70Hface.trace() < v70Fface.trace()).mean())
	print>>fout, "p(v50N.face < v70F.face) = %s" % str((v50Nface.trace() < v70Fface.trace()).mean())
	print>>fout, "p(v70H.face < v50N.face) = %s" % str((v70Hface.trace() < v50Nface.trace()).mean())
	print>>fout, "p(v90H.face < v50N.face) = %s" % str((v90Hface.trace() < v50Nface.trace()).mean())
	print>>fout, "p(v90H.face < v70H.face) = %s" % str((v90Hface.trace() < v70Hface.trace()).mean())
	print>>fout, "\n\n\n"
	
	print>>fout, "Comparing drift-rates (v) for HOUSE trials across probability cues:/n/n"
	print>>fout, "\n"
	#probability that left side of inequality is greater in magnitude
	#all values will be negative so the actual inequality test is opposite than
	#what is printed.  Just remember, that ">" is referring to absolute magnitude
	#when comparing drift rates for house trials across cue conditions
	print>>fout, "p(v90H.house > v90F.house) = %s" % str((v90Hhouse.trace() < v90Fhouse.trace()).mean())
	print>>fout, "p(v70F.house > v90F.house) = %s" % str((v70Fhouse.trace() < v90Fhouse.trace()).mean())
	print>>fout, "p(v50N.house > v90F.house) = %s" % str((v50Nhouse.trace() < v90Fhouse.trace()).mean())
	print>>fout, "p(v90H.house > v70F.house) = %s" % str((v90Hhouse.trace() < v70Fhouse.trace()).mean())
	print>>fout, "p(v70H.house > v70F.house) = %s" % str((v70Hhouse.trace() < v70Fhouse.trace()).mean())
	print>>fout, "p(v50N.house > v70F.house) = %s" % str((v50Nhouse.trace() < v70Fhouse.trace()).mean())
	print>>fout, "p(v70H.house > v50N.house) = %s" % str((v70Hhouse.trace() < v50Nhouse.trace()).mean())
	print>>fout, "p(v90H.house > v50N.house) = %s" % str((v90Hhouse.trace() < v50Nhouse.trace()).mean())
	print>>fout, "p(v90H.house > v70H.house) = %s" % str((v90Hhouse.trace() < v70Hhouse.trace()).mean())	
	print>>fout, "\n\n\n"
	
	print>>fout, "Comparing starting-points (z0) across probability cues:"
	print>>fout, "\n"
	print>>fout, "p(z90H < z90F) = %s" % str((z90H.trace() < z90F.trace()).mean())
	print>>fout, "p(z70F < z90F) = %s" % str((z70F.trace() < z90F.trace()).mean())
	print>>fout, "p(z50N < z90F) = %s" % str((z50N.trace() < z90F.trace()).mean())
	print>>fout, "p(z90H < z70F) = %s" % str((z90H.trace() < z70F.trace()).mean())
	print>>fout, "p(z70H < z70F) = %s" % str((z70H.trace() < z70F.trace()).mean())
	print>>fout, "p(z50N < z70F) = %s" % str((z50N.trace() < z70F.trace()).mean())
	print>>fout, "p(z70H < z50N) = %s" % str((z70H.trace() < z50N.trace()).mean())
	print>>fout, "p(z90H < z50N) = %s" % str((z90H.trace() < z50N.trace()).mean())
	print>>fout, "p(z90H < z70H) = %s" % str((z90H.trace() < z70H.trace()).mean())	
	print>>fout, "\n\n\n"
	
	
	fout.close()



if __name__=="__main__":
	main()