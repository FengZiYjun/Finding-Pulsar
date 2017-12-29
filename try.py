from PFDFile import *
cand = PFD("p309n_pfd/SM0001_07041.sf_0.31ms_Cand.pfd")
print('this is the subbands:')
print(cand.get_subbands())
print('this is the subints')
print(cand.get_subints())
print('this is the profiles.')
print(cand.getprofile())
 
# PFD methods
# getprofile: Obtains the profile data from the candidate file.
# plot_chi2_vs_DM: Plot (and return) an array showing the reduced-chi^2 versus DM
# calc_redchi2: Return the calculated reduced-chi^2 of the current summed profile.
# get_subbands:  Plot the interval-summed profiles vs subband.  
# get_subints: Plot the interval-summed profiles vs subband. 
# get_profs: ?

def plot_subbands(cand):
	import pylab as plt
	plt.figure(1, figsize=(9,9), dpi=150)
	plt.subplot(311)
	plt.imshow(cand.get_subbands(), origin='lower', interpolation='nearest',aspect='auto',cmap=plt.cm.Greys)
	plt.title('sub-Bands')
	plt.ylabel('Band Index')

	plt.subplot(312)
	plt.imshow(cand.get_subints(), origin='lower', interpolation='nearest',aspect='auto',cmap=plt.cm.Greys)
	plt.title('sub-ints')
	plt.ylabel('intergration Index')

	plt.subplot(313)
	plt.bar(range(len(cand.getprofile())), cand.getprofile, width=1)
	plt.xlim(0, len(cand.getprofile())
	plt.xlabel('Phase bin index')
	plt.show()

plot_subbands(cand)