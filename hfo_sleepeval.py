# hfo_sleepeval.py - Evaluates results of models to predict SOZ
#
# Copyright (C) 2021 Michael D. Nunez, <m.d.nunez@uva.nl>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 02/04/19	    Michael Nunez                           Original code
# 09/11/19      Michael Nunez           Create two subplots, 
# 01/06/20      Michael Nunez             update models to Model Type 2 (in paper), reordered
# 01/28/20      Michael Nunez         Remove non converging chains, combine similar figure from hfo_posteriordistributions1.R
# 02/18/20	    Michael Nunez           Include all patients
# 02/24/20      Michael Nunez          Include NANs in deltapower array for proper indexing
# 03/03/20      Michael Nunez      Load whether states are reordered with ''flip_labels'' boolean
#                                   Properly include only grey matter electrodes to calculate delta power
# 03/06/20      Michael Nunez        Implement one-way ANOVA to see if delta power differs between Brain States
#                                    Save out whether states are reordered with ''flip_labels'' boolean
# 03/26/20      Michael Nunez       Correction for Patient 14 
# 04/01/20      Michael Nunez          Clean up Patient1_Patient16_Patient17_sleepeval.png figure
# 04/30/20      Michael Nunez           Relabel patients
# 05/04/20      Michael Nunez          Save out summary delta power for brain state statistics
# 11/10/20      Michael Nunez         Replace one patient with qHFO model results
# 01/27/21      Michael Nunez             Load new models, remove LaTeX dependency
# 02/23/21      Michael Nunez              Add a plt.close()
# 05/21/21      Michael Nunez              Load data locally
# 06/03/21      Michael Nunez         Remove delta power 4 standard deviations away from the mean
# 06/04/21      Michael Nunez               Revert to original calculation as the outlier removal method did not clearly differentiate brain states across participants
# 07/30/21      Michael Nunez              Change labels of Brain State 1 and Brain State 2 to Brain State A and Brain State B respectively
# 08/20/20      Michael Nunez             Remove patient identifiers

##References:
#https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplots.html
#https://matplotlib.org/gallery/lines_bars_and_markers/scatter_with_legend.html#sphx-glr-gallery-lines-bars-and-markers-scatter-with-legend-py
#https://stackoverflow.com/questions/47684652/how-to-customize-marker-colors-and-shapes-in-scatter-plot
#https://stackoverflow.com/questions/43010225/logistic-regression-with-just-one-numeric-feature
#https://pythonfordatascience.org/anova-python/
#https://stackoverflow.com/questions/10996140/how-to-remove-specific-elements-in-a-numpy-array

# Imports
import numpy as np
import scipy.io as sio
from scipy import stats
import os.path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats.stats import pearsonr
from scipy.stats import f_oneway
from matplotlib import rc


## Initialize the subplots
rc('font', **{'family':'serif','serif':['Palatino']})

fontsize = 36
fontsize2 = 28

## Load data
jagsoutloc = '/data/jagsout/'
overnightloc = '[Patient data unavailable]'


sixchains = np.array([0, 1, 2, 3, 4, 5]) # All 6 chains

#Initialize lists
allmodel = []
allpowerdata = []
allkeepchains = []
allstarthour =[]
allendhour =[]
alledf =[]

#Patient 1
allmodel.append(jagsoutloc + 'jagsmodel_model12samples_Patient1_grey_qHFO_2LatentDec_02_20_12_11')
allpowerdata.append(overnightloc + 'Patient1/Patient1_powertimecourse.mat')
allkeepchains.append(sixchains)
allstarthour.append(7 - 1)
allendhour.append(19)
alledf.append(False)

#Patient 2
allmodel.append(jagsoutloc + 'jagsmodel_model12samples_Patient2_grey_qHFO_2LatentDec_23_20_11_02')
allpowerdata.append(overnightloc + 'Patient2/Patient2_powertimecourse.mat')
# allkeepchains.append(sixchains)
allkeepchains.append(np.array([0, 1, 2, 4, 5]))
allstarthour.append(25 - 1)
allendhour.append(49)
alledf.append(False)

#Patient 3
allmodel.append(jagsoutloc + 'jagsmodel_model12samples_Patient3_grey_qHFO_2LatentDec_22_20_13_52')
allpowerdata.append(overnightloc + 'Patient3/Patient3_powertimecourse.mat')
allkeepchains.append(sixchains)
allstarthour.append(46 - 46) #Account for 45 missing BESA files correction
allendhour.append(60 - 45) #Account for 45 missing BESA files correction;
alledf.append(False)

#Patient 4
allmodel.append(jagsoutloc + 'jagsmodel_model12samples_Patient4_grey_qHFO_2LatentDec_07_20_21_54')
allpowerdata.append(overnightloc + 'Patient4/Patient4_powertimecourse.mat')
allkeepchains.append(sixchains)
allstarthour.append(1 - 1)
allendhour.append(16)
alledf.append(False)

#Patient 5
allmodel.append(jagsoutloc + 'jagsmodel_model12samples_Patient5_grey_qHFO_2LatentNov_23_20_20_59')
allpowerdata.append(overnightloc + 'Patient5/Patient5_powertimecourse.mat')
allkeepchains.append(sixchains)
allstarthour.append(1 -1)
allendhour.append(12)
alledf.append(False)

#Patient 6
allmodel.append(jagsoutloc + 'jagsmodel_model12samples_Patient6_grey_qHFO_2LatentNov_12_20_16_19')
allpowerdata.append(overnightloc + 'Patient6/Patient6_powertimecourse.mat')
allkeepchains.append(sixchains)
allstarthour.append(4 -4)
allendhour.append( 8-3)
alledf.append(False)

#Patient 7
allmodel.append(jagsoutloc + 'jagsmodel_model12samples_Patient7_grey_qHFO_2LatentNov_03_20_14_55')
allpowerdata.append(overnightloc + 'Patient7/Patient7_powertimecourse.mat')
allkeepchains.append(np.array([1, 2, 3, 4, 5]))
allstarthour.append( 47 -1)
allendhour.append( 55 )
alledf.append(False)

#Patient 8
allmodel.append(jagsoutloc + 'jagsmodel_model12samples_Patient8_grey_qHFO_2LatentNov_10_20_11_03')
allpowerdata.append(overnightloc + 'Patient8/Patient8_powertimecourse.mat')
allkeepchains.append(np.array([2, 4]))
allstarthour.append(33 -5)
allendhour.append(44 -4)
alledf.append(False)

#Patient 9
allmodel.append(jagsoutloc + 'jagsmodel_model12samples_Patient9_grey_qHFO_2LatentDec_19_20_05_29')
allpowerdata.append(overnightloc + 'Patient9/Patient9_powertimecourse.mat')
allkeepchains.append(np.array([0, 1, 2, 3, 5]))
allstarthour.append(17 -8)
allendhour.append(25 -7)
alledf.append(True)

#Patient 10
allmodel.append(jagsoutloc + 'jagsmodel_model12samples_Patient10_grey_qHFO_2LatentFeb_15_21_14_36')
allpowerdata.append(overnightloc + 'Patient10/Patient10_powertimecourse.mat')
allkeepchains.append(np.array([0, 1]))
allstarthour.append(1 -1)
allendhour.append(4)
alledf.append(True)

#Patient 11
allmodel.append(jagsoutloc + 'jagsmodel_model12samples_Patient11_grey_qHFO_2LatentFeb_04_21_11_26')
allpowerdata.append(overnightloc + 'Patient11/Patient11_powertimecourse.mat')
allkeepchains.append(np.array([3, 5]))
allstarthour.append(9 -2)
allendhour.append(17-1)
alledf.append(False)

#Patient 12
allmodel.append(jagsoutloc + 'jagsmodel_model12samples_Patient12_grey_qHFO_2LatentNov_12_20_16_06')
allpowerdata.append(overnightloc + 'Patient12/Patient12_powertimecourse.mat')
allkeepchains.append(np.array([0, 1, 4]))
allstarthour.append(2 -2) #Accounting for 1 missing besa file
allendhour.append(8 -1)
alledf.append(False)

#Patient 13
allmodel.append(jagsoutloc + 'jagsmodel_model12samples_Patient13_grey_qHFO_2LatentDec_22_20_18_49')
allpowerdata.append(overnightloc + 'Patient13/Patient13_powertimecourse.mat')
allkeepchains.append(np.array([0, 1, 3, 4, 5]))
allstarthour.append(11 -7)
allendhour.append(15 -6)
alledf.append(True)

#Patient 14
allmodel.append(jagsoutloc + 'jagsmodel_model12samples_Patient14_grey_qHFO_2LatentFeb_19_21_13_40')
allpowerdata.append(overnightloc + 'Patient14/Patient14_powertimecourse.mat')
allkeepchains.append(np.array([1]))
allstarthour.append(6 -5)
allendhour.append(9 -4)
alledf.append(True)

#Patient 15
model15 = jagsoutloc + 'jagsmodel_model12samples_Patient15_grey_qHFO_2LatentDec_22_20_14_08'
allmodel.append(model15)
sleepdata15 = overnightloc + 'Patient15/BM_Patient15.mat'
powerdata15 = overnightloc + 'Patient15/Patient15_powertimecourse.mat'
allpowerdata.append(powerdata15)
patient15 = 'Patient 15'
timeindex15 = np.arange(241,1440,10) # Particular to Patient15
keepchains15 = sixchains
allkeepchains.append(keepchains15)
allstarthour.append(9 -9)
allendhour.append(18 -8)
alledf.append(True)

#Patient 16
model16 = jagsoutloc + 'jagsmodel_model12samples_Patient16_grey_qHFO_2LatentNov_23_20_20_33'
allmodel.append(model16)
sleepdata16 = overnightloc + 'Patient16/BM_Patient16.mat'
powerdata16 = overnightloc + 'Patient16/Patient16_powertimecourse.mat'
allpowerdata.append(powerdata16)
patient16 = 'Patient 16'
timeindex16 = np.arange(121,1320,10) # Particular to Patient16
# keepchains16 = np.array([1, 3, 5])
keepchains16 = np.array([0, 1, 3, 4, 5])
allkeepchains.append(keepchains16)
allstarthour.append(11 -11)
allendhour.append(20 -10)
alledf.append(True)

hours = np.arange(120)/float(12)

def loadsleepdat(model, sleepdata, powerdata, timeindex, keepchains, edfcorrection):
	patient = model[(model.find('Patient')):(model.find('Patient')+9)]
	csvloc = '/data/localization/%s_localization.csv' % (patient,patient)
	localization = np.genfromtxt(csvloc,delimiter=',',skip_header=1,dtype='int')
	grey_elecs = localization[np.array(localization[:,9],dtype='bool'),0]

	samples = sio.loadmat('%s_reordered.mat' % model)
	power = sio.loadmat(powerdata)
	fivemins = samples['stateindx'].shape[1]
	originalnchains = samples['pi'].shape[-1]
	stateindx = np.reshape(samples['stateindx'], (fivemins, 500, originalnchains))
	if os.path.isfile('%s_reordered.mat' % model):
		for n in keepchains:
			if samples['sortorder'][0,n] == 1:
				stateindx[:,:, n] = 3 - stateindx[:, :, n]
	plotstate = np.squeeze(np.mean(np.mean(stateindx[:,:,keepchains],axis=1),axis=1))
	plotstate = plotstate - 1
	deltaindx = np.squeeze((power['plotfreqs'] >= 1) & (power['plotfreqs'] <= 4))
	# deltapower_raw = np.sum(power['power'][:,deltaindx,:],axis=1)[:,grey_elecs-1] #Sum across delta power index, Note the subtraction of 1 from the index is necessary for Python
	# orgdeltapower = np.ones((deltapower_raw.shape[0]))*np.nan
	# for t in range(0,deltapower_raw.shape[0]):
	# 	whatgoodelecs = (np.abs(deltapower_raw[t,:] - np.nanmean(deltapower_raw[t,:])) < np.nanstd(deltapower_raw[t,:])*4) #Remove delta power at this time window that is 1 standard deviations away from the mean
	# 	orgdeltapower[t] = np.sum(deltapower_raw[t,whatgoodelecs]) #Sum across GOOD electrodes for total delta power across electrodes
	# standardpower = (orgdeltapower - np.nanmean(orgdeltapower))/float(np.nanstd(orgdeltapower))
	orgdeltapower = np.sum(np.sum(power['power'][:,deltaindx][:,:,grey_elecs-1],axis=2),axis=1) #Sum across electrodes then delta index for total delta power across electrodes, Note the subtraction of 1 from the index is necessary for Python
	standardpower = (orgdeltapower - np.nanmean(orgdeltapower))/float(np.nanstd(orgdeltapower))
	if edfcorrection:
		wheretrue = np.ones((standardpower.shape[0]),dtype=bool)
		wheretrue[np.arange(12,wheretrue.shape[0],13)] = False #Correction for misindexing in hfo_besaspectrum.m for edf files
		deltapower = standardpower[wheretrue]
		print('.edf file correction')
	else:
		deltapower = standardpower
	if sleepdata:
		stages = sio.loadmat(sleepdata)
		plotstages = stages['stageData'][0][0]['stages'][timeindex,0]
		slowwave = np.ones(plotstages.shape)
		slowwave[(plotstages < 1) | (plotstages > 4)] = 0 # Stages 1 through 4 of sleep are labeled "1" all other states are labeled "0"
		percentagree = np.sum(slowwave == np.round(plotstate))/float(slowwave.shape[0])
		percentslowwave = np.sum(slowwave[slowwave==1] == np.round(plotstate[slowwave==1]))/float(slowwave[slowwave==1].shape[0])
		percentother = np.sum(slowwave[slowwave==0] == np.round(plotstate[slowwave==0]))/float(slowwave[slowwave==0].shape[0])
		print('%.1f%% overlap between model and expert sleep staging' % (percentagree*100))
		print('%.1f%% of expert sleep-staged slow wave sleep found by the model' % (percentslowwave*100))
		print('%.1f%% of other expert markings found by the model' % (percentother*100))
	else:
		slowwave = []
	return (slowwave, plotstate, deltapower,power)


def savesleepdat(model,flipstate=False):
	print('Saving out flipped labels for %s ...' % (model))
	samples = sio.loadmat('%s_reordered.mat' % model)
	samples['flip_labels'] = flipstate
	sio.savemat('%s_reordered.mat' % model, samples)


def plotsleepdat(patientnum):
	plt.figure(dpi=300, figsize=(40,10))
	print('Loading sleep data for Patient %d' % (patientnum))
	(slowwave, rawstate, deltapower, power) = loadsleepdat(allmodel[patientnum-1], [], allpowerdata[patientnum-1], [], allkeepchains[patientnum-1], alledf[patientnum-1])
	hours = np.arange((allendhour[patientnum-1]-allstarthour[patientnum-1])*12)/float(12)
	if (deltapower.shape[0] < (allendhour[patientnum-1]*12)): #This correction likely only applies to Patient 14
		fixindex = deltapower.shape[0] - allendhour[patientnum-1]*12
		plothours = hours[0:fixindex]
		plotstate = rawstate[0:fixindex]
	else:
		plothours = hours
		plotstate = rawstate
	relevantdelta = deltapower[(allstarthour[patientnum-1]*12):(allendhour[patientnum-1]*12)]

	if (np.mean(relevantdelta[(plotstate>=.5)]) > np.mean(relevantdelta[(plotstate<.5)])):
		print('Brain State A already contains the largest delta power')
		flipstate = False
	else:
		print('Flipping Brain State labels so that Brain State A contains the largest delta power')
		plotstate = 1 - plotstate
		flipstate = True
	savesleepdat(allmodel[patientnum-1],flipstate)
	summarydelta = dict()
	summarydelta['state1delta'] = relevantdelta[(plotstate>=.5)]
	summarydelta['state2delta'] = relevantdelta[(plotstate<.5)]
	print('Saving summary delta power of Patient %d for statistics...' % (patientnum))
	sio.savemat('../data/Patient%d_summarydelta.mat' % (patientnum), summarydelta)
	Fstat, pval = f_oneway(relevantdelta[(plotstate>=.5)], relevantdelta[(plotstate<.5)])
	ones = np.ones(plotstate.shape[0])
	plt.plot(plothours,relevantdelta,color='k',linewidth=6)
	plt.plot(plothours[(plotstate>=.5)],ones[(plotstate>=.5)]*np.nanmax(relevantdelta),'o',color='g',markersize=5)
	plt.plot(plothours[(plotstate<.5)],ones[(plotstate<.5)]*np.nanmin(relevantdelta),'o',color='b',markersize=5)
	# plt.ylim([np.min(relevantdelta)*1.05, np.max(relevantdelta)*1.05])
	plt.tick_params(axis='both', which='major', labelsize=fontsize)
	plt.ylabel('Standardized Mean Delta (1-4 Hz) Power', fontsize=fontsize)
	plt.title('Patient %d' % (patientnum), fontsize=fontsize)

	print('Saving the figure for Patient %d ...' % (patientnum))
	plt.savefig(('Patient%d_sleepeval.png' % (patientnum)), dpi=300, format='png')
	plt.close()
	return (Fstat, pval)


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, dpi=300, figsize=(40,20), sharex=True)

markersize=15

print('Plotting Patient 15')
(slowwave15, plotstate15, deltapower15, power15) = loadsleepdat(model15, sleepdata15, powerdata15, timeindex15, keepchains15, True)
ones = np.ones(plotstate15.shape[0])
ax3.plot(hours[(slowwave15>=.5)],ones[(slowwave15>=.5)]*np.max(deltapower15[0:(hours.shape[0])]),'o',color='palegreen',markersize=markersize)
ax3.plot(hours[(slowwave15<.5)],ones[(slowwave15<.5)]*np.min(deltapower15[0:(hours.shape[0])]),'o',color='cyan',markersize=markersize)
ax3.plot(hours[(plotstate15>=.5)],ones[(plotstate15>=.5)]*np.max(deltapower15[0:(hours.shape[0])])*.85,'o',color='g',markersize=markersize)
ax3.plot(hours[(plotstate15<.5)],ones[(plotstate15<.5)]*np.min(deltapower15[0:(hours.shape[0])])*.9,'o',color='b',markersize=markersize)
ax3.plot(hours,deltapower15[0:(hours.shape[0])],color='k',linewidth=6)
ax3.set_xlim([0, 10])
ax3.set_ylim([np.min(deltapower15[0:(hours.shape[0])])*1.10, np.max(deltapower15[0:(hours.shape[0])])*1.10])
ax3.set_yticks(np.array([-2.5,-1.5, -0.5, 0.5, 1.5]))
ax3.tick_params(axis='both', which='major', labelsize=fontsize)
ax3.set_title('%s' % (patient15), fontsize=fontsize)
ax3.set_ylabel('Standardized Power', fontsize=fontsize)

print('Plotting Patient 16')
(slowwave16, plotstate16, deltapower16, power16) = loadsleepdat(model16, sleepdata16, powerdata16, timeindex16, keepchains16, True)
ones = np.ones(plotstate16.shape[0])
ax4.plot(hours[(slowwave16>=.5)],ones[(slowwave16>=.5)]*np.max(deltapower16[0:(hours.shape[0])])*1.10,'o',color='palegreen',markersize=markersize)
ax4.plot(hours[(slowwave16<.5)],ones[(slowwave16<.5)]*np.min(deltapower16[0:(hours.shape[0])]),'o',color='cyan',markersize=markersize)
ax4.plot(hours[(plotstate16>=.5)],ones[(plotstate16>=.5)]*np.max(deltapower16[0:(hours.shape[0])]),'o',color='g',markersize=markersize)
ax4.plot(hours[(plotstate16<.5)],ones[(plotstate16<.5)]*np.min(deltapower16[0:(hours.shape[0])])*.75,'o',color='b',markersize=markersize)
ax4.plot(hours,deltapower16[0:(hours.shape[0])],color='k',linewidth=6)
ax4.set_xlim([0, 10])
ax4.set_ylim([np.min(deltapower16[0:(hours.shape[0])])*1.10, np.max(deltapower16[0:(hours.shape[0])])*1.15])
ax4.set_yticks(np.array([-1.0, 0.0, 1.0, 2.0, 3.0]))
ax4.tick_params(axis='both', which='major', labelsize=fontsize)
ax4.set_title('%s' % (patient16), fontsize=fontsize)
ax4.set_xlabel('Recording hour', fontsize=fontsize)

print('Plotting Patient 1')
(slowwave1, plotstate1, deltapower1, power1) = loadsleepdat(allmodel[0], [], allpowerdata[0], [], allkeepchains[0], alledf[0])
hours1 = np.arange((allendhour[0]-allstarthour[0])*12)/float(12)
plotstate1 = 1 - plotstate1 #Patient 1 has flipped labels
ones = np.ones(plotstate1.shape[0])
ax1.plot(hours1[(plotstate1>=.5)],ones[(plotstate1>=.5)]*np.max(deltapower1[0:(hours.shape[0])]),'o',color='g',markersize=markersize)
ax1.plot(hours1[(plotstate1<.5)],ones[(plotstate1<.5)]*np.min(deltapower1[0:(hours.shape[0])]),'o',color='b',markersize=markersize)
ax1.plot(hours1[0:-1],deltapower1[(allstarthour[0]*12):(allendhour[0]*12-1)],color='k',linewidth=6) #6 hours past after using the 7th hour data to the 19th hour
ax1.set_xlim([0, 10])
ax1.set_ylim([np.min(deltapower1[0:(hours1.shape[0])])*1.10, np.max(deltapower1[0:(hours1.shape[0])])*1.10])
ax1.set_yticks(np.array([-1.0, 0.0, 1.0, 2.0, 3.0]))
ax1.tick_params(axis='both', which='major', labelsize=fontsize)
# ax1.set_ylabel('Standardized Mean Power', fontsize=fontsize)
ax1.set_title('Patient %d' % (1), fontsize=fontsize)

print('Plotting Patient 11')
(slowwave11, plotstate11, deltapower11, power11) = loadsleepdat(allmodel[10], [], allpowerdata[10], [], allkeepchains[10], alledf[10])
hours11 = np.arange((allendhour[10]-allstarthour[10])*12)/float(12)
ones = np.ones(plotstate11.shape[0])
ax2.plot(hours11[(plotstate11>=.5)],ones[(plotstate11>=.5)]*np.max(deltapower11[(allstarthour[10]*12):(allendhour[10]*12-1)]),'o',color='g',markersize=markersize)
ax2.plot(hours11[(plotstate11<.5)],ones[(plotstate11<.5)]*np.min(deltapower11[(allstarthour[10]*12):(allendhour[10]*12-1)]),'o',color='b',markersize=markersize)
ax2.plot(hours11[0:-1],deltapower11[(allstarthour[10]*12):(allendhour[10]*12-1)],color='k',linewidth=6)
ax2.set_xlim([0, 10])
ax2.set_ylim([np.min(deltapower11[(allstarthour[10]*12):(allendhour[10]*12-1)])*1.10, np.max(deltapower11[(allstarthour[10]*12):(allendhour[10]*12-1)])*1.10])
ax2.set_yticks(np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]))
ax2.tick_params(axis='both', which='major', labelsize=fontsize)
# ax2.set_ylabel('Standardized Mean Power', fontsize=fontsize)
ax2.set_title('Patient %d' % (11), fontsize=fontsize)



# Custom legend
custom_lines = [Line2D([0], [0], color='w', marker='o',markersize=markersize,markerfacecolor='g'),
                Line2D([0], [0], color='w', marker='o',markersize=markersize,markerfacecolor='b'),
                Line2D([0], [0], color='k', lw=6),
                Line2D([0], [0], color='w', marker='o',markersize=markersize,markerfacecolor='palegreen'),
                Line2D([0], [0], color='w', marker='o',markersize=markersize,markerfacecolor='cyan')]
ax1.legend(custom_lines, ['Model: Brain State A', 'Model: Brain State B','Mean delta power (1-4 Hz)',
						'Expert: NREM', 'Expert: Awake or REM'], 
                          fancybox=True, shadow=True,
                          loc=2,ncol=2,fontsize=fontsize2)

print('Saving the figure...')
plt.savefig(('Patient1_Patient11_Patient15_Patient16_sleepeval.png'), dpi=300, format='png')
plt.close()

allFstat = []
allpvals = []
for p in range(16):
	(Fstat, pval) = plotsleepdat(p+1)
	allFstat.append(Fstat)
	allpvals.append(allpvals)

# allFstat = np.asarray(allFstat)
# allpvals = np.asarray(allpvals)