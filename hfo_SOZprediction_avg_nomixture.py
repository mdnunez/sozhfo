# hfo_SOZprediction_avg_nomixture.py - Evaluates average of models to predict SOZ
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
# 04/28/21     Michael Nunez                Converted from hfo_SOZPrediction_avg.py
# 05/07/21     Michael Nunez                  Add patient numbers to ROC plots
# 05/13/21     Michael Nunez                  Print out and export patient information
# 05/21/21     Michael Nunez                     Output reorganization
# 05/25/21      Michael Nunez          t-tests and reorganization
# 07/06/21     Michael Nunez                   Print out mean and standard deviation AUCs
# 07/29/21      Michael Nunez                Change integration strategy to avoid integration errors
# 08/20/21      Michael Nunez             Remove patient identifiers

#References:
#https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/43439132#43439132
#https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
#https://stackoverflow.com/questions/34028255/set-height-and-width-of-figure-created-with-plt-subplots-in-matplotlib
#https://matplotlib.org/tutorials/intermediate/gridspec.html
#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
#https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/custom_legends.html
#https://stackoverflow.com/questions/13338550/typing-greek-letters-etc-in-python-plots
#https://matplotlib.org/gallery/recipes/transparent_legends.html
#https://stackoverflow.com/questions/63756623/how-to-remove-or-hide-y-axis-ticklabels-from-a-matplotlib-seaborn-plot
#https://www.statology.org/matplotlib-subplot-spacing/
#https://stackoverflow.com/questions/9912206/how-do-i-let-my-matplotlib-plot-go-beyond-the-axes
#https://www.statology.org/paired-samples-t-test-python/
#https://stackoverflow.com/questions/53640859/how-to-integrate-curve-from-data
#https://stackoverflow.com/questions/44915116/how-to-decide-between-scipy-integrate-simps-or-numpy-trapz

# Imports
import numpy as np
import scipy.io as sio
from pymatreader import read_mat
from scipy import stats
import os.path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats.stats import pearsonr
import seaborn as sns
from pingouin import ttest
import pandas as pd
from matplotlib import rc

rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

fontsize = 10
fontsize2 = 10
fontsizelarge = 18
rocprecision = 1000


##Average clumping parameters
clumping_with_grey = ['/data/posterior_samples/jagsmodel_model14samples_Patient1_grey_qHFOApr_01_21_20_46',
'/data/posterior_samples/jagsmodel_model14samples_Patient2_grey_qHFOMar_18_21_19_35',
'/data/posterior_samples/jagsmodel_model14samples_Patient3_grey_qHFOApr_05_21_11_30',
'/data/posterior_samples/jagsmodel_model14samples_Patient4_grey_qHFOApr_05_21_12_09',
'/data/posterior_samples/jagsmodel_model14samples_Patient9_grey_qHFOApr_06_21_12_33',
'/data/posterior_samples/jagsmodel_model14samples_Patient10_grey_qHFOApr_06_21_22_14',
'/data/posterior_samples/jagsmodel_model14samples_Patient11_grey_qHFOApr_01_21_20_47',
'/data/posterior_samples/jagsmodel_model14samples_Patient13_grey_qHFOMar_31_21_12_44',
'/data/posterior_samples/jagsmodel_model14samples_Patient14_grey_qHFOMar_18_21_14_29',
'/data/posterior_samples/jagsmodel_model14samples_Patient15_grey_qHFOApr_01_21_10_10']


patientnum_with_grey = np.array([1, 2, 3, 4, 9, 10, 11, 13, 14, 15])


clumping_with_outbrain = ['/data/posterior_samples/jagsmodel_model14samples_Patient6_grey_qHFOApr_05_21_11_27',
'/data/posterior_samples/jagsmodel_model14samples_Patient7_grey_qHFOApr_01_21_09_48',
'/data/posterior_samples/jagsmodel_model14samples_Patient8_grey_qHFOApr_01_21_20_36',
'/data/posterior_samples/jagsmodel_model14samples_Patient12_grey_qHFOApr_06_21_12_26',
'/data/posterior_samples/jagsmodel_model14samples_Patient5_grey_qHFOApr_06_21_12_29',
'/data/posterior_samples/jagsmodel_model14samples_Patient16_grey_qHFOApr_06_21_22_14']


patientnum_with_outbrain = np.array([6, 7, 8, 12, 5, 16])


allclumpingmodels = np.concatenate((clumping_with_grey, clumping_with_outbrain))
allpatientnum = np.concatenate((patientnum_with_grey, patientnum_with_outbrain))

nwithgrey = len(clumping_with_grey)
nwithoutbrain = len(clumping_with_outbrain)
nmodels = allclumpingmodels.size

true_positive_rate_clumping = np.empty((rocprecision, nmodels))
false_positive_rate_clumping = np.empty((rocprecision, nmodels))
auc_clumping = np.empty((nmodels))
perfect_clumping = np.empty((nmodels)) #Find the parameter where the True Positive Rate is 1

true_positive_rate_coef = np.empty((rocprecision, nmodels))
false_positive_rate_coef = np.empty((rocprecision, nmodels))
auc_coef = np.empty((nmodels))
perfect_coef = np.empty((nmodels)) #Find the parameter where the True Positive Rate is 1

true_positive_rate_rate = np.empty((rocprecision, nmodels))
false_positive_rate_rate = np.empty((rocprecision, nmodels))
auc_rate = np.empty((nmodels))
perfect_rate = np.empty((nmodels)) #Find the parameter where the True Positive Rate is 1

corrate = np.empty((nmodels))
corclumping = np.empty((nmodels))
corcoef = np.empty((nmodels))

allNSOZ = np.zeros((nmodels))
allNNonSOZ = np.zeros((nmodels))

allclumpingvals =[] #Track all median posterior clumping parameter values in Brain State 1
allhforates = [] #Track all median posterior HFO rates in Brain State 1
allcvs = [] #Track all mean posterior CV in Brain State 1
allsozlabels =[]

Nwindows = [] #Track the number of 5 minute windows used in analysis of interictal data

model_indx = 0
for model in allclumpingmodels:
    patient = model[(model.find('Patient')):(model.find('Patient')+9)]


    print('Loading data from patient %s' % patient)
    samples = sio.loadmat('%s.mat' % model)


    samples_relevant = dict(mean_lambda=samples['mean_lambda'], std_lambda=samples['std_lambda'],
                        mean_clumping=samples['mean_clumping'], std_clumping=samples['std_clumping'],
                        lambda2=samples['lambda'], r=samples['r'], success_prob=samples['success_prob'],
                        coef_variation=samples['coef_variation'], clumping=samples['clumping'])

    nchains = samples['mean_lambda'].shape[-1]
    totalsamps = nchains * samples['success_prob'].shape[-2]
    Nwinds = samples['lambda'].shape[0]
    Nelecs = samples['lambda'].shape[1]

    Nwindows.append(Nwinds)

    clumping_samps = np.reshape(samples['clumping'], (Nwinds, Nelecs, totalsamps))
    coef_samps = np.reshape(samples['coef_variation'], (Nwinds, Nelecs, totalsamps))
    lambda_samps = np.reshape(samples['lambda'], (Nwinds, Nelecs, totalsamps))
    allclumping = np.median(clumping_samps, axis=2)
    allcoef = np.median(coef_samps, axis=2)
    allrates = np.median(lambda_samps, axis=2)

    #Build ROC for SOZ prediction

    if 'usedchans' not in samples:
        samples['usedchans'] = samples['greychans']

    #Only include SOZ electrodes that were in the model
    samples['correctsozchans'] = np.array([
                x for x in samples['usedchans'][0] if x in samples['sozchans'][0]]).T

    for e in range(0,Nelecs):
        if samples['usedchans'][0,e] in samples['correctsozchans']:
            allsozlabels.append(1)
        else:
            allsozlabels.append(0)

    NNonSOZ = float(samples['usedchans'][0].shape[0] -
                    samples['correctsozchans'][0].shape[0])
    allNNonSOZ[model_indx] = NNonSOZ
    NSOZ = float(samples['correctsozchans'][0].shape[0])
    allNSOZ[model_indx] = NSOZ

    print("The number of non-seizure onset zone electrodes is %d" % (NNonSOZ))
    print("The number of seizure onset zone electrodes is %d" % (NSOZ))

    # Calcuate true and false positive rates for the clumping parameter
    # mean_clumping_samps = np.reshape(samples['mean_clumping'], (Nelecs, totalsamps))
    # meanclumping = np.median(mean_clumping_samps, axis=1) # Use the hierarchical mean parameter instead
    meanclumping = np.mean(allclumping,axis=0) #Mean across 5 minute time windows, for observations per electrode
    clump_cutoffs = np.linspace(np.min(meanclumping), np.max(meanclumping), num=rocprecision)
    index = 0
    foundperfect =0
    for clumping in clump_cutoffs:
        little_clumping = np.where(meanclumping <= clumping)[0]
        candidate_SOZ = samples['usedchans'][0][little_clumping]
        intersection = [
            x for x in candidate_SOZ if x in samples['correctsozchans'][0]]
        true_positive_rate_clumping[index,model_indx] = float(len(intersection)) / NSOZ
        if ((true_positive_rate_clumping[index,model_indx] > .99) & (foundperfect==0)):
            perfect_clumping[model_indx] = clumping
            foundperfect = 1
        false_SOZ = [
            x for x in candidate_SOZ if x not in samples['correctsozchans'][0]]
        false_positive_rate_clumping[index,model_indx] = float(len(false_SOZ)) / NNonSOZ
        index += 1
    auc_clumping[model_indx] = 1 - np.trapz(np.squeeze(false_positive_rate_clumping[:,model_indx]),np.squeeze(true_positive_rate_clumping[:,model_indx]))
    print("The AUC of the clumping ROC is  %.3f" % (auc_clumping[model_indx]))
    perfect_clumping[model_indx] = perfect_clumping[model_indx]
    for e in range(0,Nelecs):
        allclumpingvals.append(meanclumping[e])

    # Calcuate true and false positive rates for the coefficient of variation parameter
    meancoef = np.mean(allcoef,axis=0) #Mean across 5 minute time windows, for observations per electrode
    coef_cutoffs = np.linspace(np.min(meancoef), np.max(meancoef), num=rocprecision)
    index = 0
    foundperfect =0
    for coef in coef_cutoffs:
        little_coef = np.where(meancoef <= coef)[0]
        candidate_SOZ = samples['usedchans'][0][little_coef]
        intersection = [
            x for x in candidate_SOZ if x in samples['correctsozchans'][0]]
        true_positive_rate_coef[index,model_indx] = float(len(intersection)) / NSOZ
        if ((true_positive_rate_coef[index,model_indx] > .99) & (foundperfect==0)):
            perfect_coef[model_indx] = coef
            foundperfect = 1
        false_SOZ = [
            x for x in candidate_SOZ if x not in samples['correctsozchans'][0]]
        false_positive_rate_coef[index,model_indx] = float(len(false_SOZ)) / NNonSOZ
        index += 1
    auc_coef[model_indx] = 1 - np.trapz(np.squeeze(false_positive_rate_coef[:,model_indx]),np.squeeze(true_positive_rate_coef[:,model_indx]))
    print("The AUC of the coefficient of variation ROC is  %.3f" % (auc_coef[model_indx]))
    perfect_coef[model_indx] = perfect_coef[model_indx]
    for e in range(0,Nelecs):
        allcvs.append(meancoef[e])

    # Calcuate true and false positive rates for the HFO rate parameter
    # mean_lambda_samps = np.reshape(samples['mean_lambda'][:,:,:,keepchains], (Nelecs, totalsamps))
    # meanrate = np.median(mean_lambda_samps, axis=1)
    meanrate = np.mean(allrates,axis=0) #Mean across 5 minute time windows, for observations per electrode
    rate_cutoffs = np.linspace(np.min(meanrate), np.max(meanrate), num=rocprecision)
    index = 0
    foundperfect =0
    for rate in rate_cutoffs:
        large_rate = np.where(meanrate >= rate)[0]
        candidate_SOZ = samples['usedchans'][0][large_rate]
        intersection = [
            x for x in candidate_SOZ if x in samples['correctsozchans'][0]]
        true_positive_rate_rate[index,model_indx] = float(len(intersection)) / NSOZ
        if ((true_positive_rate_rate[index,model_indx] > .99) & (foundperfect==0)):
            perfect_rate[model_indx] = rate
            foundperfect = 1
        false_SOZ = [
            x for x in candidate_SOZ if x not in samples['correctsozchans'][0]]
        false_positive_rate_rate[index,model_indx] = float(len(false_SOZ)) / NNonSOZ
        index += 1
    auc_rate[model_indx] = 1 + np.trapz(np.squeeze(false_positive_rate_rate[:,model_indx]),np.squeeze(true_positive_rate_rate[:,model_indx]))
    print("The AUC of the HFO rate ROC is  %.3f" % (auc_rate[model_indx]))
    perfect_rate[model_indx] = perfect_rate[model_indx]
    for e in range(0,Nelecs):
        allhforates.append(meanrate[e])

    model_indx += 1



Nhours = np.asarray(Nwindows) / float(12) #Convert from 5 minute windows to hours
print("Mean and standard deviation of number hours is %.2f +/- %.2f" % (np.mean(Nhours), np.std(Nhours)))


grey_mean_false_positive = np.mean(false_positive_rate_clumping[:,0:(len(clumping_with_grey))],axis=1)
grey_mean_true_positive = np.mean(true_positive_rate_clumping[:,0:(len(clumping_with_grey))],axis=1)
out_mean_false_positive = np.mean(false_positive_rate_clumping[:,(len(clumping_with_grey)):],axis=1)
out_mean_true_positive = np.mean(true_positive_rate_clumping[:,(len(clumping_with_grey)):],axis=1)
both_mean_false_positive = np.mean(false_positive_rate_clumping[:,:],axis=1) 
both_mean_true_positive = np.mean(true_positive_rate_clumping[:,:],axis=1)

grey_mean_false_posCV = np.mean(false_positive_rate_coef[:,0:(len(clumping_with_grey))],axis=1)
grey_mean_true_posCV = np.mean(true_positive_rate_coef[:,0:(len(clumping_with_grey))],axis=1)
out_mean_false_posCV = np.mean(false_positive_rate_coef[:,(len(clumping_with_grey)):],axis=1)
out_mean_true_posCV = np.mean(true_positive_rate_coef[:,(len(clumping_with_grey)):],axis=1)
both_mean_false_posCV = np.mean(false_positive_rate_coef[:,:],axis=1)
both_mean_true_posCV = np.mean(true_positive_rate_coef[:,:],axis=1)

both_mean_false_positive_rate = np.mean(false_positive_rate_rate[:,:],axis=1) 
both_mean_true_positive_rate = np.mean(true_positive_rate_rate[:,:],axis=1)

# For what value of the false positive rate is the true positive rate 100%?
what100tpr_clumping = np.empty((nmodels))
what100tpr_coef = np.empty((nmodels))
what100tpr_rate = np.empty((nmodels))
for n in range(0,nmodels):
    where100tpr_clumping = (true_positive_rate_clumping[:,n] > .999)
    what100tpr_clumping[n] = np.min(false_positive_rate_clumping[where100tpr_clumping,n])
    where100tpr_coef = (true_positive_rate_coef[:,n] > .999)
    what100tpr_coef[n] = np.min(false_positive_rate_coef[where100tpr_coef,n])
    where100tpr_rate = (true_positive_rate_rate[:,n] > .999)
    what100tpr_rate[n] = np.min(false_positive_rate_rate[where100tpr_rate,n])




#### Compare AUC values across models ####

print('Loading summary ROC and AUC statistics from 2 state model fits...')
summaryauc_model2 = read_mat('data/AllPatients_summaryauc.mat')
summaryaucboth = dict()

summaryaucboth['what100tpr_clumping_model1'] = what100tpr_clumping
summaryaucboth['what100tpr_clumping_state1'] = summaryauc_model2['what100tpr_clumping'][1,:]
ttestoutput = ttest(summaryaucboth['what100tpr_clumping_model1'],summaryaucboth['what100tpr_clumping_state1'],paired=True)
print('The p-value and Bayes Factor of the paired samples t-test between the FPR values from clumping coefficients between Model 1 and State 1 are %.3f and %.2f respectively' % (ttestoutput['p-val'],ttestoutput['BF10']))
summaryaucboth['what100tpr_clumping_state2'] = summaryauc_model2['what100tpr_clumping'][0,:]
ttestoutput = ttest(summaryaucboth['what100tpr_clumping_model1'],summaryaucboth['what100tpr_clumping_state2'],paired=True)
print('The p-value and Bayes Factor of the paired samples t-test between the FPR values from clumping coefficients between Model 1 and State 2 are %.3f and %.2f respectively' % (ttestoutput['p-val'],ttestoutput['BF10']))
summaryaucboth['auc_clumping_model1'] = auc_clumping
summaryaucboth['auc_clumping_state1'] = summaryauc_model2['aucresort_clumping'][1,:]
ttestoutput = ttest(summaryaucboth['auc_clumping_model1'],summaryaucboth['auc_clumping_state1'],paired=True)
print('The p-value and Bayes Factor of the paired samples t-test between the AUCs from clumping values between Model 1 and State 1 are %.3f and %.2f respectively' % (ttestoutput['p-val'],ttestoutput['BF10']))
summaryaucboth['auc_clumping_state2'] = summaryauc_model2['aucresort_clumping'][0,:]
ttestoutput = ttest(summaryaucboth['auc_clumping_model1'],summaryaucboth['auc_clumping_state2'],paired=True)
print('The p-value and Bayes Factor of the paired samples t-test between the AUCs from clumping values between Model 1 and State 2 are %.3f and %.2f respectively' % (ttestoutput['p-val'],ttestoutput['BF10']))

summaryaucboth['what100tpr_coef_model1'] = what100tpr_coef
summaryaucboth['what100tpr_coef_state1'] = summaryauc_model2['what100tpr_coef'][1,:]
ttestoutput = ttest(summaryaucboth['what100tpr_coef_model1'],summaryaucboth['what100tpr_coef_state1'],paired=True)
print('The p-value and Bayes Factor of the paired samples t-test between the FPR values from coefficients of variation between Model 1 and State 1 are %.3f and %.2f respectively' % (ttestoutput['p-val'],ttestoutput['BF10']))
summaryaucboth['what100tpr_coef_state2'] = summaryauc_model2['what100tpr_coef'][0,:]
ttestoutput = ttest(summaryaucboth['what100tpr_coef_model1'],summaryaucboth['what100tpr_coef_state2'],paired=True)
print('The p-value and Bayes Factor of the paired samples t-test between the FPR values from coefficients of variation between Model 1 and State 2 are %.3f and %.2f respectively' % (ttestoutput['p-val'],ttestoutput['BF10']))
summaryaucboth['auc_coef_model1'] = auc_coef
summaryaucboth['auc_coef_state1'] = summaryauc_model2['aucresort_coef'][1,:]
ttestoutput = ttest(summaryaucboth['auc_coef_model1'],summaryaucboth['auc_coef_state1'],paired=True)
print('The p-value and Bayes Factor of the paired samples t-test between the AUCs from coefficients of variation between Model 1 and State 1 are %.3f and %.2f respectively' % (ttestoutput['p-val'],ttestoutput['BF10']))
summaryaucboth['auc_coef_state2'] = summaryauc_model2['aucresort_coef'][0,:]
ttestoutput = ttest(summaryaucboth['auc_coef_model1'],summaryaucboth['auc_coef_state2'],paired=True)
print('The p-value and Bayes Factor of the paired samples t-test between the AUCs from coefficients of variation between Model 1 and State 2 are %.3f and %.2f respectively' % (ttestoutput['p-val'],ttestoutput['BF10']))

summaryaucboth['what100tpr_rate_model1'] = what100tpr_rate
summaryaucboth['what100tpr_rate_state1'] = summaryauc_model2['what100tpr_rate'][1,:]
ttestoutput = ttest(summaryaucboth['what100tpr_rate_model1'],summaryaucboth['what100tpr_rate_state1'],paired=True)
print('The p-value and Bayes Factor of the paired samples t-test between the FPR values from HFO rates between Model 1 and State 1 are %.3f and %.2f respectively' % (ttestoutput['p-val'],ttestoutput['BF10']))
summaryaucboth['what100tpr_rate_state2'] = summaryauc_model2['what100tpr_rate'][0,:]
ttestoutput = ttest(summaryaucboth['what100tpr_rate_model1'],summaryaucboth['what100tpr_rate_state2'],paired=True)
print('The p-value and Bayes Factor of the paired samples t-test between the FPR values from HFO rates between Model 1 and State 2 are %.3f and %.2f respectively' % (ttestoutput['p-val'],ttestoutput['BF10']))
summaryaucboth['auc_rate_model1'] = auc_rate
summaryaucboth['auc_rate_state1'] = summaryauc_model2['aucresort_rate'][1,:]
ttestoutput = ttest(summaryaucboth['auc_rate_model1'],summaryaucboth['auc_rate_state1'],paired=True)
print('The p-value and Bayes Factor of the paired samples t-test between the AUCs from HFO rates between Model 1 and State 1 are %.3f and %.2f respectively' % (ttestoutput['p-val'],ttestoutput['BF10']))
summaryaucboth['auc_rate_state2'] = summaryauc_model2['aucresort_rate'][0,:]
ttestoutput = ttest(summaryaucboth['auc_rate_model1'],summaryaucboth['auc_rate_state2'],paired=True)
print('The p-value and Bayes Factor of the paired samples t-test between the AUCs from HFO rates between Model 1 and State 2 are %.3f and %.2f respectively' % (ttestoutput['p-val'],ttestoutput['BF10']))



#delta_df = pd.DataFrame({'standarddelta' : stackeddelta, 'brainstate': whichstate})
#aov = anova(dv='standarddelta',between='brainstate', data=delta_df, detailed=True)
#allFstat[p] = aov['F'][0]
#ANOVApvals[p] = aov['p-unc'][0]
#kw = kruskal(dv='standarddelta',between='brainstate', data=delta_df, detailed=True)
#allHstat[p] = kw['H'][0]
#KWpvals[p] = kw['p-unc'][0]


##### PLOTTING ####

posfacx1 = 40 # Patient number x-position 1
posfacx2 = 25 # Patient number x-position 2
posfacx3 = 60 # Patient number x-position 3
posfacx4 = 20 # Patient number x-position 3
posfacx5 = 15 # Patient number x-position 4
posfacy = 35 # Patient number y-position
widthspace = 0.1 #Space between subplots
ROCpatienty = 1.05 # Patient number position in y coordinates in ROC plot
correctROCpatientx = .015 # Patient number correction position in y coordinates in ROC plot

# Subplot figures for clumping coefficients
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))


for n in range(0,nmodels):
    ax1.plot(false_positive_rate_clumping[:, n], true_positive_rate_clumping[
         :, n], color='k', LineWidth=3,alpha=0.25)
ax1.plot(both_mean_false_positive, both_mean_true_positive, color='k', LineWidth=6)
allprobs = np.linspace(0., 1., num=rocprecision)
ax1.plot(allprobs, allprobs, color='k', LineWidth=2, LineStyle='--')
ax1.set_xlim(-.01, 1.01)
ax1.set_ylim(-.01, 1.01)
ax1.plot(what100tpr_clumping[0:nwithgrey],np.ones((nwithgrey))*ROCpatienty, 'kh', markersize=4, clip_on=False)
ax1.plot(what100tpr_clumping[nwithgrey:nmodels], np.ones((nwithoutbrain))*ROCpatienty, 'k*', markersize=4, clip_on=False)
skipit = []
for n in range(nmodels):
    if (allpatientnum[n]>9): #Correction for double digits
        posfacx = posfacx2
    else:
        posfacx = posfacx1
    if allpatientnum[n] == 13: #Overlap correction
        posfacx = posfacx3
    if n not in skipit:
        theseoverlap = (np.abs(what100tpr_clumping[n] - what100tpr_clumping) < .01) & (np.abs(what100tpr_clumping[n] - what100tpr_clumping) > 0)
        if np.any(theseoverlap):
            textstring = str(allpatientnum[n])
            for m in np.where(theseoverlap)[0]:
                textstring += ',' + str(allpatientnum[m])
                skipit.append(m)
            ax1.text(what100tpr_clumping[n]-(1/posfacx),(ROCpatienty -(1.02/posfacy)) ,textstring,fontsize=(2*fontsize/3))
        else:
            ax1.text(what100tpr_clumping[n]-(1/posfacx)+.01,(ROCpatienty -(1.02/posfacy)) ,str(allpatientnum[n]),fontsize=(2*fontsize/3))
ax1.tick_params(axis='both', which='major', labelsize=fontsize)
ax1.set_ylabel('True positive rate', fontsize=fontsize)
ax1.set_xlabel('False positive rate', fontsize=fontsize)

# Custom legend
custom_lines = [Line2D([0], [0], color='k', lw=3, alpha=0.5),
                Line2D([0], [0], color='k', lw=4),
                Line2D([0], [0], color='k', lw=2, linestyle='--')]
ax1.legend(custom_lines, ['Patient','Average','Chance'], 
                          loc=4, fontsize=fontsize, fancybox=True, framealpha=0.5)



sns.distplot(auc_clumping, hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'k'}, rug_kws={'linewidth':2, 'color': 'k'}, ax=ax2)
ymin, ymax = ax2.get_ylim()
ax2.plot(np.array([.5, .5]), np.array([ymin,ymax]), color='k', LineWidth=2, LineStyle='--')
auc_clumping_copy = np.copy(auc_clumping)
auc_clumpingorder = np.argsort(auc_clumping_copy)[::-1]
spacingit_clumping = np.linspace(start=(float(ymax-ymin)/20),stop=(3*float(ymax-ymin)/4),num=nmodels)[::-1] #Note that these y-value limits are just chosen arbitrarily to view patient numbers
markerclumping = np.empty((nmodels))
for n in range(nmodels):
    markerclumping[auc_clumpingorder[n]] = spacingit_clumping[n]
ax2.plot(auc_clumping[0:nwithgrey],markerclumping[0:nwithgrey], 'kh', markersize=4)
ax2.plot(auc_clumping[nwithgrey:nmodels],markerclumping[nwithgrey:nmodels], 'k*', markersize=4)
for n in range(nmodels):
    if (allpatientnum[n]>9):
        posfacx = posfacx2
    else:
        posfacx = posfacx1
    ax2.text(auc_clumping[n]-(1/posfacx),markerclumping[n]-(float(ymax-ymin)/posfacy),str(allpatientnum[n]),fontsize=(2*fontsize/3))
ax2.set_xlim(-.01, 1.01)
ax2.set_ylim(ymin, ymax)
ax2.set_xlabel('AUC by HFO Clumping', fontsize=fontsize)
ax2.set(ylabel=None)
ax2.set(yticklabels=[])
ax2.set(yticks=[])
ax2.tick_params(axis='both', which='major', labelsize=fontsize, left='off', right='off', labelleft='off')

custom_lines = [Line2D([0], [0], color='w', marker='h',markersize=12,markerfacecolor='k'),
                Line2D([0], [0], color='w', marker='*',markersize=12,markerfacecolor='k'),
                Line2D([0], [0], color='k', lw=2, linestyle='--')]
ax2.legend(custom_lines, ['Grey matter only', 'All iEEG','Chance'], 
                          loc=2, fontsize=fontsize, numpoints=1, ncol=1, fancybox=True, framealpha=0.5)
plt.subplots_adjust(wspace=widthspace)

print('Saving Clumping ROC+AUC distribution figures...')
plt.savefig(('figures/Clumping_ROC_AUC.png'), dpi=300, format='png')
plt.close()

# Print out information about SOZ prediction
print("The mean and standard deviation of AUC by the clumping coefficients were %.2f +/- %.2f" % (np.mean(auc_clumping),np.std(auc_clumping)))
print("The data of %d of 16 patients yielded clumping coefficients that we deemed predictive of SOZ (AUC >.60)" % (np.sum([auc_clumping > .6])))
print("In %d of 16 patients, clumping coefficients differentiated all SOZ channels for FPR less than 60%%" % (np.sum([what100tpr_clumping < .6])))
print("In %d of 16 patients, clumping coefficients differentiated all SOZ channels for FPR less than 20%%" % (np.sum([what100tpr_clumping < .2])))




# Subplot figures for coefficients of variation
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))


for n in range(0,nmodels):
    ax1.plot(false_positive_rate_coef[:, n], true_positive_rate_coef[
         :, n], color='k', LineWidth=3,alpha=0.25)
ax1.plot(both_mean_false_posCV, both_mean_true_posCV, color='k', LineWidth=6)
allprobs = np.linspace(0., 1., num=rocprecision)
ax1.plot(allprobs, allprobs, color='k', LineWidth=2, LineStyle='--')
ax1.set_xlim(-.01, 1.01)
ax1.set_ylim(-.01, 1.01)
ax1.plot(what100tpr_coef[0:nwithgrey],np.ones((nwithgrey))*ROCpatienty, 'kh', markersize=4, clip_on=False)
ax1.plot(what100tpr_coef[nwithgrey:nmodels], np.ones((nwithoutbrain))*ROCpatienty, 'k*', markersize=4, clip_on=False)
skipit = []
for fpr in np.sort(what100tpr_coef):
    thispatientnum = allpatientnum[np.where(fpr == what100tpr_coef)[0]][0]
    if thispatientnum not in skipit:
        if (thispatientnum>9): #Correction for double digits
            posfacx = posfacx2
        else:
            posfacx = posfacx1
        if (thispatientnum == 6): #Line overlap correction
            posfacx = posfacx5
        if (thispatientnum == 10): #Line overlap correction
            posfacx = posfacx1
        if thispatientnum not in skipit:
            textstring = str(thispatientnum)
        else:
            textstring = str('')
        theseoverlap = (np.abs(fpr - what100tpr_coef) < .03) & (np.abs(fpr - what100tpr_coef) > 0)
        whereoverlap = np.where(theseoverlap)[0]
        sortoverlap = np.sort(what100tpr_coef[whereoverlap])
        if np.any(theseoverlap):
            for fpr2 in sortoverlap:
                overlappatientnum = allpatientnum[np.where(fpr2 == what100tpr_coef)[0]][0]
                if overlappatientnum not in skipit:
                    if textstring is str(''):
                        textstring += str(overlappatientnum)
                    else:
                        textstring += ',' + str(overlappatientnum)
                    skipit.append(overlappatientnum)
        ax1.text(fpr-(1/posfacx),(ROCpatienty -(1.02/posfacy)) ,textstring,fontsize=(2*fontsize/3))
        skipit.append(thispatientnum)
ax1.tick_params(axis='both', which='major', labelsize=fontsize)
ax1.set_ylabel('True positive rate', fontsize=fontsize)
ax1.set_xlabel('False positive rate', fontsize=fontsize)

# Custom legend
custom_lines = [Line2D([0], [0], color='k', lw=3, alpha=0.5),
                Line2D([0], [0], color='k', lw=4),
                Line2D([0], [0], color='k', lw=2, linestyle='--')]
ax1.legend(custom_lines, ['Patient','Average','Chance'], 
                          loc=4, fontsize=fontsize, fancybox=True, framealpha=0.5)



sns.distplot(auc_coef, hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'k'}, rug_kws={'linewidth':2, 'color': 'k'}, ax=ax2)
ymin, ymax = ax2.get_ylim()
ax2.plot(np.array([.5, .5]), np.array([ymin,ymax]), color='k', LineWidth=2, LineStyle='--')
auc_coef_copy = np.copy(auc_coef)
auc_coeforder = np.argsort(auc_coef_copy)[::-1]
spacingit_coef = np.linspace(start=(float(ymax-ymin)/20),stop=(3*float(ymax-ymin)/4),num=nmodels)[::-1] #Note that these y-value limits are just chosen arbitrarily to view patient numbers
markercoef = np.empty((nmodels))
for n in range(nmodels):
    markercoef[auc_coeforder[n]] = spacingit_coef[n]
ax2.plot(auc_coef[0:nwithgrey],markercoef[0:nwithgrey], 'kh', markersize=4)
ax2.plot(auc_coef[nwithgrey:nmodels],markercoef[nwithgrey:nmodels], 'k*', markersize=4)
for n in range(nmodels):
    if (allpatientnum[n]>9):
        posfacx = posfacx2
    else:
        posfacx = posfacx1
    if (allpatientnum[n] == 12): #Line overlap correction
        posfacx = posfacx4
    ax2.text(auc_coef[n]-(1/posfacx),markercoef[n]-(float(ymax-ymin)/posfacy),str(allpatientnum[n]),fontsize=(2*fontsize/3))
ax2.set_xlim(-.01, 1.01)
ax2.set_ylim(ymin, ymax)
ax2.set_xlabel('AUC by CV', fontsize=fontsize)
ax2.set(ylabel=None)
ax2.set(yticklabels=[])
ax2.set(yticks=[])
ax2.tick_params(axis='both', which='major', labelsize=fontsize, left='off', right='off', labelleft='off')

custom_lines = [Line2D([0], [0], color='w', marker='h',markersize=12,markerfacecolor='k'),
                Line2D([0], [0], color='w', marker='*',markersize=12,markerfacecolor='k'),
                Line2D([0], [0], color='k', lw=2, linestyle='--')]
ax2.legend(custom_lines, ['Grey matter only', 'All iEEG','Chance'], 
                          loc=2, fontsize=fontsize, numpoints=1, ncol=1, fancybox=True, framealpha=0.5)
plt.subplots_adjust(wspace=widthspace)

print('Saving CV ROC+AUC distribution figures...')
plt.savefig(('figures/CV_ROC_AUC.png'), dpi=300, format='png')
plt.close()


# Print out information about SOZ prediction
print("The mean and standard deviation of AUC by the coefficients of variation were %.2f +/- %.2f" % (np.mean(auc_coef),np.std(auc_coef)))
print("The data of %d of 16 patients yielded coefficients of variation that we deemed predictive of SOZ (AUC >.60)" % (np.sum([auc_coef > .6])))
print("In %d of 16 patients, coefficients of variation differentiated all SOZ channels for FPR less than 60%%" % (np.sum([what100tpr_coef < .6])))
print("In %d of 16 patients, coefficients of variation differentiated all SOZ channels for FPR less than 20%%" % (np.sum([what100tpr_coef < .2])))



# Subplot figures for HFO rates
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))


for n in range(0,nmodels):
    ax1.plot(false_positive_rate_rate[:, n], true_positive_rate_rate[
         :, n], color='k', LineWidth=3,alpha=0.25)
ax1.plot(both_mean_false_positive_rate, both_mean_true_positive_rate, color='k', LineWidth=6)
allprobs = np.linspace(0., 1., num=rocprecision)
ax1.plot(allprobs, allprobs, color='k', LineWidth=2, LineStyle='--')
ax1.set_xlim(-.01, 1.01)
ax1.set_ylim(-.01, 1.01)
#np.mod(np.arange(nwithgrey),2)*correctROCpatienty
ax1.plot(what100tpr_rate[0:nwithgrey],np.ones((nwithgrey))*ROCpatienty, 'kh', markersize=4, clip_on=False)
ax1.plot(what100tpr_rate[nwithgrey:nmodels], np.ones((nwithoutbrain))*ROCpatienty, 'k*', markersize=4, clip_on=False)
skipit = []
for n in range(nmodels):
    if (allpatientnum[n]>9): #Correction for double digits
        posfacx = posfacx2
    else:
        posfacx = posfacx1
    if ((allpatientnum[n] == 11) | (allpatientnum[n] == 4)): #Overlap correction
        posfacx = posfacx3
    if n not in skipit:
        theseoverlap = (np.abs(what100tpr_rate[n] - what100tpr_rate) < .01) & (np.abs(what100tpr_rate[n] - what100tpr_rate) > 0)
        if np.any(theseoverlap):
            textstring = str(allpatientnum[n])
            for m in np.where(theseoverlap)[0]:
                textstring += ',' + str(allpatientnum[m])
                skipit.append(m)
            ax1.text(what100tpr_rate[n]-(1/posfacx),(ROCpatienty -(1.02/posfacy)) ,textstring,fontsize=(2*fontsize/3))
        else:
            ax1.text(what100tpr_rate[n]-(1/posfacx)+.01,(ROCpatienty -(1.02/posfacy)) ,str(allpatientnum[n]),fontsize=(2*fontsize/3))
ax1.tick_params(axis='both', which='major', labelsize=fontsize)
ax1.set_ylabel('True positive rate', fontsize=fontsize)
ax1.set_xlabel('False positive rate', fontsize=fontsize)

# Custom legend
custom_lines = [Line2D([0], [0], color='k', lw=3, alpha=0.5),
                Line2D([0], [0], color='k', lw=4),
                Line2D([0], [0], color='k', lw=2, linestyle='--')]
ax1.legend(custom_lines, ['Patient','Average','Chance'], 
                          loc=4, fontsize=fontsize, fancybox=True, framealpha=0.5)



sns.distplot(auc_rate, hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'k'}, rug_kws={'linewidth':2, 'color': 'k'}, ax=ax2)
ymin, ymax = ax2.get_ylim()
ax2.plot(np.array([.5, .5]), np.array([ymin,ymax]), color='k', LineWidth=2, LineStyle='--')
auc_rate_copy = np.copy(auc_rate)
auc_rateorder = np.argsort(auc_rate_copy)[::-1]
spacingit_rate = np.linspace(start=(float(ymax-ymin)/20),stop=(3*float(ymax-ymin)/4),num=nmodels)[::-1]
markerrate = np.empty((nmodels))
for n in range(nmodels):
    markerrate[auc_rateorder[n]] = spacingit_rate[n]
ax2.plot(auc_rate[0:nwithgrey],markerrate[0:nwithgrey], 'kh', markersize=4)
ax2.plot(auc_rate[nwithgrey:nmodels],markerrate[nwithgrey:nmodels], 'k*', markersize=4)
for n in range(nmodels):
    if (allpatientnum[n]>9): #Correction for double digits
        posfacx = posfacx2
    else:
        posfacx = posfacx1
    if ((auc_rate[n] > .5) & (auc_rate[n] < .52)):
        correctx = -.01
    else:
        correctx = 0
    ax2.text(auc_rate[n]-(1/posfacx)+correctx,markerrate[n]-(float(ymax-ymin)/posfacy),str(allpatientnum[n]),fontsize=(2*fontsize/3))
ax2.set_xlim(-.01, 1.01)
ax2.set_ylim(ymin, ymax)
ax2.set_xlabel('AUC by HFO Rate', fontsize=fontsize)
ax2.set(ylabel=None)
ax2.set(yticklabels=[])
ax2.set(yticks=[])
ax2.tick_params(axis='both', which='major', labelsize=fontsize, left='off', right='off', labelleft='off')

custom_lines = [Line2D([0], [0], color='w', marker='h',markersize=12,markerfacecolor='k'),
                Line2D([0], [0], color='w', marker='*',markersize=12,markerfacecolor='k'),
                Line2D([0], [0], color='k', lw=2, linestyle='--')]
ax2.legend(custom_lines, ['Grey matter only', 'All iEEG','Chance'], 
                          loc=2, fontsize=fontsize, numpoints=1, ncol=1, fancybox=True, framealpha=0.5)
plt.subplots_adjust(wspace=widthspace)

print('Saving Rate ROC+AUC distribution figures...')
plt.savefig(('figures/RateHFO_ROC_AUC.png'), dpi=300, format='png')
plt.close()

# Print out information about SOZ prediction
print("The mean and standard deviation of AUC by the HFO rates were %.2f +/- %.2f" % (np.mean(auc_rate),np.std(auc_rate)))
print("The data of %d of 16 patients yielded HFO rates that we deemed predictive of SOZ (AUC >.60)" % (np.sum([auc_rate > .6])))
print("In %d of 16 patients, HFO rates differentiated all SOZ channels for FPR less than 60%%" % (np.sum([what100tpr_rate < .6])))
print("In %d of 16 patients, HFO rates differentiated all SOZ channels for FPR less than 20%%" % (np.sum([what100tpr_rate < .2])))



## Aggregate ROC curves for all electrodes across patients
allclumpingvals = np.asarray(allclumpingvals)
allcvs= np.asarray(allcvs)
allhforates= np.asarray(allhforates)
sozlabels = np.asarray(allsozlabels)
NSOZ = np.sum(sozlabels == 1)
NNonSOZ = np.sum(sozlabels == 0)
rocprecision = 10000


# Aggregate ROC curves by clumping coefficient
tpr_aggregate_clumping = np.empty((rocprecision))
fpr_aggregate_clumping = np.empty((rocprecision))
aggregate_clumping_cutoffs = np.empty((rocprecision))

index=0
clump_cutoffs = np.linspace(0., np.max(allclumpingvals[ :]), num=rocprecision)
aggregate_clumping_cutoffs[:] = clump_cutoffs
for clumping in clump_cutoffs:
    little_clumping = np.sum((allclumpingvals[:] <= clumping) & (sozlabels==1))
    tpr_aggregate_clumping[ index] = float(little_clumping) / NSOZ
    false_SOZ = np.sum((allclumpingvals[:] <= clumping) & (sozlabels==0))
    fpr_aggregate_clumping[ index] = float(false_SOZ) / NNonSOZ
    index += 1



plt.figure(dpi=300)
plt.plot(fpr_aggregate_clumping, tpr_aggregate_clumping, color='k', LineWidth=6)
allprobs = np.linspace(0., 1., num=rocprecision)
plt.plot(allprobs, allprobs, color='k', LineWidth=6, LineStyle='--')
whereannotate = np.where((fpr_aggregate_clumping < .1) & (tpr_aggregate_clumping > .25))[0][0]
plt.annotate('CC1=%.2f' % aggregate_clumping_cutoffs[ whereannotate], xy=( fpr_aggregate_clumping[ whereannotate], tpr_aggregate_clumping[ whereannotate]), xycoords='data', 
        xytext=( .2, .15), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='k'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color='k')
whereannotate = np.where((fpr_aggregate_clumping < .2) & (tpr_aggregate_clumping > .5))[0][0]
plt.annotate('CC1=%.2f' % aggregate_clumping_cutoffs[ whereannotate], xy=( fpr_aggregate_clumping[ whereannotate], tpr_aggregate_clumping[ whereannotate]), xycoords='data', 
        xytext=( .3, .3), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='k'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color='k')
# whereannotate = np.where((fpr_aggregate_clumping < .3) & (tpr_aggregate_clumping > .7))[0][0]
# plt.annotate('CC1=%.2f' % aggregate_clumping_cutoffs[ whereannotate], xy=( fpr_aggregate_clumping[ whereannotate], tpr_aggregate_clumping[ whereannotate]), xycoords='data', 
#         xytext=( .6, .35), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
#            bbox=dict(boxstyle="round", fc="w", edgecolor='k'),
#             horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color='k')
whereannotate_zeta1 = np.argmin(np.abs(aggregate_clumping_cutoffs[ :] - 1))
plt.annotate('CC1=%.2f' % aggregate_clumping_cutoffs[ whereannotate_zeta1], xy=( fpr_aggregate_clumping[ whereannotate_zeta1], tpr_aggregate_clumping[ whereannotate_zeta1]), xycoords='data', 
        xytext=( .7, .5), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='k'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color='k')
# whereannotate = np.where((fpr_aggregate_clumping < .6) & (tpr_aggregate_clumping > .9))[0][0]
# plt.annotate('CC1=%.2f' % aggregate_clumping_cutoffs[ whereannotate], xy=( fpr_aggregate_clumping[ whereannotate], tpr_aggregate_clumping[ whereannotate]), xycoords='data', 
#         xytext=( .8, .65), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
#            bbox=dict(boxstyle="round", fc="w", edgecolor='k'),
#             horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color='k')
whereannotate_tpr_1 = np.where((tpr_aggregate_clumping > .999))[0][0]
plt.annotate('CC1=%.2f' % aggregate_clumping_cutoffs[ whereannotate_tpr_1], xy=( fpr_aggregate_clumping[ whereannotate_tpr_1], tpr_aggregate_clumping[ whereannotate_tpr_1]), xycoords='data', 
        xytext=( .9, .8), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='k'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color='k')
plt.xlim(-.01, 1.01)
plt.ylim(-.01, 1.01)
plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.ylabel('True positive rate', fontsize=fontsize)
plt.xlabel('False positive rate', fontsize=fontsize,labelpad=2)
# plt.title('ROC Curve by CC for all patients', fontsize=fontsize)
print('Saving aggregate ROC curve for clumping parameters...')
plt.savefig(('figures/Aggregate_ROC_Clumping_Model1.png'), dpi=300, format='png')
plt.close()

print("The total number of channels used across models was %d" % (NSOZ+NNonSOZ))
print("The false positive rate is %.2f when the clumping coefficient is equal to 1 " % (fpr_aggregate_clumping[whereannotate_zeta1]))
print("The true positive rate is %.2f when the clumping coefficient is equal to 1" % (tpr_aggregate_clumping[whereannotate_zeta1]))
print("The false positive rate is %.2f when the true positive rate is 1" % (fpr_aggregate_clumping[whereannotate_tpr_1]))



# Sort information for paper table based on patient number
patientnumindex = np.argsort(allpatientnum)
patienttable = np.vstack((allpatientnum[patientnumindex],allNSOZ[patientnumindex],allNNonSOZ[patientnumindex],Nhours[patientnumindex]))


sortedNChans = allNSOZ[patientnumindex] + allNNonSOZ[patientnumindex]
print("The minimum number of channels used in the models was %d" % (np.round(np.min(sortedNChans))))
print("The maximum number of channels used in the models was %d" % (np.round(np.max(sortedNChans))))
print("The mean and standard deviation of channels used in the models were %d +- %d" % (np.round(np.mean(sortedNChans)),np.round(np.std(sortedNChans))))

print("The minimum number of SOZ channels used in the models was %d" % (np.round(np.min(allNSOZ))))
print("The maximum number of SOZ channels used in the models was %d" % (np.round(np.max(allNSOZ))))
print("The mean and standard deviation of SOZ channels used in the models were %d +- %d" % (np.round(np.mean(allNSOZ)),np.round(np.std(allNSOZ))))
