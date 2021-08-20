# hfo_SOZprediction_avg.py - Evaluates average of models to predict SOZ
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
# 03/07/19      Michael Nunez              Converted from hfo_SOZprediction1.py
# 03/18/19      Michael Nunez                Plot individual patients
# 04/24/19      Michael Nunez                  Plot the clumping averages
# 06/14/19      Michael Nunez                Updated Poisson mixture models
#                                             Plot based on AUC
# 07/16/19      Michael Nunez                  Change line properties, 
#                                      plot clumping avg. ROC for all patients
# 07/22/19      Michael Nunez                  New clumping model fits
# 09/13/19      Michael Nunez                  New clumping model fits
# 09/24/19      Michael Nunez     Plot only those chains to keep, generate rate plots for Model 2
# 09/25/19      Michael Nunez       Output perfect prediction parameters, Fix ROC plots so that all have a point on each extreme
# 10/15/19      Michael Nunez         Stagger tick marks on AUC plots, place Engel outcome on AUC plots
# 10/17/19      Michael Nunez          Plot points on distribution plots
# 01/09/20      Michael Nunez          Export correlation values
# 01/10/20      Michael Nunez            Create figure of Engel outcomes
# 01/21/20      Michael Nunez            Fix ROC plot for Clumping parameter; Subplots
# 01/21/20      Michael Nunez        Adjustments to subplot layout
# 01/23/20      Michael Nunez          Histograms of parameters for SOZ prediction with 100% sensitivity
#                                      More precise ROC curves, aggregate clumping ROC curve
# 01/24/20      Michael Nunez            Figure cleanup
# 01/27/20      Michael Nunez          Aggregate SOZ prediction figures
# 01/29/20      Michael Nunez          Report mean and standard deviation of number of hours used
# 02/24/20      Michael Nunez              Save out sortorder
# 03/03/20      Michael Nunez      Save out whether states are reordered with ''flip_labels'' boolean
# 03/06/20      Michael Nunez      Remove ''flip_labels'' boolean
# 03/26/20      Michael Nunez        Load ''flip_labels'' from hfo_sleepeval.py, remove patient [Former patient label] (Patient9 mislabeled)
# 04/01/20      Michael Nunez         Fix connective lines in AUC distribution plots, split CV ROC plot into Localization and no Localization
# 04/02/20      Michael Nunez         Use of fancybox and transparency in legends
# 04/03/20      Michael Nunez             Keep track of electrode numbers
# 04/30/20      Michael Nunez           Label patients with numbers, fix Patient 5 which did not have localization for the analysis
# 11/10/20      Michael Nunez         Replace one patient with qHFO model results
# 01/27/21      Michael Nunez          Insert new qHFO models, use expanduser()
# 02/23/21      Michael Nunez           Replace original model fits with model fits of qHFOs
# 05/19/21      Michael Nunez        Fix output plots
# 05/25/21      Michael Nunez          t-tests and reorganization
# 07/13/21     Michael Nunez                   Print out mean and standard deviation AUCs
# 07/15/21      Michael Nunez           Relabel "State 1" as "State A" and "State 2" as "State B"
# 07/29/21      Michael Nunez                Change integration strategy to avoid integration errors
# 08/19/21      Michael Nunez             Remove patient identifiers

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
#https://github.com/matplotlib/matplotlib/issues/16911
#https://stackoverflow.com/questions/53640859/how-to-integrate-curve-from-data
#https://stackoverflow.com/questions/44915116/how-to-decide-between-scipy-integrate-simps-or-numpy-trapz

# Imports
import numpy as np
import scipy.io as sio
import os.path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats.stats import pearsonr
import seaborn as sns
from pingouin import ttest
from matplotlib import rc

rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True) #sudo apt install texlive-latex-extra cm-super dvipng

fontsize = 10
fontsize2 = 10
fontsizelarge = 18
rocprecision = 1000

# Load generated smaples from Model 3 in Supplementals
models_with_grey = ['/data/posterior_samples/jagsmodel_model8samples_Patient1_grey_2LatentNov_19_18_11_17',
'/data/posterior_samples/jagsmodel_model8samples_Patient2_grey_2LatentNov_15_18_18_19',
'/data/posterior_samples/jagsmodel_model8samples_Patient3_grey_2LatentNov_28_18_15_05',
'/data/posterior_samples/jagsmodel_model8samples_Patient4_grey_2LatentDec_05_18_14_38',
'/data/posterior_samples/jagsmodel_model8samples_Patient9_grey_2LatentJun_03_19_17_31',
'/data/posterior_samples/jagsmodel_model8samples_Patient10_grey_2LatentJun_03_19_18_11',
'/data/posterior_samples/jagsmodel_model8samples_Patient11_grey_2LatentJan_24_19_14_28',
'/data/posterior_samples/jagsmodel_model8samples_Patient13_grey_2LatentJun_07_19_18_43',
'/data/posterior_samples/jagsmodel_model8samples_Patient14_grey_2LatentJun_07_19_18_57',
'/data/posterior_samples/jagsmodel_model8samples_Patient15_grey_2LatentJun_11_19_18_42']


engel_with_grey = ['IB',
'IIIA',
'IA',
'IIB',
'IA',
'IIA',
'IIIA',
'IIIA',
'IA',
'IIIA']


models_with_outbrain = ['/data/posterior_samples/jagsmodel_model8samples_Patient6_grey_2LatentDec_10_18_15_33',
'/data/posterior_samples/jagsmodel_model8samples_Patient7_grey_2LatentJan_10_19_15_03',
'/data/posterior_samples/jagsmodel_model8samples_Patient8_grey_2LatentJan_11_19_19_02',
'/data/posterior_samples/jagsmodel_model8samples_Patient12_grey_2LatentJan_25_19_13_02',
'/data/posterior_samples/jagsmodel_model8samples_Patient5_grey_2LatentDec_07_18_13_19',
'/data/posterior_samples/jagsmodel_model8samples_Patient16_grey_2LatentJun_11_19_18_43']

engel_with_outbrain = ['',
'IIIA',
'IA',
'IIA',
'',
'IVB']

whereinbrain = np.array(np.concatenate((np.ones(len(models_with_grey)),np.zeros(len(models_with_outbrain)))),dtype=bool)
# allmodels = np.concatenate((models_with_grey, models_with_outbrain))
allengel = np.concatenate((engel_with_grey, engel_with_outbrain))


##Average clumping parameters
clumping_with_grey = ['/data/posterior_samples/jagsmodel_model12samples_Patient1_grey_qHFO_2LatentDec_02_20_12_11',
'/data/posterior_samples/jagsmodel_model12samples_Patient2_grey_qHFO_2LatentDec_23_20_11_02',
'/data/posterior_samples/jagsmodel_model12samples_Patient3_grey_qHFO_2LatentDec_22_20_13_52',
'/data/posterior_samples/jagsmodel_model12samples_Patient4_grey_qHFO_2LatentDec_07_20_21_54',
'/data/posterior_samples/jagsmodel_model12samples_Patient9_grey_qHFO_2LatentDec_19_20_05_29',
'/data/posterior_samples/jagsmodel_model12samples_Patient10_grey_qHFO_2LatentFeb_15_21_14_36',
'/data/posterior_samples/jagsmodel_model12samples_Patient11_grey_qHFO_2LatentFeb_04_21_11_26',
'/data/posterior_samples/jagsmodel_model12samples_Patient13_grey_qHFO_2LatentDec_22_20_18_49',
'/data/posterior_samples/jagsmodel_model12samples_Patient14_grey_qHFO_2LatentFeb_19_21_13_40',
'/data/posterior_samples/jagsmodel_model12samples_Patient15_grey_qHFO_2LatentDec_22_20_14_08']

sixchains = np.array([0, 1, 2, 3, 4, 5]) # All 6 chains
chains_with_grey = [sixchains,
np.array([0, 1, 2, 4, 5]),
sixchains,
sixchains,
np.array([0, 1, 2, 3, 5]),
np.array([0, 1]),
np.array([3, 5]),
np.array([0, 1, 3, 4, 5]),
np.array([1]),
sixchains]

patientnum_with_grey = np.array([1, 2, 3, 4, 9, 10, 11, 13, 14, 15])

clumping_with_outbrain = ['/data/posterior_samples/jagsmodel_model12samples_Patient6_grey_qHFO_2LatentNov_12_20_16_19',
'/data/posterior_samples/jagsmodel_model12samples_Patient7_grey_qHFO_2LatentNov_03_20_14_55',
'/data/posterior_samples/jagsmodel_model12samples_Patient8_grey_qHFO_2LatentNov_10_20_11_03',
'/data/posterior_samples/jagsmodel_model12samples_Patient12_grey_qHFO_2LatentNov_12_20_16_06',
'/data/posterior_samples/jagsmodel_model12samples_Patient5_grey_qHFO_2LatentNov_23_20_20_59',
'/data/posterior_samples/jagsmodel_model12samples_Patient16_grey_qHFO_2LatentNov_23_20_20_33']

chains_with_outbrain = [sixchains,
np.array([1, 2, 3, 4, 5]),
np.array([2, 4]),
np.array([0, 1, 4]),
sixchains,
np.array([0, 1, 3, 4, 5])]


patientnum_with_outbrain = np.array([6, 7, 8, 12, 5, 16])


allclumpingmodels = np.concatenate((clumping_with_grey, clumping_with_outbrain))
allchains = np.concatenate((chains_with_grey, chains_with_outbrain))
allpatientnum = np.concatenate((patientnum_with_grey, patientnum_with_outbrain))

nwithgrey = len(clumping_with_grey)
nwithoutbrain = len(clumping_with_outbrain)
nmodels = allclumpingmodels.size

NLatent = 2

true_positive_rate_clumping = np.empty((NLatent, rocprecision, allclumpingmodels.size))
false_positive_rate_clumping = np.empty((NLatent, rocprecision, allclumpingmodels.size))
auc_clumping = np.empty((NLatent, allclumpingmodels.size))
aucresort_clumping = np.empty((NLatent, allclumpingmodels.size))
perfect_clumping = np.empty((NLatent, allclumpingmodels.size)) #Find the parameter where the True Positive Rate is 1

true_positive_rate_coef = np.empty((NLatent, rocprecision, allclumpingmodels.size))
false_positive_rate_coef = np.empty((NLatent, rocprecision, allclumpingmodels.size))
auc_coef = np.empty((NLatent, allclumpingmodels.size))
aucresort_coef = np.empty((NLatent, allclumpingmodels.size))
perfect_coef = np.empty((NLatent, allclumpingmodels.size)) #Find the parameter where the True Positive Rate is 1

true_positive_rate = np.empty((NLatent, rocprecision, allclumpingmodels.size))
false_positive_rate = np.empty((NLatent, rocprecision, allclumpingmodels.size))
auc = np.empty((NLatent, allclumpingmodels.size))
aucresort = np.empty((NLatent, allclumpingmodels.size))
perfect_rate = np.empty((NLatent, allclumpingmodels.size)) #Find the parameter where the True Positive Rate is 1

corrate = np.empty((allclumpingmodels.size))
corclumping = np.empty((allclumpingmodels.size))
corcoef = np.empty((allclumpingmodels.size))
corrate_pval = np.empty((allclumpingmodels.size))
corclumping_pval = np.empty((allclumpingmodels.size))
corcoef_pval = np.empty((allclumpingmodels.size))

allNSOZ = np.zeros((allclumpingmodels.size))
allNNonSOZ = np.zeros((allclumpingmodels.size))

BSAallclumpingvals =[] #Track all median posterior clumping parameter values in Brain State A
BSAallhforates = [] #Track all median posterior HFO rates in Brain State A
BSAallcvs = [] #Track all mean posterior CV in Brain State A
BSBallclumpingvals =[] #Track all median posterior clumping parameter values in Brain State A
BSBallhforates = [] #Track all median posterior HFO rates in Brain State A
BSBallcvs = [] #Track all mean posterior CV in Brain State A
allsozlabels =[]

Nwindows = [] #Track the number of 5 minute windows used in analysis of interictal data

model_indx = 0
for model in allclumpingmodels:
    patient = model[(model.find('Patient')):(model.find('Patient')+9)]


    print('Loading data from patient %s' % patient)
    if os.path.isfile('%s_reordered.mat' % model):
        samples = sio.loadmat('%s_reordered.mat' % model)
    else:
        samples = sio.loadmat('%s.mat' % model)

    Nwindows.append(samples['stateindx'].shape[1])
 
    keepchains = allchains[model_indx]

    # Switch chain indices (how do we order the index ahead of time?)
    originalnchains = samples['pi'].shape[-1]
    nchains = keepchains.shape[0]
    totalsamps = nchains * samples['pi'].shape[-2]
    NLatent = samples['latent_lambda'].shape[0]
    Nelecs = samples['latent_lambda'].shape[1]

    probmeans = np.mean(np.squeeze(samples['state_lambda']), axis=1)
    sortorder = np.empty((NLatent, originalnchains), dtype='int')
    for n in keepchains:
        sortorder[:, n] = np.argsort(probmeans[:, n])
        samples['latent_lambda'][0:NLatent, :, :, n] = samples[
            'latent_lambda'][sortorder[:, n], :, :, n]
        samples['clumping'][0:NLatent, :, :, n] = samples[
            'clumping'][sortorder[:, n], :, :, n]
        samples['coef_variation'][0:NLatent, :, :, n] = samples[
            'coef_variation'][sortorder[:, n], :, :, n]
        samples['pi'][0, :, :, n] = samples['pi'][0, sortorder[:, n], :, n]
        samples['state_lambda'][0, 0:NLatent, :, n] = samples[
        'state_lambda'][0, sortorder[:, n], :, n]
        samples['state_std'][0, 0:NLatent, :, n] = samples[
        'state_std'][0, sortorder[:, n], :, n]

    if not os.path.isfile('%s_reordered.mat' % model):
        print('Saving data from patient %s with states reordered' % patient)
        samples['sortorder'] = sortorder
        sio.savemat('%s_reordered.mat' % model, samples)

    samples_relevant = dict(latent_lambda=samples[
                        'latent_lambda'][:,:,:,keepchains], pi=samples['pi'][:,:,:,keepchains],
                        state_lambda=samples['state_lambda'][:,:,:,keepchains],
                        state_std=samples['state_std'][:,:,:,keepchains],
                        clumping=samples['clumping'][:,:,:,keepchains],
                        coef_variation=samples['coef_variation'][:,:,:,keepchains])

    if keepchains.shape[0] == 1:
        print("Keeping only one chain")

    clumping_samps = np.reshape(
        samples['clumping'][:,:,:,keepchains], (NLatent, Nelecs, totalsamps))
    coef_samps = np.reshape(
        samples['coef_variation'][:,:,:,keepchains], (NLatent, Nelecs, totalsamps))
    lambda_samps = np.reshape(
        samples['latent_lambda'][:,:,:,keepchains], (NLatent, Nelecs, totalsamps))
    allclumping = np.median(clumping_samps, axis=2)
    allcoef = np.median(coef_samps, axis=2)
    allrates = np.median(lambda_samps, axis=2)

    corrval = pearsonr(allrates[0, :], allrates[1, :])
    print("The correlation of rates between the first two states is %.2f" % (corrval[0]))
    corrate[model_indx] = corrval[0] 
    corrate_pval[model_indx] = corrval[1]

    corrval = pearsonr(allclumping[0, :], allclumping[1, :])
    print("The correlation of clumping between the first two states is %.2f" % (corrval[0]))
    corclumping[model_indx] = corrval[0]
    corclumping_pval[model_indx] = corrval[1]

    corrval = pearsonr(allcoef[0, :], allcoef[1, :])
    print("The correlation of the coefficients of variation between the first two states is %.2f" % (corrval[0]))
    corcoef[model_indx] = corrval[0]
    corcoef_pval[model_indx] = corrval[1]

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

    NNonSOZ = float(samples['greychans'][0].shape[0] -
                    samples['correctsozchans'][0].shape[0])
    allNNonSOZ[model_indx] = NNonSOZ
    NSOZ = float(samples['correctsozchans'][0].shape[0])
    allNSOZ[model_indx] = NSOZ

    print("The number of non-seizure onset zone electrodes is %d" % (NNonSOZ))
    print("The number of seizure onset zone electrodes is %d" % (NSOZ))

    for n in range(0, NLatent):
        index = 0
        foundperfect =0
        clump_cutoffs = np.linspace(0., np.max(allclumping[n, :]), num=rocprecision)
        for clumping in clump_cutoffs:
            little_clumping = np.where(np.median(clumping_samps[n, :, :], axis=1) <= clumping)[0]
            candidate_SOZ = samples['usedchans'][0][little_clumping]
            intersection = [
                x for x in candidate_SOZ if x in samples['correctsozchans'][0]]
            true_positive_rate_clumping[n, index,model_indx] = float(len(intersection)) / NSOZ
            if ((true_positive_rate_clumping[n, index,model_indx] > .99) & (foundperfect==0)):
                perfect_clumping[n, model_indx] = clumping
                foundperfect = 1
            false_SOZ = [
                x for x in candidate_SOZ if x not in samples['correctsozchans'][0]]
            false_positive_rate_clumping[n, index,model_indx] = float(len(false_SOZ)) / NNonSOZ
            index += 1
        auc_clumping[n, model_indx] = 1 - np.trapz(np.squeeze(false_positive_rate_clumping[n, :,model_indx]),np.squeeze(true_positive_rate_clumping[n, :,model_indx]))
    if samples['flip_labels']: #Always plot the generator state that is more predictive of delta power as Brain State A
        true_positive_rate_clumping[:, :, model_indx] = true_positive_rate_clumping[::-1,:,model_indx] #Flip model labels 1 and 0
        false_positive_rate_clumping[:, :, model_indx] = false_positive_rate_clumping[::-1,:,model_indx] #Flip model labels 1 and 0
        aucresort_clumping[:, model_indx] = auc_clumping[::-1, model_indx]
        perfect_clumping[:, model_indx] = perfect_clumping[::-1, model_indx]
        for e in range(0,Nelecs):
            BSAallclumpingvals.append(allclumping[0,e]) #Flipped Brain State A
            BSBallclumpingvals.append(allclumping[1,e]) #Flipped Brain State B
    else:
        aucresort_clumping[:, model_indx] = auc_clumping[:, model_indx]
        perfect_clumping[:, model_indx] = perfect_clumping[:, model_indx]
        for e in range(0,Nelecs):
            BSAallclumpingvals.append(allclumping[1,e]) #Brain State A
            BSBallclumpingvals.append(allclumping[0,e]) #Brain State B

    # print('Saving data from patient %s with states reordered...' % patient)
    # samples['sortorder'] = sortorder
    # sio.savemat('%s_reordered.mat' % model, samples)


    for n in range(0, NLatent):
        index = 0
        foundperfect =0
        coef_cutoffs = np.linspace(0., np.max(allcoef[n, :]), num=rocprecision)
        for coef in coef_cutoffs:
            little_coef = np.where(np.median(coef_samps[n, :, :], axis=1) <= coef)[0]
            candidate_SOZ = samples['usedchans'][0][little_coef]
            intersection = [
                x for x in candidate_SOZ if x in samples['correctsozchans'][0]]
            true_positive_rate_coef[n, index,model_indx] = float(len(intersection)) / NSOZ
            if ((true_positive_rate_coef[n, index,model_indx] > .99) & (foundperfect==0)):
                perfect_coef[n, model_indx] = coef
                foundperfect = 1
            false_SOZ = [
                x for x in candidate_SOZ if x not in samples['correctsozchans'][0]]
            false_positive_rate_coef[n, index,model_indx] = float(len(false_SOZ)) / NNonSOZ
            index += 1
        auc_coef[n, model_indx] = 1 - np.trapz(np.squeeze(false_positive_rate_coef[n, :,model_indx]),np.squeeze(true_positive_rate_coef[n, :,model_indx]))
    if samples['flip_labels']: #Always plot the generator state that is more predictive of delta power as Brain State A
        true_positive_rate_coef[:, :, model_indx] = true_positive_rate_coef[::-1,:,model_indx] #Flip model labels 1 and 0
        false_positive_rate_coef[:, :, model_indx] = false_positive_rate_coef[::-1,:,model_indx] #Flip model labels 1 and 0
        aucresort_coef[:, model_indx] = auc_coef[::-1, model_indx]
        perfect_coef[:, model_indx] = perfect_coef[::-1, model_indx]
        for e in range(0,Nelecs):
            BSAallcvs.append(allcoef[0,e]) #Flipped Brain State A
            BSBallcvs.append(allcoef[1,e]) #Flipped Brain State B
    else:
        aucresort_coef[:, model_indx] = auc_coef[:, model_indx]
        perfect_coef[:, model_indx] = perfect_coef[:, model_indx]
        for e in range(0,Nelecs):
            BSAallcvs.append(allcoef[1,e]) #Brain State A
            BSBallcvs.append(allcoef[0,e]) #Brain State B

    for n in range(0, NLatent):
        index = 0
        foundperfect =0
        rate_cutoffs = np.linspace(np.max(allrates[n, :]), 0., num=rocprecision)
        for rate in rate_cutoffs:
            high_rates = np.where(np.median(lambda_samps[n, :, :], axis=1) >= rate)[0]
            candidate_SOZ = samples['usedchans'][0][high_rates]
            intersection = [
                x for x in candidate_SOZ if x in samples['correctsozchans'][0]]
            true_positive_rate[n, index,model_indx] = float(len(intersection)) / NSOZ
            if ((true_positive_rate[n, index,model_indx] > .99) & (foundperfect==0)):
                perfect_rate[n, model_indx] = rate
                foundperfect = 1
            false_SOZ = [
                x for x in candidate_SOZ if x not in samples['correctsozchans'][0]]
            false_positive_rate[n, index,model_indx] = float(len(false_SOZ)) / NNonSOZ
            index += 1
        auc[n, model_indx] = 1 - np.trapz(np.squeeze(false_positive_rate[n, :,model_indx]),np.squeeze(true_positive_rate[n, :,model_indx]))
    if samples['flip_labels']: #Always plot the generator state that is more predictive of delta power as Brain State A
        true_positive_rate[:, :, model_indx] = true_positive_rate[::-1,:,model_indx] #Flip model labels 1 and 0
        false_positive_rate[:, :, model_indx] = false_positive_rate[::-1,:,model_indx] #Flip model labels 1 and 0
        aucresort[:, model_indx] = auc[::-1, model_indx]
        perfect_rate[:, model_indx] = perfect_rate[::-1, model_indx]
        for e in range(0,Nelecs):
            BSAallhforates.append(allrates[0,e]) #Flipped Brain State A
            BSBallhforates.append(allrates[1,e]) #Flipped Brain State B
    else:
        aucresort[:, model_indx] = auc[:, model_indx]
        perfect_rate[:, model_indx] = perfect_rate[:, model_indx]
        for e in range(0,Nelecs):
            BSAallhforates.append(allrates[1,e]) #Brain State A
            BSBallhforates.append(allrates[0,e]) #Brain State B
    model_indx += 1


Nhours = np.asarray(Nwindows) / float(12) #Convert from 5 minute windows to hours
print("Mean and standard deviation of number hours is %.2f +/- %.2f" % (np.mean(Nhours), np.std(Nhours)))



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


ttestoutput = ttest(BSAallclumpingvals,BSBallclumpingvals,paired=True)
print('The mean and standard deviation of clumping coefficients in Brain State A is %.2f +/- %.2f' % (np.mean(BSAallclumpingvals),np.std(BSAallclumpingvals)))
print('The mean and standard deviation of clumping coefficients in Brain State B is %.2f +/- %.2f' % (np.mean(BSBallclumpingvals),np.std(BSBallclumpingvals)))
print('The p-value and Bayes Factor of the paired samples t-test between clumping coefficients (collapsed across channels and patients) between State A and State B are %.3f and %s respectively' % (ttestoutput['p-val'],np.format_float_scientific(ttestoutput['BF10'])))
ttestoutput = ttest(BSAallcvs,BSBallcvs,paired=True)
print('The mean and standard deviation of coefficients of variation in Brain State A is %.2f +/- %.2f' % (np.mean(BSAallcvs),np.std(BSAallcvs)))
print('The mean and standard deviation of coefficients of variation in Brain State B is %.2f +/- %.2f' % (np.mean(BSBallcvs),np.std(BSBallcvs)))
print('The p-value and Bayes Factor of the paired samples t-test between coefficients of variation (collapsed across channels and patients) between State A and State B are %.3f and %s respectively' % (ttestoutput['p-val'],np.format_float_scientific(ttestoutput['BF10'])))
ttestoutput = ttest(BSAallhforates,BSBallhforates,paired=True)
print('The mean and standard deviation of HFO rates in Brain State A is %.2f +/- %.2f' % (np.mean(BSAallhforates),np.std(BSAallhforates)))
print('The mean and standard deviation of HFO rates in Brain State B is %.2f +/- %.2f' % (np.mean(BSBallhforates),np.std(BSBallhforates)))
print('The p-value and Bayes Factor of the paired samples t-test between HFO rates (collapsed across channels and patients) between State A and State B are %.3f and %s respectively' % (ttestoutput['p-val'],np.format_float_scientific(ttestoutput['BF10'])))

# print("The minimum and maximum correlations between clumping coefficients of Brain States 1 and 2 were %.3f and %.3f respectively" % (np.min(corclumping),np.max(corclumping)))
# print("The minimum and maximum p-values of the correlations between clumping coefficients of Brain States 1 and 2 were %.3f and %.3f respectively" % (np.min(corclumping_pval),np.max(corclumping_pval)))
# print("The minimum and maximum correlations between coefficients of variation between Brain States 1 and 2 were %.3f and %.3f respectively" % (np.min(corcoef),np.max(corcoef)))
# print("The minimum and maximum p-values of the correlations between coefficients of variation between Brain States 1 and 2 were %.3f and %.3f respectively" % (np.min(corcoef_pval),np.max(corcoef_pval)))
# print("The minimum and maximum correlations between HFO rates between Brain States 1 and 2 were %.3f and %.3f respectively" % (np.min(corrate),np.max(corrate)))
# print("The minimum and maximum p-values of the correlations between HFO rates between Brain States 1 and 2 were %.3f and %.3f respectively" % (np.min(corrate_pval),np.max(corrate_pval)))

print("The mean and standard deviation of correlations between clumping coefficients of Brain States 1 and 2 were %.2f +/- %.2f respectively" % (np.mean(corclumping),np.std(corclumping)))
print("The mean and standard deviation of correlations between coefficients of variation of Brain States 1 and 2 were %.2f +/- %.2f respectively" % (np.mean(corcoef),np.std(corcoef)))
print("The mean and standard deviation of correlations between HFO rates of Brain States 1 and 2 were %.2f +/- %.2f respectively" % (np.mean(corrate),np.std(corrate)))

grey_mean_false_positive = np.mean(false_positive_rate_clumping[:,:,0:(len(clumping_with_grey))],axis=2)
grey_mean_true_positive = np.mean(true_positive_rate_clumping[:,:,0:(len(clumping_with_grey))],axis=2)
out_mean_false_positive = np.mean(false_positive_rate_clumping[:,:,(len(clumping_with_grey)):],axis=2)
out_mean_true_positive = np.mean(true_positive_rate_clumping[:,:,(len(clumping_with_grey)):],axis=2)
both_mean_false_positive = np.mean(false_positive_rate_clumping[:,:,:],axis=2) 
both_mean_true_positive = np.mean(true_positive_rate_clumping[:,:,:],axis=2)

grey_mean_false_posCV = np.mean(false_positive_rate_coef[:,:,0:(len(clumping_with_grey))],axis=2)
grey_mean_true_posCV = np.mean(true_positive_rate_coef[:,:,0:(len(clumping_with_grey))],axis=2)
out_mean_false_posCV = np.mean(false_positive_rate_coef[:,:,(len(clumping_with_grey)):],axis=2)
out_mean_true_posCV = np.mean(true_positive_rate_coef[:,:,(len(clumping_with_grey)):],axis=2)
both_mean_false_posCV = np.mean(false_positive_rate_coef[:,:,:],axis=2)
both_mean_true_posCV = np.mean(true_positive_rate_coef[:,:,:],axis=2)

#### For what value of the false positive rate is the true positive rate 100%? ####
what100tpr_clumping = np.empty((2,true_positive_rate_clumping.shape[2]))
for n in range(0,true_positive_rate_clumping.shape[2]):
    BSAwhere100trp = (true_positive_rate_clumping[0,:,n] > .999)
    BSBwhere100trp = (true_positive_rate_clumping[1,:,n] > .999)
    what100tpr_clumping[0,n] = np.min(false_positive_rate_clumping[0,BSAwhere100trp,n])
    what100tpr_clumping[1,n] = np.min(false_positive_rate_clumping[1,BSBwhere100trp,n])

what100tpr_coef = np.empty((2,true_positive_rate_coef.shape[2]))
for n in range(0,true_positive_rate_coef.shape[2]):
    BSAwhere100trp = (true_positive_rate_coef[0,:,n] > .999)
    BSBwhere100trp = (true_positive_rate_coef[1,:,n] > .999)
    what100tpr_coef[0,n] = np.min(false_positive_rate_coef[0,BSAwhere100trp,n])
    what100tpr_coef[1,n] = np.min(false_positive_rate_coef[1,BSBwhere100trp,n])


what100tpr_rate = np.empty((2,true_positive_rate.shape[2]))
for n in range(0,true_positive_rate.shape[2]):
    BSAwhere100trp = (true_positive_rate[0,:,n] > .999)
    BSBwhere100trp = (true_positive_rate[1,:,n] > .999)
    what100tpr_rate[0,n] = np.min(false_positive_rate[0,BSAwhere100trp,n])
    what100tpr_rate[1,n] = np.min(false_positive_rate[1,BSBwhere100trp,n])


#### Save out AUC values ####

summaryauc = dict()
summaryauc['aucresort_clumping'] = aucresort_clumping
summaryauc['what100tpr_clumping'] = what100tpr_clumping
summaryauc['aucresort_coef'] = aucresort_coef
summaryauc['what100tpr_coef'] = what100tpr_coef
summaryauc['aucresort_rate'] = aucresort
summaryauc['what100tpr_rate'] = what100tpr_rate
print('Saving summary ROC and AUC statistics...')
sio.savemat('data/AllPatients_summaryauc.mat', summaryauc)


##### PLOTTING ####

#Plotting  
posfacx1 = 40 # Patient number x-position 1
posfacx2 = 25 # Patient number x-position 2
posfacx3 = 60 # Patient number x-position 3
posfacx4 = 17 # Patient number x-position 4
posfacx5 = 100 # Patient number x-position 5
posfacy = 40 # Patient number y-position
widthspace = 0.1 #Space between subplots
ROCpatienty1 = 1.10 # Patient number position in y coordinates in ROC plot for State A
ROCpatienty0 = 1.05 # Patient number position in y coordinates in ROC plot for State B
correctROCpatientx = .015 # Patient number correction position in y coordinates in ROC plot


# Subplot figures for clumping coefficients
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))


for n in range(0,true_positive_rate_clumping.shape[2]):
    ax1.plot(false_positive_rate_clumping[0, :, n], true_positive_rate_clumping[
         0, :, n], color='b', LineWidth=3,alpha=0.25)
    ax1.plot(false_positive_rate_clumping[1, :, n], true_positive_rate_clumping[
         1, :, n], color='g', LineWidth=3,alpha=0.25)
ax1.plot(both_mean_false_positive[0, :], both_mean_true_positive[
         0, :], color='b', LineWidth=6)
allprobs = np.linspace(0., 1., num=rocprecision)
ax1.plot(allprobs, allprobs, color='k', LineWidth=1, LineStyle='--')
ax1.plot(both_mean_false_positive[1, :], both_mean_true_positive[
         1, :], color='g', LineWidth=6)
ax1.set_xlim(-.01, 1.01)
ax1.set_ylim(-.01, 1.01)
ax1.plot(what100tpr_clumping[1,0:nwithgrey],np.ones((nwithgrey))*ROCpatienty1, 'gh', markersize=4, clip_on=False)
ax1.plot(what100tpr_clumping[1,nwithgrey:nmodels], np.ones((nwithoutbrain))*ROCpatienty1, 'g*', markersize=4, clip_on=False)
ax1.plot(what100tpr_clumping[0,0:nwithgrey],np.ones((nwithgrey))*ROCpatienty0, 'bh', markersize=4, clip_on=False)
ax1.plot(what100tpr_clumping[0,nwithgrey:nmodels], np.ones((nwithoutbrain))*ROCpatienty0, 'b*', markersize=4, clip_on=False)
skipit1 = []
for fpr in np.sort(what100tpr_clumping[1,:]):
    thispatientnum = allpatientnum[np.where(fpr == what100tpr_clumping[1,:])[0]][0]
    if thispatientnum not in skipit1:
        if (thispatientnum>9): #Correction for double digits
            posfacx = posfacx2
        else:
            posfacx = posfacx1
        if (thispatientnum == 9) | (thispatientnum==11) | (thispatientnum==5): #Overlap correction
            posfacx = posfacx3
        if thispatientnum == 15: #Overlap correction
            posfacx = posfacx4
        if thispatientnum not in skipit1:
            textstring = str(thispatientnum)
        else:
            textstring = str('')
        theseoverlap = (np.abs(fpr - what100tpr_clumping[1,:]) < .03) & (np.abs(fpr - what100tpr_clumping[1,:]) > 0)
        whereoverlap = np.where(theseoverlap)[0]
        sortoverlap = np.sort(what100tpr_clumping[1,whereoverlap])
        if np.any(theseoverlap):
            for fpr2 in sortoverlap:
                overlappatientnum = allpatientnum[np.where(fpr2 == what100tpr_clumping[1,:])[0]][0]
                if overlappatientnum not in skipit1:
                    if textstring is str(''):
                        textstring += str(overlappatientnum)
                    else:
                        textstring += ',' + str(overlappatientnum)
                    skipit1.append(overlappatientnum)
        ax1.text(fpr-(1/posfacx),(ROCpatienty1 -(1.02/posfacy)) ,textstring,fontsize=(2*fontsize/3),color='g')
        skipit1.append(thispatientnum)
skipit0 = []
for fpr in np.sort(what100tpr_clumping[0,:]):
    thispatientnum = allpatientnum[np.where(fpr == what100tpr_clumping[0,:])[0]][0]
    if thispatientnum not in skipit0:
        if (thispatientnum>9): #Correction for double digits
            posfacx = posfacx2
        else:
            posfacx = posfacx1
        if (thispatientnum == 10) | (thispatientnum==3) | (thispatientnum==7) | (thispatientnum ==6): #Overlap correction
            posfacx = posfacx3
        if thispatientnum == 11: #Overlap correction
            posfacx = posfacx4
        if thispatientnum not in skipit0:
            textstring = str(thispatientnum)
        else:
            textstring = str('')
        theseoverlap = (np.abs(fpr - what100tpr_clumping[0,:]) < .03) & (np.abs(fpr - what100tpr_clumping[0,:]) > 0)
        whereoverlap = np.where(theseoverlap)[0]
        sortoverlap = np.sort(what100tpr_clumping[0,whereoverlap])
        if np.any(theseoverlap):
            for fpr2 in sortoverlap:
                overlappatientnum = allpatientnum[np.where(fpr2 == what100tpr_clumping[0,:])[0]][0]
                if overlappatientnum not in skipit0:
                    if textstring is str(''):
                        textstring += str(overlappatientnum)
                    else:
                        textstring += ',' + str(overlappatientnum)
                    skipit0.append(overlappatientnum)
        ax1.text(fpr-(1/posfacx),(ROCpatienty0 -(1.02/posfacy)) ,textstring,fontsize=(2*fontsize/3),color='b')
        skipit0.append(thispatientnum)
ax1.tick_params(axis='both', which='major', labelsize=fontsize)
ax1.set_ylabel('True positive rate', fontsize=fontsize)
ax1.set_xlabel('False positive rate', fontsize=fontsize)


# Custom legend
custom_lines = [Line2D([0], [0], color='g', lw=4),
                Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='k', lw=3, alpha=0.5),
                Line2D([0], [0], color='k', lw=4),
                Line2D([0], [0], color='k', lw=2, linestyle='--')]
ax1.legend(custom_lines, ['State A',
                          'State B', 'Patient','Average','Chance'], 
                          loc=4, fontsize=fontsize, fancybox=True, framealpha=0.5)


sns.distplot(aucresort_clumping[1,:], hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'g'}, rug_kws={'linewidth':2, 'color': 'g'}, ax=ax2)
sns.distplot(aucresort_clumping[0,:]+.01, hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'b'}, rug_kws={'linewidth':2, 'color': 'b'}, ax=ax2)
ymin, ymax = ax2.get_ylim()
ax2.plot(np.array([.5, .5]), np.array([ymin,ymax]), color='k', LineWidth=2, LineStyle='--')
aucresort_clumping_copy = np.copy(aucresort_clumping)
auc_clumpingorder1 = np.argsort(aucresort_clumping_copy[1,:])[::-1]
auc_clumpingorder0 = np.argsort(aucresort_clumping_copy[0,:])[::-1]
spacingit_clumping1 = np.linspace(start=(float(ymax-ymin)/5),stop=(3*float(ymax-ymin)/4),num=nmodels)[::-1] #Note that these y-value limits are just chosen arbitrarily to view patient numbers
spacingit_clumping0 = np.linspace(start=-.1,stop=(5*float(ymax-ymin)/12),num=nmodels)[::-1] #Note that these y-value limits are just chosen arbitrarily to view patient numbers
spacingit_clumping0[-1] = spacingit_clumping0[-3] #Fix for two points out of plot
spacingit_clumping0[-2] = spacingit_clumping0[-3] #Fix for two points out of plot
markerclumping = np.empty((NLatent,nmodels))
for n in range(nmodels):
    markerclumping[1,auc_clumpingorder1[n]] = spacingit_clumping1[n]
    markerclumping[0,auc_clumpingorder0[n]] = spacingit_clumping0[n]
ax2.plot(aucresort_clumping[1,0:nwithgrey],markerclumping[1,0:nwithgrey], 'gh', markersize=4, clip_on=False)
ax2.plot(aucresort_clumping[0,0:nwithgrey],markerclumping[0,0:nwithgrey], 'bh', markersize=4, clip_on=False)
ax2.plot(aucresort_clumping[1,nwithgrey:nmodels],markerclumping[1,nwithgrey:nmodels], 'g*', markersize=4, clip_on=False)
ax2.plot(aucresort_clumping[0,nwithgrey:nmodels],markerclumping[0,nwithgrey:nmodels], 'b*', markersize=4, clip_on=False)
for n in range(nmodels):
    if (allpatientnum[n]>9):
        posfacx = posfacx2
    else:
        posfacx = posfacx1
    if (allpatientnum[n] == 14): #Line overlap correction
        posfacx = posfacx1
    ax2.text(aucresort_clumping[1,n]-(1/posfacx),markerclumping[1,n]-(float(ymax-ymin)/posfacy),str(allpatientnum[n]),fontsize=(2*fontsize/3),color='g')
    ax2.text(aucresort_clumping[0,n]-(1/posfacx),markerclumping[0,n]-(float(ymax-ymin)/posfacy),str(allpatientnum[n]),fontsize=(2*fontsize/3),color='b')
ax2.set_xlim(-.01, 1.01)
ax2.set_ylim(ymin, ymax)
ax2.set_xlabel('AUC for classification using CC', fontsize=fontsize)
ax2.set(ylabel=None)
ax2.set(yticklabels=[])
ax2.set(yticks=[])
ax2.tick_params(axis='both', which='major', labelsize=fontsize, left='off', right='off', labelleft='off')

custom_lines = [Line2D([0], [0], color='g', lw=4),
                Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='w', marker='h',markersize=12,markerfacecolor='k'),
                Line2D([0], [0], color='w', marker='*',markersize=12,markerfacecolor='k'),
                Line2D([0], [0], color='k', lw=2, linestyle='--')]
ax2.legend(custom_lines, ['State A',
                          'State B', 'Grey matter only', 'All iEEG', 'Chance'], 
                          loc=2, fontsize=fontsize, numpoints=1, ncol=1, fancybox=True, framealpha=0.5)
plt.subplots_adjust(wspace=widthspace)

print('Saving Clumping ROC+AUC distribution figures...')
plt.savefig(('figures/Clumping_redChains_ROC_AUC.png'), dpi=300, format='png')
plt.close()


# Print out information about SOZ prediction
print("The mean and standard deviation of AUC by the clumping coefficients were %.2f +/- %.2f in Brain State A" % (np.mean(aucresort_clumping[1,:]),np.std(aucresort_clumping[1,:])))
print("The mean and standard deviation of AUC by the clumping coefficients were %.2f +/- %.2f in Brain State B" % (np.mean(aucresort_clumping[0,:]),np.std(aucresort_clumping[0,:])))
print("The data of %d of 16 patients yielded clumping coefficients that we deemed predictive of SOZ (AUC >.60) in Brain State A" % (np.sum([aucresort_clumping[1,:] > .6])))
print("The data of %d of 16 patients yielded clumping coefficients that we deemed predictive of SOZ (AUC >.60) in Brain State B" % (np.sum([aucresort_clumping[0,:] > .6])))
print("The data of %d of 16 patients yielded clumping coefficients that we deemed predictive of SOZ (AUC >.60) in at least one Brain State" % np.sum(((aucresort_clumping[0,:] > .6) | (aucresort_clumping[1,:] > .6))))
print("In %d of 16 patients, clumping coefficients differentiated all SOZ channels for FPR less than 60%% in Brain State A" % (np.sum([what100tpr_clumping[1,:] < .6])))
print("In %d of 16 patients, clumping coefficients differentiated all SOZ channels for FPR less than 20%% in Brain State A" % (np.sum([what100tpr_clumping[1,:] < .2])))
print("In %d of 16 patients, clumping coefficients differentiated all SOZ channels for FPR less than 60%% in Brain State B" % (np.sum([what100tpr_clumping[0,:] < .6])))
print("In %d of 16 patients, clumping coefficients differentiated all SOZ channels for FPR less than 20%% in Brain State B" % (np.sum([what100tpr_clumping[0,:] < .2])))



### Subplot figures for coefficients of variation
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))


for n in range(0,true_positive_rate_coef.shape[2]):
    ax1.plot(false_positive_rate_coef[0, :, n], true_positive_rate_coef[
         0, :, n], color='b', LineWidth=3,alpha=0.25)
    ax1.plot(false_positive_rate_coef[1, :, n], true_positive_rate_coef[
         1, :, n], color='g', LineWidth=3,alpha=0.25)
ax1.plot(both_mean_false_posCV[0, :], both_mean_true_posCV[
         0, :], color='b', LineWidth=6)
allprobs = np.linspace(0., 1., num=rocprecision)
ax1.plot(allprobs, allprobs, color='k', LineWidth=1, LineStyle='--')
ax1.plot(both_mean_false_posCV[1, :], both_mean_true_posCV[
         1, :], color='g', LineWidth=6)
ax1.set_xlim(-.01, 1.01)
ax1.set_ylim(-.01, 1.01)
ax1.plot(what100tpr_coef[1,0:nwithgrey],np.ones((nwithgrey))*ROCpatienty1, 'gh', markersize=4, clip_on=False)
ax1.plot(what100tpr_coef[1,nwithgrey:nmodels], np.ones((nwithoutbrain))*ROCpatienty1, 'g*', markersize=4, clip_on=False)
ax1.plot(what100tpr_coef[0,0:nwithgrey],np.ones((nwithgrey))*ROCpatienty0, 'bh', markersize=4, clip_on=False)
ax1.plot(what100tpr_coef[0,nwithgrey:nmodels], np.ones((nwithoutbrain))*ROCpatienty0, 'b*', markersize=4, clip_on=False)
skipit1 = []
for fpr in np.sort(what100tpr_coef[1,:]):
    thispatientnum = allpatientnum[np.where(fpr == what100tpr_coef[1,:])[0]][0]
    if thispatientnum not in skipit1:
        if (thispatientnum>9): #Correction for double digits
            posfacx = posfacx2
        else:
            posfacx = posfacx1
        if  (thispatientnum==14): #Overlap correction
            posfacx = posfacx3
        if (thispatientnum == 9) |(thispatientnum == 15):
            posfacx = posfacx5
        if thispatientnum not in skipit1:
            textstring = str(thispatientnum)
        else:
            textstring = str('')
        theseoverlap = (np.abs(fpr - what100tpr_coef[1,:]) < .03) & (np.abs(fpr - what100tpr_coef[1,:]) > 0)
        whereoverlap = np.where(theseoverlap)[0]
        sortoverlap = np.sort(what100tpr_coef[1,whereoverlap])
        if np.any(theseoverlap):
            for fpr2 in sortoverlap:
                overlappatientnum = allpatientnum[np.where(fpr2 == what100tpr_coef[1,:])[0]][0]
                if overlappatientnum not in skipit1:
                    if textstring is str(''):
                        textstring += str(overlappatientnum)
                    else:
                        textstring += ',' + str(overlappatientnum)
                    skipit1.append(overlappatientnum)
        ax1.text(fpr-(1/posfacx),(ROCpatienty1 -(1.02/posfacy)) ,textstring,fontsize=(2*fontsize/3),color='g')
        skipit1.append(thispatientnum)
skipit0 = []
what100tpr_coef[0,0] = 0.000001 #Plotting fix correction 
for fpr in np.sort(what100tpr_coef[0,:]):
    thispatientnum = allpatientnum[np.where(fpr == what100tpr_coef[0,:])[0]][0]
    if thispatientnum not in skipit0:
        if (thispatientnum>9): #Correction for double digits
            posfacx = posfacx2
        else:
            posfacx = posfacx1
        if (thispatientnum==14): #Overlap correction
            posfacx = posfacx3
        if (thispatientnum==3) | (thispatientnum==6): #Overlap correction
            posfacx = posfacx5
        if thispatientnum not in skipit0:
            textstring = str(thispatientnum)
        else:
            textstring = str('')
        theseoverlap = (np.abs(fpr - what100tpr_coef[0,:]) < .03) & (np.abs(fpr - what100tpr_coef[0,:]) > 0)
        whereoverlap = np.where(theseoverlap)[0]
        sortoverlap = np.sort(what100tpr_coef[0,whereoverlap])
        if np.any(theseoverlap):
            for fpr2 in sortoverlap:
                overlappatientnum = allpatientnum[np.where(fpr2 == what100tpr_coef[0,:])[0]][0]
                if overlappatientnum not in skipit0:
                    if textstring is str(''):
                        textstring += str(overlappatientnum)
                    else:
                        textstring += ',' + str(overlappatientnum)
                    skipit0.append(overlappatientnum)
        ax1.text(fpr-(1/posfacx),(ROCpatienty0 -(1.02/posfacy)) ,textstring,fontsize=(2*fontsize/3),color='b')
        skipit0.append(thispatientnum)
ax1.tick_params(axis='both', which='major', labelsize=fontsize)
ax1.set_ylabel('True positive rate', fontsize=fontsize)
ax1.set_xlabel('False positive rate', fontsize=fontsize)


# Custom legend
custom_lines = [Line2D([0], [0], color='g', lw=4),
                Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='k', lw=3, alpha=0.5),
                Line2D([0], [0], color='k', lw=4),
                Line2D([0], [0], color='k', lw=2, linestyle='--')]
ax1.legend(custom_lines, ['State A',
                          'State B', 'Patient','Average','Chance'], 
                          loc=4, fontsize=fontsize, fancybox=True, framealpha=0.5)


sns.distplot(aucresort_coef[1,:], hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'g'}, rug_kws={'linewidth':2, 'color': 'g'}, ax=ax2)
sns.distplot(aucresort_coef[0,:]+.01, hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'b'}, rug_kws={'linewidth':2, 'color': 'b'}, ax=ax2)
ymin, ymax = ax2.get_ylim()
ax2.plot(np.array([.5, .5]), np.array([ymin,ymax]), color='k', LineWidth=2, LineStyle='--')
aucresort_coef_copy = np.copy(aucresort_coef)
auc_coeforder1 = np.argsort(aucresort_coef_copy[1,:])[::-1]
auc_coeforder0 = np.argsort(aucresort_coef_copy[0,:])[::-1]
spacingit_coef1 = np.linspace(start=(float(ymax-ymin)/5),stop=(3*float(ymax-ymin)/4),num=nmodels)[::-1] #Note that these y-value limits are just chosen arbitrarily to view patient numbers
spacingit_coef0 = np.linspace(start=-.1,stop=(5*float(ymax-ymin)/12),num=nmodels)[::-1] #Note that these y-value limits are just chosen arbitrarily to view patient numbers
spacingit_coef0[-1] = spacingit_coef0[-3] #Fix for two points out of plot
spacingit_coef0[-2] = spacingit_coef0[-3] #Fix for two points out of plot
markercoef = np.empty((NLatent,nmodels))
for n in range(nmodels):
    markercoef[1,auc_coeforder1[n]] = spacingit_coef1[n]
    markercoef[0,auc_coeforder0[n]] = spacingit_coef0[n]
ax2.plot(aucresort_coef[1,0:nwithgrey],markercoef[1,0:nwithgrey], 'gh', markersize=4, clip_on=False)
ax2.plot(aucresort_coef[0,0:nwithgrey],markercoef[0,0:nwithgrey], 'bh', markersize=4, clip_on=False)
ax2.plot(aucresort_coef[1,nwithgrey:nmodels],markercoef[1,nwithgrey:nmodels], 'g*', markersize=4, clip_on=False)
ax2.plot(aucresort_coef[0,nwithgrey:nmodels],markercoef[0,nwithgrey:nmodels], 'b*', markersize=4, clip_on=False)
for n in range(nmodels):
    if (allpatientnum[n]>9):
        posfacx = posfacx2
    else:
        posfacx = posfacx1
    if (allpatientnum[n] == 23): #Line overlap correction
        posfacx = posfacx1
    ax2.text(aucresort_coef[1,n]-(1/posfacx),markercoef[1,n]-(float(ymax-ymin)/posfacy),str(allpatientnum[n]),fontsize=(2*fontsize/3),color='g')
    ax2.text(aucresort_coef[0,n]-(1/posfacx),markercoef[0,n]-(float(ymax-ymin)/posfacy),str(allpatientnum[n]),fontsize=(2*fontsize/3),color='b')
ax2.set_xlim(-.01, 1.01)
ax2.set_ylim(ymin, ymax)
ax2.set_xlabel('AUC for classification using CV', fontsize=fontsize)
ax2.set(ylabel=None)
ax2.set(yticklabels=[])
ax2.set(yticks=[])
ax2.tick_params(axis='both', which='major', labelsize=fontsize, left='off', right='off', labelleft='off')

custom_lines = [Line2D([0], [0], color='g', lw=4),
                Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='w', marker='h',markersize=12,markerfacecolor='k'),
                Line2D([0], [0], color='w', marker='*',markersize=12,markerfacecolor='k'),
                Line2D([0], [0], color='k', lw=2, linestyle='--')]
ax2.legend(custom_lines, ['State A',
                          'State B', 'Grey matter only', 'All iEEG', 'Chance'], 
                          loc=2, fontsize=fontsize, numpoints=1, ncol=1, fancybox=True, framealpha=0.5)
plt.subplots_adjust(wspace=widthspace)

print('Saving CV ROC+AUC distribution figures...')
plt.savefig(('figures/CV_redChains_ROC_AUC.png'), dpi=300, format='png')
plt.close()


# Print out information about SOZ prediction
print("The mean and standard deviation of AUC by the coefficients of variation were %.2f +/- %.2f in Brain State A" % (np.mean(aucresort_coef[1,:]),np.std(aucresort_coef[1,:])))
print("The mean and standard deviation of AUC by the coefficients of variation were %.2f +/- %.2f in Brain State B" % (np.mean(aucresort_coef[0,:]),np.std(aucresort_coef[0,:])))
print("The data of %d of 16 patients yielded coefficients of variation that we deemed predictive of SOZ (AUC >.60) in Brain State A" % (np.sum([aucresort_coef[1,:] > .6])))
print("The data of %d of 16 patients yielded coefficients of variation that we deemed predictive of SOZ (AUC >.60) in Brain State B" % (np.sum([aucresort_coef[0,:] > .6])))
print("The data of %d of 16 patients yielded coefficients of variation that we deemed predictive of SOZ (AUC >.60) in at least one Brain State" % np.sum(((aucresort_coef[0,:] > .6) | (aucresort_coef[1,:] > .6))))
print("In %d of 16 patients, coefficients of variation differentiated all SOZ channels for FPR less than 60%% in Brain State A" % (np.sum([what100tpr_coef[1,:] < .6])))
print("In %d of 16 patients, coefficients of variation differentiated all SOZ channels for FPR less than 20%% in Brain State A" % (np.sum([what100tpr_coef[1,:] < .2])))
print("In %d of 16 patients, coefficients of variation differentiated all SOZ channels for FPR less than 60%% in Brain State B" % (np.sum([what100tpr_coef[0,:] < .6])))
print("In %d of 16 patients, coefficients of variation differentiated all SOZ channels for FPR less than 20%% in Brain State B" % (np.sum([what100tpr_coef[0,:] < .2])))



## Generate HFO rate figures for Model 2

true_positive_rate_rate = true_positive_rate
false_positive_rate_rate = false_positive_rate
both_mean_false_rate = np.mean(false_positive_rate[:,:,0:(len(clumping_with_grey))],axis=2)
both_mean_true_rate = np.mean(true_positive_rate[:,:,0:(len(clumping_with_grey))],axis=2)
aucresort_rate = aucresort


### Subplot figures for HFO rates
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))


for n in range(0,true_positive_rate_rate.shape[2]):
    ax1.plot(false_positive_rate_rate[0, :, n], true_positive_rate_rate[
         0, :, n], color='b', LineWidth=3,alpha=0.25)
    ax1.plot(false_positive_rate_rate[1, :, n], true_positive_rate_rate[
         1, :, n], color='g', LineWidth=3,alpha=0.25)
ax1.plot(both_mean_false_rate[0, :], both_mean_true_rate[
         0, :], color='b', LineWidth=6)
allprobs = np.linspace(0., 1., num=rocprecision)
ax1.plot(allprobs, allprobs, color='k', LineWidth=1, LineStyle='--')
ax1.plot(both_mean_false_rate[1, :], both_mean_true_rate[
         1, :], color='g', LineWidth=6)
ax1.set_xlim(-.01, 1.01)
ax1.set_ylim(-.01, 1.01)
ax1.plot(what100tpr_rate[1,0:nwithgrey],np.ones((nwithgrey))*ROCpatienty1, 'gh', markersize=4, clip_on=False)
ax1.plot(what100tpr_rate[1,nwithgrey:nmodels], np.ones((nwithoutbrain))*ROCpatienty1, 'g*', markersize=4, clip_on=False)
ax1.plot(what100tpr_rate[0,0:nwithgrey],np.ones((nwithgrey))*ROCpatienty0, 'bh', markersize=4, clip_on=False)
ax1.plot(what100tpr_rate[0,nwithgrey:nmodels], np.ones((nwithoutbrain))*ROCpatienty0, 'b*', markersize=4, clip_on=False)
skipit1 = []
for fpr in np.sort(what100tpr_rate[1,:]):
    thispatientnum = allpatientnum[np.where(fpr == what100tpr_rate[1,:])[0]][0]
    if thispatientnum not in skipit1:
        if (thispatientnum>9): #Correction for double digits
            posfacx = posfacx2
        else:
            posfacx = posfacx1
        if  (thispatientnum==3): #Overlap correction
            posfacx = posfacx2
        if (thispatientnum == 4):
            posfacx = -posfacx3
        if thispatientnum not in skipit1:
            textstring = str(thispatientnum)
        else:
            textstring = str('')
        theseoverlap = (np.abs(fpr - what100tpr_rate[1,:]) < .03) & (np.abs(fpr - what100tpr_rate[1,:]) > 0)
        whereoverlap = np.where(theseoverlap)[0]
        sortoverlap = np.sort(what100tpr_rate[1,whereoverlap])
        if np.any(theseoverlap):
            for fpr2 in sortoverlap:
                overlappatientnum = allpatientnum[np.where(fpr2 == what100tpr_rate[1,:])[0]][0]
                if overlappatientnum not in skipit1:
                    if textstring is str(''):
                        textstring += str(overlappatientnum)
                    else:
                        textstring += ',' + str(overlappatientnum)
                    skipit1.append(overlappatientnum)
        ax1.text(fpr-(1/posfacx),(ROCpatienty1 -(1.02/posfacy)) ,textstring,fontsize=(2*fontsize/3),color='g')
        skipit1.append(thispatientnum)
skipit0 = []
for fpr in np.sort(what100tpr_rate[0,:]):
    thispatientnum = allpatientnum[np.where(fpr == what100tpr_rate[0,:])[0]][0]
    if thispatientnum not in skipit0:
        if (thispatientnum>9): #Correction for double digits
            posfacx = posfacx2
        else:
            posfacx = posfacx1
        if (thispatientnum==11): #Overlap correction
            posfacx = posfacx3
        if thispatientnum not in skipit0:
            textstring = str(thispatientnum)
        else:
            textstring = str('')
        theseoverlap = (np.abs(fpr - what100tpr_rate[0,:]) < .03) & (np.abs(fpr - what100tpr_rate[0,:]) > 0)
        whereoverlap = np.where(theseoverlap)[0]
        sortoverlap = np.sort(what100tpr_rate[0,whereoverlap])
        if np.any(theseoverlap):
            for fpr2 in sortoverlap:
                overlappatientnum = allpatientnum[np.where(fpr2 == what100tpr_rate[0,:])[0]][0]
                if overlappatientnum not in skipit0:
                    if textstring is str(''):
                        textstring += str(overlappatientnum)
                    else:
                        textstring += ',' + str(overlappatientnum)
                    skipit0.append(overlappatientnum)
        ax1.text(fpr-(1/posfacx),(ROCpatienty0 -(1.02/posfacy)) ,textstring,fontsize=(2*fontsize/3),color='b')
        skipit0.append(thispatientnum)
ax1.tick_params(axis='both', which='major', labelsize=fontsize)
ax1.set_ylabel('True positive rate', fontsize=fontsize)
ax1.set_xlabel('False positive rate', fontsize=fontsize)


# Custom legend
custom_lines = [Line2D([0], [0], color='g', lw=4),
                Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='k', lw=3, alpha=0.5),
                Line2D([0], [0], color='k', lw=4),
                Line2D([0], [0], color='k', lw=2, linestyle='--')]
ax1.legend(custom_lines, ['State A',
                          'State B', 'Patient','Average','Chance'], 
                          loc=4, fontsize=fontsize, fancybox=True, framealpha=0.5)


sns.distplot(aucresort_rate[1,:], hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'g'}, rug_kws={'linewidth':2, 'color': 'g'}, ax=ax2)
sns.distplot(aucresort_rate[0,:]+.01, hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'b'}, rug_kws={'linewidth':2, 'color': 'b'}, ax=ax2)
ymin, ymax = ax2.get_ylim()
ax2.plot(np.array([.5, .5]), np.array([ymin,ymax]), color='k', LineWidth=2, LineStyle='--')
aucresort_rate_copy = np.copy(aucresort_rate)
auc_rateorder1 = np.argsort(aucresort_rate_copy[1,:])[::-1]
auc_rateorder0 = np.argsort(aucresort_rate_copy[0,:])[::-1]
spacingit_rate1 = np.linspace(start=(float(ymax-ymin)/6),stop=(2*float(ymax-ymin)/3),num=nmodels)[::-1] #Note that these y-value limits are just chosen arbitrarily to view patient numbers
spacingit_rate0 = np.linspace(start=-.1,stop=(5*float(ymax-ymin)/12),num=nmodels)[::-1] #Note that these y-value limits are just chosen arbitrarily to view patient numbers
spacingit_rate0[-1] = spacingit_rate0[-3] #Fix for two points out of plot
spacingit_rate0[-2] = spacingit_rate0[-3] #Fix for two points out of plot
markerrate = np.empty((NLatent,nmodels))
for n in range(nmodels):
    markerrate[1,auc_rateorder1[n]] = spacingit_rate1[n]
    markerrate[0,auc_rateorder0[n]] = spacingit_rate0[n]
ax2.plot(aucresort_rate[1,0:nwithgrey],markerrate[1,0:nwithgrey], 'gh', markersize=4, clip_on=False)
ax2.plot(aucresort_rate[0,0:nwithgrey],markerrate[0,0:nwithgrey], 'bh', markersize=4, clip_on=False)
ax2.plot(aucresort_rate[1,nwithgrey:nmodels],markerrate[1,nwithgrey:nmodels], 'g*', markersize=4, clip_on=False)
ax2.plot(aucresort_rate[0,nwithgrey:nmodels],markerrate[0,nwithgrey:nmodels], 'b*', markersize=4, clip_on=False)
for n in range(nmodels):
    if (allpatientnum[n]>9):
        posfacx = posfacx2
    else:
        posfacx = posfacx1
    # if (allpatientnum[n] == 23): #Line overlap correction
    #     posfacx = posfacx1
    ax2.text(aucresort_rate[1,n]-(1/posfacx),markerrate[1,n]-(float(ymax-ymin)/posfacy),str(allpatientnum[n]),fontsize=(2*fontsize/3),color='g')
    ax2.text(aucresort_rate[0,n]-(1/posfacx),markerrate[0,n]-(float(ymax-ymin)/posfacy),str(allpatientnum[n]),fontsize=(2*fontsize/3),color='b')
ax2.set_xlim(-.01, 1.01)
ax2.set_ylim(ymin, ymax)
ax2.set_xlabel('AUC for classification using rate', fontsize=fontsize)
ax2.set(ylabel=None)
ax2.set(yticklabels=[])
ax2.set(yticks=[])
ax2.tick_params(axis='both', which='major', labelsize=fontsize, left='off', right='off', labelleft='off')

custom_lines = [Line2D([0], [0], color='g', lw=4),
                Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='w', marker='h',markersize=12,markerfacecolor='k'),
                Line2D([0], [0], color='w', marker='*',markersize=12,markerfacecolor='k'),
                Line2D([0], [0], color='k', lw=2, linestyle='--')]
ax2.legend(custom_lines, ['State A',
                          'State B', 'Grey matter only', 'All iEEG', 'Chance'], 
                          loc=2, fontsize=fontsize, numpoints=1, ncol=1, fancybox=True, framealpha=0.5)
plt.subplots_adjust(wspace=widthspace)

print('Saving HFO ROC+AUC distribution figures...')
plt.savefig(('figures/Rate_Model2_redChains_ROC_AUC.png'), dpi=300, format='png')
plt.close()


# Print out information about SOZ prediction
print("The mean and standard deviation of AUC by the HFO rates were %.2f +/- %.2f in Brain State A" % (np.mean(aucresort_rate[1,:]),np.std(aucresort_rate[1,:])))
print("The mean and standard deviation of AUC by the HFO rates were %.2f +/- %.2f in Brain State B" % (np.mean(aucresort_rate[0,:]),np.std(aucresort_rate[0,:])))
print("The data of %d of 16 patients yielded HFO rates that we deemed predictive of SOZ (AUC >.60) in Brain State A" % (np.sum([aucresort_rate[1,:] > .6])))
print("The data of %d of 16 patients yielded HFO rates that we deemed predictive of SOZ (AUC >.60) in Brain State B" % (np.sum([aucresort_rate[0,:] > .6])))
print("The data of %d of 16 patients yielded HFO rates that we deemed predictive of SOZ (AUC >.60) in at least one Brain State" % np.sum(((aucresort_rate[0,:] > .6) | (aucresort_rate[1,:] > .6))))
print("In %d of 16 patients, HFO rates differentiated all SOZ channels for FPR less than 60%% in Brain State A" % (np.sum([what100tpr_rate[1,:] < .6])))
print("In %d of 16 patients, HFO rates differentiated all SOZ channels for FPR less than 20%% in Brain State A" % (np.sum([what100tpr_rate[1,:] < .2])))
print("In %d of 16 patients, HFO rates differentiated all SOZ channels for FPR less than 60%% in Brain State B" % (np.sum([what100tpr_rate[0,:] < .6])))
print("In %d of 16 patients, HFO rates differentiated all SOZ channels for FPR less than 20%% in Brain State B" % (np.sum([what100tpr_rate[0,:] < .2])))



## Aggregate ROC curves for all electrodes across patients
allclumpingvals = np.row_stack((np.asarray(BSBallclumpingvals),np.asarray(BSAallclumpingvals)))
allcvs= np.row_stack((np.asarray(BSAallcvs),np.asarray(BSBallcvs)))
allhforates= np.row_stack((np.asarray(BSAallhforates),np.asarray(BSBallhforates)))
sozlabels = np.asarray(allsozlabels)
NSOZ = np.sum(sozlabels == 1)
NNonSOZ = np.sum(sozlabels == 0)
rocprecision = 10000


# Aggregate ROC curves by clumping coefficient
tpr_aggregate_clumping = np.empty((NLatent, rocprecision))
fpr_aggregate_clumping = np.empty((NLatent, rocprecision))
aggregate_clumping_cutoffs = np.empty((NLatent, rocprecision))

for n in range(0, NLatent):
    index = 0
    clump_cutoffs = np.linspace(0., np.max(allclumpingvals[n, :]), num=rocprecision)
    aggregate_clumping_cutoffs[n,:] = clump_cutoffs
    for clumping in clump_cutoffs:
        little_clumping = np.sum((allclumpingvals[n,:] <= clumping) & (sozlabels==1))
        tpr_aggregate_clumping[n, index] = float(little_clumping) / NSOZ
        false_SOZ = np.sum((allclumpingvals[n,:] <= clumping) & (sozlabels==0))
        fpr_aggregate_clumping[n, index] = float(false_SOZ) / NNonSOZ
        index += 1



plt.figure(dpi=300)
plt.plot(fpr_aggregate_clumping[1, :], tpr_aggregate_clumping[
         1, :], color='g', LineWidth=6)
allprobs = np.linspace(0., 1., num=rocprecision)
plt.plot(allprobs, allprobs, color='k', LineWidth=6, LineStyle='--')
plt.plot(fpr_aggregate_clumping[0, :], tpr_aggregate_clumping[
         0, :], color='b', LineWidth=6)
whereannotate = np.where((fpr_aggregate_clumping[1,:] < .1) & (tpr_aggregate_clumping[1,:] > .25))[0][0]
plt.annotate('CC2A=%.2f' % aggregate_clumping_cutoffs[1, whereannotate], xy=( fpr_aggregate_clumping[1, whereannotate], tpr_aggregate_clumping[1, whereannotate]), xycoords='data', 
        xytext=( .2, .15), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='g'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color='k')
whereannotate = np.where((fpr_aggregate_clumping[1,:] < .2) & (tpr_aggregate_clumping[1,:] > .5))[0][0]
plt.annotate('CC2A=%.2f' % aggregate_clumping_cutoffs[1, whereannotate], xy=( fpr_aggregate_clumping[1, whereannotate], tpr_aggregate_clumping[1, whereannotate]), xycoords='data', 
        xytext=( .3, .25), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='g'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color='k')
whereannotate = np.where((fpr_aggregate_clumping[1,:] < .3) & (tpr_aggregate_clumping[1,:] > .7))[0][0]
plt.annotate('CC2A=%.2f' % aggregate_clumping_cutoffs[1, whereannotate], xy=( fpr_aggregate_clumping[1, whereannotate], tpr_aggregate_clumping[1, whereannotate]), xycoords='data', 
        xytext=( .6, .35), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='g'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color='k')
whereannotate_zeta1A = np.argmin(np.abs(aggregate_clumping_cutoffs[1, :] - 1))
plt.annotate('CC2A=%.2f' % aggregate_clumping_cutoffs[1, whereannotate_zeta1A], xy=( fpr_aggregate_clumping[1, whereannotate_zeta1A], tpr_aggregate_clumping[1, whereannotate_zeta1A]), xycoords='data', 
        xytext=( .7, .5), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='g'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color='k')
whereannotate_zeta1B = np.argmin(np.abs(aggregate_clumping_cutoffs[0, :] - 1))
plt.annotate('CC2B=%.2f' % aggregate_clumping_cutoffs[0, whereannotate_zeta1B], xy=( fpr_aggregate_clumping[0, whereannotate_zeta1B], tpr_aggregate_clumping[0, whereannotate_zeta1B]), xycoords='data', 
        xytext=( .4, .35), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='b'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color='k')
whereannotate_tpr_1A = np.where((tpr_aggregate_clumping[1,:] > .999))[0][0]
plt.annotate('CC2A=%.2f' % aggregate_clumping_cutoffs[1, whereannotate_tpr_1A], xy=( fpr_aggregate_clumping[1, whereannotate_tpr_1A], tpr_aggregate_clumping[1, whereannotate_tpr_1A]), xycoords='data', 
        xytext=( .8, .65), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='g'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color='k')
whereannotate_tpr_1B = np.where((tpr_aggregate_clumping[0,:] > .999))[0][0]
plt.annotate('CC2B=%.2f' % aggregate_clumping_cutoffs[0, whereannotate_tpr_1B], xy=( fpr_aggregate_clumping[0, whereannotate_tpr_1B], tpr_aggregate_clumping[0, whereannotate_tpr_1B]), xycoords='data', 
        xytext=( .9, .8), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='b'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color='k')
plt.xlim(-.01, 1.01)
plt.ylim(-.01, 1.01)
plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.ylabel('True positive rate', fontsize=fontsize)
plt.xlabel('False positive rate', fontsize=fontsize,labelpad=2)
# plt.title('ROC Curve by CC for all patients', fontsize=fontsize)
custom_lines = [Line2D([0], [0], color='g', lw=4),
                Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='k', lw=2, linestyle='--')]
plt.legend(custom_lines, ['Aggregate State A',
                          'Aggregate State B', 'Chance'],
                          loc=4, fontsize=fontsize, fancybox=True, framealpha=0.5)
print('Saving aggregate ROC curve for clumping parameters...')
plt.savefig(('figures/Aggregate_ROC_Clumping.png'), dpi=300, format='png')
plt.close()

print("The total number of channels used across models was %d" % (np.sum(sortedNChans)))
print("The false positive rate is %.2f in Brain State A when the clumping coefficient is equal to 1 " % (fpr_aggregate_clumping[1, whereannotate_zeta1A]))
print("The true positive rate is %.2f in Brain State A when the clumping coefficient is equal to 1" % (tpr_aggregate_clumping[1, whereannotate_zeta1A]))
print("The false positive rate is %.2f in Brain State A when the true positive rate is 1" % (fpr_aggregate_clumping[1, whereannotate_tpr_1A]))

print("The false positive rate is %.2f in Brain State B when the clumping coefficient is equal to 1 " % (fpr_aggregate_clumping[0, whereannotate_zeta1B]))
print("The true positive rate is %.2f in Brain State B when the clumping coefficient is equal to 1" % (tpr_aggregate_clumping[0, whereannotate_zeta1B]))
print("The false positive rate is %.2f in Brain State B when the true positive rate is 1" % (fpr_aggregate_clumping[0, whereannotate_tpr_1B]))

