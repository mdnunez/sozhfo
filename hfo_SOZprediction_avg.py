# hfo_SOZprediction_avg.py - Evaluates average of models to predict SOZ
#
# Copyright (C) 2020 Michael D. Nunez, <mdnunez1@uci.edu>
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
# 03/26/20      Michael Nunez        Load ''flip_labels'' from hfo_sleepeval.py, remove patient [Former patient] (Patient9 mislabeled)
# 04/01/20      Michael Nunez         Fix connective lines in AUC distribution plots, split CV ROC plot into Localization and no Localization
# 04/02/20      Michael Nunez         Use of fancybox and transparency in legends
# 04/03/20      Michael Nunez             Keep track of electrode numbers
# 04/30/20      Michael Nunez           Label patients with numbers, fix Patient 5 which did not have localization for the analysis
# 05/28/20      Michael Nunez             Remove patient identifiers

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

# Imports
import numpy as np
import scipy.io as sio
from scipy import stats
import os.path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats.stats import pearsonr
import seaborn as sns
from matplotlib import rc

rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

fontsize = 10
fontsize2 = 10
fontsizelarge = 18
rocprecision = 1000

def diagnostic(insamples):
    """
    Returns Rhat (measure of convergence, less is better with an approximate
    1.10 cutoff) and Neff, number of effective samples).

    Reference: Gelman, A., Carlin, J., Stern, H., & Rubin D., (2004).
              Bayesian Data Analysis (Second Edition). Chapman & Hall/CRC:
              Boca Raton, FL.


    Parameters
    ----------
    insamples: dic
        Sampled values of monitored variables as a dictionary where keys
        are variable names and values are numpy arrays with shape:
        (dim_1, dim_n, iterations, chains). dim_1, ..., dim_n describe the
        shape of variable in JAGS model.

    Returns
    -------
    dict:
        Rhat for each variable. Prints Maximum Rhat
    """

    result = {}  # Initialize dictionary
    maxrhats = np.zeros((len(insamples.keys())), dtype=float)
    allkeys ={} # Initialize dictionary
    keyindx = 0
    for key in insamples.keys():
        if key[0] != '_':
            result[key] = {}

            possamps = insamples[key]

            nchains = possamps.shape[-1]
            nsamps = possamps.shape[-2]
            # Mean of each chain
            chainmeans = np.mean(possamps, axis=-2)
            # Global mean of each parameter
            globalmean = np.mean(chainmeans, axis=-1)
            result[key]['mean'] = globalmean
            globalmeanext = np.expand_dims(
                globalmean, axis=-1)  # Expand the last dimension
            globalmeanext = np.repeat(
                globalmeanext, nchains, axis=-1)  # For differencing
            # Between-chain variance
            between = np.sum(np.square(chainmeans - globalmeanext),
                             axis=-1) * nsamps / (nchains - 1.)
            # Mean of the variances of each chain
            within = np.mean(np.var(possamps, axis=-2), axis=-1)
            # Total estimated variance
            totalestvar = (1. - (1. / nsamps)) * \
                within + (1. / nsamps) * between
            # Rhat (Gelman-Rubin statistic)
            temprhat = np.sqrt(totalestvar / within)
            maxrhats[keyindx] = np.nanmax(temprhat) # Ignore NANs
            allkeys[keyindx] = key
            result[key]['rhat'] = temprhat
            keyindx += 1
            # Possible number of effective samples?
            # Geweke statistic?
    print "Maximum Rhat: %3.2f of variable %s" % (np.max(maxrhats),allkeys[np.argmax(maxrhats)])
    return result



# Load generated smaples from Model 1
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
allmodels = np.concatenate((models_with_grey, models_with_outbrain))
allengel = np.concatenate((engel_with_grey, engel_with_outbrain))

##Average clumping parameters
clumping_with_grey = ['/data/posterior_samples/jagsmodel_model12samples_Patient1_grey_2LatentFeb_27_19_19_53',
'/data/posterior_samples/jagsmodel_model12samples_Patient2_grey_2LatentFeb_28_19_19_04',
'/data/posterior_samples/jagsmodel_model12samples_Patient3_grey_2LatentMar_04_19_17_32',
'/data/posterior_samples/jagsmodel_model12samples_Patient4_grey_2LatentMar_15_19_19_07',
'/data/posterior_samples/jagsmodel_model12samples_Patient9_grey_2LatentApr_09_19_19_48',
'/data/posterior_samples/jagsmodel_model12samples_Patient10_grey_2LatentJul_15_19_13_31',
'/data/posterior_samples/jagsmodel_model12samples_Patient11_grey_2LatentSep_22_19_01_47',
'/data/posterior_samples/jagsmodel_model12samples_Patient13_grey_2LatentJul_18_19_12_03',
'/data/posterior_samples/jagsmodel_model12samples_Patient14_grey_2LatentJul_09_19_17_57',
'/data/posterior_samples/jagsmodel_model12samples_Patient15_grey_2LatentJul_11_19_12_50']

sixchains = np.array([0, 1, 2, 3, 4, 5]) # All 6 chains
chains_with_grey = [sixchains,
sixchains,
sixchains,
sixchains,
np.array([0, 1, 4]),
np.array([0, 2, 3, 4, 5]),
np.array([0, 1, 5]),
sixchains,
np.array([0, 1, 3, 4, 5]),
np.array([1, 2, 3, 4, 5])]

patientnum_with_grey = np.array([1, 2, 3, 4, 9, 10, 11, 13, 14, 15])


clumping_with_outbrain = ['/data/posterior_samples/jagsmodel_model12samples_Patient6_grey_2LatentMar_26_19_12_43',
'/data/posterior_samples/jagsmodel_model12samples_Patient7_grey_2LatentApr_02_19_19_39',
'/data/posterior_samples/jagsmodel_model12samples_Patient8_grey_2LatentMar_27_19_13_20',
'/data/posterior_samples/jagsmodel_model12samples_Patient12_grey_2LatentApr_09_19_19_51',
'/data/posterior_samples/jagsmodel_model12samples_Patient5_grey_2LatentMar_11_19_11_55',
'/data/posterior_samples/jagsmodel_model12samples_Patient16_grey_2LatentJul_14_19_23_05']

chains_with_outbrain = [np.array([1, 3, 4]),
np.array([0, 1, 2, 3, 5]),
np.array([1, 2, 4]),
np.array([4]),
sixchains,
np.array([1, 3, 5])]

patientnum_with_outbrain = np.array([6, 7, 8, 12, 5, 16])


allclumpingmodels = np.concatenate((clumping_with_grey, clumping_with_outbrain))
allchains = np.concatenate((chains_with_grey, chains_with_outbrain))
allpatientnum = np.concatenate((patientnum_with_grey, patientnum_with_outbrain))

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

allNSOZ = np.zeros((allclumpingmodels.size))
allNNonSOZ = np.zeros((allclumpingmodels.size))

BS1allclumpingvals =[] #Track all median posterior clumping parameter values in Brain State 1
BS1allhforates = [] #Track all median posterior HFO rates in Brain State 1
BS1allcvs = [] #Track all mean posterior CV in Brain State 1
BS2allclumpingvals =[] #Track all median posterior clumping parameter values in Brain State 1
BS2allhforates = [] #Track all median posterior HFO rates in Brain State 1
BS2allcvs = [] #Track all mean posterior CV in Brain State 1
allsozlabels =[]

Nwindows = [] #Track the number of 5 minute windows used in analysis of interictal data

model_indx = 0
for model in allclumpingmodels:
    patient = model[(model.find('Patient')):(model.find('Patient')+9)]


    print 'Loading data from patient %s' % patient
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
        print 'Saving data from patient %s with states reordered' % patient
        samples['sortorder'] = sortorder
        sio.savemat('%s_reordered.mat' % model, samples)

    samples_relevant = dict(latent_lambda=samples[
                        'latent_lambda'][:,:,:,keepchains], pi=samples['pi'][:,:,:,keepchains],
                        state_lambda=samples['state_lambda'][:,:,:,keepchains],
                        state_std=samples['state_std'][:,:,:,keepchains],
                        clumping=samples['clumping'][:,:,:,keepchains],
                        coef_variation=samples['coef_variation'][:,:,:,keepchains])

    if keepchains.shape[0] > 1:
        diags = diagnostic(samples_relevant)
    else:
        print "Keeping only one chain"

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
    print "The correlation of rates between the first two states is %.2f" % (corrval[0])
    corrate[model_indx] = corrval[0] 

    corrval = pearsonr(allclumping[0, :], allclumping[1, :])
    print "The correlation of clumping between the first two states is %.2f" % (corrval[0])
    corclumping[model_indx] = corrval[0] 

    corrval = pearsonr(allcoef[0, :], allcoef[1, :])
    print "The correlation of the coefficients of variation between the first two states is %.2f" % (corrval[0])
    corcoef[model_indx] = corrval[0] 

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

    print "The number of non-seizure onset zone electrodes is %d" % (NNonSOZ)
    print "The number of seizure onset zone electrodes is %d" % (NSOZ)

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
        auc_clumping[n, model_indx] = np.abs(np.sum(np.multiply(np.diff(np.squeeze(false_positive_rate_clumping[n, rocprecision:None:-1,model_indx])),true_positive_rate_clumping[n, rocprecision:0:-1,model_indx])))
    if samples['flip_labels']: #Always plot the generator state that is more predictive of delta power as Model 2
        true_positive_rate_clumping[:, :, model_indx] = true_positive_rate_clumping[::-1,:,model_indx] #Flip model labels 1 and 0
        false_positive_rate_clumping[:, :, model_indx] = false_positive_rate_clumping[::-1,:,model_indx] #Flip model labels 1 and 0
        aucresort_clumping[:, model_indx] = auc_clumping[::-1, model_indx]
        perfect_clumping[:, model_indx] = perfect_clumping[::-1, model_indx]
        for e in range(0,Nelecs):
            BS1allclumpingvals.append(allclumping[1,e])
            BS2allclumpingvals.append(allclumping[0,e])
    else:
        aucresort_clumping[:, model_indx] = auc_clumping[:, model_indx]
        perfect_clumping[:, model_indx] = perfect_clumping[:, model_indx]
        for e in range(0,Nelecs):
            BS1allclumpingvals.append(allclumping[0,e])
            BS2allclumpingvals.append(allclumping[1,e])
    # print 'Saving data from patient %s with states reordered...' % patient
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
        auc_coef[n, model_indx] = np.abs(np.sum(np.multiply(np.diff(np.squeeze(false_positive_rate_coef[n, rocprecision:None:-1,model_indx])),true_positive_rate_coef[n, rocprecision:0:-1,model_indx])))
    if samples['flip_labels']: #Always plot the generator state that is more predictive of delta power as Model 2
        true_positive_rate_coef[:, :, model_indx] = true_positive_rate_coef[::-1,:,model_indx] #Flip model labels 1 and 0
        false_positive_rate_coef[:, :, model_indx] = false_positive_rate_coef[::-1,:,model_indx] #Flip model labels 1 and 0
        aucresort_coef[:, model_indx] = auc_coef[::-1, model_indx]
        perfect_coef[:, model_indx] = perfect_coef[::-1, model_indx]
        for e in range(0,Nelecs):
            BS1allcvs.append(allcoef[1,e])
            BS2allcvs.append(allcoef[0,e])
    else:
        aucresort_coef[:, model_indx] = auc_coef[:, model_indx]
        perfect_coef[:, model_indx] = perfect_coef[:, model_indx]
        for e in range(0,Nelecs):
            BS1allcvs.append(allcoef[0,e])
            BS2allcvs.append(allcoef[1,e])

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
        auc[n, model_indx] = np.abs(np.sum(np.multiply(np.diff(np.squeeze(false_positive_rate[n, rocprecision:None:-1,model_indx])),true_positive_rate[n, rocprecision:0:-1,model_indx])))
    if samples['flip_labels']: #Always plot the generator state that is more predictive of delta power as Model 2
        true_positive_rate[:, :, model_indx] = true_positive_rate[::-1,:,model_indx] #Flip model labels 1 and 0
        false_positive_rate[:, :, model_indx] = false_positive_rate[::-1,:,model_indx] #Flip model labels 1 and 0
        aucresort[:, model_indx] = auc[::-1, model_indx]
        perfect_rate[:, model_indx] = perfect_rate[::-1, model_indx]
        for e in range(0,Nelecs):
            BS1allhforates.append(allrates[1,e])
            BS2allhforates.append(allrates[0,e])
    else:
        aucresort[:, model_indx] = auc[:, model_indx]
        perfect_rate[:, model_indx] = perfect_rate[:, model_indx]
        for e in range(0,Nelecs):
            BS1allhforates.append(allrates[0,e])
            BS2allhforates.append(allrates[1,e])
    model_indx += 1


Nhours = np.asarray(Nwindows) / float(12) #Convert from 5 minute windows to hours
print "Mean and standard deviation of number hours is %.2f +/- %.2f" % (np.mean(Nhours), np.std(Nhours))


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

# For what value of the false positive rate is the true positive rate 100%?
what100tpr_clumping = np.empty((2,true_positive_rate_clumping.shape[2]))
for n in range(0,true_positive_rate_clumping.shape[2]):
    BS1where100trp = (true_positive_rate_clumping[0,:,n] > .999)
    BS2where100trp = (true_positive_rate_clumping[1,:,n] > .999)
    what100tpr_clumping[0,n] = np.min(false_positive_rate_clumping[0,BS1where100trp,n])
    what100tpr_clumping[1,n] = np.min(false_positive_rate_clumping[1,BS2where100trp,n])

what100tpr_coef = np.empty((2,true_positive_rate_coef.shape[2]))
for n in range(0,true_positive_rate_coef.shape[2]):
    BS1where100trp = (true_positive_rate_coef[0,:,n] > .999)
    BS2where100trp = (true_positive_rate_coef[1,:,n] > .999)
    what100tpr_coef[0,n] = np.min(false_positive_rate_coef[0,BS1where100trp,n])
    what100tpr_coef[1,n] = np.min(false_positive_rate_coef[1,BS2where100trp,n])

# Subplot figures for clumping coefficients
fig, (ax1,ax2) = plt.subplots(2,1)

# figCC = plt.figure()
# gsCC = gridspec.GridSpec(nrows=2, ncols=2, left=0.05, wspace=0.05)
# ax1a = figCC.add_subplot(gsCC[0, 0]) #Spans the first row and first column
# ax1b = figCC.add_subplot(gsCC[0, 1]) #Spans the first row and second column
# ax2 = figCC.add_subplot(gsCC[-1, :]) #Spans the second (last) row but all columns

for n in range(0,true_positive_rate_clumping.shape[2]):
    ax1.plot(false_positive_rate_clumping[0, :, n], true_positive_rate_clumping[
         0, :, n], color='b', LineWidth=3,alpha=0.25)
    ax1.plot(false_positive_rate_clumping[1, :, n], true_positive_rate_clumping[
         1, :, n], color='g', LineWidth=3,alpha=0.25)
ax1.plot(both_mean_false_positive[0, :], both_mean_true_positive[
         0, :], color='b', LineWidth=6)
allprobs = np.linspace(0., 1., num=rocprecision)
ax1.plot(allprobs, allprobs, color='k', LineWidth=6, LineStyle='--')
ax1.plot(both_mean_false_positive[1, :], both_mean_true_positive[
         1, :], color='g', LineWidth=6)
ax1.set_xlim(-.01, 1.01)
ax1.set_ylim(-.01, 1.01)
ax1.tick_params(axis='both', which='major', labelsize=fontsize)
ax1.set_ylabel('True positive rate', fontsize=fontsize)
ax1.set_xlabel('False positive rate', fontsize=fontsize)
ax1.set_title('ROC Curves for SOZ Prediction by HFO Clumping Parameters', fontsize=fontsize)


# Custom legend
custom_lines = [Line2D([0], [0], color='g', lw=6),
                Line2D([0], [0], color='b', lw=6),
                Line2D([0], [0], color='k', lw=3, alpha=0.5),
                Line2D([0], [0], color='k', lw=6, alpha=0.5)]
ax1.legend(custom_lines, ['State 1',
                          'State 2', 'Indiv. Patient','Average'], 
                          loc=4, fontsize=fontsize, fancybox=True, framealpha=0.5)

# for n in range(0,len(clumping_with_grey)):
#     ax1a.plot(false_positive_rate_clumping[0, :, n], true_positive_rate_clumping[
#          0, :, n], color='b', LineWidth=3,alpha=0.25)
#     ax1a.plot(false_positive_rate_clumping[1, :, n], true_positive_rate_clumping[
#          1, :, n], color='g', LineWidth=3,alpha=0.25)
# ax1a.plot(grey_mean_false_positive[0, :], grey_mean_true_positive[
#          0, :], color='b', LineWidth=6)
# allprobs = np.linspace(0., 1., num=rocprecision)
# ax1a.plot(allprobs, allprobs, color='k', LineWidth=6, LineStyle='--')
# ax1a.plot(grey_mean_false_positive[1, :], grey_mean_true_positive[
#          1, :], color='g', LineWidth=6)
# ax1a.set_xlim(-.01, 1.01)
# ax1a.set_ylim(-.01, 1.01)
# ax1a.tick_params(axis='both', which='major', labelsize=fontsize)
# ax1a.set_ylabel('True positive rate', fontsize=fontsize, labelpad=2)
# ax1a.set_xlabel('False positive rate', fontsize=fontsize)
# ax1a.set_title('Clumping w/ Localization', fontsize=fontsize)

# # Custom legend
# custom_lines = [Line2D([0], [0], color='g', lw=6),
#                 Line2D([0], [0], color='b', lw=6),
#                 Line2D([0], [0], color='k', lw=6, linestyle='--'),
#                 Line2D([0], [0], color='k', lw=3, alpha=0.5),
#                 Line2D([0], [0], color='k', lw=6, alpha=0.5)]
# ax1a.legend(custom_lines, ['State 1',
#                           'State 2', 'Chance', 'Indiv. Patient','Average'], 
#                           loc=4, fontsize=fontsize, ncol=2, fancybox=True, framealpha=0.5)



# for n in range(len(clumping_with_grey),true_positive_rate.shape[2]):
#     ax1b.plot(false_positive_rate_clumping[0, :, n], true_positive_rate_clumping[
#          0, :, n], color='b', LineWidth=3,alpha=0.25)
#     ax1b.plot(false_positive_rate_clumping[1, :, n], true_positive_rate_clumping[
#          1, :, n], color='g', LineWidth=3,alpha=0.25)
# ax1b.plot(out_mean_false_positive[0, :], out_mean_true_positive[
#          0, :], color='b', LineWidth=6)
# allprobs = np.linspace(0., 1., num=rocprecision)
# ax1b.plot(allprobs, allprobs, color='k', LineWidth=6, LineStyle='--')
# ax1b.plot(out_mean_false_positive[1, :], out_mean_true_positive[
#          1, :], color='g', LineWidth=6)
# ax1b.set_xlim(-.01, 1.01)
# ax1b.set_xlim(-.01, 1.01)
# ax1b.set_ylim(-.01, 1.01)
# ax1b.tick_params(axis='both', which='major', labelsize=fontsize)
# ax1b.set_xlabel('False positive rate', fontsize=fontsize)
# ax1b.set_title('Clumping w/o Localization', fontsize=fontsize)
# ax1b.tick_params(axis='both', which='major', labelsize=fontsize, left='off', labelleft='off')


sns.distplot(aucresort_clumping[1,:], hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'g'}, rug_kws={'linewidth':2, 'color': 'g'}, ax=ax2)
sns.distplot(aucresort_clumping[0,:]+.01, hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'b'}, rug_kws={'linewidth':2, 'color': 'b'}, ax=ax2)
ymin, ymax = ax2.get_ylim()
ax2.plot(np.array([.5, .5]), np.array([ymin,ymax]), color='k', LineWidth=6, LineStyle='--')
ax2.plot(aucresort_clumping[1,0:len(clumping_with_grey)],np.ones(len(clumping_with_grey))*(float(ymax-ymin)/2), 'go', markersize=12)
ax2.plot(aucresort_clumping[0,0:len(clumping_with_grey)],np.ones(len(clumping_with_grey))*(float(ymax-ymin)/3), 'bo', markersize=12)
ax2.plot(aucresort_clumping[1,len(clumping_with_grey):true_positive_rate.shape[2]],np.ones(len(clumping_with_outbrain))*(float(ymax-ymin)/2), 'g*', markersize=12)
ax2.plot(aucresort_clumping[0,len(clumping_with_grey):true_positive_rate.shape[2]],np.ones(len(clumping_with_outbrain))*(float(ymax-ymin)/3), 'b*', markersize=12)
for n in range(0, aucresort_clumping.shape[1]):
    ax2.plot(aucresort_clumping[:,n], np.array([(float(ymax-ymin)/3), (float(ymax-ymin)/2)]), LineWidth=1, LineStyle='--',color='purple',alpha=.5)
aucsinorder = np.sort(aucresort_clumping[0,:])
aucorder = np.argsort(aucresort_clumping[0,:])
# trackit = -.09
# for n in range(0,allengel.shape[0]):
#     if allengel[aucorder[n]]:
#         trackit+=.075
#         ax2.annotate(allengel[aucorder[n]], xy=( aucresort_clumping[0,aucorder[n]], (float(ymax-ymin)/2)), xycoords='data', 
#                 xytext=( trackit, ((3)*float(ymax-ymin)/5)), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
#                    bbox=dict(boxstyle="round", fc="w"),
#                     horizontalalignment='center', verticalalignment='center', fontsize=fontsize-10, color='k')
ax2.set_xlim(-.01, 1.01)
ax2.set_ylim(ymin, ymax)
ax2.set_xlabel('Areas under HFO Clumping ROC curves', fontsize=fontsize)
ax2.set_ylabel('')
ax2.tick_params(axis='both', which='major', labelsize=fontsize, left='off', right='off', labelleft='off')

custom_lines = [Line2D([0], [0], color='g', lw=6),
                Line2D([0], [0], color='b', lw=6),
                Line2D([0], [0], color='k', lw=6, linestyle='--'),
                Line2D([0], [0], color='w', marker='o',markersize=12,markerfacecolor='k',alpha=0.5),
                Line2D([0], [0], color='w', marker='*',markersize=12,markerfacecolor='k',alpha=0.5)]
ax2.legend(custom_lines, ['State 1',
                          'State 2', 'Chance', 'Localized', 'Not Localized'], 
                          loc=2, fontsize=fontsize, numpoints=1, ncol=2, fancybox=True, framealpha=0.5)
plt.subplots_adjust(hspace=0.3)

print 'Saving Clumping ROC+AUC distribution figures...'
plt.savefig(('figures/Clumping_redChains_ROC_AUC.png'), dpi=300, format='png')


# fig, (ax1,ax2) = plt.subplots(2,1)

figCV = plt.figure()
gsCV = gridspec.GridSpec(nrows=2, ncols=2, left=0.05, wspace=0.05)
ax1a = figCV.add_subplot(gsCV[0, 0]) #Spans the first row and first column
ax1b = figCV.add_subplot(gsCV[0, 1]) #Spans the first row and second column
ax2 = figCV.add_subplot(gsCV[-1, :]) #Spans the second (last) row but all columns

# for n in range(0,true_positive_rate_coef.shape[2]):
#     ax1.plot(false_positive_rate_coef[0, :, n], true_positive_rate_coef[
#          0, :, n], color='b', LineWidth=3,alpha=0.25)
#     ax1.plot(false_positive_rate_coef[1, :, n], true_positive_rate_coef[
#          1, :, n], color='g', LineWidth=3,alpha=0.25)
# ax1.plot(both_mean_false_posCV[0, :], both_mean_true_posCV[
#          0, :], color='b', LineWidth=6)
# allprobs = np.linspace(0., 1., num=rocprecision)
# ax1.plot(allprobs, allprobs, color='k', LineWidth=6, LineStyle='--')
# ax1.plot(both_mean_false_posCV[1, :], both_mean_true_posCV[
#          1, :], color='g', LineWidth=6)
# ax1.set_xlim(-.01, 1.01)
# ax1.set_ylim(-.01, 1.01)
# ax1.tick_params(axis='both', which='major', labelsize=fontsize)
# ax1.set_ylabel('True positive rate', fontsize=fontsize)
# ax1.set_xlabel('False positive rate', fontsize=fontsize)
# ax1.set_title('ROC Curves for SOZ Prediction by HFO Coefficients of Variation (CV)', fontsize=fontsize)

for n in range(0,len(clumping_with_grey)):
    ax1a.plot(false_positive_rate_coef[0, :, n], true_positive_rate_coef[
         0, :, n], color='b', LineWidth=3,alpha=0.25)
    ax1a.plot(false_positive_rate_coef[1, :, n], true_positive_rate_coef[
         1, :, n], color='g', LineWidth=3,alpha=0.25)
ax1a.plot(grey_mean_false_posCV[0, :], grey_mean_true_posCV[
         0, :], color='b', LineWidth=6)
allprobs = np.linspace(0., 1., num=rocprecision)
ax1a.plot(allprobs, allprobs, color='k', LineWidth=6, LineStyle='--')
ax1a.plot(grey_mean_false_posCV[1, :], grey_mean_true_posCV[
         1, :], color='g', LineWidth=6)
ax1a.set_xlim(-.01, 1.01)
ax1a.set_ylim(-.01, 1.01)
ax1a.tick_params(axis='both', which='major', labelsize=fontsize)
ax1a.set_ylabel('True positive rate', fontsize=fontsize, labelpad=2)
ax1a.set_xlabel('False positive rate', fontsize=fontsize)
ax1a.set_title('Coefficients of Variation w/ Localization', fontsize=fontsize)

# Custom legend
custom_lines = [Line2D([0], [0], color='g', lw=6),
                Line2D([0], [0], color='b', lw=6),
                Line2D([0], [0], color='k', lw=3, alpha=0.5),
                Line2D([0], [0], color='k', lw=6, alpha=0.5)]
ax1a.legend(custom_lines, ['State 1',
                          'State 2', 'Indiv. Patient','Average'], 
                          loc=4, fontsize=fontsize, fancybox=True, framealpha=0.5)



for n in range(len(clumping_with_grey),true_positive_rate.shape[2]):
    ax1b.plot(false_positive_rate_coef[0, :, n], true_positive_rate_coef[
         0, :, n], color='b', LineWidth=3,alpha=0.25)
    ax1b.plot(false_positive_rate_coef[1, :, n], true_positive_rate_coef[
         1, :, n], color='g', LineWidth=3,alpha=0.25)
ax1b.plot(out_mean_false_posCV[0, :], out_mean_true_posCV[
         0, :], color='b', LineWidth=6)
allprobs = np.linspace(0., 1., num=rocprecision)
ax1b.plot(allprobs, allprobs, color='k', LineWidth=6, LineStyle='--')
ax1b.plot(out_mean_false_posCV[1, :], out_mean_true_posCV[
         1, :], color='g', LineWidth=6)
ax1b.set_xlim(-.01, 1.01)
ax1b.set_xlim(-.01, 1.01)
ax1b.set_ylim(-.01, 1.01)
ax1b.tick_params(axis='both', which='major', labelsize=fontsize)
ax1b.set_xlabel('False positive rate', fontsize=fontsize)
ax1b.set_title('Coefficients of Variation w/o Localization', fontsize=fontsize)
ax1b.tick_params(axis='both', which='major', labelsize=fontsize, left='off', labelleft='off')

# # Custom legend
# custom_lines = [Line2D([0], [0], color='g', lw=6),
#                 Line2D([0], [0], color='b', lw=6),
#                 Line2D([0], [0], color='k', lw=6, linestyle='--'),
#                 Line2D([0], [0], color='k', lw=3, alpha=0.5),
#                 Line2D([0], [0], color='k', lw=6, alpha=0.5)]
# ax1.legend(custom_lines, ['State 1',
#                           'State 2', 'Chance', 'Indiv. Patient','Average'], 
#                           loc=4, fontsize=fontsize, ncol=2, fancybox=True, framealpha=0.5)


sns.distplot(aucresort_coef[1,:], hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'g'}, rug_kws={'linewidth':2, 'color': 'g'}, ax=ax2)
sns.distplot(aucresort_coef[0,:]+.01, hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'b'}, rug_kws={'linewidth':2, 'color': 'b'}, ax=ax2)
ymin, ymax = ax2.get_ylim()
ax2.plot(np.array([.5, .5]), np.array([ymin,ymax]), color='k', LineWidth=6, LineStyle='--')
ax2.plot(aucresort_coef[1,0:len(clumping_with_grey)],np.ones(len(clumping_with_grey))*(float(ymax-ymin)/2), 'go', markersize=12)
ax2.plot(aucresort_coef[0,0:len(clumping_with_grey)],np.ones(len(clumping_with_grey))*(float(ymax-ymin)/3), 'bo', markersize=12)
ax2.plot(aucresort_coef[1,len(clumping_with_grey):true_positive_rate.shape[2]],np.ones(len(clumping_with_outbrain))*(float(ymax-ymin)/2), 'g*', markersize=12)
ax2.plot(aucresort_coef[0,len(clumping_with_grey):true_positive_rate.shape[2]],np.ones(len(clumping_with_outbrain))*(float(ymax-ymin)/3), 'b*', markersize=12)
for n in range(0, aucresort_coef.shape[1]):
    ax2.plot(aucresort_coef[:,n], np.array([(float(ymax-ymin)/3), (float(ymax-ymin)/2)]), LineWidth=1, LineStyle='--',color='purple',alpha=.5)
aucsinorder = np.sort(aucresort_coef[0,:])
aucorder = np.argsort(aucresort_coef[0,:])
ax2.set_xlim(-.01, 1.01)
ax2.set_ylim(ymin, ymax)
ax2.set_xlabel('Areas under HFO CV ROC curves', fontsize=fontsize)
ax2.set_ylabel('')
ax2.tick_params(axis='both', which='major', labelsize=fontsize, left='off', right='off', labelleft='off')

custom_lines = [Line2D([0], [0], color='g', lw=6),
                Line2D([0], [0], color='b', lw=6),
                Line2D([0], [0], color='k', lw=6, linestyle='--'),
                Line2D([0], [0], color='w', marker='o',markersize=12,markerfacecolor='k',alpha=0.5),
                Line2D([0], [0], color='w', marker='*',markersize=12,markerfacecolor='k',alpha=0.5)]
ax2.legend(custom_lines, ['State 1',
                          'State 2', 'Chance', 'Localized', 'Not Localized'], 
                          loc=2, fontsize=fontsize, numpoints=1, ncol=2, fancybox=True, framealpha=0.5)
plt.subplots_adjust(hspace=0.3)

print 'Saving CV ROC+AUC distribution figures...'
plt.savefig(('figures/CV_redChains_ROC_AUC.png'), dpi=300, format='png')

## Generate HFO rate figures for Model 2

grey_mean_false_positive = np.mean(false_positive_rate[:,:,0:(len(clumping_with_grey))],axis=2)
grey_mean_true_positive = np.mean(true_positive_rate[:,:,0:(len(clumping_with_grey))],axis=2)
out_mean_false_positive = np.mean(false_positive_rate[:,:,(len(clumping_with_grey)):],axis=2)
out_mean_true_positive = np.mean(true_positive_rate[:,:,(len(clumping_with_grey)):],axis=2)




figRATE = plt.figure()
gsRATE= gridspec.GridSpec(nrows=2, ncols=2, left=0.05, wspace=0.05)
axR1 = figRATE.add_subplot(gsRATE[0, 0]) #Spans the first row and first column
axR2 = figRATE.add_subplot(gsRATE[0, 1]) #Spans the first row and second column
axR3 = figRATE.add_subplot(gsRATE[-1, :]) #Spans the second (last) row but all columns

for n in range(0,len(clumping_with_grey)):
    axR1.plot(false_positive_rate[0, :, n], true_positive_rate[
         0, :, n], color='b', LineWidth=3,alpha=0.25)
    axR1.plot(false_positive_rate[1, :, n], true_positive_rate[
         1, :, n], color='g', LineWidth=3,alpha=0.25)
axR1.plot(grey_mean_false_positive[0, :], grey_mean_true_positive[
         0, :], color='b', LineWidth=6)
allprobs = np.linspace(0., 1., num=rocprecision)
axR1.plot(allprobs, allprobs, color='k', LineWidth=6, LineStyle='--')
axR1.plot(grey_mean_false_positive[1, :], grey_mean_true_positive[
         1, :], color='g', LineWidth=6)
axR1.set_xlim(-.01, 1.01)
axR1.set_ylim(-.01, 1.01)
axR1.tick_params(axis='both', which='major', labelsize=fontsize)
axR1.set_ylabel('True positive rate', fontsize=fontsize, labelpad=2)
axR1.set_xlabel('False positive rate', fontsize=fontsize)
axR1.set_title('HFO Rates w/ Localization', fontsize=fontsize)

# Custom legend
custom_lines = [Line2D([0], [0], color='g', lw=6),
                Line2D([0], [0], color='b', lw=6),
                Line2D([0], [0], color='k', lw=3, alpha=0.5),
                Line2D([0], [0], color='k', lw=6, alpha=0.5)]
axR1.legend(custom_lines, ['State 1',
                          'State 2', 'Indiv. Patient','Average'], 
                          loc=4, fontsize=fontsize, fancybox=True, framealpha=0.5)



for n in range(len(clumping_with_grey),true_positive_rate.shape[2]):
    axR2.plot(false_positive_rate[0, :, n], true_positive_rate[
         0, :, n], color='b', LineWidth=3,alpha=0.25)
    axR2.plot(false_positive_rate[1, :, n], true_positive_rate[
         1, :, n], color='g', LineWidth=3,alpha=0.25)
axR2.plot(out_mean_false_positive[0, :], out_mean_true_positive[
         0, :], color='b', LineWidth=6)
allprobs = np.linspace(0., 1., num=rocprecision)
axR2.plot(allprobs, allprobs, color='k', LineWidth=6, LineStyle='--')
axR2.plot(out_mean_false_positive[1, :], out_mean_true_positive[
         1, :], color='g', LineWidth=6)
axR2.set_xlim(-.01, 1.01)
axR2.set_xlim(-.01, 1.01)
axR2.set_ylim(-.01, 1.01)
axR2.tick_params(axis='both', which='major', labelsize=fontsize)
axR2.set_xlabel('False positive rate', fontsize=fontsize)
axR2.set_title('HFO Rates w/o Localization', fontsize=fontsize)
axR2.tick_params(axis='both', which='major', labelsize=fontsize, left='off', labelleft='off')


#Subplot 3
sns.distplot(aucresort[1,:], hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'g'}, rug_kws={'linewidth':2, 'color': 'g'}, ax=axR3)
sns.distplot(aucresort[0,:]+.01, hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'b'}, rug_kws={'linewidth':2, 'color': 'b'}, ax=axR3)
ymin, ymax = axR3.get_ylim()
axR3.plot(np.array([.5, .5]), np.array([ymin,ymax]), color='k', LineWidth=6, LineStyle='--')
axR3.plot(aucresort[1,0:len(clumping_with_grey)],np.ones(len(clumping_with_grey))*(float(ymax-ymin)/2), 'go', markersize=12)
axR3.plot(aucresort[0,0:len(clumping_with_grey)],np.ones(len(clumping_with_grey))*(float(ymax-ymin)/3), 'bo', markersize=12)
axR3.plot(aucresort[1,len(clumping_with_grey):true_positive_rate.shape[2]],np.ones(len(clumping_with_outbrain))*(float(ymax-ymin)/2), 'g*', markersize=12)
axR3.plot(aucresort[0,len(clumping_with_grey):true_positive_rate.shape[2]],np.ones(len(clumping_with_outbrain))*(float(ymax-ymin)/3), 'b*', markersize=12)
for n in range(0, aucresort.shape[1]):
    axR3.plot(aucresort[:,n], np.array([(float(ymax-ymin)/3), (float(ymax-ymin)/2)]), LineWidth=1, LineStyle='--',color='purple',alpha=.5)
aucsinorder = np.sort(aucresort[0,:])
aucorder = np.argsort(aucresort[0,:])
axR3.set_xlim(-.01, 1.01)
axR3.set_ylim(ymin, ymax)
axR3.set_xlabel('Areas under HFO Rate ROC curves', fontsize=fontsize)
axR3.set_ylabel('')
axR3.tick_params(axis='both', which='major', labelsize=fontsize, left='off', right='off', labelleft='off')

custom_lines = [Line2D([0], [0], color='g', lw=6),
                Line2D([0], [0], color='b', lw=6),
                Line2D([0], [0], color='k', lw=6, linestyle='--'),
                Line2D([0], [0], color='w', marker='o',markersize=12,markerfacecolor='k',alpha=0.5),
                Line2D([0], [0], color='w', marker='*',markersize=12,markerfacecolor='k',alpha=0.5)]
axR3.legend(custom_lines, ['State 1',
                          'State 2', 'Chance', 'Localized', 'Not Localized'], 
                          loc=2, fontsize=fontsize, numpoints=1, ncol=2, fancybox=True, framealpha=0.5)
plt.subplots_adjust(hspace=0.3)

print 'Saving HFO ROC+AUC distribution figures...'
plt.savefig(('figures/Rate_Model2_redChains_ROC_AUC.png'), dpi=300, format='png')



# Engel outcome figures
possible_engel =('IA', 'IB', 'IIA', 'IIB', 'IIIA', 'IVB')

plt.figure(dpi=300)
xaxistrack = 0
aucmeans = np.empty((2,len(possible_engel)))
X =[]
Y0 =[]
Y1 =[]
for engel in possible_engel:
    whereauc_in = (engel==allengel) & whereinbrain
    whereauc_out = (engel==allengel) & np.invert(whereinbrain)
    howmany_in = np.sum((whereauc_in))
    howmany_out = np.sum((whereauc_out))
    plt.plot(np.ones((howmany_in))*xaxistrack, aucresort_clumping[0,whereauc_in], 'go', markersize=12)
    plt.plot(np.ones((howmany_in))*xaxistrack, aucresort_clumping[1,whereauc_in], 'bo', markersize=12)
    plt.plot(np.ones((howmany_out))*xaxistrack, aucresort_clumping[0,whereauc_out], 'g*', markersize=12)
    plt.plot(np.ones((howmany_out))*xaxistrack, aucresort_clumping[1,whereauc_out], 'b*', markersize=12)
    for n in range(0,howmany_in):
        X.append(xaxistrack)
        Y0.append(aucresort_clumping[0,whereauc_in[n]])
        Y1.append(aucresort_clumping[1,whereauc_in[n]])
    for n in range(0,howmany_out):
        X.append(xaxistrack)
        Y0.append(aucresort_clumping[0,whereauc_out[n]])
        Y1.append(aucresort_clumping[1,whereauc_out[n]])
    aucmeans[0,xaxistrack] = np.mean(np.concatenate((aucresort_clumping[0,whereauc_in],aucresort_clumping[0,whereauc_out])))
    aucmeans[1,xaxistrack] = np.mean(np.concatenate((aucresort_clumping[1,whereauc_in],aucresort_clumping[1,whereauc_out])))
    xaxistrack += 1
X = np.array(X)
Y0 = np.array(Y0)
Y1 = np.array(Y1)
beta10, beta00 = np.linalg.lstsq(np.vstack([X, np.ones(len(X))]).T,Y0)[0] #Simple least squares regression
plt.plot(X,beta10*X + beta00, linewidth=4, color='b', alpha=.5)
beta11, beta01 = np.linalg.lstsq(np.vstack([X, np.ones(len(X))]).T,Y1)[0] #Simple least squares regression
plt.plot(X,beta11*X + beta01, linewidth=4, color='g', alpha=.5)
plt.ylabel('AUC from HFO Clumping Prediction', fontsize=fontsizelarge)
plt.xlabel('Engel Outcome', fontsize=fontsizelarge)
plt.ylim(-.01, 1.01)
plt.tick_params(axis='both', which='major', labelsize=fontsizelarge)
plt.xticks(np.arange(len(possible_engel)),possible_engel,fontsize=fontsizelarge)
plt.xlim(-0.5,len(possible_engel)+.5)

custom_lines = [Line2D([0], [0], color='g', lw=6),
                Line2D([0], [0], color='b', lw=6),
                Line2D([0], [0], color='w', marker='o',markersize=12,markerfacecolor='k',alpha=0.5),
                Line2D([0], [0], color='w', marker='*',markersize=12,markerfacecolor='k',alpha=0.5)]
plt.legend(custom_lines, ['State 1',
                          'State 2', 'Localized', 'Not Localized'],
                        loc=4, fontsize=fontsizelarge, numpoints=1, fancybox=True, framealpha=0.5) 
plt.tight_layout()

print 'Saving Engel versus AUC figure...'
plt.savefig(('figures/Model2_Engel_outcomes_clumping.png'), dpi=300, format='png')

plt.figure(dpi=300)
xaxistrack = 0
aucmeans = np.empty((2,len(possible_engel)))
X =[]
Y0 =[]
Y1 =[]
for engel in possible_engel:
    whereauc_in = (engel==allengel) & whereinbrain
    whereauc_out = (engel==allengel) & np.invert(whereinbrain)
    howmany_in = np.sum((whereauc_in))
    howmany_out = np.sum((whereauc_out))
    plt.plot(np.ones((howmany_in))*xaxistrack, aucresort_coef[0,whereauc_in], 'go', markersize=12)
    plt.plot(np.ones((howmany_in))*xaxistrack, aucresort_coef[1,whereauc_in], 'bo', markersize=12)
    plt.plot(np.ones((howmany_out))*xaxistrack, aucresort_coef[0,whereauc_out], 'g*', markersize=12)
    plt.plot(np.ones((howmany_out))*xaxistrack, aucresort_coef[1,whereauc_out], 'b*', markersize=12)
    for n in range(0,howmany_in):
        X.append(xaxistrack)
        Y0.append(aucresort_coef[0,whereauc_in[n]])
        Y1.append(aucresort_coef[1,whereauc_in[n]])
    for n in range(0,howmany_out):
        X.append(xaxistrack)
        Y0.append(aucresort_coef[0,whereauc_out[n]])
        Y1.append(aucresort_coef[1,whereauc_out[n]])
    aucmeans[0,xaxistrack] = np.mean(np.concatenate((aucresort_coef[0,whereauc_in],aucresort_coef[0,whereauc_out])))
    aucmeans[1,xaxistrack] = np.mean(np.concatenate((aucresort_coef[1,whereauc_in],aucresort_coef[1,whereauc_out])))
    xaxistrack += 1
X = np.array(X)
Y0 = np.array(Y0)
Y1 = np.array(Y1)
beta10, beta00 = np.linalg.lstsq(np.vstack([X, np.ones(len(X))]).T,Y0)[0] #Simple least squares regression
plt.plot(X,beta10*X + beta00, linewidth=4, color='b', alpha=.5)
beta11, beta01 = np.linalg.lstsq(np.vstack([X, np.ones(len(X))]).T,Y1)[0] #Simple least squares regression
plt.plot(X,beta11*X + beta01, linewidth=4, color='g', alpha=.5)
plt.ylabel('AUC from HFO CV Prediction', fontsize=fontsizelarge)
plt.xlabel('Engel Outcome', fontsize=fontsizelarge)
plt.ylim(-.01, 1.01)
plt.tick_params(axis='both', which='major', labelsize=fontsizelarge)
plt.xticks(np.arange(len(possible_engel)),possible_engel,fontsize=fontsizelarge)
plt.xlim(-0.5,len(possible_engel)+.5)

custom_lines = [Line2D([0], [0], color='g', lw=6),
                Line2D([0], [0], color='b', lw=6),
                Line2D([0], [0], color='w', marker='o',markersize=12,markerfacecolor='k',alpha=0.5),
                Line2D([0], [0], color='w', marker='*',markersize=12,markerfacecolor='k',alpha=0.5)]
plt.legend(custom_lines, ['State 1',
                          'State 2', 'Localized', 'Not Localized'],
                        loc=4, fontsize=fontsizelarge, numpoints=1, fancybox=True, framealpha=0.5)
plt.tight_layout()


print 'Saving Engel versus AUC of coef figure...'
plt.savefig(('figures/Model2_Engel_outcomes_coef.png'), dpi=300, format='png')

plt.figure(dpi=300)
xaxistrack = 0
aucmeans = np.empty((2,len(possible_engel)))
X =[]
Y0 =[]
Y1 =[]
for engel in possible_engel:
    whereauc_in = (engel==allengel) & whereinbrain
    whereauc_out = (engel==allengel) & np.invert(whereinbrain)
    howmany_in = np.sum((whereauc_in))
    howmany_out = np.sum((whereauc_out))
    plt.plot(np.ones((howmany_in))*xaxistrack, aucresort[0,whereauc_in], 'go', markersize=12)
    plt.plot(np.ones((howmany_in))*xaxistrack, aucresort[1,whereauc_in], 'bo', markersize=12)
    plt.plot(np.ones((howmany_out))*xaxistrack, aucresort[0,whereauc_out], 'g*', markersize=12)
    plt.plot(np.ones((howmany_out))*xaxistrack, aucresort[1,whereauc_out], 'b*', markersize=12)
    for n in range(0,howmany_in):
        X.append(xaxistrack)
        Y0.append(aucresort[0,whereauc_in[n]])
        Y1.append(aucresort[1,whereauc_in[n]])
    for n in range(0,howmany_out):
        X.append(xaxistrack)
        Y0.append(aucresort[0,whereauc_out[n]])
        Y1.append(aucresort[1,whereauc_out[n]])
    aucmeans[0,xaxistrack] = np.mean(np.concatenate((aucresort[0,whereauc_in],aucresort[0,whereauc_out])))
    aucmeans[1,xaxistrack] = np.mean(np.concatenate((aucresort[1,whereauc_in],aucresort[1,whereauc_out])))
    xaxistrack += 1
X = np.array(X)
Y0 = np.array(Y0)
Y1 = np.array(Y1)
beta10, beta00 = np.linalg.lstsq(np.vstack([X, np.ones(len(X))]).T,Y0)[0] #Simple least squares regression
plt.plot(X,beta10*X + beta00, linewidth=4, color='b', alpha=.5)
beta11, beta01 = np.linalg.lstsq(np.vstack([X, np.ones(len(X))]).T,Y1)[0] #Simple least squares regression
plt.plot(X,beta11*X + beta01, linewidth=4, color='g', alpha=.5)
plt.ylabel('AUC from HFO Rate Prediction', fontsize=fontsizelarge)
plt.xlabel('Engel Outcome', fontsize=fontsizelarge)
plt.ylim(-.01, 1.01)
plt.tick_params(axis='both', which='major', labelsize=fontsizelarge)
plt.xticks(np.arange(len(possible_engel)),possible_engel,fontsize=fontsizelarge)
plt.xlim(-0.5,len(possible_engel)+.5)

custom_lines = [Line2D([0], [0], color='g', lw=6),
                Line2D([0], [0], color='b', lw=6),
                Line2D([0], [0], color='w', marker='o',markersize=12,markerfacecolor='k',alpha=0.5),
                Line2D([0], [0], color='w', marker='*',markersize=12,markerfacecolor='k',alpha=0.5)]
plt.legend(custom_lines, ['State 1',
                          'State 2', 'Localized', 'Not Localized'],
                        loc=4, fontsize=fontsizelarge, numpoints=1, fancybox=True, framealpha=0.5)
plt.tight_layout()


print 'Saving Engel versus AUC of rate figure...'
plt.savefig(('figures/Model2_Engel_outcomes_rate.png'), dpi=300, format='png')


# Histograms of parameters for SOZ prediction with 100% sensitivity
plt.figure(dpi=300)
sns.distplot(perfect_clumping[1,:]+.01, hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'g'}, rug_kws={'linewidth':2, 'color': 'g'})
sns.distplot(perfect_clumping[0,:], hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'b'}, rug_kws={'linewidth':2, 'color': 'b'})
axes = plt.gca()
ymin, ymax = axes.get_ylim()
xmin, xmax = axes.get_xlim()
plt.plot(perfect_clumping[1,0:len(clumping_with_grey)],np.ones(len(clumping_with_grey))*(float(ymax-ymin)/2), 'go', markersize=12)
plt.plot(perfect_clumping[0,0:len(clumping_with_grey)],np.ones(len(clumping_with_grey))*(float(ymax-ymin)/3), 'bo', markersize=12)
plt.plot(perfect_clumping[1,len(clumping_with_grey):true_positive_rate.shape[2]],np.ones(len(clumping_with_outbrain))*(float(ymax-ymin)/2), 'g*', markersize=12)
plt.plot(perfect_clumping[0,len(clumping_with_grey):true_positive_rate.shape[2]],np.ones(len(clumping_with_outbrain))*(float(ymax-ymin)/3), 'b*', markersize=12)
for n in range(0, perfect_clumping.shape[1]):
    plt.plot(perfect_clumping[:,n], np.array([(float(ymax-ymin)/3), (float(ymax-ymin)/2)]), LineWidth=1, LineStyle='--',color='purple',alpha=.5)
plt.ylim(ymin, ymax)
plt.xlim(0, xmax)
plt.xlabel('Clumping parameters for SOZ prediction w/ 100% sensitivity', fontsize=fontsize)
plt.ylabel('')
plt.tick_params(axis='both', which='major', labelsize=fontsize, left='off', right='off', labelleft='off')

custom_lines = [Line2D([0], [0], color='g', lw=6),
                Line2D([0], [0], color='b', lw=6),
                Line2D([0], [0], color='w', marker='o',markersize=12,markerfacecolor='k',alpha=0.5),
                Line2D([0], [0], color='w', marker='*',markersize=12,markerfacecolor='k',alpha=0.5)]
plt.legend(custom_lines, ['State 1',
                          'State 2', 'Localized', 'Not Localized'], 
                          loc=2, fontsize=fontsize, numpoints=1, fancybox=True, framealpha=0.5)
print 'Saving histogram of clumping parameters for SOZ prediction w/ 100% sensitivity...'
plt.savefig(('figures/Clumping_100_sensitivity.png'), dpi=300, format='png')



plt.figure(dpi=300)
perfect_coef[:, (np.sum(perfect_coef,axis=0) > 100)] = None #Remove outliers 
sns.distplot(perfect_coef[1,:]+.01, hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'g'}, rug_kws={'linewidth':2, 'color': 'g'})
sns.distplot(perfect_coef[0,:], hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'b'}, rug_kws={'linewidth':2, 'color': 'b'})
axes = plt.gca()
ymin, ymax = axes.get_ylim()
xmin, xmax = axes.get_xlim()
plt.plot(perfect_coef[1,0:len(clumping_with_grey)],np.ones(len(clumping_with_grey))*(float(ymax-ymin)/2), 'go', markersize=12)
plt.plot(perfect_coef[0,0:len(clumping_with_grey)],np.ones(len(clumping_with_grey))*(float(ymax-ymin)/3), 'bo', markersize=12)
plt.plot(perfect_coef[1,len(clumping_with_grey):true_positive_rate.shape[2]],np.ones(len(clumping_with_outbrain))*(float(ymax-ymin)/2), 'g*', markersize=12)
plt.plot(perfect_coef[0,len(clumping_with_grey):true_positive_rate.shape[2]],np.ones(len(clumping_with_outbrain))*(float(ymax-ymin)/3), 'b*', markersize=12)
for n in range(0, perfect_coef.shape[1]):
    plt.plot(perfect_coef[:,n], np.array([(float(ymax-ymin)/3), (float(ymax-ymin)/2)]), LineWidth=1, LineStyle='--',color='purple',alpha=.5)
plt.ylim(ymin, ymax)
plt.xlim(0,5)
plt.xlabel('CVs for SOZ prediction w/ 100% sensitivity', fontsize=fontsize)
plt.ylabel('')
plt.tick_params(axis='both', which='major', labelsize=fontsize, left='off', right='off', labelleft='off')

custom_lines = [Line2D([0], [0], color='g', lw=6),
                Line2D([0], [0], color='b', lw=6),
                Line2D([0], [0], color='w', marker='o',markersize=12,markerfacecolor='k',alpha=0.5),
                Line2D([0], [0], color='w', marker='*',markersize=12,markerfacecolor='k',alpha=0.5)]
plt.legend(custom_lines, ['State 1',
                          'State 2', 'Localized', 'Not Localized'], 
                          loc=2, fontsize=fontsize, numpoints=1, fancybox=True, framealpha=0.5)
print 'Saving histogram of CV parameters for SOZ prediction w/ 100% sensitivity...'
plt.savefig(('figures/CV_100_sensitivity.png'), dpi=300, format='png')



plt.figure(dpi=300)
sns.distplot(perfect_rate[1,:]+.01, hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'g'}, rug_kws={'linewidth':2, 'color': 'g'})
sns.distplot(perfect_rate[0,:], hist=False, kde=True, rug=False, kde_kws={'shade': True, 'linewidth':3, 'color': 'b'}, rug_kws={'linewidth':2, 'color': 'b'})
axes = plt.gca()
ymin, ymax = axes.get_ylim()
xmin, xmax = axes.get_xlim()
plt.plot(perfect_rate[1,0:len(clumping_with_grey)],np.ones(len(clumping_with_grey))*(float(ymax-ymin)/2), 'go', markersize=12)
plt.plot(perfect_rate[0,0:len(clumping_with_grey)],np.ones(len(clumping_with_grey))*(float(ymax-ymin)/3), 'bo', markersize=12)
plt.plot(perfect_rate[1,len(clumping_with_grey):true_positive_rate.shape[2]],np.ones(len(clumping_with_outbrain))*(float(ymax-ymin)/2), 'g*', markersize=12)
plt.plot(perfect_rate[0,len(clumping_with_grey):true_positive_rate.shape[2]],np.ones(len(clumping_with_outbrain))*(float(ymax-ymin)/3), 'b*', markersize=12)
for n in range(0, perfect_rate.shape[1]):
    plt.plot(perfect_rate[:,n], np.array([(float(ymax-ymin)/3), (float(ymax-ymin)/2)]), LineWidth=1, LineStyle='--',color='purple',alpha=.5)
plt.ylim(ymin, ymax)
plt.xlim(0, xmax)
plt.xlabel('HFO rates for SOZ prediction w/ 100% sensitivity', fontsize=fontsize)
plt.ylabel('')
plt.tick_params(axis='both', which='major', labelsize=fontsize, left='off', right='off', labelleft='off')

custom_lines = [Line2D([0], [0], color='g', lw=6),
                Line2D([0], [0], color='b', lw=6),
                Line2D([0], [0], color='w', marker='o',markersize=12,markerfacecolor='k',alpha=0.5),
                Line2D([0], [0], color='w', marker='*',markersize=12,markerfacecolor='k',alpha=0.5)]
plt.legend(custom_lines, ['State 1',
                          'State 2', 'Localized', 'Not Localized'], 
                          loc=2, fontsize=fontsize, numpoints=1, fancybox=True, framealpha=0.5)
print 'Saving histogram of rate parameters for SOZ prediction w/ 100% sensitivity...'
plt.savefig(('figures/Rate_100_sensitivity.png'), dpi=300, format='png')


## Aggregate ROC curves for all electrodes across patients
allclumpingvals = np.row_stack((np.asarray(BS1allclumpingvals),np.asarray(BS2allclumpingvals)))
allcvs= np.row_stack((np.asarray(BS1allcvs),np.asarray(BS2allcvs)))
allhforates= np.row_stack((np.asarray(BS1allhforates),np.asarray(BS2allhforates)))
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
plt.annotate('$\zeta=%.2f$' % aggregate_clumping_cutoffs[1, whereannotate], xy=( fpr_aggregate_clumping[1, whereannotate], tpr_aggregate_clumping[1, whereannotate]), xycoords='data', 
        xytext=( .2, .15), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='g'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsizelarge, color='k')
whereannotate = np.where((fpr_aggregate_clumping[1,:] < .2) & (tpr_aggregate_clumping[1,:] > .5))[0][0]
plt.annotate('$\zeta=%.2f$' % aggregate_clumping_cutoffs[1, whereannotate], xy=( fpr_aggregate_clumping[1, whereannotate], tpr_aggregate_clumping[1, whereannotate]), xycoords='data', 
        xytext=( .3, .3), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='g'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsizelarge, color='k')
whereannotate = np.where((fpr_aggregate_clumping[1,:] < .3) & (tpr_aggregate_clumping[1,:] > .7))[0][0]
plt.annotate('$\zeta=%.2f$' % aggregate_clumping_cutoffs[1, whereannotate], xy=( fpr_aggregate_clumping[1, whereannotate], tpr_aggregate_clumping[1, whereannotate]), xycoords='data', 
        xytext=( .6, .35), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='g'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsizelarge, color='k')
whereannotate = np.argmin(np.abs(aggregate_clumping_cutoffs[1, :] - 1))
plt.annotate('$\zeta=%.2f$' % aggregate_clumping_cutoffs[1, whereannotate], xy=( fpr_aggregate_clumping[1, whereannotate], tpr_aggregate_clumping[1, whereannotate]), xycoords='data', 
        xytext=( .7, .5), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='g'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsizelarge, color='k')
whereannotate = np.where((fpr_aggregate_clumping[1,:] < .6) & (tpr_aggregate_clumping[1,:] > .9))[0][0]
plt.annotate('$\zeta=%.2f$' % aggregate_clumping_cutoffs[1, whereannotate], xy=( fpr_aggregate_clumping[1, whereannotate], tpr_aggregate_clumping[1, whereannotate]), xycoords='data', 
        xytext=( .8, .65), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='g'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsizelarge, color='k')
whereannotate = np.where((tpr_aggregate_clumping[1,:] > .999))[0][0]
plt.annotate('$\zeta=%.2f$' % aggregate_clumping_cutoffs[1, whereannotate], xy=( fpr_aggregate_clumping[1, whereannotate], tpr_aggregate_clumping[1, whereannotate]), xycoords='data', 
        xytext=( .9, .8), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='g'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsizelarge, color='k')
plt.xlim(-.01, 1.01)
plt.ylim(-.01, 1.01)
plt.tick_params(axis='both', which='major', labelsize=fontsizelarge)
plt.ylabel('True positive rate', fontsize=fontsizelarge)
plt.xlabel('False positive rate', fontsize=fontsizelarge,labelpad=2)
plt.title('ROC Curve by Clumping prediction for all patients', fontsize=fontsizelarge)
custom_lines = [Line2D([0], [0], color='g', lw=6),
                Line2D([0], [0], color='b', lw=6)]
plt.legend(custom_lines, ['Aggregate State 1',
                          'Aggregate State 2'], 
                          loc=4, fontsize=fontsizelarge, fancybox=True, framealpha=0.5)
print 'Saving aggregate ROC curve for clumping parameters...'
plt.savefig(('figures/Aggregate_ROC_Clumping.png'), dpi=300, format='png')


# Aggregate ROC curves by coefficient of variation
tpr_aggregate_coef = np.empty((NLatent, rocprecision))
fpr_aggregate_coef = np.empty((NLatent, rocprecision))
aggregate_coef_cutoffs = np.empty((NLatent, rocprecision))

for n in range(0, NLatent):
    index = 0
    coef_cutoffs = np.linspace(0., np.max(allcvs[n, :]), num=rocprecision)
    aggregate_coef_cutoffs[n,:] = coef_cutoffs
    for coef in coef_cutoffs:
        little_coef = np.sum((allcvs[n,:] <= coef) & (sozlabels==1))
        tpr_aggregate_coef[n, index] = float(little_coef) / NSOZ
        false_SOZ = np.sum((allcvs[n,:] <= coef) & (sozlabels==0))
        fpr_aggregate_coef[n, index] = float(false_SOZ) / NNonSOZ
        index += 1



plt.figure(dpi=300)
plt.plot(fpr_aggregate_coef[1, :], tpr_aggregate_coef[
         1, :], color='g', LineWidth=6)
allprobs = np.linspace(0., 1., num=rocprecision)
plt.plot(allprobs, allprobs, color='k', LineWidth=6, LineStyle='--')
plt.plot(fpr_aggregate_coef[0, :], tpr_aggregate_coef[
         0, :], color='b', LineWidth=6)
whereannotate = np.where((fpr_aggregate_coef[1,:] < .1) & (tpr_aggregate_coef[1,:] > .25))[0][0]
plt.annotate('$\gamma=%.2f$' % aggregate_coef_cutoffs[1, whereannotate], xy=( fpr_aggregate_coef[1, whereannotate], tpr_aggregate_coef[1, whereannotate]), xycoords='data', 
        xytext=( .4, .15), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='g'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsizelarge, color='k')
whereannotate = np.where((fpr_aggregate_coef[1,:] < .2) & (tpr_aggregate_coef[1,:] > .5))[0][0]
plt.annotate('$\gamma=%.2f$' % aggregate_coef_cutoffs[1, whereannotate], xy=( fpr_aggregate_coef[1, whereannotate], tpr_aggregate_coef[1, whereannotate]), xycoords='data', 
        xytext=( .6, .2), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='g'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsizelarge, color='k')
whereannotate = np.argmin(np.abs(aggregate_coef_cutoffs[1, :] - 1))
plt.annotate('$\gamma=%.2f$' % aggregate_coef_cutoffs[1, whereannotate], xy=( fpr_aggregate_coef[1, whereannotate], tpr_aggregate_coef[1, whereannotate]), xycoords='data', 
        xytext=( .7, .5), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='g'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsizelarge, color='k')
whereannotate = np.where((tpr_aggregate_coef[1,:] > .99))[0][0]
plt.annotate('$\gamma=%.2f$' % aggregate_coef_cutoffs[1, whereannotate], xy=( fpr_aggregate_coef[1, whereannotate], tpr_aggregate_coef[1, whereannotate]), xycoords='data', 
        xytext=( .9, .8), arrowprops=dict(facecolor='black', width=1, headwidth=3, alpha=.5),
           bbox=dict(boxstyle="round", fc="w", edgecolor='g'),
            horizontalalignment='center', verticalalignment='center', fontsize=fontsizelarge, color='k')
plt.xlim(-.01, 1.01)
plt.ylim(-.01, 1.01)
plt.tick_params(axis='both', which='major', labelsize=fontsizelarge)
plt.ylabel('True positive rate', fontsize=fontsizelarge)
plt.xlabel('False positive rate', fontsize=fontsizelarge)
plt.title('ROC Curve by CV prediction for all patients', fontsize=fontsizelarge)
custom_lines = [Line2D([0], [0], color='g', lw=6),
                Line2D([0], [0], color='b', lw=6)]
plt.legend(custom_lines, ['Aggregate State 1',
                          'Aggregate State 2'], 
                          loc=4, fontsize=fontsizelarge, fancybox=True, framealpha=0.5)
print 'Saving aggregate ROC curve for coefficients of variation...'
plt.savefig(('figures/Aggregate_ROC_CV.png'), dpi=300, format='png')

# Sort information for paper table based on patient number
patientnumindex = np.argsort(allpatientnum)
patienttable = np.vstack((allpatientnum[patientnumindex],allNSOZ[patientnumindex],allNNonSOZ[patientnumindex],Nhours[patientnumindex]))
