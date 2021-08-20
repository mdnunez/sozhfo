# hfo_SOZprediction_avg_boot.py - Finds predictive AUC from Model B by randomly mixing labels for each patient
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
# 08/06/21       Michael Nunez                      Converted from hfo_SOZprediction_avg.py
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
#https://github.com/matplotlib/matplotlib/issues/16911
#https://stackoverflow.com/questions/53640859/how-to-integrate-curve-from-data
#https://stackoverflow.com/questions/44915116/how-to-decide-between-scipy-integrate-simps-or-numpy-trapz

# Imports
import numpy as np
import scipy.io as sio
import os.path
from scipy.stats.stats import pearsonr
from pymatreader import read_mat
from pingouin import ttest

rocprecision = 1000
bootprecision = 1000

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

true_positive_rate_clumping = np.empty((NLatent, rocprecision, nmodels))
false_positive_rate_clumping = np.empty((NLatent, rocprecision, nmodels))
auc_clumping = np.empty((NLatent, nmodels))
aucresort_clumping = np.empty((NLatent, nmodels))
perfect_clumping = np.empty((NLatent, nmodels)) #Find the parameter where the True Positive Rate is 1

true_positive_rate_clumping_boot = np.empty((NLatent, rocprecision, nmodels, bootprecision))
false_positive_rate_clumping_boot = np.empty((NLatent, rocprecision, nmodels, bootprecision))
auc_clumping_boot = np.empty((NLatent, nmodels, bootprecision))
aucresort_clumping_boot = np.empty((NLatent, nmodels, bootprecision))
auc_clumping_boot_cutoff = np.empty((NLatent, nmodels))

true_positive_rate_coef = np.empty((NLatent, rocprecision, nmodels))
false_positive_rate_coef = np.empty((NLatent, rocprecision, nmodels))
auc_coef = np.empty((NLatent, nmodels))
aucresort_coef = np.empty((NLatent, nmodels))
perfect_coef = np.empty((NLatent, nmodels)) #Find the parameter where the True Positive Rate is 1

true_positive_rate_coef_boot = np.empty((NLatent, rocprecision, nmodels, bootprecision))
false_positive_rate_coef_boot = np.empty((NLatent, rocprecision, nmodels, bootprecision))
auc_coef_boot = np.empty((NLatent, nmodels, bootprecision))
aucresort_coef_boot = np.empty((NLatent, nmodels, bootprecision))
auc_coef_boot_cutoff = np.empty((NLatent, nmodels)) 

true_positive_rate_rate = np.empty((NLatent, rocprecision, nmodels))
false_positive_rate_rate = np.empty((NLatent, rocprecision, nmodels))
auc_rate = np.empty((NLatent, nmodels))
aucresort_rate = np.empty((NLatent, nmodels))
perfect_rate = np.empty((NLatent, nmodels)) #Find the parameter where the True Positive Rate is 1

true_positive_rate_rate_boot = np.empty((NLatent, rocprecision, nmodels, bootprecision))
false_positive_rate_rate_boot = np.empty((NLatent, rocprecision, nmodels, bootprecision))
auc_rate_boot = np.empty((NLatent, nmodels, bootprecision))
aucresort_rate_boot = np.empty((NLatent, nmodels, bootprecision))
auc_rate_boot_cutoff = np.empty((NLatent, nmodels)) 


corrate = np.empty((nmodels))
corclumping = np.empty((nmodels))
corcoef = np.empty((nmodels))
corrate_pval = np.empty((nmodels))
corclumping_pval = np.empty((nmodels))
corcoef_pval = np.empty((nmodels))

allNSOZ = np.zeros((nmodels))
allNNonSOZ = np.zeros((nmodels))

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

    # Set up bootstrapping for all three statistics
    fakesoz = []
    for b in range(0, bootprecision):
        channeldraw = np.random.choice(int(NNonSOZ + NSOZ), size=int(NSOZ),
                                       replace=False)  # Randomly label some electrodes as SOZ
        fakesoz.append(samples['usedchans'][0, channeldraw])


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
            # Bootstrapping
            for b in range(0, bootprecision):
                bootintersection = [x for x in candidate_SOZ if x in fakesoz[b]]
                true_positive_rate_clumping_boot[n,index, model_indx, b] = float(len(bootintersection)) / NSOZ
                false_SOZ_boot = [x for x in candidate_SOZ if x not in fakesoz[b]]
                false_positive_rate_clumping_boot[n,index, model_indx, b] = float(len(false_SOZ_boot)) / NNonSOZ
            index += 1
        auc_clumping[n, model_indx] = 1 - np.trapz(np.squeeze(false_positive_rate_clumping[n, :,model_indx]),np.squeeze(true_positive_rate_clumping[n, :,model_indx]))
        # Find bootstrapping range
        for b in range(0, bootprecision):
            auc_clumping_boot[n, model_indx, b] = 1 - np.trapz(np.squeeze(false_positive_rate_clumping_boot[n,:,model_indx,b]),np.squeeze(true_positive_rate_clumping_boot[n,:,model_indx,b]))

    if samples['flip_labels']: #Always plot the generator state that is more predictive of delta power as Brain State A
        true_positive_rate_clumping[:, :, model_indx] = true_positive_rate_clumping[::-1,:,model_indx] #Flip model labels 1 and 0
        false_positive_rate_clumping[:, :, model_indx] = false_positive_rate_clumping[::-1,:,model_indx] #Flip model labels 1 and 0
        aucresort_clumping[:, model_indx] = auc_clumping[::-1, model_indx]
        aucresort_clumping_boot[:,model_indx,:] = auc_clumping_boot[::-1,model_indx,:]
        perfect_clumping[:, model_indx] = perfect_clumping[::-1, model_indx]
        for e in range(0,Nelecs):
            BSAallclumpingvals.append(allclumping[0,e]) #Flipped Brain State A
            BSBallclumpingvals.append(allclumping[1,e]) #Flipped Brain State B
    else:
        aucresort_clumping[:, model_indx] = auc_clumping[:, model_indx]
        aucresort_clumping_boot[:,model_indx,:] = auc_clumping_boot[:,model_indx,:]
        perfect_clumping[:, model_indx] = perfect_clumping[:, model_indx]
        for e in range(0,Nelecs):
            BSAallclumpingvals.append(allclumping[1,e]) #Brain State A
            BSBallclumpingvals.append(allclumping[0,e]) #Brain State B
    auc_clumping_boot_cutoff[0,model_indx] = np.sort(auc_clumping_boot[0,model_indx, :])[int(np.round(bootprecision * .95))]
    auc_clumping_boot_cutoff[1,model_indx] = np.sort(auc_clumping_boot[1,model_indx, :])[int(np.round(bootprecision * .95))]
    print("The AUC of the clumping ROC is  %.3f using brain state A" % (aucresort_clumping[1,model_indx]))
    print("The AUC of the clumping ROC is  %.3f using brain state B" % (aucresort_clumping[0,model_indx]))
    if aucresort_clumping[1,model_indx] > auc_clumping_boot_cutoff[1,model_indx]:
        print(
            "The AUC using brain state A was deemed predictive such that the AUC of %.3f was greater than the permutation cutoff of %.3f" % (
            aucresort_clumping[1,model_indx], auc_clumping_boot_cutoff[1,model_indx]))
    else:
        print(
            "The AUC using brain state A was deemed NOT predictive such that the AUC of %.3f was less than the permutation cutoff of %.3f" % (
            auc_clumping[1,model_indx], auc_clumping_boot_cutoff[1,model_indx]))
    if aucresort_clumping[0,model_indx] > auc_clumping_boot_cutoff[0,model_indx]:
        print(
            "The AUC using brain state B was deemed predictive such that the AUC of %.3f was greater than the permutation cutoff of %.3f" % (
            aucresort_clumping[0,model_indx], auc_clumping_boot_cutoff[0,model_indx]))
    else:
        print(
            "The AUC using brain state B was deemed NOT predictive such that the AUC of %.3f was less than the permutation cutoff of %.3f" % (
            auc_clumping[0,model_indx], auc_clumping_boot_cutoff[0,model_indx]))


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
            # Bootstrapping
            for b in range(0, bootprecision):
                bootintersection = [x for x in candidate_SOZ if x in fakesoz[b]]
                true_positive_rate_coef_boot[n,index, model_indx, b] = float(len(bootintersection)) / NSOZ
                false_SOZ_boot = [x for x in candidate_SOZ if x not in fakesoz[b]]
                false_positive_rate_coef_boot[n,index, model_indx, b] = float(len(false_SOZ_boot)) / NNonSOZ
            index += 1
        auc_coef[n, model_indx] = 1 - np.trapz(np.squeeze(false_positive_rate_coef[n, :,model_indx]),np.squeeze(true_positive_rate_coef[n, :,model_indx]))
        # Find bootstrapping range
        for b in range(0, bootprecision):
            auc_coef_boot[n, model_indx, b] = 1 - np.trapz(np.squeeze(false_positive_rate_coef_boot[n,:,model_indx,b]),np.squeeze(true_positive_rate_coef_boot[n,:,model_indx,b]))
    if samples['flip_labels']: #Always plot the generator state that is more predictive of delta power as Brain State A
        true_positive_rate_coef[:, :, model_indx] = true_positive_rate_coef[::-1,:,model_indx] #Flip model labels 1 and 0
        false_positive_rate_coef[:, :, model_indx] = false_positive_rate_coef[::-1,:,model_indx] #Flip model labels 1 and 0
        aucresort_coef[:, model_indx] = auc_coef[::-1, model_indx]
        aucresort_coef_boot[:,model_indx,:] = auc_coef_boot[::-1,model_indx,:]
        perfect_coef[:, model_indx] = perfect_coef[::-1, model_indx]
        for e in range(0,Nelecs):
            BSAallcvs.append(allcoef[0,e]) #Flipped Brain State A
            BSBallcvs.append(allcoef[1,e]) #Flipped Brain State B
    else:
        aucresort_coef[:, model_indx] = auc_coef[:, model_indx]
        aucresort_coef_boot[:,model_indx,:] = auc_coef_boot[:,model_indx,:]
        perfect_coef[:, model_indx] = perfect_coef[:, model_indx]
        for e in range(0,Nelecs):
            BSAallcvs.append(allcoef[1,e]) #Brain State A
            BSBallcvs.append(allcoef[0,e]) #Brain State B
    auc_coef_boot_cutoff[0,model_indx] = np.sort(auc_coef_boot[0,model_indx, :])[int(np.round(bootprecision * .95))]
    auc_coef_boot_cutoff[1,model_indx] = np.sort(auc_coef_boot[1,model_indx, :])[int(np.round(bootprecision * .95))]
    print("The AUC of the coef ROC is  %.3f using brain state A" % (aucresort_coef[1,model_indx]))
    print("The AUC of the coef ROC is  %.3f using brain state B" % (aucresort_coef[0,model_indx]))
    if aucresort_coef[1,model_indx] > auc_coef_boot_cutoff[1,model_indx]:
        print(
            "The AUC using brain state A was deemed predictive such that the AUC of %.3f was greater than the permutation cutoff of %.3f" % (
            aucresort_coef[1,model_indx], auc_coef_boot_cutoff[1,model_indx]))
    else:
        print(
            "The AUC using brain state A was deemed NOT predictive such that the AUC of %.3f was less than the permutation cutoff of %.3f" % (
            auc_coef[1,model_indx], auc_coef_boot_cutoff[1,model_indx]))
    if aucresort_coef[0,model_indx] > auc_coef_boot_cutoff[0,model_indx]:
        print(
            "The AUC using brain state B was deemed predictive such that the AUC of %.3f was greater than the permutation cutoff of %.3f" % (
            aucresort_coef[0,model_indx], auc_coef_boot_cutoff[0,model_indx]))
    else:
        print(
            "The AUC using brain state B was deemed NOT predictive such that the AUC of %.3f was less than the permutation cutoff of %.3f" % (
            auc_coef[0,model_indx], auc_coef_boot_cutoff[0,model_indx]))



    for n in range(0, NLatent):
        index = 0
        foundperfect =0
        rate_cutoffs = np.linspace(np.max(allrates[n, :]), 0., num=rocprecision)
        for rate in rate_cutoffs:
            high_rates = np.where(np.median(lambda_samps[n, :, :], axis=1) >= rate)[0]
            candidate_SOZ = samples['usedchans'][0][high_rates]
            intersection = [
                x for x in candidate_SOZ if x in samples['correctsozchans'][0]]
            true_positive_rate_rate[n, index,model_indx] = float(len(intersection)) / NSOZ
            if ((true_positive_rate_rate[n, index,model_indx] > .99) & (foundperfect==0)):
                perfect_rate[n, model_indx] = rate
                foundperfect = 1
            false_SOZ = [
                x for x in candidate_SOZ if x not in samples['correctsozchans'][0]]
            false_positive_rate_rate[n, index,model_indx] = float(len(false_SOZ)) / NNonSOZ
            # Bootstrapping
            for b in range(0, bootprecision):
                bootintersection = [x for x in candidate_SOZ if x in fakesoz[b]]
                true_positive_rate_rate_boot[n,index, model_indx, b] = float(len(bootintersection)) / NSOZ
                false_SOZ_boot = [x for x in candidate_SOZ if x not in fakesoz[b]]
                false_positive_rate_rate_boot[n,index, model_indx, b] = float(len(false_SOZ_boot)) / NNonSOZ
            index += 1
        auc_rate[n, model_indx] = 1 - np.trapz(np.squeeze(false_positive_rate_rate[n, :,model_indx]),np.squeeze(true_positive_rate_rate[n, :,model_indx]))
        # Find bootstrapping range
        for b in range(0, bootprecision):
            auc_rate_boot[n, model_indx, b] = 1 - np.trapz(np.squeeze(false_positive_rate_rate_boot[n,:,model_indx,b]),np.squeeze(true_positive_rate_rate_boot[n,:,model_indx,b]))
    if samples['flip_labels']: #Always plot the generator state that is more predictive of delta power as Brain State A
        true_positive_rate_rate[:, :, model_indx] = true_positive_rate_rate[::-1,:,model_indx] #Flip model labels 1 and 0
        false_positive_rate_rate[:, :, model_indx] = false_positive_rate_rate[::-1,:,model_indx] #Flip model labels 1 and 0
        aucresort_rate[:, model_indx] = auc_rate[::-1, model_indx]
        aucresort_rate_boot[:,model_indx,:] = auc_rate_boot[::-1,model_indx,:]
        perfect_rate[:, model_indx] = perfect_rate[::-1, model_indx]
        for e in range(0,Nelecs):
            BSAallhforates.append(allrates[0,e]) #Flipped Brain State A
            BSBallhforates.append(allrates[1,e]) #Flipped Brain State B
    else:
        aucresort_rate[:, model_indx] = auc_rate[:, model_indx]
        aucresort_rate_boot[:,model_indx,:] = auc_rate_boot[:,model_indx,:]
        perfect_rate[:, model_indx] = perfect_rate[:, model_indx]
        for e in range(0,Nelecs):
            BSAallhforates.append(allrates[1,e]) #Brain State A
            BSBallhforates.append(allrates[0,e]) #Brain State B
    auc_rate_boot_cutoff[0,model_indx] = np.sort(auc_rate_boot[0,model_indx, :])[int(np.round(bootprecision * .95))]
    auc_rate_boot_cutoff[1,model_indx] = np.sort(auc_rate_boot[1,model_indx, :])[int(np.round(bootprecision * .95))]
    print("The AUC of the rate ROC is  %.3f using brain state A" % (aucresort_rate[1,model_indx]))
    print("The AUC of the rate ROC is  %.3f using brain state B" % (aucresort_rate[0,model_indx]))
    if aucresort_rate[1,model_indx] > auc_rate_boot_cutoff[1,model_indx]:
        print(
            "The AUC using brain state A was deemed predictive such that the AUC of %.3f was greater than the permutation cutoff of %.3f" % (
            aucresort_rate[1,model_indx], auc_rate_boot_cutoff[1,model_indx]))
    else:
        print(
            "The AUC using brain state A was deemed NOT predictive such that the AUC of %.3f was less than the permutation cutoff of %.3f" % (
            auc_rate[1,model_indx], auc_rate_boot_cutoff[1,model_indx]))
    if aucresort_rate[0,model_indx] > auc_rate_boot_cutoff[0,model_indx]:
        print(
            "The AUC using brain state B was deemed predictive such that the AUC of %.3f was greater than the permutation cutoff of %.3f" % (
            aucresort_rate[0,model_indx], auc_rate_boot_cutoff[0,model_indx]))
    else:
        print(
            "The AUC using brain state B was deemed NOT predictive such that the AUC of %.3f was less than the permutation cutoff of %.3f" % (
            auc_rate[0,model_indx], auc_rate_boot_cutoff[0,model_indx]))
    model_indx += 1


Nhours = np.asarray(Nwindows) / float(12) #Convert from 5 minute windows to hours
print("Mean and standard deviation of number hours is %.2f +/- %.2f" % (np.mean(Nhours), np.std(Nhours)))

patientnumindex = np.argsort(allpatientnum)
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


what100tpr_rate = np.empty((2,true_positive_rate_rate.shape[2]))
for n in range(0,true_positive_rate_rate.shape[2]):
    BSAwhere100trp = (true_positive_rate_rate[0,:,n] > .999)
    BSBwhere100trp = (true_positive_rate_rate[1,:,n] > .999)
    what100tpr_rate[0,n] = np.min(false_positive_rate_rate[0,BSAwhere100trp,n])
    what100tpr_rate[1,n] = np.min(false_positive_rate_rate[1,BSBwhere100trp,n])


#### Save out AUC values ####

summaryauc = dict()
summaryauc['aucresort_clumping'] = aucresort_clumping
summaryauc['what100tpr_clumping'] = what100tpr_clumping
summaryauc['aucresort_coef'] = aucresort_coef
summaryauc['what100tpr_coef'] = what100tpr_coef
summaryauc['aucresort_rate'] = aucresort_rate
summaryauc['what100tpr_rate'] = what100tpr_rate
print('Saving summary ROC and AUC statistics...')
sio.savemat('data/AllPatients_summaryauc.mat', summaryauc)


# Print out information about SOZ prediction
print("The mean and standard deviation of AUC by the clumping coefficients were %.2f +/- %.2f in Brain State A" % (np.mean(aucresort_clumping[1,:]),np.std(aucresort_clumping[1,:])))
print("The mean and standard deviation of AUC by the clumping coefficients were %.2f +/- %.2f in Brain State B" % (np.mean(aucresort_clumping[0,:]),np.std(aucresort_clumping[0,:])))
print("The data of %d of 16 patients yielded clumping coefficients that we deemed predictive of SOZ (AUC >.60) in Brain State A" % (np.sum([aucresort_clumping[1,:] > .6])))
print("The data of %d of 16 patients yielded clumping coefficients that we deemed strongly predictive of SOZ (AUC > boot_cutoff) in Brain State A" % (np.sum([aucresort_clumping[1,:] > auc_clumping_boot_cutoff[1,:]])))
print("The data of %d of 16 patients yielded clumping coefficients that we deemed predictive of SOZ (AUC >.60) in Brain State B" % (np.sum([aucresort_clumping[0,:] > .6])))
print("The data of %d of 16 patients yielded clumping coefficients that we deemed strongly predictive of SOZ (AUC > boot_cutoff) in Brain State B" % (np.sum([aucresort_clumping[0,:] > auc_clumping_boot_cutoff[0,:]])))
print("The data of %d of 16 patients yielded clumping coefficients that we deemed predictive of SOZ (AUC >.60) in at least one Brain State" % np.sum(((aucresort_clumping[0,:] > .6) | (aucresort_clumping[1,:] > .6))))
print("In %d of 16 patients, clumping coefficients differentiated all SOZ channels for FPR less than 60%% in Brain State A" % (np.sum([what100tpr_clumping[1,:] < .6])))
print("In %d of 16 patients, clumping coefficients differentiated all SOZ channels for FPR less than 20%% in Brain State A" % (np.sum([what100tpr_clumping[1,:] < .2])))
print("In %d of 16 patients, clumping coefficients differentiated all SOZ channels for FPR less than 60%% in Brain State B" % (np.sum([what100tpr_clumping[0,:] < .6])))
print("In %d of 16 patients, clumping coefficients differentiated all SOZ channels for FPR less than 20%% in Brain State B" % (np.sum([what100tpr_clumping[0,:] < .2])))


# Print out information about SOZ prediction
print("The mean and standard deviation of AUC by the coefficients of variation were %.2f +/- %.2f in Brain State A" % (np.mean(aucresort_coef[1,:]),np.std(aucresort_coef[1,:])))
print("The mean and standard deviation of AUC by the coefficients of variation were %.2f +/- %.2f in Brain State B" % (np.mean(aucresort_coef[0,:]),np.std(aucresort_coef[0,:])))
print("The data of %d of 16 patients yielded coefficients of variation that we deemed predictive of SOZ (AUC >.60) in Brain State A" % (np.sum([aucresort_coef[1,:] > .6])))
print("The data of %d of 16 patients yielded coefficients of variation that we deemed strongly predictive of SOZ (AUC > boot_cutoff) in Brain State A" % (np.sum([aucresort_coef[1,:] > auc_coef_boot_cutoff[1,:]])))
print("The data of %d of 16 patients yielded coefficients of variation that we deemed predictive of SOZ (AUC >.60) in Brain State B" % (np.sum([aucresort_coef[0,:] > .6])))
print("The data of %d of 16 patients yielded coefficients of variation that we deemed strongly predictive of SOZ (AUC > boot_cutoff) in Brain State B" % (np.sum([aucresort_coef[0,:] > auc_coef_boot_cutoff[0,:]])))
print("The data of %d of 16 patients yielded coefficients of variation that we deemed predictive of SOZ (AUC >.60) in at least one Brain State" % np.sum(((aucresort_coef[0,:] > .6) | (aucresort_coef[1,:] > .6))))
print("In %d of 16 patients, coefficients of variation differentiated all SOZ channels for FPR less than 60%% in Brain State A" % (np.sum([what100tpr_coef[1,:] < .6])))
print("In %d of 16 patients, coefficients of variation differentiated all SOZ channels for FPR less than 20%% in Brain State A" % (np.sum([what100tpr_coef[1,:] < .2])))
print("In %d of 16 patients, coefficients of variation differentiated all SOZ channels for FPR less than 60%% in Brain State B" % (np.sum([what100tpr_coef[0,:] < .6])))
print("In %d of 16 patients, coefficients of variation differentiated all SOZ channels for FPR less than 20%% in Brain State B" % (np.sum([what100tpr_coef[0,:] < .2])))

# Print out information about SOZ prediction
print("The mean and standard deviation of AUC by the HFO rates were %.2f +/- %.2f in Brain State A" % (np.mean(aucresort_rate[1,:]),np.std(aucresort_rate[1,:])))
print("The mean and standard deviation of AUC by the HFO rates were %.2f +/- %.2f in Brain State B" % (np.mean(aucresort_rate[0,:]),np.std(aucresort_rate[0,:])))
print("The data of %d of 16 patients yielded HFO rates that we deemed predictive of SOZ (AUC >.60) in Brain State A" % (np.sum([aucresort_rate[1,:] > .6])))
print("The data of %d of 16 patients yielded HFO rates that we deemed strongly predictive of SOZ (AUC > boot_cutoff) in Brain State A" % (np.sum([aucresort_rate[1,:] > auc_rate_boot_cutoff[1,:]])))
print("The data of %d of 16 patients yielded HFO rates that we deemed predictive of SOZ (AUC >.60) in Brain State B" % (np.sum([aucresort_rate[0,:] > .6])))
print("The data of %d of 16 patients yielded HFO rates that we deemed strongly predictive of SOZ (AUC > boot_cutoff) in Brain State B" % (np.sum([aucresort_rate[0,:] > auc_rate_boot_cutoff[0,:]])))
print("The data of %d of 16 patients yielded HFO rates that we deemed predictive of SOZ (AUC >.60) in at least one Brain State" % np.sum(((aucresort_rate[0,:] > .6) | (aucresort_rate[1,:] > .6))))
print("In %d of 16 patients, HFO rates differentiated all SOZ channels for FPR less than 60%% in Brain State A" % (np.sum([what100tpr_rate[1,:] < .6])))
print("In %d of 16 patients, HFO rates differentiated all SOZ channels for FPR less than 20%% in Brain State A" % (np.sum([what100tpr_rate[1,:] < .2])))
print("In %d of 16 patients, HFO rates differentiated all SOZ channels for FPR less than 60%% in Brain State B" % (np.sum([what100tpr_rate[0,:] < .6])))
print("In %d of 16 patients, HFO rates differentiated all SOZ channels for FPR less than 20%% in Brain State B" % (np.sum([what100tpr_rate[0,:] < .2])))


print("The total number of channels used across models was %d" % (np.sum(sortedNChans)))
print("The false positive rate is %.2f in Brain State A when the clumping coefficient is equal to 1 " % (fpr_aggregate_clumping[1, whereannotate_zeta1A]))
print("The true positive rate is %.2f in Brain State A when the clumping coefficient is equal to 1" % (tpr_aggregate_clumping[1, whereannotate_zeta1A]))
print("The false positive rate is %.2f in Brain State A when the true positive rate is 1" % (fpr_aggregate_clumping[1, whereannotate_tpr_1A]))

print("The false positive rate is %.2f in Brain State B when the clumping coefficient is equal to 1 " % (fpr_aggregate_clumping[0, whereannotate_zeta1B]))
print("The true positive rate is %.2f in Brain State B when the clumping coefficient is equal to 1" % (tpr_aggregate_clumping[0, whereannotate_zeta1B]))
print("The false positive rate is %.2f in Brain State B when the true positive rate is 1" % (fpr_aggregate_clumping[0, whereannotate_tpr_1B]))

