# hfo_SOZprediction_avg_nomix_boot.py - Finds predictive AUC from Model A by randomly mixing labels for each patient
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
# 07/13/21     Michael Nunez                Converted from hfo_SOZPrediction_avg.py
# 07/29/21     Michael Nunez                  Remove plotted figures
# 07/30/21      Michael Nunez                 Change integration strategy to avoid integration errors
# 08/20/21      Michael Nunez             Remove patient identifiers

# References:
#https://www.statology.org/paired-samples-t-test-python/
#https://stackoverflow.com/questions/53640859/how-to-integrate-curve-from-data
#https://stackoverflow.com/questions/44915116/how-to-decide-between-scipy-integrate-simps-or-numpy-trapz

# Imports
import numpy as np
import scipy.io as sio
from pymatreader import read_mat
from pingouin import ttest

rocprecision = 1000
bootprecision = 1000


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
perfect_clumping = np.empty((nmodels))  # Find the parameter where the True Positive Rate is 1

true_positive_rate_clumping_boot = np.empty((rocprecision, nmodels, bootprecision))
false_positive_rate_clumping_boot = np.empty((rocprecision, nmodels, bootprecision))
auc_clumping_boot = np.empty((nmodels, bootprecision))
auc_clumping_boot_cutoff = np.empty((nmodels))

true_positive_rate_coef = np.empty((rocprecision, nmodels))
false_positive_rate_coef = np.empty((rocprecision, nmodels))
auc_coef = np.empty((nmodels))
perfect_coef = np.empty((nmodels))  # Find the parameter where the True Positive Rate is 1

true_positive_rate_coef_boot = np.empty((rocprecision, nmodels, bootprecision))
false_positive_rate_coef_boot = np.empty((rocprecision, nmodels, bootprecision))
auc_coef_boot = np.empty((nmodels, bootprecision))
auc_coef_boot_cutoff = np.empty((nmodels))

true_positive_rate_rate = np.empty((rocprecision, nmodels))
false_positive_rate_rate = np.empty((rocprecision, nmodels))
auc_rate = np.empty((nmodels))
perfect_rate = np.empty((nmodels))  # Find the parameter where the True Positive Rate is 1

true_positive_rate_rate_boot = np.empty((rocprecision, nmodels, bootprecision))
false_positive_rate_rate_boot = np.empty((rocprecision, nmodels, bootprecision))
auc_rate_boot = np.empty((nmodels, bootprecision))
auc_rate_boot_cutoff = np.empty((nmodels))

corrate = np.empty((nmodels))
corclumping = np.empty((nmodels))
corcoef = np.empty((nmodels))

allNSOZ = np.zeros((nmodels))
allNNonSOZ = np.zeros((nmodels))

allclumpingvals = []  # Track all median posterior clumping parameter values in Brain State 1
allhforates = []  # Track all median posterior HFO rates in Brain State 1
allcvs = []  # Track all mean posterior CV in Brain State 1
allsozlabels = []

Nwindows = []  # Track the number of 5 minute windows used in analysis of interictal data

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

    # Build ROC for SOZ prediction

    if 'usedchans' not in samples:
        samples['usedchans'] = samples['greychans']

    # Only include SOZ electrodes that were in the model
    samples['correctsozchans'] = np.array([
        x for x in samples['usedchans'][0] if x in samples['sozchans'][0]]).T

    for e in range(0, Nelecs):
        if samples['usedchans'][0, e] in samples['correctsozchans']:
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

    # Set up bootstrapping for all three statistics
    fakesoz = []
    for b in range(0, bootprecision):
        channeldraw = np.random.choice(int(NNonSOZ + NSOZ), size=int(NSOZ),
                                       replace=False)  # Randomly label some electrodes as SOZ
        fakesoz.append(samples['usedchans'][0, channeldraw])

    # Calcuate true and false positive rates for the clumping parameter
    # mean_clumping_samps = np.reshape(samples['mean_clumping'], (Nelecs, totalsamps))
    # meanclumping = np.median(mean_clumping_samps, axis=1) # Use the hierarchical mean parameter instead
    meanclumping = np.mean(allclumping, axis=0)  # Mean across 5 minute time windows, for observations per electrode
    clump_cutoffs = np.linspace(np.min(meanclumping), np.max(meanclumping), num=rocprecision)
    index = 0
    foundperfect = 0
    for clumping in clump_cutoffs:
        little_clumping = np.where(meanclumping <= clumping)[0]
        candidate_SOZ = samples['usedchans'][0][little_clumping]
        intersection = [
            x for x in candidate_SOZ if x in samples['correctsozchans'][0]]
        true_positive_rate_clumping[index, model_indx] = float(len(intersection)) / NSOZ
        if ((true_positive_rate_clumping[index, model_indx] > .99) & (foundperfect == 0)):
            perfect_clumping[model_indx] = clumping
            foundperfect = 1
        false_SOZ = [
            x for x in candidate_SOZ if x not in samples['correctsozchans'][0]]
        false_positive_rate_clumping[index, model_indx] = float(len(false_SOZ)) / NNonSOZ
        # Bootstrapping
        for b in range(0, bootprecision):
            bootintersection = [x for x in candidate_SOZ if x in fakesoz[b]]
            true_positive_rate_clumping_boot[index, model_indx, b] = float(len(bootintersection)) / NSOZ
            false_SOZ_boot = [x for x in candidate_SOZ if x not in fakesoz[b]]
            false_positive_rate_clumping_boot[index, model_indx, b] = float(len(false_SOZ_boot)) / NNonSOZ
        index += 1
    auc_clumping[model_indx] = 1 - np.trapz(np.squeeze(false_positive_rate_clumping[:,model_indx]),np.squeeze(true_positive_rate_clumping[:,model_indx]))
    print("The AUC of the clumping ROC is  %.3f" % (auc_clumping[model_indx]))
    perfect_clumping[model_indx] = perfect_clumping[model_indx]
    for e in range(0, Nelecs):
        allclumpingvals.append(meanclumping[e])

    # Find bootstrapping range
    for b in range(0, bootprecision):
        auc_clumping_boot[model_indx, b] = 1 - np.trapz(np.squeeze(false_positive_rate_clumping_boot[:,model_indx,b]),np.squeeze(true_positive_rate_clumping_boot[:,model_indx,b]))
    auc_clumping_boot_cutoff[model_indx] = np.sort(auc_clumping_boot[model_indx, :])[int(np.round(bootprecision * .95))]
    if auc_clumping[model_indx] > auc_clumping_boot_cutoff[model_indx]:
        print(
            "The AUC was deemed predictive such that the AUC of %.3f was greater than the permutation cutoff of %.3f" % (
            auc_clumping[model_indx], auc_clumping_boot_cutoff[model_indx]))
    else:
        print(
            "The AUC was deemed NOT predictive such that the AUC of %.3f was less than the permutation cutoff of %.3f" % (
            auc_clumping[model_indx], auc_clumping_boot_cutoff[model_indx]))

    # Calcuate true and false positive rates for the coefficient of variation parameter
    meancoef = np.mean(allcoef, axis=0)  # Mean across 5 minute time windows, for observations per electrode
    coef_cutoffs = np.linspace(np.min(meancoef), np.max(meancoef), num=rocprecision)
    index = 0
    foundperfect = 0
    for coef in coef_cutoffs:
        little_coef = np.where(meancoef <= coef)[0]
        candidate_SOZ = samples['usedchans'][0][little_coef]
        intersection = [
            x for x in candidate_SOZ if x in samples['correctsozchans'][0]]
        true_positive_rate_coef[index, model_indx] = float(len(intersection)) / NSOZ
        if ((true_positive_rate_coef[index, model_indx] > .99) & (foundperfect == 0)):
            perfect_coef[model_indx] = coef
            foundperfect = 1
        false_SOZ = [
            x for x in candidate_SOZ if x not in samples['correctsozchans'][0]]
        false_positive_rate_coef[index, model_indx] = float(len(false_SOZ)) / NNonSOZ
        # Bootstrapping
        for b in range(0, bootprecision):
            bootintersection = [x for x in candidate_SOZ if x in fakesoz[b]]
            true_positive_rate_coef_boot[index, model_indx, b] = float(len(bootintersection)) / NSOZ
            false_SOZ_boot = [x for x in candidate_SOZ if x not in fakesoz[b]]
            false_positive_rate_coef_boot[index, model_indx, b] = float(len(false_SOZ_boot)) / NNonSOZ
        index += 1
    auc_coef[model_indx] = 1 - np.trapz(np.squeeze(false_positive_rate_coef[:,model_indx]),np.squeeze(true_positive_rate_coef[:,model_indx]))
    print("The AUC of the coefficient of variation ROC is  %.3f" % (auc_coef[model_indx]))
    perfect_coef[model_indx] = perfect_coef[model_indx]
    for e in range(0, Nelecs):
        allcvs.append(meancoef[e])

    for b in range(0, bootprecision):
        auc_coef_boot[model_indx, b] = 1 - np.trapz(np.squeeze(false_positive_rate_coef_boot[:,model_indx,b]),np.squeeze(true_positive_rate_coef_boot[:,model_indx,b]))
    auc_coef_boot_cutoff[model_indx] = np.sort(auc_coef_boot[model_indx, :])[int(np.round(bootprecision * .95))]
    if auc_coef[model_indx] > auc_coef_boot_cutoff[model_indx]:
        print(
            "The AUC was deemed predictive such that the AUC of %.3f was greater than the permutation cutoff of %.3f" % (
            auc_coef[model_indx], auc_coef_boot_cutoff[model_indx]))
    else:
        print(
            "The AUC was deemed NOT predictive such that the AUC of %.3f was less than the permutation cutoff of %.3f" % (
            auc_coef[model_indx], auc_coef_boot_cutoff[model_indx]))

    # Calcuate true and false positive rates for the HFO rate parameter
    # mean_lambda_samps = np.reshape(samples['mean_lambda'][:,:,:,keepchains], (Nelecs, totalsamps))
    # meanrate = np.median(mean_lambda_samps, axis=1)
    meanrate = np.mean(allrates, axis=0)  # Mean across 5 minute time windows, for observations per electrode
    rate_cutoffs = np.linspace(np.min(meanrate), np.max(meanrate), num=rocprecision)
    index = 0
    foundperfect = 0
    for rate in rate_cutoffs:
        large_rate = np.where(meanrate >= rate)[0]
        candidate_SOZ = samples['usedchans'][0][large_rate]
        intersection = [
            x for x in candidate_SOZ if x in samples['correctsozchans'][0]]
        true_positive_rate_rate[index, model_indx] = float(len(intersection)) / NSOZ
        if ((true_positive_rate_rate[index, model_indx] > .99) & (foundperfect == 0)):
            perfect_rate[model_indx] = rate
            foundperfect = 1
        false_SOZ = [
            x for x in candidate_SOZ if x not in samples['correctsozchans'][0]]
        false_positive_rate_rate[index, model_indx] = float(len(false_SOZ)) / NNonSOZ
        # Bootstrapping
        for b in range(0, bootprecision):
            bootintersection = [x for x in candidate_SOZ if x in fakesoz[b]]
            true_positive_rate_rate_boot[index, model_indx, b] = float(len(bootintersection)) / NSOZ
            false_SOZ_boot = [x for x in candidate_SOZ if x not in fakesoz[b]]
            false_positive_rate_rate_boot[index, model_indx, b] = float(len(false_SOZ_boot)) / NNonSOZ
        index += 1
    auc_rate[model_indx] = 1 + np.trapz(np.squeeze(false_positive_rate_rate[:,model_indx]),np.squeeze(true_positive_rate_rate[:,model_indx]))
    print("The AUC of the HFO rate ROC is  %.3f" % (auc_rate[model_indx]))
    perfect_rate[model_indx] = perfect_rate[model_indx]
    for e in range(0, Nelecs):
        allhforates.append(meanrate[e])

    for b in range(0, bootprecision):
        auc_rate_boot[model_indx, b] = 1 + np.trapz(np.squeeze(false_positive_rate_rate_boot[:,model_indx,b]),np.squeeze(true_positive_rate_rate_boot[:,model_indx,b]))
    auc_rate_boot_cutoff[model_indx] = np.sort(auc_rate_boot[model_indx, :])[int(np.round(bootprecision * .95))]
    if auc_rate[model_indx] > auc_rate_boot_cutoff[model_indx]:
        print(
            "The AUC was deemed predictive such that the AUC of %.3f was greater than the permutation cutoff of %.3f" % (
            auc_rate[model_indx], auc_rate_boot_cutoff[model_indx]))
    else:
        print(
            "The AUC was deemed NOT predictive such that the AUC of %.3f was less than the permutation cutoff of %.3f" % (
            auc_rate[model_indx], auc_rate_boot_cutoff[model_indx]))

    model_indx += 1

Nhours = np.asarray(Nwindows) / float(12)  # Convert from 5 minute windows to hours
print("Mean and standard deviation of number hours is %.2f +/- %.2f" % (np.mean(Nhours), np.std(Nhours)))

grey_mean_false_positive = np.mean(false_positive_rate_clumping[:, 0:(len(clumping_with_grey))], axis=1)
grey_mean_true_positive = np.mean(true_positive_rate_clumping[:, 0:(len(clumping_with_grey))], axis=1)
out_mean_false_positive = np.mean(false_positive_rate_clumping[:, (len(clumping_with_grey)):], axis=1)
out_mean_true_positive = np.mean(true_positive_rate_clumping[:, (len(clumping_with_grey)):], axis=1)
both_mean_false_positive = np.mean(false_positive_rate_clumping[:, :], axis=1)
both_mean_true_positive = np.mean(true_positive_rate_clumping[:, :], axis=1)

grey_mean_false_posCV = np.mean(false_positive_rate_coef[:, 0:(len(clumping_with_grey))], axis=1)
grey_mean_true_posCV = np.mean(true_positive_rate_coef[:, 0:(len(clumping_with_grey))], axis=1)
out_mean_false_posCV = np.mean(false_positive_rate_coef[:, (len(clumping_with_grey)):], axis=1)
out_mean_true_posCV = np.mean(true_positive_rate_coef[:, (len(clumping_with_grey)):], axis=1)
both_mean_false_posCV = np.mean(false_positive_rate_coef[:, :], axis=1)
both_mean_true_posCV = np.mean(true_positive_rate_coef[:, :], axis=1)

both_mean_false_positive_rate = np.mean(false_positive_rate_rate[:, :], axis=1)
both_mean_true_positive_rate = np.mean(true_positive_rate_rate[:, :], axis=1)

# For what value of the false positive rate is the true positive rate 100%?
what100tpr_clumping = np.empty((nmodels))
what100tpr_coef = np.empty((nmodels))
what100tpr_rate = np.empty((nmodels))
for n in range(0, nmodels):
    where100tpr_clumping = (true_positive_rate_clumping[:, n] > .999)
    what100tpr_clumping[n] = np.min(false_positive_rate_clumping[where100tpr_clumping, n])
    where100tpr_coef = (true_positive_rate_coef[:, n] > .999)
    what100tpr_coef[n] = np.min(false_positive_rate_coef[where100tpr_coef, n])
    where100tpr_rate = (true_positive_rate_rate[:, n] > .999)
    what100tpr_rate[n] = np.min(false_positive_rate_rate[where100tpr_rate, n])

#### Compare AUC values across models ####

print('Loading summary ROC and AUC statistics from 2 state model fits...')
summaryauc_model2 = read_mat('data/AllPatients_summaryauc.mat')
summaryaucboth = dict()

summaryaucboth['what100tpr_clumping_model1'] = what100tpr_clumping
summaryaucboth['what100tpr_clumping_state1'] = summaryauc_model2['what100tpr_clumping'][1, :]
ttestoutput = ttest(summaryaucboth['what100tpr_clumping_model1'], summaryaucboth['what100tpr_clumping_state1'],
                    paired=True)
print(
    'The p-value and Bayes Factor of the paired samples t-test between the FPR values from clumping coefficients between Model 1 and State 1 are %.3f and %.2f respectively' % (
    ttestoutput['p-val'], ttestoutput['BF10']))
summaryaucboth['what100tpr_clumping_state2'] = summaryauc_model2['what100tpr_clumping'][0, :]
ttestoutput = ttest(summaryaucboth['what100tpr_clumping_model1'], summaryaucboth['what100tpr_clumping_state2'],
                    paired=True)
print(
    'The p-value and Bayes Factor of the paired samples t-test between the FPR values from clumping coefficients between Model 1 and State 2 are %.3f and %.2f respectively' % (
    ttestoutput['p-val'], ttestoutput['BF10']))
summaryaucboth['auc_clumping_model1'] = auc_clumping
summaryaucboth['auc_clumping_state1'] = summaryauc_model2['aucresort_clumping'][1, :]
ttestoutput = ttest(summaryaucboth['auc_clumping_model1'], summaryaucboth['auc_clumping_state1'], paired=True)
print(
    'The p-value and Bayes Factor of the paired samples t-test between the AUCs from clumping values between Model 1 and State 1 are %.3f and %.2f respectively' % (
    ttestoutput['p-val'], ttestoutput['BF10']))
summaryaucboth['auc_clumping_state2'] = summaryauc_model2['aucresort_clumping'][0, :]
ttestoutput = ttest(summaryaucboth['auc_clumping_model1'], summaryaucboth['auc_clumping_state2'], paired=True)
print(
    'The p-value and Bayes Factor of the paired samples t-test between the AUCs from clumping values between Model 1 and State 2 are %.3f and %.2f respectively' % (
    ttestoutput['p-val'], ttestoutput['BF10']))

summaryaucboth['what100tpr_coef_model1'] = what100tpr_coef
summaryaucboth['what100tpr_coef_state1'] = summaryauc_model2['what100tpr_coef'][1, :]
ttestoutput = ttest(summaryaucboth['what100tpr_coef_model1'], summaryaucboth['what100tpr_coef_state1'], paired=True)
print(
    'The p-value and Bayes Factor of the paired samples t-test between the FPR values from coefficients of variation between Model 1 and State 1 are %.3f and %.2f respectively' % (
    ttestoutput['p-val'], ttestoutput['BF10']))
summaryaucboth['what100tpr_coef_state2'] = summaryauc_model2['what100tpr_coef'][0, :]
ttestoutput = ttest(summaryaucboth['what100tpr_coef_model1'], summaryaucboth['what100tpr_coef_state2'], paired=True)
print(
    'The p-value and Bayes Factor of the paired samples t-test between the FPR values from coefficients of variation between Model 1 and State 2 are %.3f and %.2f respectively' % (
    ttestoutput['p-val'], ttestoutput['BF10']))
summaryaucboth['auc_coef_model1'] = auc_coef
summaryaucboth['auc_coef_state1'] = summaryauc_model2['aucresort_coef'][1, :]
ttestoutput = ttest(summaryaucboth['auc_coef_model1'], summaryaucboth['auc_coef_state1'], paired=True)
print(
    'The p-value and Bayes Factor of the paired samples t-test between the AUCs from coefficients of variation between Model 1 and State 1 are %.3f and %.2f respectively' % (
    ttestoutput['p-val'], ttestoutput['BF10']))
summaryaucboth['auc_coef_state2'] = summaryauc_model2['aucresort_coef'][0, :]
ttestoutput = ttest(summaryaucboth['auc_coef_model1'], summaryaucboth['auc_coef_state2'], paired=True)
print(
    'The p-value and Bayes Factor of the paired samples t-test between the AUCs from coefficients of variation between Model 1 and State 2 are %.3f and %.2f respectively' % (
    ttestoutput['p-val'], ttestoutput['BF10']))

summaryaucboth['what100tpr_rate_model1'] = what100tpr_rate
summaryaucboth['what100tpr_rate_state1'] = summaryauc_model2['what100tpr_rate'][1, :]
ttestoutput = ttest(summaryaucboth['what100tpr_rate_model1'], summaryaucboth['what100tpr_rate_state1'], paired=True)
print(
    'The p-value and Bayes Factor of the paired samples t-test between the FPR values from HFO rates between Model 1 and State 1 are %.3f and %.2f respectively' % (
    ttestoutput['p-val'], ttestoutput['BF10']))
summaryaucboth['what100tpr_rate_state2'] = summaryauc_model2['what100tpr_rate'][0, :]
ttestoutput = ttest(summaryaucboth['what100tpr_rate_model1'], summaryaucboth['what100tpr_rate_state2'], paired=True)
print(
    'The p-value and Bayes Factor of the paired samples t-test between the FPR values from HFO rates between Model 1 and State 2 are %.3f and %.2f respectively' % (
    ttestoutput['p-val'], ttestoutput['BF10']))
summaryaucboth['auc_rate_model1'] = auc_rate
summaryaucboth['auc_rate_state1'] = summaryauc_model2['aucresort_rate'][1, :]
ttestoutput = ttest(summaryaucboth['auc_rate_model1'], summaryaucboth['auc_rate_state1'], paired=True)
print(
    'The p-value and Bayes Factor of the paired samples t-test between the AUCs from HFO rates between Model 1 and State 1 are %.3f and %.2f respectively' % (
    ttestoutput['p-val'], ttestoutput['BF10']))
summaryaucboth['auc_rate_state2'] = summaryauc_model2['aucresort_rate'][0, :]
ttestoutput = ttest(summaryaucboth['auc_rate_model1'], summaryaucboth['auc_rate_state2'], paired=True)
print(
    'The p-value and Bayes Factor of the paired samples t-test between the AUCs from HFO rates between Model 1 and State 2 are %.3f and %.2f respectively' % (
    ttestoutput['p-val'], ttestoutput['BF10']))

# delta_df = pd.DataFrame({'standarddelta' : stackeddelta, 'brainstate': whichstate})
# aov = anova(dv='standarddelta',between='brainstate', data=delta_df, detailed=True)
# allFstat[p] = aov['F'][0]
# ANOVApvals[p] = aov['p-unc'][0]
# kw = kruskal(dv='standarddelta',between='brainstate', data=delta_df, detailed=True)
# allHstat[p] = kw['H'][0]
# KWpvals[p] = kw['p-unc'][0]


# Print out information about SOZ prediction
print("The mean and standard deviation of AUC by the clumping coefficients were %.2f +/- %.2f" % (
np.mean(auc_clumping), np.std(auc_clumping)))
print(
    "The data of %d of 16 patients yielded clumping coefficients that we deemed predictive of SOZ (AUC > AUC_boot)" % (
        np.sum([auc_clumping > auc_clumping_boot_cutoff])))
print("The data of %d of 16 patients yielded clumping coefficients that we deemed predictive of SOZ (AUC >.60)" % (
    np.sum([auc_clumping > .6])))
print("In %d of 16 patients, clumping coefficients differentiated all SOZ channels for FPR less than 60%%" % (
    np.sum([what100tpr_clumping < .6])))
print("In %d of 16 patients, clumping coefficients differentiated all SOZ channels for FPR less than 20%%" % (
    np.sum([what100tpr_clumping < .2])))

# Print out information about SOZ prediction
print("The mean and standard deviation of AUC by the coefficients of variation were %.2f +/- %.2f" % (
np.mean(auc_coef), np.std(auc_coef)))
print(
    "The data of %d of 16 patients yielded coefficients of variation that we deemed predictive of SOZ (AUC > AUC_boot)" % (
        np.sum([auc_coef > auc_coef_boot_cutoff])))
print("The data of %d of 16 patients yielded coefficients of variation that we deemed predictive of SOZ (AUC >.60)" % (
    np.sum([auc_coef > .6])))
print("In %d of 16 patients, coefficients of variation differentiated all SOZ channels for FPR less than 60%%" % (
    np.sum([what100tpr_coef < .6])))
print("In %d of 16 patients, coefficients of variation differentiated all SOZ channels for FPR less than 20%%" % (
    np.sum([what100tpr_coef < .2])))

# Print out information about SOZ prediction
print("The mean and standard deviation of AUC by the HFO rates were %.2f +/- %.2f" % (
np.mean(auc_rate), np.std(auc_rate)))
print("The data of %d of 16 patients yielded HFO rates that we deemed predictive of SOZ (AUC > AUC_boot)" % (
    np.sum([auc_rate > auc_rate_boot_cutoff])))
print("The data of %d of 16 patients yielded HFO rates that we deemed predictive of SOZ (AUC >.60)" % (
    np.sum([auc_rate > .6])))
print("In %d of 16 patients, HFO rates differentiated all SOZ channels for FPR less than 60%%" % (
    np.sum([what100tpr_rate < .6])))
print("In %d of 16 patients, HFO rates differentiated all SOZ channels for FPR less than 20%%" % (
    np.sum([what100tpr_rate < .2])))

sortedNChans = allNSOZ[patientnumindex] + allNNonSOZ[patientnumindex]
print("The minimum number of channels used in the models was %d" % (np.round(np.min(sortedNChans))))
print("The maximum number of channels used in the models was %d" % (np.round(np.max(sortedNChans))))
print("The mean and standard deviation of channels used in the models were %d +- %d" % (
np.round(np.mean(sortedNChans)), np.round(np.std(sortedNChans))))

print("The minimum number of SOZ channels used in the models was %d" % (np.round(np.min(allNSOZ))))
print("The maximum number of SOZ channels used in the models was %d" % (np.round(np.max(allNSOZ))))
print("The mean and standard deviation of SOZ channels used in the models were %d +- %d" % (
np.round(np.mean(allNSOZ)), np.round(np.std(allNSOZ))))
