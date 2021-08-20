# hfo_fitmodel14results.py - Evaluates results of models
#   accounting for possible overdispersion in the HFO count data without latent mixtures
#
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
# 03/17/21      Michael Nunez              Converted from hfo_fitmodel13results.py
# 04/01/21      Michael Nunez         Use model fit parameters for lambda and clumping means
# 08/20/21      Michael Nunez             Remove patient identifiers


# Imports
import numpy as np
import scipy.io as sio
from scipy import stats
import os.path
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from os.path import expanduser
import sys
from pyhddmjagsutils import jellyfish, diagnostic


def load_results(filename,keepchains=np.array([0, 1, 2, 3, 4, 5])):
    # Load model results
    jagsoutloc = '/data/jagsout/'
    patientlab = filename[(model.find('Patient')):(model.find('Patient')+34)]

    ## Load model results
    print('Loading patient %s parameter fits from %s ...' % (patientlab, jagsoutloc + filename))

    samples = sio.loadmat('%s.mat' % (jagsoutloc + filename))

    return (samples, filename)


def run_diagnostic(samples,keepchains=np.array([0, 1, 2, 3, 4, 5])):
    samples_relevant = dict(mean_lambda=samples['mean_lambda'][:,:,:,keepchains], std_lambda=samples['std_lambda'][:,:,:,keepchains],
                        mean_clumping=samples['mean_clumping'][:,:,:,keepchains], std_clumping=samples['std_clumping'][:,:,:,keepchains],
                        lambda2=samples['lambda'][:,:,:,keepchains], r=samples['r'][:,:,:,keepchains], success_prob=samples['success_prob'][:,:,:,keepchains],
                        coef_variation=samples['coef_variation'][:,:,:,keepchains], clumping=samples['clumping'][:,:,:,keepchains])

    if keepchains.shape[0] > 1:
        diags = diagnostic(samples_relevant)
    else:
        print("Keeping only one chain")
        diags = []
    return diags


def save_figures(samples,filename,keepchains=np.array([0, 1, 2, 3, 4, 5])):
    patientlab = filename[(model.find('Patient')):(model.find('Patient')+34)]

    nchains = keepchains.shape[0]
    totalsamps = nchains * samples['success_prob'].shape[-2]
    Nwinds = samples['lambda'].shape[0]
    Nelecs = samples['lambda'].shape[1]

    plt.figure()
    jellyfish(samples['mean_lambda'][:,:,:,keepchains])
    plt.title('Poisson rates per electrode')
    plt.savefig(('../figures/model14results/%s_mean_lambda.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()

    plt.figure()
    jellyfish(samples['std_lambda'][:,:,:,keepchains])
    plt.title('Standard deviation of Poisson rates across 5 minute windows')
    plt.savefig(('../figures/model14results/%s_std_lambda.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()

    plt.figure()
    jellyfish(samples['mean_clumping'][:,:,:,keepchains])
    plt.title('Clumping parameter per electrode')
    plt.savefig(('../figures/model14results/%s_mean_clumping.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()

    plt.figure()
    jellyfish(samples['std_clumping'][:,:,:,keepchains])
    plt.title('Standard deviation of clumping parameters across 5 minute windows')
    plt.savefig(('../figures/model14results/%s_std_clumping.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()

    clumping_samps = np.reshape(samples['clumping'][:,:,:,keepchains], (Nwinds, Nelecs, totalsamps))
    coef_samps = np.reshape(samples['coef_variation'][:,:,:,keepchains], (Nwinds, Nelecs, totalsamps))
    lambda_samps = np.reshape(samples['lambda'][:,:,:,keepchains], (Nwinds, Nelecs, totalsamps))
    allclumping = np.median(clumping_samps, axis=2)
    allcoef = np.median(coef_samps, axis=2)
    allrates = np.median(lambda_samps, axis=2)

    plt.figure()
    plt.plot(np.arange(Nwinds)/12,allrates)
    plt.xlabel('Recording Hour')
    plt.ylabel('Poisson rate per second')
    plt.savefig(('../figures/model14results/%s_lambda_timecourse.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(np.arange(Nwinds)/12,allclumping)
    plt.xlabel('Recording Hour')
    plt.ylabel('Clumping parameters')
    plt.savefig(('../figures/model14results/%s_clumping_timecourse.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(np.arange(Nwinds)/12,allcoef)
    plt.xlabel('Recording Hour')
    plt.ylabel('Coefficients of variation')
    plt.savefig(('../figures/model14results/%s_coef_variation_timecourse.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()

    #Build ROC for SOZ prediction

    if 'usedchans' not in samples:
        samples['usedchans'] = samples['greychans']

    #Only include SOZ electrodes that were in the model
    samples['correctsozchans'] = np.array([
                x for x in samples['usedchans'][0] if x in samples['sozchans'][0]]).T

    NNonSOZ = float(samples['greychans'][0].shape[0] -
                    samples['correctsozchans'][0].shape[0])
    NSOZ = float(samples['correctsozchans'][0].shape[0])

    print("The number of non-seizure onset zone electrodes is %d" % (NNonSOZ))
    print("The number of seizure onset zone electrodes is %d" % (NSOZ))

    rocprecision = 1000

    # Calcuate true and false positive rates for the rate parameter
    # mean_lambda_samps = np.reshape(samples['mean_lambda'][:,:,:,keepchains], (Nelecs, totalsamps))
    # meanrate = np.median(mean_lambda_samps, axis=1)
    meanrate = np.mean(allrates,axis=0) #Mean across 5 minute time windows, for observations per electrode
    rate_cutoffs = np.linspace(np.min(meanrate), np.max(meanrate), num=rocprecision)
    true_positive_rate_rate = np.empty((rocprecision))
    false_positive_rate_rate = np.empty((rocprecision))
    index = 0
    for rate in rate_cutoffs:
        large_rate = np.where(meanrate >= rate)[0]
        candidate_SOZ = samples['usedchans'][0][large_rate]
        intersection = [
            x for x in candidate_SOZ if x in samples['correctsozchans'][0]]
        true_positive_rate_rate[index] = float(len(intersection)) / NSOZ
        false_SOZ = [
            x for x in candidate_SOZ if x not in samples['correctsozchans'][0]]
        false_positive_rate_rate[index] = float(len(false_SOZ)) / NNonSOZ
        index += 1
    auc_rate= np.abs(np.sum(np.multiply(np.diff(false_positive_rate_rate[rocprecision:None:-1]),true_positive_rate_rate[rocprecision:0:-1])))
    print("The AUC of the rate ROC is  %.3f" % (auc_rate))

    # Calcuate true and false positive rates for the clumping parameter
    # mean_clumping_samps = np.reshape(samples['mean_clumping'][:,:,:,keepchains], (Nelecs, totalsamps))
    # meanclumping = np.median(mean_clumping_samps, axis=1)
    meanclumping = np.mean(allclumping,axis=0) #Mean across 5 minute time windows, for observations per electrode
    clump_cutoffs = np.linspace(np.min(meanclumping), np.max(meanclumping), num=rocprecision)
    true_positive_rate_clumping = np.empty((rocprecision))
    false_positive_rate_clumping = np.empty((rocprecision))
    index = 0
    for clumping in clump_cutoffs:
        little_clumping = np.where(meanclumping <= clumping)[0]
        candidate_SOZ = samples['usedchans'][0][little_clumping]
        intersection = [
            x for x in candidate_SOZ if x in samples['correctsozchans'][0]]
        true_positive_rate_clumping[index] = float(len(intersection)) / NSOZ
        false_SOZ = [
            x for x in candidate_SOZ if x not in samples['correctsozchans'][0]]
        false_positive_rate_clumping[index] = float(len(false_SOZ)) / NNonSOZ
        index += 1
    auc_clumping= np.abs(np.sum(np.multiply(np.diff(false_positive_rate_clumping[rocprecision:None:-1]),true_positive_rate_clumping[rocprecision:0:-1])))
    print("The AUC of the clumping ROC is  %.3f" % (auc_clumping))

    # Calcuate true and false positive rates for the coefficient of variation
    meancoef = np.mean(allcoef,axis=0) #Mean across 5 minute time windows, for observations per electrode
    coef_cutoffs = np.linspace(np.min(meancoef), np.max(meancoef), num=rocprecision)
    true_positive_rate_coef = np.empty((rocprecision))
    false_positive_rate_coef = np.empty((rocprecision))
    index = 0
    for coef in coef_cutoffs:
        little_coef = np.where(meancoef <= coef)[0]
        candidate_SOZ = samples['usedchans'][0][little_coef]
        intersection = [
            x for x in candidate_SOZ if x in samples['correctsozchans'][0]]
        true_positive_rate_coef[index] = float(len(intersection)) / NSOZ
        false_SOZ = [
            x for x in candidate_SOZ if x not in samples['correctsozchans'][0]]
        false_positive_rate_coef[index] = float(len(false_SOZ)) / NNonSOZ
        index += 1
    auc_coef= np.abs(np.sum(np.multiply(np.diff(false_positive_rate_coef[rocprecision:None:-1]),true_positive_rate_coef[rocprecision:0:-1])))
    print("The AUC of the cofficient of variation ROC is  %.3f" % (auc_coef))


    #Plot the ROCs

    plt.figure()
    plt.plot(false_positive_rate_rate, true_positive_rate_rate, LineWidth=3)
    allprobs = np.linspace(0., 1., num=rocprecision)
    plt.plot(allprobs, allprobs, color='k', LineWidth=6, LineStyle='--')
    plt.xlim(-.01, 1.01)
    plt.ylim(-.01, 1.01)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curves for SOZ Prediction by HFO Poisson rate')
    # plt.savefig(('../figures/model14results/%s_lambda_ROC2.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.savefig(('../figures/model14results/%s_lambda_ROC.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(false_positive_rate_clumping, true_positive_rate_clumping, LineWidth=3)
    allprobs = np.linspace(0., 1., num=rocprecision)
    plt.plot(allprobs, allprobs, color='k', LineWidth=6, LineStyle='--')
    plt.xlim(-.01, 1.01)
    plt.ylim(-.01, 1.01)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curves for SOZ Prediction by HFO Clumping Parameters')
    # plt.savefig(('../figures/model14results/%s_clumping_ROC2.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.savefig(('../figures/model14results/%s_clumping_ROC.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(false_positive_rate_coef, true_positive_rate_coef, LineWidth=3)
    allprobs = np.linspace(0., 1., num=rocprecision)
    plt.plot(allprobs, allprobs, color='k', LineWidth=6, LineStyle='--')
    plt.xlim(-.01, 1.01)
    plt.ylim(-.01, 1.01)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curves for SOZ Prediction by HFO Clumping Parameters')
    plt.savefig(('../figures/model14results/%s_coef_variation_ROC.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()



if __name__ == '__main__':
    #When hfo_fitmodel14results.py is run as a script, do this
    if (len(sys.argv) > 2):
        keepchains = np.array(sys.argv[2])
    else:
        keepchains = np.array([0, 1, 2, 3, 4, 5])
    (samples, filename) = load_results(filename=sys.argv[1],keepchains=keepchains)
    diags = run_diagnostic(samples,keepchains=keepchains)
    save_figures(samples, filename,keepchains=keepchains)
