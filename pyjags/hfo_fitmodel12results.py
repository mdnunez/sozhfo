# hfo_fitmodel12results.py - Evaluates simulation results of models
#                         fit to real data that assumes latent states in HFO rates
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
# 03/04/19      Michael Nunez              Converted from hfo_fitmodel11results.py
# 09/12/19      Michael Nunez            Plot chains for state_lambda
# 09/19/19      Michael Nunez                  Label chains
# 09/23/19      Michael Nunez                Plot only those chains to keep
# 12/28/20      Michael Nunez      Use functions in pyhddmjagsutils
# 01/27/21      Michael Nunez             Load multiple models
# 02/12/21      Michael Nunez               Two additional models
# 02/18/21      Michael Nunez             One additional model and better print updates
# 02/23/21      Michael Nunez              Minor print updates
# 08/20/21      Michael Nunez             Remove patient identifiers

# Imports
import numpy as np
import scipy.io as sio
from scipy import stats
import os.path
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from pyhddmjagsutils import jellyfish, diagnostic

## Set up loading of data
sixchains = np.array([0, 1, 2, 3, 4, 5]) # All 6 chains
jagsoutloc = '/data/jagsout/'


### Models with qHFOs (possible artifacts removed) ###
allmodels = []
allkeepchains = []

##Patient 1
allmodels.append('jagsmodel_model12samples_Patient1_grey_qHFO_2LatentDec_02_20_12_11')
allkeepchains.append(sixchains)

##Patient 2
allmodels.append('jagsmodel_model12samples_Patient2_grey_qHFO_2LatentDec_23_20_11_02')
allkeepchains.append(np.array([0, 1, 2, 4, 5]))
# allkeepchains.append(sixchains)

##Patient 3 
allmodels.append('jagsmodel_model12samples_Patient3_grey_qHFO_2LatentDec_22_20_13_52')
allkeepchains.append(sixchains)

#Patient 4
allmodels.append('jagsmodel_model12samples_Patient4_grey_qHFO_2LatentDec_07_20_21_54')
allkeepchains.append(sixchains)

#Patient 5
allmodels.append('jagsmodel_model12samples_Patient5_grey_qHFO_2LatentNov_23_20_20_59')
allkeepchains.append(sixchains)

##Patient 6
allmodels.append('jagsmodel_model12samples_Patient6_grey_qHFO_2LatentNov_12_20_16_19')
allkeepchains.append(sixchains)

##Patient 7
allmodels.append('jagsmodel_model12samples_Patient7_grey_qHFO_2LatentNov_03_20_14_55')
allkeepchains.append(np.array([1, 2, 3, 4, 5]))
# allkeepchains.append(sixchains)

##Patient 8
allmodels.append('jagsmodel_model12samples_Patient8_grey_qHFO_2LatentNov_10_20_11_03')
allkeepchains.append(np.array([2, 4]))
# allkeepchains.append(sixchains)

#Patient 9
allmodels.append('jagsmodel_model12samples_Patient9_grey_qHFO_2LatentDec_19_20_05_29')
allkeepchains.append(np.array([0, 1, 2, 3, 5]))
# allkeepchains.append(sixchains)

#Patient 10
allmodels.append('jagsmodel_model12samples_Patient10_grey_qHFO_2LatentFeb_15_21_14_36')
allkeepchains.append(np.array([0, 1]))
# allkeepchains.append(sixchains)

#Patient 11
allmodels.append('jagsmodel_model12samples_Patient11_grey_qHFO_2LatentFeb_04_21_11_26')
allkeepchains.append(np.array([3, 5])) # Could have picked one of 3 pairs of chains
# allkeepchains.append(sixchains)

#Patient 12
allmodels.append('jagsmodel_model12samples_Patient12_grey_qHFO_2LatentNov_12_20_16_06')
allkeepchains.append(np.array([0, 1, 4]))
# allkeepchains.append(sixchains)

#Patient 13
allmodels.append('jagsmodel_model12samples_Patient13_grey_qHFO_2LatentDec_22_20_18_49')
allkeepchains.append(np.array([0, 1, 3, 4, 5]))
# allkeepchains.append(sixchains)

#Patient 14
allmodels.append('jagsmodel_model12samples_Patient14_grey_qHFO_2LatentFeb_19_21_13_40')
allkeepchains.append(np.array([1]))


#Patient 15
allmodels.append('jagsmodel_model12samples_Patient15_grey_qHFO_2LatentDec_22_20_14_08')
allkeepchains.append(sixchains)

#Patient 16
allmodels.append('jagsmodel_model12samples_Patient16_grey_qHFO_2LatentNov_23_20_20_33')
allkeepchains.append(np.array([0, 1, 3, 4, 5]))
# allkeepchains.append(sixchains)

modeltrack = 0
for model in allmodels:
    patientlab = model[(model.find('Patient')):(model.find('Patient')+9)]
    keepchains = allkeepchains[modeltrack]
    modeltrack += 1

    ## Load data
    print('Loading patient %s data from %s ...' % (patientlab, jagsoutloc + model))

    samples = sio.loadmat('%s.mat' % (jagsoutloc + model))

    # Switch chain indices (how do we order the index ahead of time?)
    originalnchains = samples['pi'].shape[-1]

    try:
        keepchains
    except NameError:
        keepchains = np.arange(0,originalnchains)
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


    if not os.path.isfile('%s_reordered.mat' % (jagsoutloc + model)):
        samples['sortorder'] = sortorder
        sio.savemat('%s_reordered.mat' % (jagsoutloc + model), samples)


    samples_relevant = dict(latent_lambda=samples[
                            'latent_lambda'][:,:,:,keepchains], pi=samples['pi'][:,:,:,keepchains],
                            state_lambda=samples['state_lambda'][:,:,:,keepchains],
                            state_std=samples['state_std'][:,:,:,keepchains],
                            clumping=samples['clumping'][:,:,:,keepchains],
                            coef_variation=samples['coef_variation'][:,:,:,keepchains])

    if keepchains.shape[0] > 1:
        diags = diagnostic(samples_relevant)
    else:
        print("Keeping only one chain")



    plt.figure()
    jellyfish(samples['latent_lambda'][:,:,:,keepchains])
    plt.title('Poisson rates per second of latent states')
    plt.savefig(('../figures/model12results/%s_latent_lambda.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()

    plt.figure()
    jellyfish(samples['clumping'][:,:,:,keepchains])
    plt.title('Clumping parameter of latent states')
    plt.savefig(('../figures/model12results/%s_clumping.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()

    plt.figure()
    jellyfish(samples['coef_variation'][:,:,:,keepchains])
    plt.title('Coefficient of variation of latent states')
    plt.savefig(('../figures/model12results/%s_coef_variation.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()

    plt.figure()
    jellyfish(samples['pi'][:,:,:,keepchains])
    plt.title('Probability of latent states')
    plt.savefig(('../figures/model12results/%s_pi.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()


    plt.figure()
    fivemins = samples['stateindx'].shape[1]
    stateindx_samps = np.reshape(samples['stateindx'], (fivemins, 500, originalnchains))
    for n in keepchains:
        plt.plot(np.squeeze(np.mean(stateindx_samps[:, :, n], axis=1)), label=('chain %d' % (n+1)))
    plt.title('Recovery of Sleep states', fontsize=16)
    plt.xlabel('5 Minute Window #', fontsize=16)
    plt.ylabel('Average State Label', fontsize=16)
    plt.legend()
    plt.savefig(('../figures/model12results/%s_stateindx.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()

    plt.figure()
    jellyfish(samples['state_lambda'][:,:,:,keepchains])
    plt.title('Average Poisson rates over all electrodes')
    plt.savefig(('../figures/model12results/%s_state_lambda.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()

    plt.figure()
    jellyfish(samples['state_std'][:,:,:,keepchains])
    plt.title('Std of Poisson rates over all electrodes')
    plt.savefig(('../figures/model12results/%s_state_std.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()

    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    for n in keepchains:
        ax1.plot(samples['state_lambda'][0,0,:,n], label=('chain %d' % (n+1)))
        ax2.plot(samples['state_lambda'][0,1,:,n], label=('chain %d' % (n+1)))
    ax1.set_title('Chains for State_lambda k=1')
    ax2.set_title('Chains for state_lambda k=2')
    ax2.legend()
    plt.savefig(('../figures/model12results/%s_state_lambda_chains.png' % (patientlab)), format='png',bbox_inches="tight")
    plt.close()


    clumping_samps = np.reshape(
        samples['clumping'][:,:,:,keepchains], (NLatent, Nelecs, totalsamps))
    coef_samps = np.reshape(
        samples['coef_variation'][:,:,:,keepchains], (NLatent, Nelecs, totalsamps))

    lambda_samps = np.reshape(
        samples['latent_lambda'][:,:,:,keepchains], (NLatent, Nelecs, totalsamps))
    rate_cutoff = 1.75
    high_rates = np.where(np.median(lambda_samps[1, :, :], axis=1) > rate_cutoff)[0]
    print(samples['greychans'][0][high_rates])

    allrates = np.median(lambda_samps, axis=2)
    allclumping = np.median(clumping_samps, axis=2)
    allcoef = np.median(coef_samps, axis=2)


    corrval = pearsonr(allrates[0, :], allrates[1, :])
    print("The correlation of rates between the first two states is %.2f" % (corrval[0]))

    corrval = pearsonr(allclumping[0, :], allclumping[1, :])
    print("The correlation of clumping between the first two states is %.2f" % (corrval[0]))

    corrval = pearsonr(allcoef[0, :], allcoef[1, :])
    print("The correlation of the coefficients of variation between the first two states is %.2f" % (corrval[0]))

    corrval = pearsonr(allrates[0, :], allclumping[0, :])
    print("The correlation of rates and clumping in the first state is %.2f" % (corrval[0]))

    corrval = pearsonr(allrates[1, :], allclumping[1, :])
    print("The correlation of rates and clumping in the second state is %.2f" % (corrval[0]))
