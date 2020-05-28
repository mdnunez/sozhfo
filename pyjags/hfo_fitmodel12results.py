# hfo_fitmodel12results.py - Evaluates simulation results of models
#                         fit to real data that assumes latent states in HFO rates
#
#
# Copyright (C) 2019 Michael D. Nunez, <mdnunez1@uci.edu>
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
# 05/28/20      Michael Nunez             Remove patient identifiers

# Imports
import numpy as np
import scipy.io as sio
from scipy import stats
import os.path
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
# from IPython import get_ipython  # Run magic functions from script
# get_ipython().magic('pylab')  # Initialize ipython matplotlib plotting graphics




def jellyfish(possamps):  # jellyfish plots
    """Plots posterior distributions of given posterior samples in a jellyfish
    plot. Jellyfish plots are posterior distributions (mirrored over their
    horizontal axes) with 99% and 95% credible intervals (currently plotted
    from the .5% and 99.5% & 2.5% and 97.5% percentiles respectively.
    Also plotted are the median and mean of the posterior distributions"

    Parameters
    ----------
    possamps : ndarray of posterior chains where the last dimension is
    the number of chains, the second to last dimension is the number of samples
    in each chain, all other dimensions describe the shape of the parameter
    """

    # Number of chains
    nchains = possamps.shape[-1]

    # Number of samples per chain
    nsamps = possamps.shape[-2]

    # Number of dimensions
    ndims = possamps.ndim - 2

    # Number of variables to plot
    nvars = np.prod(possamps.shape[0:-2])

    # Index of variables
    varindx = np.arange(nvars).reshape(possamps.shape[0:-2])

    # Reshape data
    alldata = np.reshape(possamps, (nvars, nchains, nsamps))
    alldata = np.reshape(alldata, (nvars, nchains * nsamps))

    # Plot properties
    LineWidths = np.array([2, 5])
    teal = np.array([0, .7, .7])
    blue = np.array([0, 0, 1])
    orange = np.array([1, .3, 0])
    Colors = [teal, blue]

    # Initialize ylabels list
    ylabels = ['']

    for v in range(0, nvars):
        # Create ylabel
        whereis = np.where(varindx == v)
        newlabel = ''
        for l in range(0, ndims):
            newlabel = newlabel + ('_%i' % whereis[l][0])

        ylabels.append(newlabel)

        # Compute posterior density curves
        kde = stats.gaussian_kde(alldata[v, :])
        bounds = stats.scoreatpercentile(alldata[v, :], (.5, 2.5, 97.5, 99.5))
        for b in range(0, 2):
            # Bound by .5th percentile and 99.5th percentile
            x = np.linspace(bounds[b], bounds[-1 - b], 100)
            p = kde(x)

            # Scale distributions down
            maxp = np.max(p)

            # Plot jellyfish
            upper = .25 * p / maxp + v + 1
            lower = -.25 * p / maxp + v + 1
            lines = plt.plot(x, upper, x, lower)
            plt.setp(lines, color=Colors[b], linewidth=LineWidths[b])
            if b == 1:
                # Mark mode
                wheremaxp = np.argmax(p)
                mmode = plt.plot(np.array([1., 1.]) * x[wheremaxp],
                                 np.array([lower[wheremaxp], upper[wheremaxp]]))
                plt.setp(mmode, linewidth=3, color=orange)
                # Mark median
                mmedian = plt.plot(np.median(alldata[v, :]), v + 1, 'ko')
                plt.setp(mmedian, markersize=10, color=[0., 0., 0.])
                # Mark mean
                mmean = plt.plot(np.mean(alldata[v, :]), v + 1, '*')
                plt.setp(mmean, markersize=10, color=teal)

    # Display plot
    plt.setp(plt.gca(), yticklabels=ylabels, yticks=np.arange(0, nvars + 1))


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


# Load generated smaples
model = '/data/jagsout/jagsmodel_model12samples_Patient7_grey_2LatentApr_02_19_19_39';
keepchains = np.array([0, 1, 2, 3, 5])
# model = '/data/jagsout/jagsmodel_model12samples_Patient2_grey_2LatentFeb_28_19_19_04',
# model = '/data/jagsout/jagsmodel_model12samples_Patient8_grey_2LatentMar_27_19_13_20';
# # keepchains = np.array([1, 2, 4])
# model = '/data/jagsout/jagsmodel_model12samples_Patient1_grey_2LatentFeb_27_19_19_53';
# model = '/data/jagsout/jagsmodel_model12samples_Patient11_grey_2LatentSep_22_19_01_47';
# keepchains = np.array([0, 1, 5])
# model = '/data/jagsout/jagsmodel_model12samples_Patient6_grey_2LatentMar_26_19_12_43';
# keepchains = np.array([1, 3, 4])
# keepchains = np.array([0, 2, 5])
# model = '/data/jagsout/jagsmodel_model12samples_Patient3_grey_2LatentMar_04_19_17_32';
# model = '/data/jagsout/jagsmodel_model12samples_Patient4_grey_2LatentMar_15_19_19_07';
# model = '/data/jagsout/jagsmodel_model12samples_Patient12_grey_2LatentApr_09_19_19_51';
# keepchains = np.array([4])
# model = '/data/jagsout/jagsmodel_model12samples_Patient5_grey_2LatentMar_11_19_11_55';
# model = '/data/jagsout/jagsmodel_model12samples_Patient9_grey_2LatentApr_09_19_19_48';
# keepchains = np.array([0, 1, 4])
# model = '/data/jagsout/jagsmodel_model12samples_Patient10_grey_2LatentJul_15_19_13_31';
# keepchains = np.array([0, 2, 3, 4, 5])
# model = '/data/jagsout/jagsmodel_model12samples_Patient13_grey_2LatentJul_18_19_12_03';
# model = '/data/jagsout/jagsmodel_model12samples_Patient14_grey_2LatentJul_09_19_17_57';
# keepchains = np.array([0, 1, 3, 4, 5])
# model = '/data/jagsout/jagsmodel_model12samples_Patient15_grey_2LatentJul_11_19_12_50';
# keepchains = np.array([1, 2, 3, 4, 5])
# model = '/data/jagsout/jagsmodel_model12samples_Patient16_grey_2LatentJul_14_19_23_05';
# keepchains = np.array([1, 3, 5])

samples = sio.loadmat('%s.mat' % model)

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


if not os.path.isfile('%s_reordered.mat' % model):
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



plt.figure()
jellyfish(samples['latent_lambda'][:,:,:,keepchains])
plt.title('Poisson rates per second of latent states')

plt.figure()
jellyfish(samples['clumping'][:,:,:,keepchains])
plt.title('Clumping parameter of latent states')

plt.figure()
jellyfish(samples['coef_variation'][:,:,:,keepchains])
plt.title('Coefficient of variation of latent states')

plt.figure()
jellyfish(samples['pi'][:,:,:,keepchains])
plt.title('Probability of latent states')


plt.figure()
fivemins = samples['stateindx'].shape[1]
stateindx_samps = np.reshape(samples['stateindx'], (fivemins, 500, originalnchains))
for n in keepchains:
    plt.plot(np.squeeze(np.mean(stateindx_samps[:, :, n], axis=1)), label=('chain %d' % (n+1)))
plt.title('Recovery of Sleep states', fontsize=16)
plt.xlabel('5 Minute Window #', fontsize=16)
plt.ylabel('Average State Label', fontsize=16)
plt.legend()

plt.figure()
jellyfish(samples['state_lambda'][:,:,:,keepchains])
plt.title('Average Poisson rates over all electrodes')
plt.figure()
jellyfish(samples['state_std'][:,:,:,keepchains])
plt.title('Std of Poisson rates over all electrodes')

plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
for n in keepchains:
    ax1.plot(samples['state_lambda'][0,0,:,n], label=('chain %d' % (n+1)))
    ax2.plot(samples['state_lambda'][0,1,:,n], label=('chain %d' % (n+1)))
ax1.set_title('Chains for State_lambda k=1')
ax2.set_title('Chains for state_lambda k=2')
ax2.legend()


clumping_samps = np.reshape(
    samples['clumping'][:,:,:,keepchains], (NLatent, Nelecs, totalsamps))
coef_samps = np.reshape(
    samples['coef_variation'][:,:,:,keepchains], (NLatent, Nelecs, totalsamps))

lambda_samps = np.reshape(
    samples['latent_lambda'][:,:,:,keepchains], (NLatent, Nelecs, totalsamps))
rate_cutoff = 1.75
high_rates = np.where(np.median(lambda_samps[1, :, :], axis=1) > rate_cutoff)[0]
print samples['greychans'][0][high_rates]

allrates = np.median(lambda_samps, axis=2)
allclumping = np.median(clumping_samps, axis=2)
allcoef = np.median(coef_samps, axis=2)


corrval = pearsonr(allrates[0, :], allrates[1, :])
print "The correlation of rates between the first two states is %.2f" % (corrval[0])

corrval = pearsonr(allclumping[0, :], allclumping[1, :])
print "The correlation of clumping between the first two states is %.2f" % (corrval[0])

corrval = pearsonr(allcoef[0, :], allcoef[1, :])
print "The correlation of the coefficients of variation between the first two states is %.2f" % (corrval[0])

corrval = pearsonr(allrates[0, :], allclumping[0, :])
print "The correlation of rates and clumping in the first state is %.2f" % (corrval[0])

corrval = pearsonr(allrates[1, :], allclumping[1, :])
print "The correlation of rates and clumping in the second state is %.2f" % (corrval[0])


plt.show()
