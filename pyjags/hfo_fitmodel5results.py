# hfo_fitmodel5results.py - Evaluates simulation results of models
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
# 07/27/18      Michael Nunez              Converted from pdm5b_simmodel5results.py
# 07/31/18      Michael NUnez                  Updated results for Patient2
# 09/11/18      Michael Nunez                  Save out reordered data
# 09/13/18      Michael Nunez                   Load from new data storage location
# 10/25/18      Michael Nunez                         New results
# 11/19/18      Michael Nunez        Plot model results from Model type 8
# 12/06/18      Michael Nunez   Sort chains by variable 'state_lambda' if it exists
# 01/11/19      Michael Nunez           Plot new result, update copyright year
# 01/29/19      Michael Nunez         Remove dependency on ipython
# 03/07/19      Michael Nunez       Return variable with maximum Rhat
# 09/12/19      Michael Nunez            Plot chains for state_lambda
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

model = '/data/jagsout/jagsmodel_model8samples_Patient1_grey_2LatentNov_19_18_11_17'




samples = sio.loadmat('%s.mat' % model)

# Switch chain indices (how do we order the index ahead of time?)
nchains = samples['pi'].shape[-1]
totalsamps = nchains * samples['pi'].shape[-2]
NLatent = samples['pi'].shape[1]
Nelecs = samples['latent_lambda'].shape[1]

if 'state_lambda' in samples:
    probmeans = np.mean(np.squeeze(samples['state_lambda']), axis=1)
else:
    probmeans = np.mean(np.squeeze(samples['pi']), axis=1)
sortorder = np.empty((NLatent, nchains), dtype='int')
for n in range(0, nchains):
    sortorder[:, n] = np.argsort(probmeans[:, n])
    samples['latent_lambda'][0:NLatent, :, :, n] = samples[
        'latent_lambda'][sortorder[:, n], :, :, n]
    samples['pi'][0, :, :, n] = samples['pi'][0, sortorder[:, n], :, n]
    tempsamps = np.copy(samples['stateindx'][:, :, :, n])
    if 'state_lambda' in samples:
        samples['state_lambda'][0, 0:NLatent, :, n] = samples[
        'state_lambda'][0, sortorder[:, n], :, n]
        samples['state_std'][0, 0:NLatent, :, n] = samples[
        'state_std'][0, sortorder[:, n], :, n]

if not os.path.isfile('%s_reordered.mat' % model):
    samples['sortorder'] = sortorder
    sio.savemat('%s_reordered.mat' % model, samples)
if 'state_lambda' in samples:
    samples_relevant = dict(latent_lambda=samples[
                            'latent_lambda'], pi=samples['pi'],
                            state_lambda=samples['state_lambda'],
                            state_std=samples['state_std'])
else:
    samples_relevant = dict(latent_lambda=samples[
                            'latent_lambda'], pi=samples['pi'])
diags = diagnostic(samples_relevant)

# # Load generated data
# jagsin = sio.loadmat('../data/simmodel5_data.mat')
# E = jagsin['E'][0][0]
# T = jagsin['T'][0][0]
# count = jagsin['count']
# wind = jagsin['wind']


plt.figure()
jellyfish(samples['latent_lambda'])
plt.title('Poisson rates per second of latent states')

plt.figure()
jellyfish(samples['pi'])
plt.title('Probability of latent states')

plt.figure()
fivemins = samples['stateindx'].shape[1]
stateindx_samps = np.reshape(samples['stateindx'], (fivemins, 500, 6))
plt.plot(np.squeeze(np.mean(stateindx_samps[:, :, ], axis=1)))
plt.title('Recovery of Sleep states', fontsize=16)
plt.xlabel('5 Minute Window #', fontsize=16)
plt.ylabel('Average State Label', fontsize=16)

if 'state_lambda' in samples:
    plt.figure()
    jellyfish(samples['state_lambda'])
    plt.title('Average Poisson rates over all electrodes')
    plt.figure()
    jellyfish(samples['state_std'])
    plt.title('Std of Poisson rates over all electrodes')

    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(samples['state_lambda'][0,0,:,:])
    ax1.set_title('Chains for State_lambda k=1')
    ax2.plot(samples['state_lambda'][0,1,:,:])
    ax2.set_title('Chains for state_lambda k=2')


lambda_samps = np.reshape(
    samples['latent_lambda'], (NLatent, Nelecs, totalsamps))
rate_cutoff = 1.75
high_rates = np.where(np.mean(lambda_samps[1, :, :], axis=1) > rate_cutoff)[0]
print samples['greychans'][0][high_rates]

allrates = np.empty((NLatent, Nelecs))
for n in range(0, NLatent):
    allrates[n, :] = np.mean(lambda_samps[n, :, :], axis=1)

corrval = pearsonr(allrates[0, :], allrates[1, :])
print "The correlation of rates between the first two states is %.2f" % (corrval[0])


plt.show()
