# pyhddmjagsutils.py - Definitions for simulation, model diagnostics, and parameter recovery
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
# 06/29/20      Michael Nunez                             Original code
# 12/04/20      Michael Nunez              Update explanation of summary output


# Modules
import numpy as np
import numpy.matlib
import pyjags
import scipy.io as sio
from scipy import stats
import warnings
import os
import matplotlib.pyplot as plt

### Definitions ###

def diagnostic(insamples):
    """
    Returns two versions of Rhat (measure of convergence, less is better with an approximate
    1.10 cutoff) and Neff, number of effective samples). Note that 'rhat' is more diagnostic than 'oldrhat' according to 
    Gelman et al. (2014).

    Reference for preferred Rhat calculation (split chains) and number of effective sample calculation: 
        Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. & Rubin, D. B. (2014). 
        Bayesian data analysis (Third Edition). CRC Press:
        Boca Raton, FL

    Reference for original Rhat calculation:
        Gelman, A., Carlin, J., Stern, H., & Rubin D., (2004).
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
        rhat, oldrhat, neff, posterior mean, and posterior std for each variable. Prints maximum Rhat and minimum Neff across all variables
    """

    result = {}  # Initialize dictionary
    maxrhatsold = np.zeros((len(insamples.keys())), dtype=float)
    maxrhatsnew = np.zeros((len(insamples.keys())), dtype=float)
    minneff = np.ones((len(insamples.keys())), dtype=float)*np.inf
    allkeys ={} # Initialize dictionary
    keyindx = 0
    for key in insamples.keys():
        if key[0] != '_':
            result[key] = {}
            
            possamps = insamples[key]
            
            # Number of chains
            nchains = possamps.shape[-1]
            
            # Number of samples per chain
            nsamps = possamps.shape[-2]
            
            # Number of variables per key
            nvars = np.prod(possamps.shape[0:-2])
            
            # Reshape data
            allsamps = np.reshape(possamps, possamps.shape[:-2] + (nchains * nsamps,))

            # Reshape data to preduce R_hatnew
            possampsnew = np.empty(possamps.shape[:-2] + (int(nsamps/2), nchains * 2,))
            newc=0
            for c in range(nchains):
                possampsnew[...,newc] = np.take(np.take(possamps,np.arange(0,int(nsamps/2)),axis=-2),c,axis=-1)
                possampsnew[...,newc+1] = np.take(np.take(possamps,np.arange(int(nsamps/2),nsamps),axis=-2),c,axis=-1)
                newc += 2

            # Index of variables
            varindx = np.arange(nvars).reshape(possamps.shape[0:-2])
            
            # Reshape data
            alldata = np.reshape(possamps, (nvars, nsamps, nchains))
                    
            # Mean of each chain for rhat
            chainmeans = np.mean(possamps, axis=-2)
            # Mean of each chain for rhatnew
            chainmeansnew = np.mean(possampsnew, axis=-2)
            # Global mean of each parameter for rhat
            globalmean = np.mean(chainmeans, axis=-1)
            globalmeannew = np.mean(chainmeansnew, axis=-1)
            result[key]['mean'] = globalmean
            result[key]['std'] = np.std(allsamps, axis=-1)
            globalmeanext = np.expand_dims(
                globalmean, axis=-1)  # Expand the last dimension
            globalmeanext = np.repeat(
                globalmeanext, nchains, axis=-1)  # For differencing
            globalmeanextnew = np.expand_dims(
                globalmeannew, axis=-1)  # Expand the last dimension
            globalmeanextnew = np.repeat(
                globalmeanextnew, nchains*2, axis=-1)  # For differencing
            # Between-chain variance for rhat
            between = np.sum(np.square(chainmeans - globalmeanext),
                             axis=-1) * nsamps / (nchains - 1.)
            # Mean of the variances of each chain for rhat
            within = np.mean(np.var(possamps, axis=-2), axis=-1)
            # Total estimated variance for rhat
            totalestvar = (1. - (1. / nsamps)) * \
                within + (1. / nsamps) * between
            # Rhat (original Gelman-Rubin statistic)
            temprhat = np.sqrt(totalestvar / within)
            maxrhatsold[keyindx] = np.nanmax(temprhat) # Ignore NANs
            allkeys[keyindx] = key
            result[key]['oldrhat'] = temprhat
            # Between-chain variance for rhatnew
            betweennew = np.sum(np.square(chainmeansnew - globalmeanextnew),
                             axis=-1) * (nsamps/2) / ((nchains*2) - 1.)
            # Mean of the variances of each chain for rhatnew
            withinnew = np.mean(np.var(possampsnew, axis=-2), axis=-1)
            # Total estimated variance
            totalestvarnew = (1. - (1. / (nsamps/2))) * \
                withinnew + (1. / (nsamps/2)) * betweennew
            # Rhatnew (Gelman-Rubin statistic from Gelman et al., 2013)
            temprhatnew = np.sqrt(totalestvarnew / withinnew)
            maxrhatsnew[keyindx] = np.nanmax(temprhatnew) # Ignore NANs
            result[key]['rhat'] = temprhatnew
            # Number of effective samples from Gelman et al. (2013) 286-288
            neff = np.empty(possamps.shape[0:-2])
            for v in range(0, nvars):
                whereis = np.where(varindx == v)
                rho_hat = []
                rho_hat_even = 0
                rho_hat_odd = 0
                t = 2
                while ((t < nsamps - 2) & (float(rho_hat_even) + float(rho_hat_odd) >= 0)):
                    variogram_odd = np.mean(np.mean(np.power(alldata[v,(t-1):nsamps,:] - alldata[v,0:(nsamps-t+1),:],2),axis=0)) # above equation (11.7) in Gelman et al., 2013
                    rho_hat_odd = 1 - np.divide(variogram_odd, 2*totalestvar[whereis]) # Equation (11.7) in Gelman et al., 2013
                    rho_hat.append(rho_hat_odd)
                    variogram_even = np.mean(np.mean(np.power(alldata[v,t:nsamps,:] - alldata[v,0:(nsamps-t),:],2),axis=0)) # above equation (11.7) in Gelman et al., 2013
                    rho_hat_even = 1 - np.divide(variogram_even, 2*totalestvar[whereis]) # Equation (11.7) in Gelman et al., 2013
                    rho_hat.append(rho_hat_even)
                    t += 2
                rho_hat = np.asarray(rho_hat)
                neff[whereis] = np.divide(nchains*nsamps, 1 + 2*np.sum(rho_hat)) # Equation (11.8) in Gelman et al., 2013
            result[key]['neff'] = np.round(neff) 
            minneff[keyindx] = np.nanmin(np.round(neff))
            keyindx += 1

            # Geweke statistic?
    # print("Maximum old Rhat was %3.2f for variable %s" % (np.max(maxrhatsold),allkeys[np.argmax(maxrhatsold)]))
    maxrhatkey = allkeys[np.argmax(maxrhatsnew)]
    maxrhatindx = np.unravel_index( np.argmax(result[maxrhatkey]['rhat']) , result[maxrhatkey]['rhat'].shape)
    print("Maximum Rhat was %3.2f for variable %s at index %s" % (np.max(maxrhatsnew), maxrhatkey, maxrhatindx))
    minneffkey = allkeys[np.argmin(minneff)]
    minneffindx = np.unravel_index( np.argmin(result[minneffkey]['neff']) , result[minneffkey]['neff'].shape)
    print("Minimum number of effective samples was %d for variable %s at index %s" % (np.min(minneff), minneffkey, minneffindx))
    return result


def summary(insamples):
    """
    Returns parameter estimates for each posterior distribution (mean and median posteriors) as well as 95% and 99% credible intervals (.5th, 2.5th, 97.5th, 99.5th percentiles)

    Parameters
    ----------
    insamples: dic
        Sampled values of monitored variables as a dictionary where keys
        are variable names and values are numpy arrays with shape:
        (dim_1, dim_n, iterations, chains). dim_1, ..., dim_n describe the
        shape of variable in JAGS model.
    """

    result = {}  # Initialize dictionary
    maxrhats = np.zeros((len(insamples.keys())), dtype=float)
    maxrhatsnew = np.zeros((len(insamples.keys())), dtype=float)
    minneff = np.ones((len(insamples.keys())), dtype=float)*np.inf
    allkeys ={} # Initialize dictionary
    keyindx = 0
    for key in insamples.keys():
        if key[0] != '_':
            result[key] = {}
            
            possamps = insamples[key]
            
            # Number of chains
            nchains = possamps.shape[-1]
            
            # Number of samples per chain
            nsamps = possamps.shape[-2]
            
            # Number of variables per key
            nvars = np.prod(possamps.shape[0:-2])
            
            # Reshape data
            allsamps = np.reshape(possamps, possamps.shape[:-2] + (nchains * nsamps,))

            # Reshape data to preduce R_hatnew
            possampsnew = np.empty(possamps.shape[:-2] + (int(nsamps/2), nchains * 2,))
            newc=0
            for c in range(nchains):
                possampsnew[...,newc] = np.take(np.take(possamps,np.arange(0,int(nsamps/2)),axis=-2),c,axis=-1)
                possampsnew[...,newc+1] = np.take(np.take(possamps,np.arange(int(nsamps/2),nsamps),axis=-2),c,axis=-1)
                newc += 2

            result[key]['mean'] = np.mean(allsamps, axis=-1)
            result[key]['std'] = np.std(allsamps, axis=-1)
            result[key]['median'] = np.quantile(allsamps,0.5, axis=-1)
            result[key]['95lower'] = np.quantile(allsamps,0.025, axis=-1)
            result[key]['95upper'] = np.quantile(allsamps,0.975, axis=-1)
            result[key]['99lower'] = np.quantile(allsamps,0.005, axis=-1)
            result[key]['99upper'] = np.quantile(allsamps,0.995, axis=-1)
    return result


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


def recovery(possamps, truevals):  # Parameter recovery plots
    """Plots true parameters versus 99% and 95% credible intervals of recovered
    parameters. Also plotted are the median (circles) and mean (stars) of the posterior
    distributions.

    Parameters
    ----------
    possamps : ndarray of posterior chains where the last dimension is the
    number of chains, the second to last dimension is the number of samples in
    each chain, all other dimensions must match the dimensions of truevals

    truevals : ndarray of true parameter values
    """

    # Number of chains
    nchains = possamps.shape[-1]

    # Number of samples per chain
    nsamps = possamps.shape[-2]

    # Number of variables to plot
    nvars = np.prod(possamps.shape[0:-2])

    # Reshape data
    alldata = np.reshape(possamps, (nvars, nchains, nsamps))
    alldata = np.reshape(alldata, (nvars, nchains * nsamps))
    truevals = np.reshape(truevals, (nvars))

    # Plot properties
    LineWidths = np.array([2, 5])
    teal = np.array([0, .7, .7])
    blue = np.array([0, 0, 1])
    orange = np.array([1, .3, 0])
    Colors = [teal, blue]

    for v in range(0, nvars):
        # Compute percentiles
        bounds = stats.scoreatpercentile(alldata[v, :], (.5, 2.5, 97.5, 99.5))
        for b in range(0, 2):
            # Plot credible intervals
            credint = np.ones(100) * truevals[v]
            y = np.linspace(bounds[b], bounds[-1 - b], 100)
            lines = plt.plot(credint, y)
            plt.setp(lines, color=Colors[b], linewidth=LineWidths[b])
            if b == 1:
                # Mark median
                mmedian = plt.plot(truevals[v], np.median(alldata[v, :]), 'o')
                plt.setp(mmedian, markersize=10, color=[0., 0., 0.])
                # Mark mean
                mmean = plt.plot(truevals[v], np.mean(alldata[v, :]), '*')
                plt.setp(mmean, markersize=10, color=teal)
    # Plot line y = x
    tempx = np.linspace(np.min(truevals), np.max(
        truevals), num=100)
    recoverline = plt.plot(tempx, tempx)
    plt.setp(recoverline, linewidth=3, color=orange)


def rsquared_pred(trueval,predval):
    """
    RSQUARED_PRED  Calculates R^2_prediction for data and statistics derived from data
    """
    divisor = np.sum(np.isfinite(trueval)) -1
    # Mean squared error of prediction
    MSEP = np.nansum(np.power(trueval - predval,2)) / divisor
    # Variance estimate of the true values
    vartrue = np.nansum(np.power(trueval - np.nanmean(trueval),2)) / divisor
    # R-squared definition
    rsquared = 1 - (MSEP / vartrue)
    return rsquared