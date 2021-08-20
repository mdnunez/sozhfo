# hfo_simmodel14.py - Script simulates and fits Model 14 using JAGS with simulated HFO count data
#     Accounting for possible overdispersion in the HFO count data without latent mixtures
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
# 04/06/21       Michael Nunez              Converted from hfo_fitmodel14.py
# 04/08/21       Michael Nunez            Finish simulation figure and export recovery figures
# 07/28/21       Michael Nunez                Correct label of time axis to minutes

# References:
# https://georgederpa.github.io/teaching/countModels.html
# https://www.johndcook.com/negative_binomial.pdf
# http://www.flutterbys.com.au/stats/tut/tut10.6b.html
# Label switching problem:
# https://stats.stackexchange.com/questions/152/is-there-a-standard-method-to-deal-with-label-switching-problem-in-mcmc-estimati
# https://sourceforge.net/p/mcmc-jags/discussion/610037/thread/e6850b93/
# https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
# https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
# https://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib


# Imports
import numpy as np
import numpy.ma as ma
import scipy.io as sio
import pyjags
from scipy.stats import nbinom, expon
from scipy.signal import savgol_filter
from time import strftime
import os
import csv
import matplotlib.pyplot as plt
from os.path import expanduser
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from hfo_fitmodel14results import run_diagnostic
from pyhddmjagsutils import recovery



#Plotting definitions

def add_subplot_axes(ax,rect):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax



### Simulations ###

# Generate HFO occurences through a Negative Binomial / Poisson process

if not os.path.exists('../data/simmodel14_data.mat'):

  # Set random seed
  np.random.seed(2021)

  # Number of simulated electrodes
  E = 300

  timestepsize = 1 #in seconds
  windowsize = 300 #in seconds (5 minutes)
  T = timestepsize*60*60*5 # 5 hours
  W = np.int32(T/windowsize) #Number of windows

  wind = np.empty((T),dtype='int32') #Window index
  for w in range(1,W+1):
    wind[((w-1)*windowsize):(w*windowsize)] = np.ones((1,windowsize))*w

  mean_lambda = np.empty((E))
  mean_clumping = np.empty((E))
  lambdas = np.empty((W,E))
  clumping = np.empty((W,E))
  r = np.empty((W,E))
  success_prob = np.empty((W,E))

  count = np.empty((T,E))

  for e in range(0,E-3):
    mean_lambda[e] = expon.rvs(size=1,scale=.5) # Generate a mean HFO rate of each simulated electrode
    mean_clumping[e] = expon.rvs(size=1,scale=5) # Generate a mean HFO clumping parameter for each simulated electrode
    for w in range(0,W):
      lambdas[w,e] = expon.rvs(size=1,scale=mean_lambda[e]) #Generate a HFO rate for each 5 minute window
      clumping[w,e] = expon.rvs(size=1,scale=mean_clumping[e]) # Generate a HFO clumping parameter for each 5 minute window

  mean_lambda[E-3] = .1 # Fixed value for simulation figure
  mean_lambda[E-2] = .1 # Fixed value for simulation figure
  mean_lambda[E-1] = .1 # Fixed value for simulation figure
  mean_clumping[E-3] = .01 # Fixed value for simulation figure
  mean_clumping[E-2] = 1 # Fixed value for simulation figure
  mean_clumping[E-1] = 10 # Fixed value for simulation figure
  for e in range(E-3,E):
    for w in range(0,W):
      lambdas[w,e] = mean_lambda[e] # Fixed values for simulation figure
      clumping[w,e] = mean_clumping[e] # Fixed values for simulation figure
  
  r = 1 / clumping # Note as clumping apporaches 0, r (number of failures) approaches infinity, and the negative binomial distribution approaches a Poission distribution
  success_prob = r / (r + lambdas)

  for e in range(0,E):
    for t in range(0,T):
      count[t,e] = nbinom.rvs(size=1,n=r[wind[t]-1,e],p=success_prob[wind[t]-1,e])


  genparam = dict()
  genparam['mean_lambda'] = mean_lambda
  genparam['mean_clumping'] = mean_clumping
  genparam['lambdas'] = lambdas
  genparam['clumping'] = clumping
  genparam['r'] = r
  genparam['success_prob'] = success_prob
  genparam['count'] = count
  sio.savemat('../data/simmodel14_data.mat', genparam)
  genparam = sio.loadmat('../data/simmodel14_data.mat')
else:
  genparam = sio.loadmat('../data/simmodel14_data.mat')


### Plot simulations ###
fontsize = 36

windowsize = 300 #in seconds (5 minutes)
T = genparam['count'].shape[0]
W = np.int32(T/windowsize) #State window in minutes
E = genparam['count'].shape[1] #Number of electrodes

fig, (ax1, ax2, ax3) = plt.subplots(3,1, dpi=300, figsize=(40,20), sharex=True)

plottimesize = 3600 #in seconds (60 minutes)
secindex = np.arange(0,plottimesize)
minindex = secindex/60
ax1.plot(minindex,genparam['count'][0:plottimesize,E-3],color='k',linewidth=1)
ax1.plot(minindex,savgol_filter(genparam['count'][0:plottimesize,E-3],51,3),color='b',linewidth=3)
ax2.plot(minindex,genparam['count'][0:plottimesize,E-2],color='k',linewidth=1)
ax2.plot(minindex,savgol_filter(genparam['count'][0:plottimesize,E-2],51,3),color='b',linewidth=3)
ax3.plot(minindex,genparam['count'][0:plottimesize,E-1],color='k',linewidth=1)
ax3.plot(minindex,savgol_filter(genparam['count'][0:plottimesize,E-1],51,3),color='b',linewidth=3)
ax1.set_title('Clumping parameter of .01', fontsize=fontsize)
ax2.set_title('Clumping parameter of 1',fontsize=fontsize)
ax3.set_title('Clumping parameter of 10',fontsize=fontsize)
ax1.tick_params(axis='both', which='major', labelsize=fontsize)
ax2.tick_params(axis='both', which='major', labelsize=fontsize)
ax3.tick_params(axis='both', which='major', labelsize=fontsize)
ax3.set_xlabel('Time (minutes)',fontsize=fontsize)
ax1.set_ylabel('Instantaneous Rate', fontsize=fontsize)
ax1.set_xlim([0, 60])
ax1.set_ylim([0, 7])
ax2.set_ylabel('Instantaneous Rate', fontsize=fontsize)
ax2.set_xlim([0, 60])
ax2.set_ylim([0, 7])
ax3.set_ylabel('Instantaneous Rate', fontsize=fontsize)
ax3.set_xlim([0, 60])
ax3.set_ylim([0, 7])
plt.savefig(('../figures/simulations/Clumping_60min.png'), format='png',bbox_inches="tight")
plt.close()


### Fit model 14 ###
timestepsize = 1 #in seconds


T = genparam['count'].shape[0]
W = np.int32(T/windowsize) #State window in minutes
E = genparam['count'].shape[1] #Number of electrodes


wind = np.empty((1,T),dtype='int32')
for w in range(1,W+1):
  wind[0,((w-1)*windowsize):(w*windowsize)] = np.ones((1,windowsize))*w

thismodel = '''
model {
  for (e in 1:E) {
    mean_lambda[1,e] ~ dnorm(1, pow(.5, -2))
    std_lambda[1,e] ~ dgamma(1,1)
    mean_clumping[1,e] ~ dnorm(10, pow(5, -2))
    std_clumping[1,e] ~ dexp(0.25)
     for (w in 1:W) {
        lambda[w, e] ~ dnorm(mean_lambda[1,e], pow(std_lambda[1,e], -2)) T(0, )
        clumping[w, e] ~ dnorm(mean_clumping[1,e], pow(std_clumping[1,e], -2)) T(0, )
        success_prob[w, e] <- r[w, e] / (r[w, e] + lambda[w, e])
        coef_variation[w, e] <- pow( sqrt(r[w,e]*(1-success_prob[w, e])) , -1)
        r[w,e] <- pow(clumping[w,e], -1)
     }
     for (t in 1:T) {
        count[t,e] ~ dnegbin(success_prob[ wind[1,t], e ], r[ wind[1,t], e ])
     }
  }


} # End of model
'''

# pyjags code

# Make sure $LD_LIBRARY_PATH sees /usr/local/lib
pyjags.modules.load_module('dic')
pyjags.modules.list_modules()

nchains = 6
threads = 6
chains_per_thread = np.int64(np.ceil(nchains/threads))
burnin = 200  # Note that scientific notation breaks pyjags
nsamps = 5000

trackvars = ['mean_lambda', 'std_lambda', 'mean_clumping', 'std_clumping', 'lambda', 'r', 'success_prob', 'coef_variation', 'clumping']


initials = []
for c in range(0, nchains):
    chaininit = {
        'mean_lambda': np.random.uniform(.1, 2, size=(1, E)),
        'std_lambda': np.random.uniform(.01, 1., size=(1, E)),
        'lambda': np.random.uniform(.1, 2, size=(W, E)),
        'mean_clumping': np.random.uniform(0, 20, size=(1, E)),
        'std_clumping': np.random.uniform(1, 10, size=(1, E)),
        'clumping': np.random.uniform(0, 20, size=(W, E))
    }
    initials.append(chaininit)


# Run JAGS model

# Choose JAGS model type
saveloc = '/data/jagsout/'
modelname = 'model14samples_sim1'

# Save model
timestart = strftime('%b') + '_' + strftime('%d') + '_' + \
    strftime('%y') + '_' + strftime('%H') + '_' + strftime('%M')
modelfile = saveloc + 'jagsmodel_' + modelname + timestart + '.jags'
f = open(modelfile, 'w')
f.write(thismodel)
f.close()
print('Fitting model %s ...' % (modelname + timestart))


indata = dict(E=E, T=T, W=W, count=genparam['count'], wind=wind)

threaded = pyjags.Model(file=modelfile, init=initials,
                        data=indata,
                        chains=nchains, adapt=burnin, threads=threads,
                        chains_per_thread=chains_per_thread, progress_bar=True)


samples = threaded.sample(nsamps, vars=trackvars, thin=10)

savestring = saveloc + "jagsmodel_" + \
    modelname + timestart + ".mat"

print('Saving results to: \n %s' % savestring)
sio.savemat(savestring, samples)
diags = run_diagnostic(samples)
samples = sio.loadmat(savestring)


### Recovery figures ###

plt.figure()
recovery(samples['lambda'],genparam['lambdas'])
plt.title('Recovery of HFO rates')
plt.savefig(('../figures/model14results/lambdas_recovery_model14.png'), format='png',bbox_inches="tight")
plt.close()

plt.figure()
recovery(samples['clumping'],genparam['clumping'])
plt.title('Recovery of HFO clumping parameters')
plt.savefig(('../figures/model14results/clumping_recovery_model14.png'), format='png',bbox_inches="tight")
plt.close()

plt.figure()
recovery(samples['clumping'],genparam['clumping'])
plt.xlim(0,5)
plt.ylim(0,50)
plt.title('Recovery of HFO clumping parameters')
plt.savefig(('../figures/model14results/clumping_recovery_zoomed_model14.png'), format='png',bbox_inches="tight")
plt.close()

nchains = samples['success_prob'].shape[-1]
totalsamps = nchains * samples['success_prob'].shape[-2]
Nwinds = samples['lambda'].shape[0]
Nelecs = samples['lambda'].shape[1]
clumping_samps = np.reshape(samples['clumping'][:,:,:,:], (Nwinds, Nelecs, totalsamps))
lambda_samps = np.reshape(samples['lambda'][:,:,:,:], (Nwinds, Nelecs, totalsamps))
allclumping = np.median(clumping_samps, axis=2)
allrates = np.median(lambda_samps, axis=2)


fig, (ax1, ax2) = plt.subplots(1,2, dpi=300, figsize=(40,20))

ax1.plot(np.mean(allrates,axis=0),np.mean(allrates,axis=0),linewidth=5, color='k')
ax1.plot(np.squeeze(genparam['mean_lambda']),np.mean(allrates,axis=0),'o',color='b',markersize=12)
ax1.set_xlabel('True mean HFO rate',fontsize=fontsize)
ax1.set_ylabel('Model-derived mean HFO rate',fontsize=fontsize)
ax1.tick_params(axis='both', which='major', labelsize=fontsize)

ax2.plot(np.mean(allclumping,axis=0),np.mean(allclumping,axis=0),linewidth=5, color='k')
ax2.plot(np.squeeze(genparam['mean_clumping']),np.mean(allclumping,axis=0),'o',color='b',markersize=12)
ax2.set_xlabel('True mean HFO clumping parameter',fontsize=fontsize)
ax2.set_ylabel('Model-derived mean HFO clumping parameter',fontsize=fontsize)
ax2.tick_params(axis='both', which='major', labelsize=fontsize)

subpos = [0.65,0.05,0.3,0.3]

subax1 = add_subplot_axes(ax1,subpos)
subax1.plot(np.mean(allrates,axis=0),np.mean(allrates,axis=0),linewidth=5, color='k')
subax1.plot(np.squeeze(genparam['mean_lambda']),np.mean(allrates,axis=0),'o',color='b',markersize=12)
subax1.set_xlim(0,0.5)
subax1.set_ylim(0,0.5)
subax1.set_xticks([0,0.25,0.5])
subax1.set_yticks([0.25,0.5])
subax1.tick_params(axis='both', which='major', labelsize=fontsize)
mark_inset(ax1, subax1, loc1=2, loc2=4, fc='none', ec="0.5")

subax2 = add_subplot_axes(ax2,subpos)
subax2.plot(np.mean(allclumping,axis=0),np.mean(allclumping,axis=0),linewidth=5, color='k')
subax2.plot(np.squeeze(genparam['mean_clumping']),np.mean(allclumping,axis=0),'o',color='b',markersize=12)
subax2.set_xlim(0,5)
subax2.set_ylim(0,5)
subax2.set_xticks([0,2.5,5])
subax2.set_yticks([2.5,5])
subax2.tick_params(axis='both', which='major', labelsize=fontsize)
mark_inset(ax2, subax2, loc1=2, loc2=4, fc='none', ec="0.5",linewidth=3)


plt.savefig(('../figures/model14results/mean_lambdas_clumping_recovery_model14.png'), format='png',bbox_inches="tight")
plt.close()