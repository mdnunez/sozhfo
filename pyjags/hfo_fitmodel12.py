# hfo_fitmodel12.py - Script fits Model 12 using JAGS with real HFO count data
#                     Accounting for possible overdispersion in the HFO count data
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
# 02/27/19       Michael Nunez              Converted from hfo_fitmodel11.py
# 06/14/19       Michael Nunez        All channels flag to see the effect of noisy channels on prediction
# 07/09/19       Michael Nunez              For loop for batch processing
# 09/12/19       Michael Nunez                   Fix first window to state 1
# 05/28/20      Michael Nunez             Remove patient identifiers
# 08/20/21      Michael Nunez             Remove patient identifiers


# References:
# https://georgederpa.github.io/teaching/countModels.html
# https://www.johndcook.com/negative_binomial.pdf
# http://www.flutterbys.com.au/stats/tut/tut10.6b.html
# Label switching problem:
# https://stats.stackexchange.com/questions/152/is-there-a-standard-method-to-deal-with-label-switching-problem-in-mcmc-estimati
# https://sourceforge.net/p/mcmc-jags/discussion/610037/thread/e6850b93/


# Imports
from __future__ import division
import numpy as np
import numpy.ma as ma
import scipy.io as sio
import pyjags
from scipy import stats
from time import strftime
import os
import csv


usingchans = 'grey'

# patient = 'Patient1'
# starthour = 7 - 1;
# endhour = 19;
# NLatent = 2; #Number of latent states
# patient = 'Patient2'
# starthour = 25 - 1;
# endhour = 49;
# NLatent = 2; #Number of latent states
# patient = 'Patient3'
# starthour = 46 - 17; #Account for 16 missing BESA files correction
# endhour = 60 - 16; #Account for 16 missing BESA files correction;
# NLatent = 2; #Number of latent states
# patient = 'Patient4'
# starthour = 1 - 1
# endhour = 16
# NLatent = 2 #Number of latent states
# patient = 'Patient12'
# starthour = 2 -2 #Accounting for 1 missing besa file
# endhour = 8 -1
# NLatent = 2 #Number of latent states
# patient = 'Patient5'
# starthour = 1 -1
# endhour = 12
# NLatent = 2 #Number of latent states
# patient = 'Patient9'
# starthour = 17 -8 #Accounting for 7 missing besa files
# endhour = 25 -7
# NLatent = 2 #Number of latent states
# patient = 'Patient13'
# starthour = 11 -7 #Accounting for 6 missing besa files
# endhour = 15 -6
# NLatent = 2 #Number of latent states


# patients = ['Patient14', 'Patient15', 'Patient16']
# starthours = [5-5, 9-9, 11-11]
# endhours = [9-4, 18-8, 20-10]
# NLatent = 2

# patients = ['Patient7', 'Patient11', 'Patient10']
# starthours = [47 -1, 9 -2, 1 -1]
# endhours = [55, 17 - 1, 4]
# NLatent = 2

patients = ['Patient6', 'Patient8']
starthours = [4 -4, 33 -5]
endhours = [8-3, 44 -4]
NLatent = 4

qHFO = True #Flag for usage of HFO artifact

for p in range(0,len(patients)):

  patient = patients[p]
  starthour = starthours[p]
  endhour = endhours[p]

  csvloc = '/data/localization/%s_localization.csv' % (patient,patient)
  localization = np.genfromtxt(csvloc,delimiter=',',skip_header=1,dtype='int')
  grey_elecs = localization[np.array(localization[:,9],dtype='bool'),0]
  soz_elecs = localization[np.array(localization[:,10],dtype='bool'),0]
  fpz_elecs = localization[np.array(localization[:,11],dtype='bool'),0]
  inbrain_elecs = localization[np.logical_or((np.array(localization[:,12],dtype='bool')),(np.array(localization[:,9],dtype='bool'))),0] #White or grey matter
  all_elecs = localization[:,0]
  irr_elecs = np.intersect1d(np.union1d((soz_elecs),(fpz_elecs)),(grey_elecs))

  if usingchans is 'inbrain':
    used_elecs = inbrain_elecs
  elif usingchans is 'all':
    used_elecs = all_elecs
  else:
    usingchans = 'grey'
    used_elecs = grey_elecs


  # Load data
  if qHFO:
    countsdic = sio.loadmat('/data/hfos/%s_qHFOcounts_total.mat' % (patient))
  else:
    countsdic = sio.loadmat('/data/hfos/%s_HFOcounts_total.mat' % (patient))

  timestepsize = 1 #in seconds
  windowsize = 300 #in seconds (5 minutes)

  count = countsdic['count'][(starthour*3600):(endhour*3600),used_elecs-1] #Note the subtraction of 1 from the index is necessary for Python
  greychans = countsdic['whichchans'][0,grey_elecs-1]
  sozchans = countsdic['whichchans'][0,soz_elecs-1]
  irrchans = countsdic['whichchans'][0,irr_elecs-1]
  usedchans = countsdic['whichchans'][0,used_elecs-1]
  T = count.shape[0]
  W = np.int32(T/windowsize) #State window in minutes
  E = count.shape[1] #Number of electrodes


  wind = np.empty((1,T),dtype='int32')
  for w in range(1,W+1):
    wind[0,((w-1)*windowsize):(w*windowsize)] = np.ones((1,windowsize))*w

  # JAGS code


  thismodel = '''
  model {
    pi[1, 1:NLatent] ~ ddirich(rep(1,NLatent))
    stateindx[1, 1] <- 1 # Constraint such that all chains will match state labels if converged
    for (w in 2:W) {
      stateindx[1, w] ~ dcat(pi[1, 1:NLatent])
    }
    for (n in 1:NLatent) {
      state_lambda[1,n] ~ dnorm(1, pow(.5, -2))
      state_std[1,n] ~ dgamma(1,1)
    }
    for (e in 1:E) {
       for (n in 1:NLatent) {
          latent_lambda[n, e] ~ dnorm(state_lambda[1,n], pow(state_std[1,n], -2)) T(0, )
          r[n, e] ~ dunif(0, 50)
          success_prob[n, e] <- r[n, e] / (r[n, e] + latent_lambda[n, e])
          coef_variation[n, e] <- 1 / (.001 + sqrt(r[n,e]*(1-success_prob[n, e]))) # Placed constant .001 in equation to avoid numerical errors when r==0
          clumping[n, e] <- 1 / (.001 + r[n,e]) # Placed constant .001 in equation to avoid numerical errors when r==0
       }
       for (t in 1:T) {
          count[t,e] ~ dnegbin(success_prob[ stateindx[1, wind[1,t]], e ], r[ stateindx[1, wind[1,t]], e ])
       }
    }


  } # End of model
  '''

  # pyjags code

  # Make sure $LD_LIBRARY_PATH sees /usr/local/lib
  pyjags.modules.load_module('dic')
  pyjags.modules.list_modules()

  nchains = 6
  threads = 3
  chains_per_thread = np.int64(np.ceil(nchains/threads))
  burnin = 200  # Note that scientific notation breaks pyjags
  nsamps = 5000

  trackvars = ['latent_lambda', 'clumping', 'coef_variation', 'pi', 'stateindx', 'state_lambda', 'state_std']


  initials = []
  for c in range(0, nchains):
      chaininit = {
          'state_lambda': np.random.uniform(.1, 2, size=(1, NLatent)),
          'state_std': np.random.uniform(.01, 1., size=(1, NLatent)),
          'latent_lambda': np.random.uniform(.1, 2, size=(NLatent, E)),
          'pi': np.reshape(np.random.dirichlet(np.ones(NLatent)),(1,NLatent)),
          'r': np.random.uniform(.1, 50, size=(NLatent, E)),
      }
      initials.append(chaininit)


  # Run JAGS model

  # Choose JAGS model type
  saveloc = '/data/jagsout/'
  if qHFO:
    modelname = 'model12samples_%s_%s_qHFO_%dLatent' % (patient,usingchans,NLatent)
  else:
    modelname = 'model12samples_%s_%s_%dLatent' % (patient,usingchans,NLatent)

  # Save model
  timestart = strftime('%b') + '_' + strftime('%d') + '_' + \
      strftime('%y') + '_' + strftime('%H') + '_' + strftime('%M')
  modelfile = saveloc + 'jagsmodel_' + modelname + timestart + '.jags'
  f = open(modelfile, 'w')
  f.write(thismodel)
  f.close()
  print 'Fitting model %s ...' % (modelname + timestart)


  indata = dict(E=E, T=T, W=W, NLatent=NLatent, count=count, wind=wind)

  threaded = pyjags.Model(file=modelfile, init=initials,
                          data=indata,
                          chains=nchains, adapt=burnin, threads=threads,
                          chains_per_thread=chains_per_thread, progress_bar=True)


  samples = threaded.sample(nsamps, vars=trackvars, thin=10)
  samples['greychans'] = greychans
  samples['sozchans'] = sozchans
  samples['irrchans'] = irrchans
  samples['usedchans'] = usedchans
  samples['usingchans'] = usingchans
  samples['starthour'] = starthour
  samples['endhour'] = endhour

  savestring = saveloc + "jagsmodel_" + \
      modelname + timestart + ".mat"

  print 'Saving results to: \n %s' % savestring

  sio.savemat(savestring, samples)
