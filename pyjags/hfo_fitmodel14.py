# hfo_fitmodel14.py - Script fits Model 14 using JAGS with real HFO count data
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
# 03/17/21       Michael Nunez              Converted from hfo_fitmodel13.py
# 03/18/21       Michael Nunez                    Remove try loop
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
from hfo_fitmodel14results import run_diagnostic, save_figures


usingchans = 'grey'

# patients = ['Patient15', 'Patient16']
# starthours = [9-9, 11-11]
# endhours = [18-8, 20-10]

# patients = ['Patient7', 'Patient11', 'Patient10']
# starthours = [47 -1, 9 -2, 1 -1]
# endhours = [55, 17 - 1, 4]

# patients = ['Patient6', 'Patient8']
# starthours = [4 -4, 33 -5]
# endhours = [8-3, 44 -4]
# NLatent = 4

# patients = ['Patient8', 'Patient7']
# starthours = [33 -5, 47 -1]
# endhours = [44 -4, 55]

# patients = ['Patient7']
# starthours = [47 -1]
# endhours = [55]

# patients = ['Patient8']
# starthours = [33 -5]
# endhours = [44 -4]

# patients = ['Patient12']
# starthours = [2 -2] #Accounting for 1 missing besa file
# endhours = [8 -1]

# patients = ['Patient6']
# starthours = [4 -4]
# endhours = [8 - 3]

# patients = ['Patient1']
# starthours = [7 - 1]
# endhours = [19]

patients = ['Patient16']
starthours = [11 -11] #Accounting for 10 missing besa files
endhours = [20 -10]

# patients = ['Patient5']
# starthours = [1 -1]
# endhours = [12]

# patients = ['Patient4', 'Patient9']
# starthours = [1 - 1, 17-8] #Account for 16 missing BESA files correction
# endhours = [16, 25-7] #Account for 16 missing BESA files correction;

# patients = [ 'Patient10']
# starthours = [1 -1]
# endhours = [4]

# patients = ['Patient3', 'Patient14']
# starthours = [46 - 46, 5 - 5] #Account for 45 missing BESA files correction
# endhours = [60 - 45, 9 - 4] #Account for 45 missing BESA files correction;

# patients = ['Patient2', 'Patient13']
# starthours = [25 - 1, 11 - 7]
# endhours = [49, 15 - 6]

# patients = ['Patient15']
# starthours = [9-9]
# endhours = [18-8]

# patients = ['Patient13']
# starthours = [11 - 7]
# endhours = [15 - 6]

# patients = ['Patient14']
# starthours = [5 - 5] #Account for 4 missing BESA files correction
# endhours = [9 - 4] #Account for 4 missing BESA files correction;

# patients = ['Patient2']
# starthours = [25 - 1]
# endhours = [49]

# patients = ['Patient11']
# starthours = [9 -2] #Accounting for 1 missing besa file
# endhours = [17 - 1]

# patients = ['Patient10']
# starthours = [1 -1]
# endhours = [4]

# patients = ['Patient14']
# starthours = [6 - 5] #Account for 4 missing BESA files correction
# endhours = [9 - 4] #Account for 4 missing BESA files correction;

# patients = ['Patient3']
# starthours = [46 - 46] #Account for 45 missing BESA files correction
# endhours = [60 - 45] #Account for 45 missing BESA files correction;

# patients = ['Patient4']
# starthours = [1 - 1] #Account for 16 missing BESA files correction
# endhours = [16] #Account for 16 missing BESA files correction;

# patients = ['Patient9']
# starthours = [17-8] #Account for 16 missing BESA files correction
# endhours = [25-7] #Account for 16 missing BESA files correction;


qHFO = True #Flag for usage of HFO artifact

for p in range(0,len(patients)):

  patient = patients[p]
  starthour = starthours[p]
  endhour = endhours[p]

  csvloc = '/data/localization/%s_localization.csv' % (patient)
  localization = np.genfromtxt(csvloc,delimiter=',',skip_header=1,dtype='int')
  grey_elecs = localization[np.array(localization[:,9],dtype='bool'),0]
  soz_elecs = localization[np.array(localization[:,10],dtype='bool'),0]
  fpz_elecs = localization[np.array(localization[:,11],dtype='bool'),0]
  inbrain_elecs = localization[np.logical_or((np.array(localization[:,12],dtype='bool')),(np.array(localization[:,9],dtype='bool'))),0] #White or grey matter
  all_elecs = localization[:,0]
  irr_elecs = np.intersect1d(np.union1d((soz_elecs),(fpz_elecs)),(grey_elecs))

  if usingchans=='inbrain':
    used_elecs = inbrain_elecs
  elif usingchans=='all':
    used_elecs = all_elecs
  else:
    usingchans=='grey'
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


  # thismodel = '''
  # model {
  #   for (e in 1:E) {
  #     mean_lambda[1,e] ~ dnorm(1, pow(.5, -2))
  #     std_lambda[1,e] ~ dgamma(1,1)
  #     mean_clumping[1,e] ~ dnorm(10, pow(5, -2))
  #     std_clumping[1,e] ~ dgamma(1,1)
  #      for (w in 1:W) {
  #         lambda[w, e] ~ dnorm(mean_lambda[1,e], pow(std_lambda[1,e], -2)) T(0, )
  #         clumping[w, e] ~ dnorm(mean_clumping[1,e], pow(std_clumping[1,e], -2)) T(0, )
  #         success_prob[w, e] <- r[w, e] / (r[w, e] + lambda[w, e])
  #         coef_variation[w, e] <- 1 / (.001 + sqrt(r[w,e]*(1-success_prob[w, e]))) # Placed constant .001 in equation to avoid numerical errors when r==0
  #         r[w,e] <- 1 / (.001 + clumping[w, e]) # Placed constant .001 in equation to avoid numerical errors when clumping==0
  #      }
  #      for (t in 1:T) {
  #         count[t,e] ~ dnegbin(success_prob[ wind[1,t], e ], r[ wind[1,t], e ])
  #      }
  #   }


  # } # End of model
  # '''

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
  if qHFO:
    modelname = 'model14samples_%s_%s_qHFO' % (patient,usingchans)
  else:
    modelname = 'model14samples_%s_%s' % (patient,usingchans)

  # Save model
  timestart = strftime('%b') + '_' + strftime('%d') + '_' + \
      strftime('%y') + '_' + strftime('%H') + '_' + strftime('%M')
  modelfile = saveloc + 'jagsmodel_' + modelname + timestart + '.jags'
  f = open(modelfile, 'w')
  f.write(thismodel)
  f.close()
  print('Fitting model %s ...' % (modelname + timestart))


  indata = dict(E=E, T=T, W=W, count=count, wind=wind)

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

  print('Saving results to: \n %s' % savestring)
  sio.savemat(savestring, samples)
  diags = run_diagnostic(samples)
  samples = sio.loadmat(savestring)
  save_figures(samples, 'jagsmodel_' + modelname + timestart)

