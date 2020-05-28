# hfo_fitmodel8.py - Script fits Model 8 using JAGS with real HFO count data
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
# 11/15/18       Michael Nunez              Converted from hfo_fitmodel5.py
# 05/19/19       Michael Nunez        All channels flag to see the effect of noisy channels on prediction
# 05/28/20      Michael Nunez             Remove patient identifiers

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
import pandas

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


csvloc = '/data/localization/%s_localization.csv' % (patient,patient)
df = pandas.read_csv(csvloc)
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
# df['Channel_Label'][grey_elecs]
# df['Channel_Label'][soz_elecs]

# Load data
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
  for (w in 1:W) {
    stateindx[1, w] ~ dcat(pi[1, 1:NLatent])
  }
  for (n in 1:NLatent) {
    state_lambda[1,n] ~ dnorm(1, pow(.5, -2))
    state_std[1,n] ~ dgamma(1,1)
  }
  for (e in 1:E) {
     for (n in 1:NLatent) {
        latent_lambda[n, e] ~ dnorm(state_lambda[1,n], pow(state_std[1,n], -2)) T(0, )
     }
     for (t in 1:T) {
        count[t,e] ~ dpois(latent_lambda[ stateindx[1, wind[1,t]], e ])
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

trackvars = ['latent_lambda', 'pi', 'stateindx', 'state_lambda', 'state_std']


initials = []
for c in range(0, nchains):
    chaininit = {
        'state_lambda': np.random.uniform(.1, 2, size=(1, NLatent)),
        'state_std': np.random.uniform(.01, 1., size=(1, NLatent)),
        'latent_lambda': np.random.uniform(.1, 2, size=(NLatent, E)),
        'pi': np.reshape(np.random.dirichlet(np.ones(NLatent)),(1,NLatent)),
    }
    initials.append(chaininit)


# Run JAGS model

# Choose JAGS model type
saveloc = '/data/jagsout/';
modelname = 'model8samples_%s_%s_%dLatent' % (patient,usingchans,NLatent)

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