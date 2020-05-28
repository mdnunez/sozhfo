# hfo_deltastatetests.py - Evaluates model-derived brain states' ability to describe delta power
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
# 05/04/20	    Michael Nunez                       Original code for Python 3

##References:
#https://pingouin-stats.org/generated/pingouin.anova.html
#https://pythonfordatascience.org/anova-python/
#https://pingouin-stats.org/generated/pingouin.kruskal.html

# Imports
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import f_oneway
from pingouin import kruskal, anova


npatients = 16


## Load data
allFstat = np.empty((npatients))
ANOVApvals = np.empty((npatients))
allHstat = np.empty((npatients))
KWpvals = np.empty((npatients))
for p in range(npatients):
	print(f'Loading summary delta data for Patient {p+1}')
	summarydelta = sio.loadmat(f'data/Patient{p+1}_summarydelta')
	stackeddelta = np.hstack((np.squeeze(summarydelta['state1delta']),np.squeeze(summarydelta['state2delta'])))
	whichstate = np.ones(stackeddelta.shape[0])*2
	whichstate[0:(np.squeeze(summarydelta['state1delta']).size)] = 1
	delta_df = pd.DataFrame({'standarddelta' : stackeddelta, 'brainstate': whichstate})
	aov = anova(dv='standarddelta',between='brainstate', data=delta_df, detailed=True)
	allFstat[p] = aov['F'][0]
	ANOVApvals[p] = aov['p-unc'][0]
	kw = kruskal(dv='standarddelta',between='brainstate', data=delta_df, detailed=True)
	allHstat[p] = kw['H'][0]
	KWpvals[p] = kw['p-unc'][0]

nANOVAsig001 = np.sum(ANOVApvals < .001)
print(f'There were {nANOVAsig001} significant differences by ANOVA (alpha = .001) of {npatients} patients between mean standardized delta across both brain-derived states')
nANOVAsig01 = np.sum(ANOVApvals < .01)
print(f'There were {nANOVAsig01} significant differences by ANOVA (alpha = .01) of {npatients} patients between mean standardized delta across both brain-derived states')
nANOVAsig05 = np.sum(ANOVApvals < .05)
print(f'There were {nANOVAsig05} significant differences by ANOVA (alpha = .05) of {npatients} patients between mean standardized delta across both brain-derived states')

nKWsig001 = np.sum(KWpvals < .001)
print(f'There were {nKWsig001} significant differences by Kruskal-Wallis (alpha = .001) of {npatients} patients between mean standardized delta across both brain-derived states')
nKWsig01 = np.sum(KWpvals < .01)
print(f'There were {nKWsig01} significant differences by Kruskal-Wallis (alpha = .01) of {npatients} patients between mean standardized delta across both brain-derived states')
nKWsig05 = np.sum(KWpvals < .05)
print(f'There were {nKWsig05} significant differences by Kruskal-Wallis (alpha = .05) of {npatients} patients between mean standardized delta across both brain-derived states')