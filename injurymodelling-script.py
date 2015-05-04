import pandas as pd
import numpy as np
import pystan

torndata = pd.read_csv('AM207-Twister/1950-2014_torn.warnings.csv', header=None)
torndata.columns = ['om', 'yr', 'mo', 'dy', 'date', 'time', 'tz', 'st', 'stf', 'stn', 'f', 'inj', 'fat', 'loss', 'closs', 'slat', 'slon', 'elat', 'elon', 'len', 'wid', 'ns', 'sn', 'sg', 'f1', 'f2', 'f3', 'f4', 'touchdown', 'has_warning', 'has_watch', 'warning_time']

torndatasel = torndata[(torndata['sn'] == 1) & (torndata['yr'] >= 1986) & (torndata['f'] == 3)]

landdata = pd.read_csv('DEC_10_DP_G001_with_ann.csv', skiprows=1, header=0, index_col='Id2')
landdata = landdata['AREA CHARACTERISTICS - Area (Land)']

popdata = pd.read_csv('PEP_2014_PEPANNRES_with_ann.csv', skiprows=1, header=0, index_col='Id2')
popdata = popdata['April 1, 2010 - Census']


torndatasel['pop_dens'] = 0 # need better way to make column?

for i in torndatasel.index:
    population = 0.
    area = 0.
    density = 0.
    row = torndatasel.loc[i]
    for column in ['f1', 'f2', 'f3', 'f4']:
        county = row[column]
        if county != 0:
            county += row['stf'] * 1000
            population += popdata[county]
            area += landdata[county]
    torndatasel.loc[i, 'pop_dens'] = population / float(area)#dens
torndatasel.head()

earlywarning = np.logical_and(torndatasel['has_warning'], torndatasel['warning_time'] > 0)

states = torndatasel['st'].unique()
tornsts = np.empty((torndatasel.index.size, states.size), dtype=np.bool)
for i in xrange(states.size):
    tornsts[:, i] = (torndatasel['st'] == states[i])

years_used = range(1986, 2015)
tornyrs = np.empty((torndatasel.index.size, len(years_used)), dtype=np.bool)
for i in xrange(len(years_used)):
    tornyrs[:,i] = torndatasel['yr'] == years_used[i]

tornado_model = """
data {
    int N_torn; // Number of tornados
    int N_yr; // number of time periods
    int N_st; // number of states
    vector[N_torn] logpopdens; // effective population density
    int y[N_torn]; // number of injuries
    vector[N_torn] earlywarning;
    matrix[N_torn, N_yr] tornyrs;
    matrix[N_torn, N_st] tornsts;
}
parameters {
    real const_factor;
    real popdens_power;
    vector[N_torn] noise_factor;
    real<lower=0> noise_hp_sig;
    
    real warning_factor;
    
    vector[N_yr] year_factor;
    real<lower=0> yr_hp_sig;
    
    vector[N_st] state_factor;
    
    real<lower=0> phi;
}
transformed parameters {
    vector[N_torn] lambda;
    lambda <- exp(const_factor + popdens_power * logpopdens + tornsts * state_factor + tornyrs * year_factor +
        warning_factor * earlywarning + noise_factor);
}
model  {
    phi ~ cauchy(0, 1);
    y ~ neg_binomial_2(lambda, phi);
    noise_factor ~ normal(0, noise_hp_sig);
    noise_hp_sig ~ cauchy(0, 1);
    year_factor ~ normal(0, yr_hp_sig);
    yr_hp_sig ~ cauchy(0, 1);
    state_factor ~ cauchy(0, 1);
}
"""

tornado_dat = {'N_torn':len(torndatasel.index),
               'N_yr': len(years_used),
               'N_st':states.size,
               'tornsts':tornsts.astype(int),
               'tornyrs':tornyrs.astype(int),
               'earlywarning':earlywarning.astype(int),
               'logpopdens':np.log(torndatasel['pop_dens'].values),
               'y':torndatasel['inj'].values}

niter = 1000
fit = pystan.stan(model_code=tornado_model, data=tornado_dat, iter=2*niter, chains=1)

extract = fit.extract(permuted=True)

import pickle
f = open('injurymodelling.pkl', 'w')
pickle.dump(states.astype(list), f)
pickle.dump(extract, f)
f.close()
