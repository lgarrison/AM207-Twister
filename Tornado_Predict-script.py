
# coding: utf-8

# In[9]:

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pystan
import pickle
import vincent
from IPython.display import display


# In[10]:

pd.set_option('display.max_columns', 50)


# In[11]:

import us
state_map = us.states.mapping('abbr', 'name')


# In[12]:

names = ('Tornado Number','Year','Month','Day','Date','Time','Time Zone','State','State ID','State Number',         'Scale','Injuries','Fatalities','Property Loss','Crop Loss','Starting Lat','Starting Lon',         'Ending Lat','Ending Lon','Length','Width','Number of States','State Flag','Segment Number',         '1st County','2nd County','3rd County','4th County','Wind')


# In[13]:

data = pd.read_csv('1950-2014_torn.csv',names=names)
data = data.query('Year >= 1985')


# In[14]:

states = data['State'].unique()
years = data['Year'].unique()
state_IDs = data['State ID'].unique()
state_dict = {}
for i,state in enumerate(states):
    state_dict[states[i]]=i
state_ID_dict = {}
for i,ID in enumerate(state_IDs):
    state_ID_dict[ID]=i
year_dict = {}
j=0
for i,year in enumerate(years):
    year_dict[year]=j
    if i%3==2:
        j+=1
    
counties = np.array(list(set([str(x).zfill(2)+str(y).zfill(3) for x,y in zip(data['State ID'].values,data['1st County'].values) if y!=0]+    [str(x).zfill(2)+str(y).zfill(3) for x,y in zip(data['State ID'].values,data['2nd County'].values) if y!=0]+    [str(x).zfill(2)+str(y).zfill(3) for x,y in zip(data['State ID'].values,data['3rd County'].values) if y!=0]+    [str(x).zfill(2)+str(y).zfill(3) for x,y in zip(data['State ID'].values,data['4th County'].values) if y!=0])))

county_dict = {}
for i,k in enumerate(counties):
    county_dict[k]=i

t_data_full = np.zeros((len(counties),len(years)),dtype=int)


# In[15]:

# t_data_full = np.zeros((len(counties),len(years),6),dtype=int)
# for i in data.index:
#     row = data.ix[i]
#     scale = row['Scale']
#     if row['State Flag'] ==1 and scale>=0:
#         if row['1st County']!=0:
#             t_data_full[county_dict[str(row['State ID']).zfill(2)+str(row['1st County']).zfill(3)]][year_dict[row['Year']]][scale]+=1
#         if row['2nd County']!=0:
#             t_data_full[county_dict[str(row['State ID']).zfill(2)+str(row['2nd County']).zfill(3)]][year_dict[row['Year']]][scale]+=1
#         if row['3rd County']!=0:
#             t_data_full[county_dict[str(row['State ID']).zfill(2)+str(row['3rd County']).zfill(3)]][year_dict[row['Year']]][scale]+=1  
#         if row['4th County']!=0:
#             t_data_full[county_dict[str(row['State ID']).zfill(2)+str(row['4th County']).zfill(3)]][year_dict[row['Year']]][scale]+=1


# In[16]:

t_data_full = np.zeros((len(counties),len(np.unique(year_dict.values())),3),dtype=int)
for i in data.index:
    row = data.ix[i]
    scale = row['Scale']
    if row['State Flag'] ==1 and scale>=0:
        if row['1st County']!=0:
            if scale ==0 or scale==1:
                t_data_full[county_dict[str(row['State ID']).zfill(2)+str(row['1st County']).zfill(3)]][year_dict[row['Year']]][0]+=1
            if scale ==2 or scale==3:
                t_data_full[county_dict[str(row['State ID']).zfill(2)+str(row['1st County']).zfill(3)]][year_dict[row['Year']]][1]+=1
            if scale ==4 or scale==5 or scale >5:
                t_data_full[county_dict[str(row['State ID']).zfill(2)+str(row['1st County']).zfill(3)]][year_dict[row['Year']]][2]+=1
#         if row['2nd County']!=0:
#             t_data_full[county_dict[str(row['State ID']).zfill(2)+str(row['2nd County']).zfill(3)]][year_dict[row['Year']]][scale]+=1
#         if row['3rd County']!=0:
#             t_data_full[county_dict[str(row['State ID']).zfill(2)+str(row['3rd County']).zfill(3)]][year_dict[row['Year']]][scale]+=1  
#         if row['4th County']!=0:
#             t_data_full[county_dict[str(row['State ID']).zfill(2)+str(row['4th County']).zfill(3)]][year_dict[row['Year']]][scale]+=1


# In[17]:

county_land_mass_df = pd.read_csv('County_land_mass.csv')
states_used = ['MT','WY','CO','ND','SD','NE','KS','OK','MN','IA','MO','AR']
#states_used=[st for st in states if st not in ['DC']]

county_state_list = []
used_inds = []
for i in range(len(counties)):
    if states[state_ID_dict[int(counties[i][0:2])]] in states_used and int(counties[i]) in county_land_mass_df['STCOU'].values        and (int(counties[i])<51500 or int(counties[i])>=52000):
        used_inds.append(i)
        county_state_list.append(state_ID_dict[int(counties[i][0:2])])
t_data = t_data_full[used_inds]
county_state_list = np.array(county_state_list)

county_state_map = np.zeros_like(county_state_list)
for i,st in enumerate(county_state_list):
    county_state_map[i] = np.where(st==np.unique(county_state_list))[0]
    
county_land_mass = np.zeros(len(used_inds))
for i,ct in enumerate(counties[used_inds]):
    county_land_mass[i] = county_land_mass_df[county_land_mass_df['STCOU'] == int(ct)]['Land_mass'].values[0]


# In[18]:

tornado_code = """
data {
    int<lower=0> N_obs; // Number of observations
    int<lower=0> N_yr; // Number of years
    int<lower=0> N_st; // Number of states
    int<lower=0> N_ct; // Number of counties
    int<lower=0> N_sc; // Number of f-scales
    real land_mass[N_ct]; // Land mass of counties
    int<lower=0> y[N_obs]; // Tornado number
    int<lower=0> county_state[N_ct]; // State in which county is in
}
parameters {
  //  real land_mass_factor;
  
    vector[N_sc*N_ct] county_factor;
    vector[N_sc*N_st] county_mu;
    vector<lower=0>[N_sc*N_st] county_sig;
    
    vector[N_sc*N_yr] year_factor;
    vector<lower=0>[N_sc] year_sig;

    vector[N_obs] noise_factor;
    vector<lower=0>[N_sc] noise_sig;
 
}
transformed parameters {
    //vector[N_obs] lambda;
    vector[N_obs] lambda;
    for (i in 1:N_ct) {
        for (j in 1:N_yr) {
            for (f in 1:N_sc) {
                lambda[(i-1)*N_yr*N_sc + (j-1)*N_sc + f] <- exp(
                year_factor[(j-1)*N_sc + f] + county_factor[(i-1)*N_sc + f] + log(land_mass[i]/100.0)
                + noise_factor[(i-1)*N_yr*N_sc + (j-1)*N_sc + f]) ;
            }
        }
    }
}
model  {
 //   county_mu ~ normal(0,5);
    county_sig ~ cauchy(0,0.05);
    for (i in 1:N_ct) {
        for (f in 1:N_sc) {
            county_factor[(i-1)*N_sc + f] ~ 
            normal(county_mu[(county_state[i]-1)*N_sc + f],county_sig[(county_state[i]-1)*N_sc + f]);
        }
    }
    year_sig ~ cauchy(0,0.1);
    for (j in 1:N_yr) {
        for (f in 1:N_sc) {
            year_factor[(j-1)*N_sc + f] ~ normal(0,year_sig[f]);
        }
    }
    noise_sig ~ cauchy(0,0.1);
    for (i in 1:N_ct) {
        for (j in 1:N_yr) {
            for (f in 1:N_sc) {
                noise_factor[(i-1)*N_yr*N_sc + (j-1)*N_sc + f] ~ normal(0,noise_sig[f]);
            }
        }
    }
    y ~ poisson(lambda);

}
"""


# In[19]:

tornado_dat = {'N_obs':t_data.shape[0]*t_data.shape[1]*t_data.shape[2],
                'N_yr':t_data.shape[1],
               'N_st':len(states_used),
               'N_ct':t_data.shape[0],
               'N_sc':t_data.shape[2],
               'land_mass':county_land_mass,
               'y':t_data.flatten(),
              'county_state':np.array(county_state_map+1,dtype=int)}


# In[ ]:

fit = pystan.stan(model_code=tornado_code, data=tornado_dat,iter=1000,chains=32, n_jobs=32)  #, init='random', init_r=[-1.,1.]


# In[165]:

post_means = fit.get_posterior_mean()


# In[166]:

pickle.dump(post_means, open( "post_means.pickle", "wb" ) )


# In[69]:

la = fit.extract(permuted=True)


# In[88]:

pickle.dump(la, open( "la.pickle", "wb" ) )

