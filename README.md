# PUI2015_agungel

# coding: utf-8

# In[7]:

import pylab as pl
import pandas as pd
import numpy as np
get_ipython().magic(u'pylab inline')

import os
import json


# In[10]:

#February data 
df=pd.read_csv('201502-citibike-tripdata.csv')
print df.columns


# In[11]:

df.head()


# In[12]:

print df.describe()


# In[13]:

df['ageM'] = 2015-df['birth year'][(df['usertype'] == 'Subscriber') & (df['gender'] == 1)]
df['ageF'] = 2015-df['birth year'][(df['usertype'] == 'Subscriber') & (df['gender'] == 2)]


# In[14]:

bins = np.arange(10, 99, 10)
df.ageM.groupby(pd.cut(df.ageM, bins)).agg([count_nonzero]).plot(kind='bar')
df.ageF.groupby(pd.cut(df.ageF, bins)).agg([count_nonzero]).plot(kind='bar')


# In[15]:

#create a cumulative distribution
csM=df.ageM.groupby(pd.cut(df.ageM, bins)).agg([count_nonzero]).cumsum()

csF=df.ageF.groupby(pd.cut(df.ageF, bins)).agg([count_nonzero]).cumsum()

print np.abs(csM / csM.max()-csF / csF.max())

pl.plot(bins[:-1] + 5, csM / csM.max(), label = "M")
pl.plot(bins[:-1] + 5, csF / csF.max(), label = "F")
pl.legend()


# In[18]:

#KS Test
import scipy.stats


# In[19]:

ks=scipy.stats.ks_2samp(df.ageM, df.ageF)


# In[20]:

print ks


# In[ ]:

#The result shows that p value is equal to zero which means that they do not have the same distribution.
#The test result is 0,652500... which is smaller than 0,05. The null hypothesis must be rejected. 


# In[21]:

#Pearson test
import scipy.stats


# In[22]:

scipy.stats.pearsonr(df.ageM, df.ageF)


# In[ ]:

#Pearson test measures the degree of the relationship between linear variables.Here,the Pearson test value is nan.
#The value of the coefficient ratio r should be between -1 and 1.So the value of the covariance is irrelevant.
#In the calculation process, pearson uses in the ratio part multiplicatio of the two std of the Male and Female age. One of them may be "0" for getting the result nan.


# In[ ]:

#Spearman test


# In[23]:

scipy.stats.spearmanr(df.ageM, df.ageF, axis=0)


# In[ ]:

#When the variables are not normally distributed or the relationship between the variables is not linear, 
#it is more appropriate to use the Spearman rank correlation method.
#Here,t is the power of the spearman's test and value : -0,41368...shows that these two distribution are negatively correlated and when one group age increases other group age decreases as coefficient value: -0,4136... 
#P value,which is zero,indicates that the two variables do not vary together at all.


# In[24]:

import pylab as pl
import pandas as pd
import numpy as np
get_ipython().magic(u'pylab inline')
from datetime import datetime

import os

#dataframe for starttime,hour
df['starttime'] = pd.to_datetime(df['starttime'])
df['hour'] = df['starttime'].dt.hour


# In[28]:

#day and night riders age distribution. Night riders are defined between 7pm-8am. Day riders are defined between 8am-7pm.
df['ageN'] = 2015-df['birth year'][(df['usertype'] == 'Subscriber') & (df['hour'] > 19) | (df['hour'] < 8 )]
df['ageD'] = 2015-df['birth year'][(df['usertype'] == 'Subscriber') & ((df['hour'] <= 19) & (df['hour'] >= 8 ))]


# In[27]:

bins = np.arange(10, 99, 10)
df.ageN.groupby(pd.cut(df.ageN, bins)).agg([count_nonzero]).plot(kind='bar')
df.ageD.groupby(pd.cut(df.ageD, bins)).agg([count_nonzero]).plot(kind='bar')


# In[29]:

#cumulative distribution

csN=df.ageN.groupby(pd.cut(df.ageN, bins)).agg([count_nonzero]).cumsum()

csD=df.ageD.groupby(pd.cut(df.ageD, bins)).agg([count_nonzero]).cumsum()

print np.abs(csD / csD.max()-csN / csN.max())

pl.plot(bins[:-1] + 5, csN / csN.max(), label = "N")
pl.plot(bins[:-1] + 5, csD / csD.max(), label = "D")
pl.legend()


# In[ ]:

#KS test


# In[31]:

ks=scipy.stats.ks_2samp(df.ageN, df.ageD)


# In[32]:

print ks


# In[30]:

#P-value is equal to 0 and it means that two sample do not come from the same parent distribution.
#Our test result is 0.58646.... and this value smaller than the table value c(alpha) at 0.05 alpha value 1,36. As a result we must reject the null hypothesis


# In[ ]:

#Pearson test


# In[34]:

ageD_arr = df.ageD
ageN_arr = df.ageN
ageD_arr = ageD_arr[~numpy.isnan(ageD_arr)]
ageN_arr = ageN_arr[~numpy.isnan(ageN_arr)]
ageD_arr = np.random.choice(ageD_arr,20000, replace=False)
ageN_arr = np.random.choice(ageN_arr,20000, replace=False)

print ageD_arr
print ageN_arr


# In[35]:

scipy.stats.pearsonr (ageD_arr, ageN_arr)


# In[ ]:

#The correlation coefficient is 0,0025 (around 0) , which means that these two distributions are not correlated.


# In[36]:

scipy.stats.spearmanr(ageD_arr, ageN_arr, axis=0)

![Alt text](agungel_bash.png)
![Alt text](setup_env.png)

--
# coding: utf-8

# In[1]:

import pylab as pl
import pandas as pd
import numpy as np
get_ipython().magic(u'pylab inline')

import os

import scipy.stats


# In[2]:

dist_n = np.random.randn(500)

#anderson and ks test
print "normal on normal", scipy.stats.kstest(dist_n,'norm')
print "normal on normal", scipy.stats.anderson(dist_n, dist='norm')
print "" 

dist_p = np.random.poisson(1, 500)

print "poisson on normal", scipy.stats.kstest(dist_p,'norm')
print "poisson on normal", scipy.stats.anderson(dist_p, dist='norm')

threshold = scipy.stats.anderson(dist_n, dist='norm')[1][scipy.stats.anderson(dist_n, dist='norm')[2]==[1.0]]
print threshold


# In[3]:

distpdf_n, mybins_n, = np.histogram(dist_n, density=True)
distpdf_p, mybins_p, = np.histogram(dist_p, density=True)
#notice the extra comma on the left side of the '=' sign: that tells numpy take the first two values returned, and throw away the rest


# In[4]:

bincenters_n = mybins_n[:-1] + 0.5*(mybins_n[1] - mybins_n[0])
bincenters_p = mybins_p[:-1] + 0.5*(mybins_p[1] - mybins_p[0])
print "normal on normal", scipy.stats.entropy(distpdf_n, scipy.stats.norm.pdf(bincenters_n))  
print "poisson on normal", scipy.stats.entropy(distpdf_p, scipy.stats.norm.pdf(bincenters_p)) 


# In[ ]:

#Poisson normal has a high value,null hypothesis must be rejected.


# In[5]:

#Binomial distribution 
narray = range(1,60,1)
ks_b = np.zeros(len(narray))
ad_b = np.zeros(len(narray))
kl_b = np.zeros(len(narray))
chi2_b = np.zeros(len(narray))


# In[6]:

def mynorm (x, mu, var):
    return scipy.stats.norm.cdf(x, loc=mu, scale=var)


# In[7]:

p=0.5
for i,n in enumerate(narray):
    p=0.1 #parameter for the binomial, my arbitrary choice
    #generate the distribution
    dist = np.random.binomial(n, p, 500)
    #run the tests. 
 
    ks_b[i] = scipy.stats.kstest(dist, mynorm, args=(n*p, n*p*(1.0-p)))[0]
    ad_b[i] = scipy.stats.anderson(dist, dist='norm')[0]
    mybins=np.linspace(min(dist),max(dist), 5) 
    bincenters = mybins[:-1]+0.5*(mybins[1]-mybins[0])
    kl_b [i] =  scipy.stats.entropy(np.histogram(dist, bins=mybins)[0], scipy.stats.norm.pdf(bincenters, loc=n*p, scale=n*p*(1.0-p)))


# In[ ]:

#Try different plot types


# In[8]:

fig = pl.figure(figsize = (18,5))
fig.add_subplot(131)
pl.plot(narray, ks_b, '--r', label='KS TEst')
plt.title('KS Test')
pl.legend()

fig.add_subplot(132)
pl.plot(narray, ad_b, 'bo', label='AD')
pl.plot([narray[0], narray[-1]],[threshold, threshold])
plt.title('AD Test')
pl.plot()
pl.legend()

fig.add_subplot(133)
pl.plot(narray, kl_b, label='K-L ')
plt.title('K-L Test')
pl.legend()


# In[9]:

pl.figure(figsize=(15,15))

p = lambda x, mu : scipy.stats.distributions.poisson.pmf(x,mu)
q = lambda x, mu : scipy.stats.distributions.norm.pdf(x-mu)
c = lambda x, mu : scipy.stats.distributions.chi2.pdf(x, df=mu)
f = lambda x, mu : scipy.stats.distributions.f.pdf(x,50,50)

Dkl = lambda x ,mu, p :  p(x, mu)* np.log10(q(x, mu)) + p(x, mu) *np.log10(p(x, mu))

def model(x) : 
    return 1./(s*np.sqrt(2*np.pi))*exp(-((x-m)/2./2./s)**2)

x=np.linspace(-20,20,100)
pl.plot(x, q(x,1), label='norm')
pl.plot(x, p(x,1), label='poisson')
pl.plot(x, Dkl(x, 1, p), '-.', label = 'KL - poisson')

#print np.nansum(Dkl(np.linspace(0,30,1000), 1, p))


pl.plot(x, c(x,1), label='f')
pl.plot(x,Dkl(x, 1, c), '--bo', label = 'KL - chi2')

---
# coding: utf-8

# In[1]:

import pylab as pl
import pandas as pd
import numpy as np
get_ipython().magic(u'pylab inline')

import os
import scipy.stats


# In[5]:

#winter and summer riders
dfW=pd.read_csv('201502-citibike-tripdata.csv')
dfS=pd.read_csv('201506-citibike-tripdata.csv')


# In[ ]:

print dfW.columns, dfS.columns


# In[ ]:

#Winter and Summer age dataframe
dfW['age'] = 2015-dfW['birth year'][(dfW['usertype'] == 'Subscriber')]
dfS['age'] = 2015-dfS['birth year'][(dfS['usertype'] == 'Subscriber')]


# In[ ]:

bins = np.arange(10, 99, 10)
dfW.age.groupby(pd.cut(dfW.age, bins)).agg([count_nonzero]).plot(kind='bar', title="Winter")
W_age_dist = dfW.age.groupby(pd.cut(dfW.age, bins)).agg([count_nonzero])
dfS.age.groupby(pd.cut(dfS.age, bins)).agg([count_nonzero]).plot(kind='bar', title="Summer")
S_age_dist = dfS.age.groupby(pd.cut(dfS.age, bins)).agg([count_nonzero])


# In[ ]:

# Ks test normal distribution

ksW=scipy.stats.kstest(W_age_dist, 'norm')
ksS=scipy.stats.kstest(S_age_dist, 'norm')
print "winter, normal fit", ksW
print "summer, normal fit", ksS

# poisson distribution 

ksW=scipy.stats.kstest(W_age_dist, 'cauchy')
ksS=scipy.stats.kstest(S_age_dist, 'cauchy')
print "winter, normal fit", ksW
print "summer, normal fit", ksS


# In[ ]:

#Anderson Test
adW=scipy.stats.anderson(W_age_dist, 'gumbel')
adS=scipy.stats.anderson(S_age_dist, 'gumbel')
print "winter, gumbel fit", adW
print "summer, gumbel fit", adS

# normal distribution

adW=scipy.stats.anderson(W_age_dist, 'extreme1')
adS=scipy.stats.anderson(S_age_dist, 'extreme1')
print "winter, extrem1 fit", adW
print "summer, extreme1 fit", adS


# In[ ]:

import pandas as pd
import numpy as np
import urllib2
import json

import statsmodels.formula.api as smf
import matplotlib.pyplot as plt



