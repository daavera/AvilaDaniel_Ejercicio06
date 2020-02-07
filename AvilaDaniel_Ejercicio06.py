
# coding: utf-8

# In[119]:


import numpy as np
import matplotlib.pyplot as plt


# In[120]:


x = [4.6, 6.0, 2.0, 5.8] 
sigma = [2.0, 1.5, 5.0, 1.0]


# In[121]:


mu = np.linspace(np.min(x),1.5*np.max(x),1000)
def interval(m):
    return 1/(np.max(m)-np.min(m))*np.ones(len(m))


# In[122]:


def gaussian(x_0, mu_0, sig):
    return (1/np.sqrt(2*np.pi*(sig**2)))*np.exp(-0.5*((x_0-mu_0)/sig)**2)


# In[123]:


def L(mu):
    ele = np.zeros(1000)
    for i in np.arange(len(x)):
        ele += np.log(gaussian(x[i],mu, sigma[i]))
    return ele


# In[124]:


L = L(mu)
dmu = mu[1] - mu[0]
d1_L = (L[:-1]-L[1:])/dmu
d1_zeros = np.where(np.abs(d1_L)==np.min(np.abs(d1_L)))
mu_max = mu[d1_zeros[0][0]]

d2_L = (d1_L[:-1]-d1_L[1:])/dmu
sigma = 1/np.sqrt(-d2_L[d1_zeros][0])


# In[125]:


plt.plot(mu,np.exp(L)/np.trapz(np.exp(L),mu))
plt.title("$\mu$ = " + str('{:2f}'.format(float(mu_max))) + "$\pm$" + str('{:2f}'.format(float(sigma))))
plt.xlabel("$\mu$")
plt.ylabel("posterior")
plt.savefig("mean.png")

