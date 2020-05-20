#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np 
from sklearn import cluster 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from patch import getpatchedprogress
progress = getpatchedprogress()
from progress.bar import IncrementalBar


# In[7]:


#Creates N brownian particle trajectories with diffusion D sampled every dt seconds for t seconds.
def trajectory(t,N,D,dt):
    rmatrix = np.random.normal(0,np.sqrt(2*D*dt),(N,int(t/dt)))
    xn = [np.cumsum(rands) for rands in rmatrix]
    return xn


# In[8]:


#Uses MSD to determine Diffusivity.
def Diffusion(x1,y1,dt):
    x1 = np.array(x1)
    y1 = np.array(y1)
    D_n = np.mean(np.diff(np.sqrt(x1**2 + y1**2))**2, axis = 1)/(2*dt)
    return list(D_n)


# In[9]:


# Find Clusters using Diffusivity and DBSCAN. 
def DB_clusters(D_n):
    D_n = D_n.reshape(-1,1)
    I = cluster.DBSCAN().fit_predict(D_n)
    N = len(set(I))
    diffusion = []
    diff = [[] for _ in range(N)]
        
    for i,n in enumerate(I):
        diff[n].append(D_n[i])

    diffusion = [np.mean(diff[i]) for i in range(N)]

    return diffusion, N


# In[10]:


# Find Clusters using Diffusivity and K-means.
def K_clusters(D_n,K):
    D_n = D_n.reshape(-1,1)
    I = cluster.KMeans(2).fit_predict(D_n)
    diffusion = []
    
    diff = [[] for _ in range(K)]
        
    for i,n in enumerate(I):
        diff[n].append(D_n[i])
        
    diffusion = [np.mean(diff[i]) for i in range(K)]

    return diffusion, K


# In[ ]:


def main(t,dt,N1,D_1,D_2,k):
    phi = []
    Dout = []
    Kout = []
    bar = IncrementalBar('simulation progress', max = len(N1))
    
    for n1 in N1:
        bar.next()
        n2 = 1001 - n1
        phi.append([n1/(n1+n2) for _ in range(len(D_2))])
        Dpass_fail = []
        Kpass_fail = []
        for d2 in D_2:
        
            x1 = trajectory(t,n1,D_1,dt)
            y1 = trajectory(t,n1,D_1,dt)

            x2 = trajectory(t,n2,d2,dt)
            y2 = trajectory(t,n2,d2,dt)

            Dest1 = Diffusion(x1,y1,dt)
            Dest2 = Diffusion(x2,y2,dt)
        
            D_n = np.array(Dest1 + Dest2)
            np.random.shuffle(D_n)

            [kdiffusion,num] = K_clusters(D_n,k)
            [diffusion,num] = DB_clusters(D_n)
        
            diffusion.sort()
            kdiffusion.sort()
        
            D = [D_1,d2]
            D.sort()

        
            if len(diffusion) == k:
                if (abs(diffusion[0] - D[0]) < .1) and (abs(diffusion[1] - D[1]) < .1):
                    Dpass_fail.append(1)
                else:
                    Dpass_fail.append(0)
            else:
                Dpass_fail.append(0)
            
            
            if (abs(kdiffusion[0] - D[0]) < .1) and (abs(kdiffusion[1] - D[1]) < .1):
                Kpass_fail.append(1)
            else:
                Kpass_fail.append(0)
        
        pass_failDcopy = Dpass_fail.copy()
        pass_failKcopy = Kpass_fail.copy()
        Dout.append(pass_failDcopy)
        Kout.append(pass_failKcopy)
    bar.finish()
    return Dout, Kout, phi


# In[ ]:


# Initialize values for diffusion simulation.
print("suggested values for the length of the simulation and the time")
print("step for the particle trajectories are 1 and 0.001.")
t = float(input("Input simulation run time: "))
dt = float(input("Input simulation step size: "))
N1 = np.arange(1,1000,20)
D_1 = 1
D_2 = np.geomspace(0.1,10,50)

k = 2


# In[ ]:


dout,kout,phi = main(t,dt,N1,D_1,D_2,k)


# In[ ]:


dout = np.array(dout)
kout = np.array(kout)
m,n = np.shape(dout)

size = m*n


dout_flat = dout.reshape(size,1)

kout_flat = kout.reshape(size,1)

dcorr = (np.sum(dout_flat)/size)*100
kcorr = (np.sum(kout_flat)/size)*100

print(f"DBSCAN was {dcorr}% accurate")

print(f"k-means was {kcorr}% accurate")
print(" ")
print("Graphing...")


# In[ ]:


for i in range(len(dout)):
    for j in range(len(dout[i])):
        if dout[i][j] == 0:
            C = "Red"
        else:
            C = "Blue"
        plt.scatter((D_1/D_2)[j],phi[i][j], color = C);
        plt.xscale("log");
plt.title("DBSCAN error");
plt.xlabel("D1/D2");
plt.ylabel("phi");
plt.show();
plt.savefig("DBSCAN_error.jpg")


# In[ ]:


for i in range(len(kout)):
    for j in range(len(kout[i])):
        if kout[i][j] == 0:
            C = "Red"
        else:
            C = "Blue"
        plt.scatter((D_1/D_2)[j],phi[i][j], color = C);
        plt.xscale("log");
plt.title("K-means error.");
plt.xlabel("D1/D2");
plt.ylabel("phi");
plt.show();
plt.savefig("K-means_error.jpg")

