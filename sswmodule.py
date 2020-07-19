import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter

from datetime import datetime,timedelta
from netCDF4 import Dataset,num2date,date2num

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator

import sswmodule as swm

from time import sleep
import gc

## 'global' variables _________________________________##_

plev = np.zeros(37) #length = 37, unit: Pa
zlev = np.zeros(37) #length = 37, unit: m
ylat = np.zeros(121) #length = 121
xlon = np.zeros(240) #length = 240
cos_phi = np.zeros(121) #cosine(lat)
sin_phi = np.zeros(121) #sine(lat)
dS_coef = np.zeros(121)
rho = np.zeros(37)


sswdatetime = [datetime(1981,3,4),datetime(1981,12,4),datetime(1984,2,24),datetime(1985,1,1),datetime(1987,1,23),datetime(1987,12,8),datetime(1988,3,14),datetime(1989,2,21),datetime(1998,12,15),datetime(1999,2,26),datetime(2000,3,20),datetime(2001,2,11),datetime(2001,12,30),datetime(2002,2,17),datetime(2003,1,18),datetime(2004,1,5),datetime(2006,1,21),datetime(2007,2,24),datetime(2008,2,22),datetime(2008,3,13),datetime(2008,3,29),datetime(2009,1,24),datetime(2010,2,9),datetime(2010,3,24),datetime(2013,1,7)]

sswdate_as_str= ['']*25
for i in range(0,25):
    sswdate_as_str[i] = sswdatetime[i].strftime('%Y%m%d')


initsetting = True

if initsetting == True:
    Tdata = Dataset('//home/storage-soonsin/REANA/ERAINTERIM/4xdaily/T/T.197901.nc')

    plev = Tdata.variables['lev'][:] # as Pa
    zlev = -7000.0*np.log(plev/100000.0) #7000m
    zlev[0] = 0.0000

    ylat = Tdata.variables['lat'][:] # as deg
    xlon = Tdata.variables['lon'][:] # as deg

    cos_phi = np.cos(ylat*math.pi/180.0)
    sin_phi = np.sin(ylat*math.pi/180.0)

    ylat_rad = ylat*math.pi/180.0
    dum = np.zeros(122)
    for i in range(1,121):
        dum[i] = math.sin((ylat_rad[i-1] + ylat_rad[i])/2)
    dum[0] = 1.0
    dum[121] = -1.0
    for i in range(0,121):
        dS_coef[i] = dum[i] - dum[i+1]

    np.savez('./init/global_arrays',plev=plev,zlev=zlev,ylat=ylat,xlon=xlon,cos_phi=cos_phi,sin_phi=sin_phi,dS_coef=dS_coef)

else:
    load = np.load('./init/global_arrays.npz')
    plev = load['plev']
    zlev = load['zlev']
    ylat = load['ylat']
    xlon = load['xlon']
    cos_phi = load['cos_phi']
    sin_phi = load['sin_phi']
    dS_coef = load['dS_coef']
    load.close()

rho = plev/100.0

##__________
##_____________________________________________________##
##_____________________________________________________##
def replace(arr,subarr,index,axis):
    #call by assignment

    if axis==0:
        arr[index,...] = subarr

    elif axis==1:
        arr[:,index,...] = subarr

    elif axis==2:
        arr[:,:,index,...] = subarr

    elif axis==3:
        arr[:,:,:,index] = subarr

##_____________________________________

def D__(x,f,daxis,increasingaxis = True): #x as 1D array, f as multiD array.

    #'(partial)diffrential operator' D_ makes 'D_x f' array on original grid.

    if len(x)!=f.shape[daxis]:
        return print('diffrential: grid array and value array size error!!')

    N = len(x)
    df = np.zeros(f.shape)
    dum = np.zeros(np.take(f,0,axis=daxis).shape)

    for i in range(1,N-1):
        h = x[i+1]-x[i]
        h_ = x[i]-x[i-1]

        dum = (h_*h_*np.take(f,i+1,axis=daxis)-h*h*np.take(f,i-1,axis=daxis)+(h*h-h_*h_)*np.take(f,i,axis=daxis))/(h*h_*(h+h_))
        replace(df,dum,i,daxis)

    h = x[1]-x[0]
    dum = (np.take(f,1,axis=daxis) - np.take(f,0,axis=daxis))/h
    replace(df,dum,0,daxis)

    h = x[N-1]-x[N-2]
    dum = (np.take(f,N-1,axis=daxis) - np.take(f,N-2,axis=daxis))/h
    replace(df,dum,N-1,daxis)

    return df

##_____________________________________
def smoothD_(x,f,daxis):

    #makes h^4 order diffrentiation with x array with fixed interval.

    N = len(x)
    h = x[1]-x[0] #(equal with x[2]-x[1],...,x[N-1]-x[N-2])
    df = np.zeros(f.shape)
    dum = np.zeros(np.take(f,0,axis=daxis).shape)

    for i in range(2,N-2):
        dum = (-1.0*(np.take(f,i+2,axis=daxis)-np.take(f,i-2,axis=daxis)) + 8.0*(np.take(f,i+1,axis=daxis)-np.take(f,i-1,axis=daxis)))/(12.0*h)
        replace(df,dum,i,daxis)

    dum = (-25.0*np.take(f,0,axis=daxis)+48.0*np.take(f,1,axis=daxis)-36.0*np.take(f,2,axis=daxis)+16.0*np.take(f,3,axis=daxis)-3.0*np.take(f,4,axis=daxis))/(12.0*h)
    replace(df,dum,0,daxis)

    dum = (-25.0*np.take(f,N-1,axis=daxis)+48.0*np.take(f,N-2,axis=daxis)-36.0*np.take(f,N-3,axis=daxis)+16.0*np.take(f,N-4,axis=daxis)-3.0*np.take(f,N-5,axis=daxis))/(-12.0*h)
    replace(df,dum,N-1,daxis)

    dum = (-3.0*np.take(f,0,axis=daxis)-10.0*np.take(f,1,axis=daxis)+18.0*np.take(f,2,axis=daxis)-6.0*np.take(f,3,axis=daxis)+1.0*np.take(f,4,axis=daxis))/(12.0*h)
    replace(df,dum,1,daxis)

    dum = (-3.0*np.take(f,N-1,axis=daxis)-10.0*np.take(f,N-2,axis=daxis)+18.0*np.take(f,N-3,axis=daxis)-6.0*np.take(f,N-4,axis=daxis)+1.0*np.take(f,N-5,axis=daxis))/(-12.0*h)
    replace(df,dum,N-2,daxis)

    return df
##_____________________________________________________##
##_____________________________________________________##
'''
def get_equivlat(q_field,Q_level,asdeg = True): 

    #q_field is QGPV on lonlat field.(in fact it is latlon, with size (240,121).)
    #Q_level is mono.incri.arr of QGPV value want to calculate equiv latitude AS DEGREE.
    
    lonlatmap = np.zeros((121,240))

    for Q in Q_level:
        lonlatmap += (q_field > Q).astype(int)
    lonlatmap -= 1
    
    N = len(Q_level)
    dum = np.zeros(N)
    dS = pow(1.5*math.pi/180.0,2)
    
    for i in range(0,N):
        dum[i] = np.inner(cos_phi,np.sum(np.where(lonlatmap>=i,1,0),axis=1))*dS
    dum = np.arcsin(1 - dum/(2*math.pi))
    dum[0] = -math.pi/2

    if asdeg == True:
        dum *= 180.0/math.pi
    
    return dum


##_____________________________________

def get_Q_ylat(q_field):
        
    # makes Q array with respect to ylat.
    
    Q_level = np.concatenate((np.linspace(q_field.min(),0,50),np.delete(np.linspace(0,q_field.max(),500),0)))
    lat_level = get_equivlat(q_field,Q_level)
    Q_lat = np.zeros(121) # Q(latitude)

    eps = 1.0e-6

    for i in range(1,120):
        idx = find_nearest(lat_level,ylat[i])
        
        if lat_level[idx]>ylat[i]:
            h2 = lat_level[idx] - ylat[i]
            h1 = ylat[i] - lat_level[idx-1]
            if((math.fabs(h1)<=eps) and (math.fabs(h2)<=eps)) or math.fabs(h1+h2)<=eps:
                Q_lat[i] = (Q_level[idx] + Q_level[idx-1])/2.0
            else:
                Q_lat[i] = (Q_level[idx]*h1 + Q_level[idx-1]*h2)/(h1+h2)
        
        elif (math.fabs(lat_level[idx] -  ylat[i])<eps):
            Q_lat[i] = Q_level[idx]

        else:
            h2 = lat_level[idx+1] - ylat[i]
            h1 = ylat[i] - lat_level[idx]
            if((math.fabs(h1)<=eps) and (math.fabs(h2)<=eps)) or math.fabs(h1+h2)<=eps:
                Q_lat[i] = (Q_level[idx] + Q_level[idx+1])/2.0
            else:
                Q_lat[i] = (Q_level[idx+1]*h1 + Q_level[idx]*h2)/(h1+h2)
    
    Q_lat[0] = q_field.max()
    Q_lat[120] = q_field.min()

    return Q_lat
'''
#_______________________________________________
def get_equivlat_NH(q_field,Q_level,asdeg = True):

    #q_field is QGPV on lonlat field.(in fact it is latlon, with size (240,121).)
    #Q_level is mono.incri.arr of QGPV value want to calculate equiv latitude AS DEGREE.

    lonlatmap = np.zeros((61,240))

    for Q in Q_level:
        lonlatmap += (q_field[0:61,:] > Q).astype(int)
    lonlatmap -= 1

    N = len(Q_level)
    dum = np.zeros(N)
    dS = pow(1.5*math.pi/180.0,2)

    for i in range(0,N):
        dum[i] = np.inner(dS_coef[0:61],np.sum(np.where(lonlatmap>=i,1,0),axis=1))*dS
    const = dum[0]
    dum = np.arcsin(1 - dum/(const))

    if asdeg == True:
        dum *= 180.0/math.pi

    return dum

def find_nearest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index

def get_Q_ylat_NH(q_field):

    # makes Q array with respect to ylat.

    #Q_level = np.concatenate((np.linspace(q_field.min(),0,50),np.delete(np.linspace(0,q_field.max(),500),0)))
    NHqmax = (q_field[0:61,:]).max()
    NHqmin = (q_field[0:61,:]).min()

    Q_level = np.linspace(NHqmin,NHqmax,num=300)
    lat_level = get_equivlat_NH(q_field,Q_level)
    Q_lat = np.zeros(121) # Q(latitude)

    eps = 1.0e-6

    for i in range(0,61):
        sleep(0.0001)
        idx = find_nearest(lat_level,ylat[i])

        if lat_level[idx]>ylat[i]:
            if idx == 0:
                Q_lat[i] = NHqmin

            else:
                h2 = lat_level[idx] - ylat[i]
                h1 = ylat[i] - lat_level[idx-1]
                if((math.fabs(h1)<=eps) and (math.fabs(h2)<=eps)) or math.fabs(h1+h2)<=eps:
                    Q_lat[i] = (Q_level[idx] + Q_level[idx-1])/2.0
                else:
                    Q_lat[i] = (Q_level[idx]*h1 + Q_level[idx-1]*h2)/(h1+h2)

        elif (math.fabs(lat_level[idx] -  ylat[i])<eps):
            Q_lat[i] = Q_level[idx]

        else:
            if idx == 299:
                Q_lat[i] = NHqmax
            else:
                h2 = lat_level[idx+1] - ylat[i]
                h1 = ylat[i] - lat_level[idx]
                if((math.fabs(h1)<=eps) and (math.fabs(h2)<=eps)) or math.fabs(h1+h2)<=eps:
                    Q_lat[i] = (Q_level[idx] + Q_level[idx+1])/2.0
                else:
                    Q_lat[i] = (Q_level[idx+1]*h1 + Q_level[idx]*h2)/(h1+h2)

    return Q_lat
##_____________________________________________________##
##_____________________________________________________##
def calc_vorticity(U,V):

    N = V.shape[0]
    VO = np.zeros((N,37,121,240))

    for i in range(0,N):
        for k in range(1,120):
            sleep(0.005)
            VO[i,:,k,:] = U[i,:,k+1,:]*cos_phi[k+1] - U[i,:,k-1,:]*cos_phi[k-1]

    VO[:,:,0,:] = ((ylat[0] - ylat[1])/(ylat[1] - ylat[2]))*(VO[:,:,1,:]-VO[:,:,2,:]) + VO[:,:,1,:]
    VO[:,:,120,:] = ((ylat[120] - ylat[119])/(ylat[118] - ylat[119]))*(VO[:,:,118,:]-VO[:,:,119,:]) + VO[:,:,119,:]

    sleep(5.0)

    for i in range(0,N):
        for l in range(1,239):
            sleep(0.005)
            VO[i,:,:,l] += V[i,:,:,l+1] - V[i,:,:,l-1]

        VO[i,:,:,0] += V[i,:,:,1] - V[i,:,:,239]
        VO[i,:,:,239] += V[i,:,:,0] - V[i,:,:,238]

    sleep(5.0)

    VO = np.einsum('ijkl,k->ijkl',VO,1.0/cos_phi)
    VO[:,:,0,:] = 0.0
    VO[:,:,120,:] = 0.0

    VO /= (2*6378.0*1000.0*1.5*math.pi/180.0)

    return VO

coef = np.power((100000.0/plev),0.286)
def calc_PT(T):
    PT = np.einsum('ijkl,j->ijkl',T,coef)
    return PT
'''
def calc_QGPV(T,VO):
    
    PT = calc_PT(T)
    PTz = np.mean((np.einsum('ijkl,k->ijkl',PT,dS_coef))[:,:,0:61,:],axis=(2,3))/np.mean(dS_coef[0:61])
    QGPV = PT - PTz[:,:,np.newaxis,np.newaxis]

    Dz_PTz = D__(zlev,PTz,daxis=1)
    QGPV = D__(zlev,np.einsum('ijkl,ij->ijkl',QGPV,1.0/Dz_PTz),daxis=1)
    QGPV = np.einsum('ijkl,j->ijkl',QGPV,rho)
    QGPV = np.einsum('ijkl,j->ijkl',D__(zlev,QGPV,daxis=1),1.0/rho)

    QGPV = np.einsum('ijkl,k->ijkl',QGPV,sin_phi*1.458*1.0e-4)
    
    QGPV += VO
    QGPV += (sin_phi*1.458*1.0e-4)[np.newaxis,np.newaxis,:,np.newaxis]

    return QGPV
'''
def calc_QGPV_new(T,VO):
    gap = 1.3
    N = T.shape[0]

    dum = np.zeros((N,37,121,240))
    PT = calc_PT(T)
    PTz = np.mean((np.einsum('ijkl,k->ijkl',PT,dS_coef))[:,:,0:61,:],axis=(2,3))/np.mean(dS_coef[0:61])

    sleep(gap)
    dum1 = (np.roll(PTz,-2,axis=1) - PTz)
    dum1[:,35] = 2*(PTz[:,36] - PTz[:,35])
    sleep(gap)

    for i in range(0,N):
        for j in range(0,36):
            sleep(0.005)
            dum[i,j,:,:] = PT[i,j+1,:,:] - PTz[i,j+1]

    dum1 = np.einsum('ijkl,ij->ijkl',dum,1.0/dum1)
    sleep(gap)
    dum1 = np.einsum('ijkl,j->ijkl',dum1,np.roll(rho,-1))

    sleep(gap)
    dum2 = (PTz - np.roll(PTz,2,axis=1))
    dum2[:,1] = 2*(PTz[:,1] - PTz[:,0])
    sleep(gap)

    for i in range(0,N):
        for j in range(1,37):
            sleep(0.005)
            dum[i,j,:,:] = PT[i,j-1,:,:] - PTz[i,j-1]

    dum2 = np.einsum('ijkl,ij->ijkl',dum,1.0/dum2)
    sleep(gap)
    dum2 = np.einsum('ijkl,j->ijkl',dum2,np.roll(rho,1))

    sleep(gap)
    f = (sin_phi*1.458*1.0e-4)

    for i in range(0,N):
        for j in range(0,37):
            sleep(0.005)
            dum[i,j,:,:] = dum1[i,j,:,:] - dum2[i,j,:,:]

    QGPV = np.einsum('ijkl,j->ijkl',dum,1.0/rho)
    QGPV = np.einsum('ijkl,k->ijkl',QGPV,f)

    QGPV[:,0,:,:] = ((zlev[0] - zlev[1])/(zlev[1] - zlev[2]))*(QGPV[:,1,:,:]-QGPV[:,2,:,:]) + QGPV[:,1,:,:]
    QGPV[:,36,:,:] = ((zlev[36] - zlev[35])/(zlev[34] - zlev[35]))*(QGPV[:,34,:,:]-QGPV[:,35,:,:]) + QGPV[:,35,:,:]



    for i in range(0,121):
        for k in range(0,N):
            sleep(0.005)
            QGPV[k,:,i,:] += f[i]
            QGPV[k,:,i,:] += VO[k,:,i,:]

    sleep(gap)

    return QGPV

'''
def calc_FAWA_Q(QGPV,check=False):
    
    N = QGPV.shape[0]
    Q = np.zeros((N,37,121))

    for i in range(0,N):
        for j in range(0,37):
            Q[i,j,:] = get_Q_ylat(QGPV[i,j,:,:])
            if check==True:
                print(i,'-',j)
    return Q
'''

def calc_FAWA_Q_NH(QGPV,check=False,timeaxis=True):

    if timeaxis == True:
        N = QGPV.shape[0]
        Q = np.zeros((N,37,121))

        for i in range(0,N):
            for j in range(0,37):
                Q[i,j,:] = get_Q_ylat_NH(QGPV[i,j,:,:])
                sleep(0.005)
                if check == True:
                    print(i,'-',j)
    else:
        Q = np.zeros((37,121))

        for j in range(0,37):
            Q[j,:] = get_Q_ylat_NH(QGPV[j,:,:])
            if check==True:
                print('-',j)
    return Q

'''
def calc_FAWA_A(QGPV,Q):
    
    dS = pow(1.5*math.pi/180.0,2)

    N = QGPV.shape[0]
    FAWA = np.zeros((N,37,121))
    for i in range(0,121):
        FAWA[:,:,i] = np.einsum('ijk,k->ij',np.sum(np.where(QGPV - Q[:,:,i].reshape(N,37,1,1) >=0,QGPV,0),axis=3),cos_phi)
        
        FAWA[:,:,i] -= np.einsum('ijk,k->ij',np.sum(QGPV[:,:,0:(i+1),:],axis=3),dS_coef[0:(i+1)])

        FAWA[:,:,i] *= (dS*6378.0*1000.0 / (2*math.pi*cos_phi[i]))

    return FAWA
'''

def calc_FAWA_A_NH(QGPV,Q):

    d = pow(1.5*math.pi/180.0,2)

    N = QGPV.shape[0]
    FAWA = np.zeros((N,37,121))

    QGPV = QGPV[:,:,0:61,:]
    Q = Q[:,:,0:61]

    for i in range(0,N):
        qdum = QGPV[i,:,:,:]
        gc.collect()
        for k in range(0,61):
            FAWA[i,:,k] = np.einsum('jk,k->j',np.sum(np.where((qdum - Q[i,:,k].reshape(1,37,1)) >=0,qdum,0),axis=2),cos_phi[0:61])

            FAWA[i,:,k] -= np.einsum('jk,k->j',np.sum(qdum[:,0:(k+1),:],axis=2),cos_phi[0:(k+1)])

            FAWA[i,:,k] *= (d*6378.0*1000.0 / (2*math.pi*cos_phi[k]))

            sleep(0.005)

    return FAWA

##_____________________________________________________##

def calc_EPfluxdiv(QGPV,V):
    gap = 0.1

    N = V.shape[0]
    vq_bar = np.zeros((N,37,121))
    EPdiv = np.zeros((N,37,121))

    for i in range(0,N):
        for j in range(0,37):
            V[i,j,:,:] -= np.mean(V[i,j,:,:],axis=1).reshape(121,1)
            sleep(0.005)
            QGPV[i,j,:,:] -= np.mean(QGPV[i,j,:,:],axis=1).reshape(121,1)

            vq_bar[i,j,:] = np.mean(np.multiply(V[i,j,:,:],QGPV[i,j,:,:]),axis=1)

    EPdiv = np.einsum('ijk,k->ijk',vq_bar,cos_phi)

    return EPdiv

def calc_EPfluxdiv2(T,U,V):
    gap = 0.1 #(s)

    PT = calc_PT(T)
    PTz = np.mean((np.einsum('ijkl,k->ijkl',PT,dS_coef))[:,:,0:61,:],axis=(2,3))/np.mean(dS_coef[0:61])

    N = T.shape[0]
    vPT_bar = np.zeros((N,37,121))
    vu_bar = np.zeros((N,37,121))
    EPdiv = np.zeros((N,37,121))

    for i in range(0,N):
        for j in range(0,37):
            V[i,j,:,:] -= np.mean(V[i,j,:,:],axis=1).reshape(121,1)
            sleep(0.005)
            U[i,j,:,:] -= np.mean(U[i,j,:,:],axis=1).reshape(121,1)
            sleep(0.005)
            PT[i,j,:,:] -= np.mean(PT[i,j,:,:],axis=1).reshape(121,1)
            # Now array V,U,PT indicate V',U',PT' respectively.

            vu_bar[i,j,:] = np.mean(np.multiply(V[i,j,:,:],U[i,j,:,:]),axis=1)
            vPT_bar[i,j,:] = np.mean(np.multiply(V[i,j,:,:],PT[i,j,:,:]),axis=1)


    sleep(gap)
    dum1 = (np.roll(PTz,-2,axis=1) - PTz)
    dum1[:,35] = 2*(PTz[:,36] - PTz[:,35])
    sleep(gap)
    dum1 = np.einsum('ijk,ij->ijk',np.roll(vPT_bar,-1,axis=1),1.0/dum1)
    sleep(gap)
    dum1 = np.einsum('ijk,j->ijk',dum1,np.roll(rho,-1))

    sleep(gap)
    dum2 = (PTz - np.roll(PTz,2,axis=1))
    dum2[:,1] = 2*(PTz[:,1] - PTz[:,0])
    sleep(gap)
    dum2 = np.einsum('ijk,ij->ijk',np.roll(vPT_bar,1,axis=1),1.0/dum2)
    sleep(gap)
    dum2 = np.einsum('ijk,j->ijk',dum2,np.roll(rho,1))

    sleep(gap)
    for i in range(0,N):
        for j in range(0,37):
            sleep(0.005)
            EPdiv[i,j,:] = dum1[i,j,:] - dum2[i,j,:]

    f = (sin_phi*1.458*1.0e-4)
    EPdiv = np.einsum('ijk,j->ijk',EPdiv,1.0/rho)
    EPdiv = np.einsum('ijk,k->ijk',EPdiv,np.multiply(f,cos_phi))

    EPdiv[:,0,:] = ((zlev[0] - zlev[1])/(zlev[1] - zlev[2]))*(EPdiv[:,1,:]-EPdiv[:,2,:]) + EPdiv[:,1,:]
    EPdiv[:,36,:] = ((zlev[36] - zlev[35])/(zlev[34] - zlev[35]))*(EPdiv[:,34,:]-EPdiv[:,35,:]) + EPdiv[:,35,:]

    #__________________#

    dum3 = np.zeros((N,37,121))
    cossq = np.power(cos_phi,2)
    for i in range(0,N):
        for k in range(1,120):
            sleep(0.005)
            dum3[i,:,k] = vu_bar[i,:,k-1]*cossq[k-1] - vu_bar[i,:,k+1]*cossq[k+1]

    dum3[:,:,0] = ((ylat[0] - ylat[1])/(ylat[1] - ylat[2]))*(dum3[:,:,1]-dum3[:,:,2]) + dum3[:,:,1]
    dum3[:,:,120] = ((ylat[120] - ylat[119])/(ylat[118] - ylat[119]))*(dum3[:,:,118]-dum3[:,:,119]) + dum3[:,:,119]

    dum3 = np.einsum('ijk,k->ijk',dum3,1.0/cos_phi)
    dum3 /= (2*6378.0*1000.0*1.5*math.pi/180.0)

    sleep(gap)
    for i in range(0,N):
        for j in range(0,37):
            sleep(0.005)
            EPdiv[i,j,:] -= dum3[i,j,:]

    EPdiv[:,:,0] = 0.0
    EPdiv[:,:,120] = 0.0

    return EPdiv

##(((((((((((((((((((((((((()))))))))))))))))))))))))))##

def calc_equivlength(Q,q):
    # q is QGPV

    N = Q.shape[0]
    q = q[:,:,0:61,:]
    Q = Q[:,:,0:61]

    a = 6378000.0
    d = 1.5*math.pi/180.0

    DqDq = np.zeros((N,37,121,240))
    I_DqDq = np.zeros((N,37,121))

    ## Calculate Dq^2 ##

    for i in range(0,N):
        gc.collect()
        qdum = q[i]

        dum1 = np.roll(qdum,-1,axis=2) - np.roll(qdum,1,axis=2)
        dum1 = np.einsum('jkl,k->jkl',dum1,1/(2.0*a*d*cos_phi))
        dum1 = np.multiply(dum1,dum1)

        dum2 = np.roll(qdum,-1,axis=1) - np.roll(qdum,1,axis=2)
        dum2 *= 1/(2.0*a*d)
        dum2 = np.multiply(dum2,dum2)

        DqDq[i] = dum1 + dum2
        DqDq[i,:,0,:] = 0.0
        DqDq[i,:,120,:] = 0.0

    # ____________  #


    ## Calculate integral of Dq^2 ##

    for i in range(0,N):
        gc.collect()
        qdum = q[i]
        for k in range(0,61):
            I_DqDq[i,:,k] = np.einsum('jk,k->j',np.sum(np.where((qdum - Q[i,:,k].reshape(1,37,1)) >=0,DqDq,0),axis=2),cos_phi[0:61])
        sleep(0.005)

    I_DqDq *= pow((d*a),2)

    # _____________  #

##_____________________________________________________##
##_____________________________________________________##

def get_sswevent(centraldate,var,N=320,center=160):

    #centraldate as datetime format
    #var as 'T' 'VO' etc...

    dates = ['']*N
    timeindex = np.zeros(N)

    for i in range(0,N):
        d = centraldate+(i-center)*timedelta(hours=6)
        dates[i] = d.strftime('%Y%m')
        timeindex[i] = int((int(d.strftime('%d'))-1)*4 + int(d.strftime('%H'))/6)

    if var=='VO':
        directory = ''.join('adress in server')
    else:
        directory = ''.join('adress in server')

    arr = np.zeros((N,37,121,240))
    for i in range(0,N):
        indata = Dataset(''.join([directory,dates[i],'.nc']))
        arr[i,:,:,:] = indata.variables[var][timeindex[i],:,:,:]
        indata.close()

    return arr

def get_var(month,year,var):

    #month as 1979, 198004 etc...
    monthyear = str(int(year))+str(int(month)).zfill(2)

    if var=='VO':
        directory = ''.join('adress in server')
    else:
        directory = ''.join('adress in server')

    indata = Dataset(directory)

    return indata.variables[var]

##_____________________________________________________##
##_____________________________________________________##

def set_save_variable_tzyx(write,opt=True,N = 320,center=160):

    #write as Dataset(dir,'w',format='NETCDF4')

    write.createDimension('lon',240)
    write.createDimension('lat',121)
    write.createDimension('lev',37)
    write.createDimension('time',None)
    time = write.createVariable('time','f4',('time',))
    lev = write.createVariable('lev','f4',('lev',))
    lat = write.createVariable('lat','f4',('lat',))
    lon = write.createVariable('lon','f4',('lon',))
    time.setncatts({'standard_name' : u"time",'units': u"hours with origin at date(central of ssw)-00:00:00)"})
    lev.setncatts({'standard_name' : u"air_pressure",'units': u"Pa"})
    lat.setncatts({'standard_name' : u"latitude",'units': u"degrees_north"})
    lon.setncatts({'standard_name' : u"longitude",'units': u"degrees_east"})

    if opt==True:
        time[:] = (np.asarray(range(0,320))-160)*6
    else:
        time[:] = (np.asarray(range(0,N))-center)*6

    lev[:] = plev
    lat[:] = ylat
    lon[:] = xlon

def set_save_variable_tzy(write,opt=True,N=320,center=160):

    #write as Dataset(dir,'w',format='NETCDF4')

    write.createDimension('lat',121)
    write.createDimension('lev',37)
    write.createDimension('time',None)
    time = write.createVariable('time','f4',('time',))
    lev = write.createVariable('lev','f4',('lev',))
    lat = write.createVariable('lat','f4',('lat',))
    time.setncatts({'standard_name' : u"time",'units': u"hours with origin at date(central of ssw)-00:00:00)"})
    lev.setncatts({'standard_name' : u"air_pressure",'units': u"Pa"})
    lat.setncatts({'standard_name' : u"latitude",'units': u"degrees_north"})

    if opt==True:
        time[:] = (np.asarray(range(0,320))-160)*6
    else:
        time[:] = (np.asarray(range(0,N))-center)*6

    lev[:] = plev
    lat[:] = ylat
