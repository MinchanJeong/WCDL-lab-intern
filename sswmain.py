
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import numpy as np
import math

from datetime import datetime,timedelta
from netCDF4 import Dataset,num2date,date2num

import sswmodule as swm

from time import sleep
import gc

## 'global' variables _________________________________

plev = np.zeros(37) #length = 37, unit: Pa
zlev = np.zeros(37) #length = 37, unit: m
ylat = np.zeros(121) #length = 121
xlon = np.zeros(240) #length = 240

Cos_phi = np.zeros(121) #cosine(lat)
sin_phi = np.zeros(121)
dS_coef = np.zeros(121) # sin(phi_f) - sin(phi_i) ; [sin(90)-sin(89.25),sin(89.25)-sin(87.75),..]
rho = np.zeros(37) #= plev/100.0

sswdatetime = [datetime(1981,3,4),datetime(1981,12,4),datetime(1984,2,24),datetime(1985,1,1),datetime(1987,1,23),datetime(1987,12,8),datetime(1988,3,14),datetime(1989,2,21),datetime(1998,12,15),datetime(1999,2,26),datetime(2000,3,20),datetime(2001,2,11),datetime(2001,12,30),datetime(2002,2,17),datetime(2003,1,18),datetime(2004,1,5),datetime(2006,1,21),datetime(2007,2,24),datetime(2008,2,22),datetime(2008,3,13),datetime(2008,3,29),datetime(2009,1,24),datetime(2010,2,9),datetime(2010,3,24),datetime(2013,1,7)]

sswdate_as_str= ['']*25
for i in range(0,25):
    sswdate_as_str[i] = sswdatetime[i].strftime('%Y%m%d')

load = np.load('./init/global_arrays.npz')
plev = load['plev']
zlev = load['zlev']
ylat = load['ylat']
xlon = load['xlon']
cos_phi = load['cos_phi']
sin_phi = load['sin_phi']
rho = plev/100.0
dS_coef = load['dS_coef']
load.close()

##_________________________________
calc_vorticity = False
calc_QGPV = False
repair_QGPV = False
calc_FAWA_Q = False
calc_FAWA_A = False

calc_EPdiv = False
calc_EPdiv2 = True
##_____________________________________________________
##_____________________________________________________

if calc_vorticity == True:

    print('start & end?')
    start = int(input())
    end = int(input())
    print(start,'to',end)

    for i in range(start,end+1):

        U = swm.get_sswevent(sswdatetime[i],'U')
        V = swm.get_sswevent(sswdatetime[i],'V')
        VO = np.zeros((320,37,121,240))
        print('VO'+sswdate_as_str[i]+' read completed')

        mark = np.arange(0,321,10) #0~319
        L = len(mark)
        for n in range(0,L-1):
            VO[mark[n]:mark[n+1],:,:,:] = swm.calc_vorticity(U[mark[n]:mark[n+1],:,:,:],V[mark[n]:mark[n+1],:,:,:])
            print(mark[n],'~',mark[n+1]-1)
            gc.collect()
            sleep(1.0)

        writeVO = Dataset('./ssw/VO/VO.'+sswdate_as_str[i]+'.nc','w',format='NETCDF4')
        swm.set_save_variable_tzyx(writeVO)
        VO_data = writeVO.createVariable('VO','f4',('time','lev','lat','lon',))
        VO_data[:,:,:,:] = VO
        VO_data.setncatts({'standard_name' : u"Vorticity",'units' : u"1/s"})
        writeVO.close()
        print('VO'+sswdate_as_str[i])

        gc.collect()
        sleep(1.0)

if calc_QGPV == True:

    print('start & end?')
    start = int(input())
    end = int(input())
    print(start,'to',end)
    for i in range(start,end+1):

        T = swm.get_sswevent(sswdatetime[i],'T')
        VO = Dataset('./ssw/VO/VO.'+sswdate_as_str[i]+'.nc').variables['VO'][:,:,:,:]
        QGPV = np.zeros((320,37,121,240))
        print('QGPV'+sswdate_as_str[i]+' read completed')

        mark = np.arange(0,321,10) #0~319
        L = len(mark)
        for n in range(0,L-1):
            QGPV[mark[n]:mark[n+1],:,:,:] = swm.calc_QGPV_new(T[mark[n]:mark[n+1],:,:,:],VO[mark[n]:mark[n+1],:,:,:])
            print(mark[n],'~',mark[n+1]-1)
            gc.collect()
            sleep(1.0)

        writeQGPV = Dataset('./ssw/QGPV_NH/QGPV.'+sswdate_as_str[i]+'.nc','w',format='NETCDF4')
        swm.set_save_variable_tzyx(writeQGPV)
        QGPV_data = writeQGPV.createVariable('QGPV','f4',('time','lev','lat','lon',))
        QGPV_data[:,:,:,:] = QGPV
        QGPV_data.setncatts({'standard_name' : u"qusigeostropic potential vorticity",'units' : u"1/s"})
        writeQGPV.close()
        print('QGPV'+sswdate_as_str[i])

        gc.collect()
        sleep(1.0)

if repair_QGPV == True:

    start,end = 15,24

    for i in range(start,end+1):

        QGPV = Dataset('./ssw/QGPV/QGPV.'+sswdate_as_str[i]+'.nc').variables['QGPV'][:,:,:,:]

        QGPV[:,0,:,:] = ((zlev[0] - zlev[1])/(zlev[1] - zlev[2]))*(QGPV[:,1,:,:]-QGPV[:,2,:,:]) + QGPV[:,1,:,:]
        QGPV[:,36,:,:] = ((zlev[36] - zlev[35])/(zlev[34] - zlev[35]))*(QGPV[:,34,:,:]-QGPV[:,35,:,:]) + QGPV[:,35,:,:]

        writeQGPV = Dataset('./ssw/QGPV2/QGPV.'+sswdate_as_str[i]+'.nc','w',format='NETCDF4')
        swm.set_save_variable_tzyx(writeQGPV)
        QGPV_data = writeQGPV.createVariable('QGPV','f4',('time','lev','lat','lon',))
        QGPV_data[:,:,:,:] = QGPV
        QGPV_data.setncatts({'standard_name' : u"qusigeostropic potential vorticity",'units' : u"1/s"})
        writeQGPV.close()
        print('QGPV'+sswdate_as_str[i])

##_________________________________

if calc_FAWA_Q == True:

    print('start & end?')
    start = int(input())
    end = int(input())
    print(start,'to',end)
    Q = np.zeros((320,37,121))

    for i in range(start,end+1):
        q = Dataset('./ssw/QGPV_NH/QGPV.'+sswdate_as_str[i]+'.nc').variables['QGPV'][:,:,:,:]
        print('Q'+sswdate_as_str[i]+' read completed')

        mark = np.arange(0,321,10) #0~319
        L = len(mark)
        for n in range(0,L-1):
            Q[mark[n]:mark[n+1],:,:] = swm.calc_FAWA_Q_NH(q[mark[n]:mark[n+1],:,:,:],check=True)
            print(mark[n],'~',mark[n+1]-1)
            gc.collect()
            sleep(1.0)

        writeQ = Dataset('./ssw/Q_NH/Q.'+sswdate_as_str[i]+'.nc','w',format='NETCDF4')
        swm.set_save_variable_tzy(writeQ)
        Q_data = writeQ.createVariable('Q','f4',('time','lev','lat',))
        Q_data[:,:,:] = Q
        Q_data.setncatts({'standard_name' : u"Lagrangian-mean QGPV with respect to equivalend latitude",'units' : u"1/s"})
        writeQ.close()
        print('Q'+sswdate_as_str[i])

##_________________________________

if calc_FAWA_A == True:

    q = np.zeros((320,37,121,240))
    Q = np.zeros((320,37,121))

    print('start & end?')
    start = int(input())
    end = int(input())
    print(start,'to',end)

    y50N = np.where(ylat==49.5)[0][0]
    y70N = np.where(ylat==70.5)[0][0]

    for n in range(start,end+1):
        q = Dataset('./ssw/QGPV_past/QGPV.'+sswdate_as_str[n]+'.nc').variables['QGPV'][:,:,:,:]
        Q = Dataset('./ssw/Q_past/Q.'+sswdate_as_str[n]+'.nc').variables['Q'][:,:,:]

        FAWA = swm.calc_FAWA_A_NH(q,Q)

        writeFAWA = Dataset('./ssw/FAWA_past/FAWA.'+sswdate_as_str[n]+'.nc','w',format='NETCDF4')
        swm.set_save_variable_tzy(writeFAWA)
        FAWA_data = writeFAWA.createVariable('FAWA','f4',('time','lev','lat',))
        FAWA_data[:,:,:] = FAWA
        FAWA_data.setncatts({'standard_name' : u"Finite Amplitude Wave Activity",'units' : u"m/s"})
        writeFAWA.close()
        print('FAWA'+sswdate_as_str[n])

##_____________________________________________________
##_____________________________________________________

if calc_EPdiv == True:

    print('start & end?')
    start = int(input())
    end = int(input())
    print(start,'to',end)

    EPdiv = np.zeros((320,37,121))

    for i in range(start,end+1):
        QGPV = Dataset('./ssw/QGPV_NH/QGPV.'+sswdate_as_str[i]+'.nc').variables['QGPV'][:,:,:,:]
        V = swm.get_sswevent(sswdatetime[i],'V')
        print('EPdiv'+sswdate_as_str[i]+' read completed')

        mark = np.arange(0,321,10) #0~319
        L = len(mark)
        for n in range(0,L-1):
            EPdiv[mark[n]:mark[n+1],:,:] = swm.calc_EPfluxdiv(QGPV[mark[n]:mark[n+1],:,:,:],V[mark[n]:mark[n+1],:,:,:])
            print(mark[n],'~',mark[n+1]-1)
            gc.collect()
            sleep(1.0)

        writeEPdiv = Dataset('./ssw/EPdiv/EPdiv.'+sswdate_as_str[i]+'.nc','w',format='NETCDF4')
        swm.set_save_variable_tzy(writeEPdiv)
        EPdiv_data = writeEPdiv.createVariable('EPdiv','f4',('time','lev','lat',))
        EPdiv_data[:,:,:] = EPdiv
        EPdiv_data.setncatts({'standard_name' : u"Divergence of E-P flux divided by air density ",'units' : u"m/s^2"})
        writeEPdiv.close()
        print('EPdiv'+sswdate_as_str[i])


if calc_EPdiv2 == True:

    print('start & end?')
    start = int(input())
    end = int(input())
    print(start,'to',end)

    EPdiv = np.zeros((320,37,121))

    for i in range(start,end+1):
        T = swm.get_sswevent(sswdatetime[i],'T')
        U = swm.get_sswevent(sswdatetime[i],'U')
        V = swm.get_sswevent(sswdatetime[i],'V')

        mark = np.arange(0,321,10) #0~319
        L = len(mark)
        for n in range(0,L-1):
            EPdiv[mark[n]:mark[n+1],:,:] = swm.calc_EPfluxdiv2(T[mark[n]:mark[n+1],:,:,:],U[mark[n]:mark[n+1],:,:,:],V[mark[n]:mark[n+1],:,:,:])
            print(mark[n],'~',mark[n+1]-1)
            gc.collect()
            sleep(1.0)

        writeEPdiv = Dataset('./ssw/EPdiv2/EPdiv.'+sswdate_as_str[i]+'.nc','w',format='NETCDF4')
        swm.set_save_variable_tzy(writeEPdiv)
        EPdiv_data = writeEPdiv.createVariable('EPdiv','f4',('time','lev','lat',))
        EPdiv_data[:,:,:] = EPdiv
        EPdiv_data.setncatts({'standard_name' : u"Divergence of E-P flux divided by air density ",'units' : u"m/s^2"})
        writeEPdiv.close()
        print('EPdiv'+sswdate_as_str[i])
