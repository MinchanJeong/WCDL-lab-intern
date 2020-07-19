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

## 'global' variables _________________________________

plev = np.zeros(37) #length = 37, unit: Pa
zlev = np.zeros(37) #length = 37, unit: m
ylat = np.zeros(121) #length = 121
xlon = np.zeros(240) #length = 240

cos_phi = np.zeros(121) #cosine(lat)
sin_phi = np.zeros(121)
dS_coef = np.zeros(121) # sin(phi_f) - sin(phi_i) ; [sin(90)-sin(89.25),sin(89.25)-sin(87.75),..]
rho = np.zeros(37) #= plev/100.0

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

#_______________________________________________________

save_DJF_QGPV = False
qREF_qbar = False
bar__qREF_q = False
diff = False

plotabove = True

#_______________________________________________________
if save_DJF_QGPV == True:

    monthlist = np.asarray([1,2,12]).astype(int)

    print('Starting year and month?')
    y = int(input())
    m = int(input())
    print(m)
    start = 3*(y - 1979) + np.where(monthlist == m)[0]
    start = int(start)

    for n in np.arange(start,120):

        year = 1979 + int(int(n)/3) #1979 ~ 2018
        month = monthlist[int(n)%3]

        print('START:',year,str(month).zfill(2))

        T = swm.get_var(month,year,'T')
        VO = swm.get_var(month,year,'VO')
        QGPV = swm.calc_QGPV_new(T,VO)
        writeQGPV = Dataset('./climat/QGPV_NH/QGPV.'+str(year)+str(month).zfill(2)+'.nc','w',format='NETCDF4')
        swm.set_save_variable_tzyx(writeQGPV)
        QGPV_data = writeQGPV.createVariable('QGPV','f4',('time','lev','lat','lon',))
        QGPV_data[:,:,:,:] = QGPV
        QGPV_data.setncatts({'standard_name' : u"qusigeostropic potential vorticity",'units' : u"1/s"})
        writeQGPV.close()

        print('END:',year,str(month).zfill(2),'\n')


if qREF_qbar == True:

    N = 119

    monthlist = np.asarray([1,2,12]).astype(int)

    count = 0
    dum = np.zeros((37,121,240))
    for n in np.arange(0,N+1):

        year = 1979 + int(int(n)/3) #1979 ~ 2018
        month = monthlist[int(n)%3]

        QGPV = Dataset('./climat/QGPV_NH/QGPV.'+str(year)+str(month).zfill(2)+'.nc').variables['QGPV'][:,:,:,:]
        dum += np.sum(QGPV,axis=0)

        count += QGPV.shape[0]
        print(n,'/',N)

    dum /= count

    qREF = swm.calc_FAWA_Q_NH(dum,check=True,timeaxis=False)

    np.savez('./climat/qREF_qbar',qREF = qREF)

if plotabove == True:

    load = np.load('./climat/qREF_qbar.npz')
    qREF = load['qREF']
    load.close()

    print(qREF)

    arr = qREF
    #____________

    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    from matplotlib import ticker
    from matplotlib.ticker import MaxNLocator

    import simpleplot as smp

    is900hPa = np.where(plev == 90000)[0][0]
    is30N = np.where(ylat == 30)[0][0]

    xaxis = ylat[:(is30N+1)]
    yaxis = -7.0*np.log(plev[is900hPa:]/100000.0)
    arr = arr[is900hPa:,:(is30N+1)]

    cmap = cm.rainbow
    plt.figure(figsize=(6,5))
    plt.title('qREF of time-meaned q',fontsize=20,pad=10)

    vmin = arr.min()
    vmax = arr.max()

    levels = MaxNLocator(nbins=27).tick_values(0,0.001)
    #levels = np.asarray([0,12,24,36,48,60,72,84,96])*0.00001

    #cmap = smp.shiftedColorMap(cmap,midpoint=-vmin/(vmax-vmin),name='shifted')
    cf = plt.contourf(xaxis,yaxis,arr,levels=levels,cmap=cmap)

    cbar = plt.colorbar(cf,extend='both',fraction=0.046,pad=0.04)
    cbar.ax.set_title('[1/s]',fontsize=8)

    plt.ylabel('Pseudoheight [km]',fontsize = 15)

    yticklist=[1,6,11,16,21,26,31,36,41,46]
    plt.yticks(yticklist,yticklist)

    plt.xlabel('Latitude [deg]',fontsize = 15)
    plt.xticks([30,40,50,60,70,80,90],[30,40,50,60,70,80,90])

    plt.show()
