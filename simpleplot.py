
import numpy as np
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
import matplotlib
##__________________________________________##

def shiftedColorMap(cmap,start=0.0,midpoint=0.5,stop=1.0,name='shifted'):
    cdict = {
            'red': [],
            'green': [],
            'blue': [],
            'alpha': []
    }

    #regular index to compute the colors
    reg_index = np.linspace(start,stop,257)

    #shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0,midpoint,128,endpoint=False),
        np.linspace(midpoint,1.0,129,endpoint=True)
        ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si,r,r))
        cdict['green'].append((si,g,g))
        cdict['blue'].append((si,b,b))
        cdict['alpha'].append((si,a,a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name,cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

class plotOnmap:

    def __init__(self,x_axis,y_axis,title,cmap=cm.coolwarm):
        self.x_axis = np.asarray(x_axis)
        self.y_axis = np.asarray(y_axis)
        self.cmap = cmap

        self.lonmin=x_axis.min()
        self.lonmax=x_axis.max()
        self.latmin=y_axis.min()
        self.latmax=y_axis.max()

        self.m = Basemap(llcrnrlon=self.lonmin,llcrnrlat=self.latmin,urcrnrlon=self.lonmax,urcrnrlat=self.latmax,fix_aspect=True,projection='mill',resolution='l')

        self.m.drawcoastlines(linewidth = 0.6,color='black')
        self.m.drawmapboundary()
        #self.m.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0],fontsize=10)
        #self.m.drawmeridians(np.arange(0,360,60),labels=[0,0,0,1],fontsize=10)

        plt.title(title,fontsize=20)

    def shading(self,arr,name,nbins,setorigin=1,cutoff = False,fmt=False):
        if (self.lonmin==0 and self.lonmax==385.5):
            v,xl = addcyclic(arr,self.x_axis)
        else:
            v = arr
            xl = self.x_axis

        yl = self.y_axis
        lon,lat = np.meshgrid(xl,yl)
        self.x,self.y = self.m(lon, lat)


        avg = np.mean(arr)
        std = np.std(arr)
        r = 4.5
        if cutoff==True:
            arr = np.where(arr<r*std,arr,r*std)
            arr = np.where(arr>-1.0*r*std,arr,-1.0*r*std)

        vmin = arr.min()
        vmax = arr.max()
        axis = np.arange(avg-1.8*std,avg+1.8*std,.2*std)

        if fmt == False:
            fmt = '%f'
        else:
            fmt=ticker.FuncFormatter(fmt)

        if setorigin==1:
            self.cmap = shiftedColorMap(self.cmap,midpoint= -vmin/(vmax-vmin))

        levels = MaxNLocator(nbins=nbins).tick_values(vmin,vmax)

        cs = self.m.contourf(self.x,self.y,v,axis,levels=levels,cmap=self.cmap,extend='both')
        cb = self.m.colorbar(cs,location='bottom',extend='both',spacing='proportional',fraction=0.046,pad=0.30,shrink=0.6,format=fmt)
        cb.ax.tick_params(direction='out',length=6,width=2,labelsize=7,grid_alpha=0.1)
        cb.set_label(name,size=15)

    def contour(self,arr,nbins,d,fmt=False):

        if (self.lonmin==0 and self.lonmax==385.5):
            v,xl = addcyclic(arr,self.x_axis)
        else:
            v = arr
            xl = self.x_axis

        yl = self.y_axis
        lon,lat = np.meshgrid(xl,yl)
        self.x,self.y = self.m(lon, lat)

        levels = MaxNLocator(nbins=nbins).tick_values(arr.min(),arr.max())
        cs = self.m.contour(self.x,self.y,v,levels=levels,colors='black',linewidths=1,liinestyles='solid')

        if fmt == False:
            fmt = '%f'
        else:
            fmt=ticker.FuncFormatter(fmt)

        clevels = levels[::d]
        clb=plt.clabel(cs,clevels,fontsize=7,fmt=fmt)

    def ploting(self):
        print('\n***press [X] for end***\n')
        plt.show()

##______________________##

class plotOnlatlev:
    def __init__(self,ylat,zlev,title,cmap=cm.coolwarm):

        self.x_axis = np.asarray(ylat)
        self.y_axis = np.asarray(zlev)
        self.cmap = cmap

        plt.figure(figsize=(9,5))

        plt.title(title,fontsize = 20,pad=10)
        #plt.tight_layout()

    def shading(self,arr,name,nbins,setorigin,fmt=False):
        arr = np.asarray(arr)
        vmin = arr.min()
        vmax = arr.max()
        if setorigin==1:
            self.cmap = shiftedColorMap(self.cmap,midpoint= -vmin/(vmax-vmin))

        levels = MaxNLocator(nbins=nbins).tick_values(vmin,vmax)
        cf = plt.contourf(self.x_axis,self.y_axis,arr,levels=levels,cmap=self.cmap)

        if fmt == False:
            fmt = '%f'
        else:
            fmt = ticker.FuncFormatter(fmt)
        cbar = plt.colorbar(cf,extend='both',spacing='proportional',fraction=0.046,pad=0.04,format=fmt)
        cbar.ax.tick_params(direction='out',length=2,width =1, labelsize=10,grid_alpha = 1.0)
        cbar.set_label(name,size=15)

    def contour(self,arr,nbins,d,fmt=False):
        arr = np.asarray(arr)
        levels = MaxNLocator(nbins=nbins).tick_values(arr.min(),arr.max())
        cs = plt.contour(self.x_axis,self.y_axis,arr,levels=levels,colors='black',linewidths=.7,linestyles='solid')
        clevels=levels[::d]
        if fmt == False:
            fmt = '%f'
        else:
            fmt=ticker.FuncFormatter(fmt)

        clb = plt.clabel(cs,clevels,fontsize=8,fmt=fmt)

    def ploting(self):

        plt.ylabel('pressure height (mbar)',fontsize = 15)
        plt.gca().invert_yaxis()
        plt.yscale('log')
        yticklist=[1,2,5,10,20,50,100,200,500,1000]
        plt.yticks(yticklist,yticklist)

        plt.xlabel('latitude',fontsize = 15)
        plt.xticks([-90,-60,-30,0,30,60,90],['90°S','60°S','30°S','0','30°N','60°N','90°N'])

        print('\n***press [X] for end***\n')
        plt.show()

