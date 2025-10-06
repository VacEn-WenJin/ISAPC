# from vg_stack2.py
# use 3 parts
# according to r50
# 
# stack spectra within one galaxy
# into 5 concentric rings
# ZZ@NAOC
# Sep 1, 2019
#
from time import perf_counter as clock
from os import path
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from astropy.io import fits
import numpy as np
import sys,glob
from scipy import ndimage

listpath='/data/work/virgo3d/plots/'
glistfile=listpath+'galaxylist.txt'

i = int(sys.argv[1])

### input this
x0 = 11.
y0 = 43.
anns = np.array([0.,0.3,0.7,1.])
###

# extract information of the target galaxy
with open(glistfile) as f:
 for tmpi,line in enumerate(f):
  if tmpi == i+1:
   tmp=line.split()
   name=tmp[0]
   rz=float(tmp[1])
   r50=float(tmp[9])

print(i,name,rz,r50)
anns = anns * r50

# input spectra path
stack_dir = listpath

# input spectra file
file = stack_dir+name+'_stack.fits'
c = 299792.458  # spped of light [km/s]
        

# read input spectra
with fits.open(file) as hdu:
    flux= hdu[0].data
    hdr = hdu[0].header
    wave = hdr['CRVAL3']+np.arange(hdr['NAXIS3'])*hdr['CD3_3']
    #wave *= np.median(util.vac_to_air(wave)/wave)
    
xscale = 2.65
yscale = 0.58

# make an index array
indarray = np.indices(flux.shape[1:3])
xind = indarray[0,:,:] - x0
yind = indarray[1,:,:] - y0
dist = np.sqrt((xind*xscale)**2 + (yind*yscale)**2)

flux_stack = np.zeros([flux.shape[0],anns.shape[0]])

plt.figure(figsize=(14,10))
plt.ylim(0.3,1.25)
plt.xlim(4827,5210)
plt.xlabel('Wave [Angstrom]',size=20)
plt.ylabel('Normalized Flux',size=20)
plt.title(name,size=20)
plt.tick_params(labelsize=20)
#plt.rcParams["font.size"] = 20
#plt.rcParams["font.family"] = "Times New Roman"
for ind_ann in range(flux_stack.shape[1]-1):
    indgood = np.where((dist >= anns[ind_ann]) & (dist < anns[ind_ann+1]))
    flux_stack[:,ind_ann] = np.median(flux[:,indgood[0],indgood[1]],axis = 1)
    tmpflux = flux_stack[:,ind_ann]/np.median(flux_stack[200:-200,ind_ann])+ind_ann/5.-0.4
    tmpflux2 = ndimage.gaussian_filter1d(tmpflux, 3)
    plt.plot(wave/(1.+rz),tmpflux2,label=str(ind_ann+1))

indgood = np.where(dist < 30.)
flux_stack[:,anns.shape[0]-1] = np.median(flux[:,indgood[0],indgood[1]],axis = 1)
tmpflux = flux_stack[:,anns.shape[0]-1]/np.median(flux_stack[200:-200,anns.shape[0]-1])+(anns.shape[0]-1.)/5.-0.4
tmpflux2 = ndimage.gaussian_filter1d(tmpflux, 1)
#plt.plot(wave/(1.+rz),tmpflux2)

plt.axvline(x=4827.875,color='k',linestyle='--')
plt.axvline(x=4847.875,color='k',linestyle='--')
plt.axvline(x=4876.625,color='k',linestyle='--')
plt.axvline(x=4891.625,color='k',linestyle='--')
plt.axvline(x=4946.625,color='k',linestyle='--')
plt.axvline(x=4977.75,color='k',linestyle='--')
plt.axvline(x=5054.0,color='k',linestyle='--')
plt.axvline(x=5065.25,color='k',linestyle='--')
plt.axvline(x=5142.625,color='k',linestyle='--')
plt.axvline(x=5160.125,color='k',linestyle='--')
plt.axvline(x=5192.625,color='k',linestyle='--')
plt.axvline(x=5206.375,color='k',linestyle='--')
#plt.show()
plt.savefig(stack_dir+name+'_ann_stack3.png')
                                
#hdu1 = fits.PrimaryHDU(flux_stack)

hdu = fits.BinTableHDU.from_columns(
      [fits.Column(name='wave',format='E',array=wave/(1+rz)),
       fits.Column(name='flux1',format='E',array=flux_stack[:,0]),
       fits.Column(name='flux2',format='E',array=flux_stack[:,1]),
       fits.Column(name='flux3',format='E',array=flux_stack[:,2]),
       fits.Column(name='flux_all',format='E',array=flux_stack[:,3])])
hdu.writeto(stack_dir+name+'_ann_stack3.fits',overwrite=True)
