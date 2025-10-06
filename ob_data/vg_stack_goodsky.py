'''

from vg_stack.py
for galaxies with bad sky subtractions
ZZ@NAOC
Dec 3, 2019

'''
import numpy as np
from pandas import DataFrame
from scipy import interpolate
from scipy import optimize
from astropy.io import fits
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.axis as ax
import re,sys

###################
# These functions are written by Weijia Sun @PKU
#
def moments(m_stack):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    m_total = m_stack.sum()
    X, Y = np.indices(m_stack.shape)
    x = (X * m_stack).sum() / m_total
    y = (Y * m_stack).sum() / m_total
    m_col = m_stack[:, int(y)]
    m_width_x = np.sqrt(abs((np.arange(m_col.size) - y) ** 2 * m_col).sum() / m_col.sum())
    m_row = m_stack[int(x), :]
    m_width_y = np.sqrt(abs((np.arange(m_row.size) - x) ** 2 * m_row).sum() / m_row.sum())
    m_height = m_stack.max()
    return m_height, x, y, m_width_x, m_width_y

def gaussian_2d(m_height, m_center_x, m_center_y, m_width_x, m_width_y):
    """Returns a gaussian function with the given parameters"""
    m_width_x = float(m_width_x)
    m_width_y = float(m_width_y)
    return lambda x, y: m_height * np.exp(
                -(((m_center_x - x) / m_width_x) ** 2 + ((m_center_y - y) / m_width_y) ** 2) / 2)

def fit_gaussian(m_stack, m_x_range=[0, 0], m_y_range=[0, 0]):
    """Derive the center coordinate using 2D fitting
    
    Args:
        m_stack (2D array): 2D data
        m_x_range (list): range of x used in calculation
        m_y_range (list): range of y used in calculation
    """
    if m_x_range == [0, 0]:
        m_x_range = [0, m_stack.shape[0]]
    if m_y_range == [0, 0]:
        m_y_range = [0, m_stack.shape[1]]
    m_stack = m_stack[m_x_range[0]: m_x_range[1], m_y_range[0]: m_y_range[1]]
    m_params = moments(m_stack)
    errorfunction = lambda p: np.ravel(gaussian_2d(*p)(*np.indices(m_stack.shape)) - m_stack)
    p, success = optimize.leastsq(errorfunction, m_params)
    return p + [0, m_x_range[0], m_y_range[0], 0, 0]

def shift_data(m_data, m_x_center, m_y_center, m_x_target, m_y_target):
    """Shift the data to target coordinates

    Args:
        m_data (DataFrame): 2D data
        m_x_center (float): center of raw 2D data in x
        m_y_center (float): center of raw 2D data in y
        m_x_target (float): center of target 2D data in x
        m_y_target (float): center of target 2D data in y
    """
    m_x_delta = m_x_target - m_x_center
    m_y_delta = m_y_target - m_y_center
    m_x_1 = np.arange(m_data.shape[1])
    m_y_1 = np.arange(m_data.shape[2])
    m_x_2 = m_x_1 - m_x_delta
    m_y_2 = m_y_1 - m_y_delta
    m_data_shift = np.zeros([m_data.shape[0], m_data.shape[1], m_data.shape[2]])
    for i in range(m_data.shape[0]):
        f = interpolate.interp2d(m_x_1, m_y_1, m_data[i, :, :].T, kind='cubic')
        m_data_shift[i, :, :] = f(m_x_2, m_y_2).T
    return m_data_shift

def integrate_frame(m_data_shift):
    """Combine the frames together

    Use after function shift_data
    """
    m_data_int = np.zeros(m_data_shift[0].shape)
    for i in range(len(m_data_shift)):
        m_data_int = m_data_int + m_data_shift[i]
    return m_data_int

###################
# below are written by ZZ@NAOC

# read pcwi.link 

indi=int(sys.argv[1])

inpaths=[
 '/home/zz/work/virgo3d/obs/2016Apr/reduce/160405/',\
 '/home/zz/work/virgo3d/obs/2017Mar/reduce/170328/',\
 '/home/zz/work/virgo3d/obs/2018Apr/reduce/11_redux/',\
 '/home/zz/work/virgo3d/obs/2018Apr/reduce/12_redux/']

inpath=inpaths[indi]

pathout='/home/zz/work/virgo3d/plots_badsky/'
linkfile=inpath+'redux/pcwi.link'
links=open(linkfile,'r')
imgnum=[]
targets=[]
names=[]
for link in links:
    tmpstr=link.split()
    #print(len(tmpstr),tmpstr[0],tmpstr[7],tmpstr[-1][0:3])
    if (tmpstr[7] != '-1') and (tmpstr[-1][0:3] == 'VCC'):
        #print(len(tmpstr),tmpstr[0],tmpstr[7],tmpstr[-1][0:3])
        imgnum.append(tmpstr[0])
        targets.append(tmpstr[-1])
        if tmpstr[-1] in names:
            continue
        else:
            names.append(tmpstr[-1])
#print(imgnum,targets)
files=[inpath+'redux/image' + tmpimgnum + '_icubes.fits' for tmpimgnum in imgnum]


targets=np.array(targets)
files=np.array(files)
imgnum=np.array(imgnum)
m_x_target = 11
m_y_target = 43
cutwav= 2. # [Ang], cut the edges of the spectra to stack, use 2 for normal observations
#cutwav=5. # [Ang], use this for Medrez observations
for name in names:
    print(name)
    if name == 'VCC990':
        cutwav=3.
    if name=='VCC1499':
        cutwav=5.
    indname = np.where(targets == name)[0]
    tmpfiles=files[indname]
    tmpimgnum=imgnum[indname]
    nf = len(tmpfiles)
    for indf,file in enumerate(tmpfiles):
        print(indf, file)
        if tmpimgnum[indf]=='24648':
            continue
        with fits.open(file) as hdu:
            data=hdu[0].data
            wav0=hdu[0].header['CRVAL3']
            dw=hdu[0].header['CD3_3']
            y0=hdu[0].header['CRPIX1']
            x0=hdu[0].header['CRPIX2']
            nw=hdu[0].header['NAXIS3']
            header=hdu[0].header
        wave=wav0+np.arange(nw)*dw
        datasum=data[300:-300,:,:].sum(axis=0)
        plt.figure(figsize=(16,10))
        plt.imsave(pathout+name+'_'+tmpimgnum[indf]+'_img.png',datasum)
        plt.close()
        datashape=datasum.shape
        p=fit_gaussian(datasum,[m_x_target-5,m_x_target+5],[m_y_target-15,m_y_target+15])
        m_x_center=p[1]
        m_y_center=p[2]
        #print(m_x_center,m_y_center,x0,y0,wav0)
        m_data_shift=shift_data(data, m_x_center, m_y_center, m_x_target, m_y_target)
        datasum_shifted=m_data_shift.sum(axis=0)
        plt.figure(figsize=(16,10))
        plt.imsave(pathout+name+'_'+tmpimgnum[indf]+'_img_shifted.png',datasum)
        plt.close()
        if indf == 0:
            wav1=np.min(wave)+cutwav
            wav2=np.max(wave)-cutwav
            indwav=np.where((wave>wav1) & (wave<wav2))[0]
            stackdata=np.empty(shape=[nf, indwav.shape[0],m_x_target*2+1,m_y_target*2+1])
        else:
            indwav=np.where((wave>wav1) & (wave<wav2))[0]
        if (datashape[0] >= m_x_target*2+1) and (datashape[1] >= m_y_target*2+1):
            stackdata[indf,:,:,:]=m_data_shift[indwav,0:2*m_x_target+1,0:2*m_y_target+1]
        if (datashape[0] < m_x_target*2+1) and (datashape[1] >= m_y_target*2+1):
            stackdata[indf,:,m_x_target-int(datashape[0]/2):m_x_target+int(datashape[0]/2),:]= \
          m_data_shift[indwav,m_x_target-int(datashape[0]/2):m_x_target+int(datashape[0]/2),0:2*m_y_target+1]
        if (datashape[0] >= m_x_target*2+1) and (datashape[1] < m_y_target*2+1):
            stackdata[indf,:,:,m_y_target-int(datashape[1]/2):m_y_target+int(datashape[1]/2)]= \
          m_data_shift[indwav,0:2*m_x_target+1,m_y_target-int(datashape[1]/2):m_y_target+int(datashape[1]/2)]
        if (datashape[0] < m_x_target*2+1) and (datashape[1] < m_y_target*2+1):
            stackdata[indf,:,m_x_target-int(datashape[0]/2):m_x_target+int(datashape[0]/2), \
          m_y_target-int(datashape[1]/2):m_y_target+int(datashape[1]/2)]= \
          m_data_shift[indwav,m_x_target-int(datashape[0]/2):m_x_target+int(datashape[0]/2), \
          m_y_target-int(datashape[1]/2):m_y_target+int(datashape[1]/2)]
        outhdu=fits.PrimaryHDU(m_data_shift)
        header['CRVAL3'] = np.min(wave[indwav])
        header['NAXIS3'] = len(indwav)
        header['CRPIX1'] = m_x_target
        header['CRPIX2'] = m_y_target
        outhdu.header=header
        outhdu.writeto(pathout+name+'_'+tmpimgnum[indf]+'_shifted.fits',overwrite=True)
    #print(m_data_shift.shape)
    stackdata.shape
    stackdataout=np.nanmedian(stackdata,axis=0)
    stackhdu=fits.PrimaryHDU(stackdataout)
    stackhdu.header=header
    stackhdu.writeto(pathout+name+'_stack.fits',overwrite=True)


# examine the shift result
    stackimg=stackdataout.sum(axis=0)
    plt.figure(figsize=(16,10))
    plt.imsave(pathout+name+'_stack.png',stackimg)
    plt.close()
