'''
Spectra plottings
'''

import numpy as np
import matplotlib.pyplot as plt


def singles(wn, spec):
    '''
    Individual spectra plot. 

    Parameters
    ----------
    wn : ndarray 
        Wavenumber of shape [n_points].
    spec : ndarray
        Spectra of shape [n_spectra, n_points].

    Returns
    -------
    None.

    '''
    
    fig, ax = plt.subplots()
    plt.xlabel('Wavenumber ($\mathrm{cm^{-1}}$)')
    plt.ylabel('Absorbance')
    plt.xlim(np.ceil(wn.max()), np.floor(wn.min()))
    plt.plot(wn, spec.T)
    plt.grid(False) 


def means(wn, spec, label=False, std=False):
    '''
    Mean spectra plot. 
    If label is passed, one mean per group.
    If std, plot mean + standard deviation.

    Parameters
    ----------
    wn : ndarray 
        Wavenumber of shape [n_points].
    spec : ndarray
        Spectra of shape [n_spectra, n_points].
    label : list of str, optional
        Labels of shape [n_sepctra]. The default is False.
    std : boolean, optional
        If True, plot mean + standard deviation. The default is False.

    Returns
    -------
    None.

    '''

    if label:
        label_set = list(set(label))
        specmean = []
        specstd = []
        for i in range(len(label_set)):
            mask = [None]*len(label)
            for j in range(len(label)):
                mask[j] = (label[j] == label_set[i])
            specmean += [np.mean(spec[mask,:], axis=0)]
            specstd += [np.std(spec[mask,:], axis=0)]
        specmean = np.array(specmean)     
        specstd = np.array(specstd)  

    else:
        specmean = np.mean(spec, axis=0)
        specstd = np.std(spec, axis=0)
                       
    fig, ax = plt.subplots()
    plt.grid()
    plt.xlabel('Wavenumber ($\mathrm{cm^{-1}}$)')
    plt.ylabel('Absorbance')
    plt.xlim(np.ceil(wn.max()), np.floor(wn.min()))
    plt.plot(wn, specmean.T)
    if std:
        if len(specstd.shape) == 1:
            plt.fill_between(wn, (specmean-1*specstd), (specmean+1*specstd), alpha=0.25)
        else:
            for i in range(specstd.shape[0]):
                plt.fill_between(wn, (specmean[i,:]-1*specstd[i,:]), (specmean[i,:]+1*specstd[i,:]), alpha=0.25)
    if label:
        plt.legend(label_set)
    plt.grid(False)   
    
    
def img(values, n_sample, fpa_size):
    '''
    Plot images given values such as area, intensities.

    Parameters
    ----------
    values : ndarray
        Values to be plotted as image
    n_sample : int
        Sample to be plotted
    fpa_size : int
        Size of the FPA.

    Returns
    -------
    None.

    '''

    # Reshape to an image of shape fpa_size x fpa_size
    values_img = np.reshape(
        values[0+(fpa_size*fpa_size*n_sample)
               :fpa_size*fpa_size+(fpa_size*fpa_size*n_sample)],
        (fpa_size,fpa_size)
        )
    
    # Define zero (outliers) as black
    cmap = plt.cm.jet
    cmap.set_under(color='black')  
    
    # Define the scale min as the values_img min
    vmin = np.min(values_img[np.nonzero(values_img)])-0.000001
    
    # Plot image
    fig, ax = plt.subplots()
    plt.imshow(values_img, cmap=cmap, vmin=vmin)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.colorbar()
    
    