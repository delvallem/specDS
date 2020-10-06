'''
Data pre-processing
'''

import numpy as np

def concat2D(data):
    '''
    Concatenate all spectra opened with agilentFPA_multiple or agilentFPA where
    mode = 'mosaic'. Useful for processing.

    Create a label for each spectrum. Useful for group analysis, model 
    training/testing.
    
    Parameters
    ----------
    data : list of dict or dict
       Data files opened with agilentFPA_multiple (single or mosaic) or
       one data file opened with agilentFPA (mosaic)

    Returns
    -------
    spec_concat : ndarray
        All spectra in the shape [n_spectra, n_points].
        
    label_concat : list
        List of one filename per each spectrum.
    '''
    
    spec_all = []
    label_concat = []
    
    # If only one mosaic, create a list with only it
    if isinstance(data, dict) == True:
        data = [data]   
    
    # Get spectra
    for oneFPA in data:
        spec_one = oneFPA['spec']
        
        # If single, add the spectra in the spectra list 
        if len(np.shape(spec_one)) == 2:
            spec_all += [spec_one]
        
        # If mosaic, reshape tiles in 2D array  
        # then add the spectra in the spectra list 
        elif len(np.shape(spec_one)) == 3:
            spec_all += [
                np.reshape(
                    spec_one,
                    (
                        (oneFPA['fpa_size']
                         * oneFPA['fpa_size']
                         * oneFPA['tiles_x']
                         * oneFPA['tiles_y']),
                        len(oneFPA['wn'])
                        )
                    )
                ]
        
        # Add one label for each spectrum in the label list     
        label = ([oneFPA['filename']]
                 * (oneFPA['fpa_size']
                    * oneFPA['tiles_x']
                    * oneFPA['fpa_size']
                    * oneFPA['tiles_y']))
        label_concat += label
    
    # All spectra to one 2D array
    spec_concat = np.concatenate(spec_all, axis=0)
    
    return spec_concat, label_concat


def wnnear(wn, value):
    '''
    Wavenumbers are not integers, so find the nearest.

    Parameters
    ----------
    wn : ndarray 
        Wavenumber of shape [n_points].
    value : int
        Point to be found.

    Returns
    -------
    idx : int
        Index of the closest wavenumber point.

    '''

    idx = (np.abs(wn - value)).argmin()
   
    return idx


def cut_inner(wn, spec, init, final):
    '''
    Cut the spectra and the wavenumber in the style:    
    
    XXXXX----------XXXXX

    where:
        X: removed value
        
        -: held value
    
    Parameters
    ----------
    wn : ndarray 
        Wavenumber of shape [n_points].
    spec : ndarray
        Spectra of shape [n_spectra, n_points].
    init : int
        Initial value to hold.
    final : int
        Final value to hold.

    Returns
    -------
    wn : ndarray 
        Cutted wavenumber of shape [n_points].
    spec : ndarray
        Cutted spectra of shape [n_spectra, n_points].

    '''
    
    wn_i = wnnear(wn, init)
    wn_f = wnnear(wn, final)
    
    wn = wn[wn_i:wn_f+1]
    spec = spec[:,wn_i:wn_f+1]
    
    return wn, spec


def cut_outer(wn, spec, init, final):    
    '''
    Cut the spectra and the wavenumber in the style: 
    
    ----------XXXXX----------

    where:
        X: removed value
        
        -: held value
    
    Parameters
    ----------
    wn : ndarray 
        Wavenumber of shape [n_points].
    spec : ndarray
        Spectra of shape [n_spectra, n_points].
    init : int
        Initial value to remove.
    final : int
        Final value to remove.

    Returns
    -------
    wn : ndarray 
        Cutted wavenumber of shape [n_points].
    spec : ndarray
        Cutted spectra of shape [n_spectra, n_points].

    '''
    
    wn_i = wnnear(wn, init)
    wn_f = wnnear(wn, final)
    
    wn = np.delete(wn, np.s_[wn_i:wn_f+1])
    spec = np.delete(spec, np.s_[wn_i:wn_f+1], axis=1)
    
    return wn, spec


def vector(spec):
    '''
    Vector normalization

    Parameters
    ----------
    spec : ndarray
        Spectra of shape [n_spectra, n_points].

    Returns
    -------
    spec : ndarray
        Normalized spectra of same shape.

    '''
    
    spec = np.divide(
        spec,
        np.sqrt(np.sum(np.square(spec),axis=1))[:,None]
        )
    
    return spec


def snv(spec):
    '''
    Standard Normal Variate

    Parameters
    ----------
    spec : ndarray
        Spectra of shape [n_spectra, n_points].

    Returns
    -------
    spec : ndarray
        Normalized spectra of same shape.

    '''
    
    spec = np.divide(
        spec-np.mean(spec, axis=1)[:,None],
        np.std(spec,axis=1)[:,None]
        )
    
    return spec

    
def minmax(spec):
    '''
    Min-max normalization

    Parameters
    ----------
    spec : ndarray
        Spectra of shape [n_spectra, n_points].

    Returns
    -------
    spec : ndarray
        Normalized spectra of same shape.

    '''
    
    spec = np.divide(
        spec - np.min(spec, axis=1)[:,None],
        (np.max(spec, axis=1) - np.min(spec, axis=1))[:,None]
        )
    
    return spec    
 
    
def meancenter(spec):
    '''
    Mean center the spectrum

    Parameters
    ----------
    spec : ndarray
        Spectra of shape [n_spectra, n_points].

    Returns
    -------
    spec : ndarray
        Mean centered spectra of same shape.

    '''
    
    spec = spec - np.mean(spec, axis=1)[:,None]

    return spec

def emsc(spec, degree = 2, norm = True):
    '''
    Extended multiplicative signal correction (EMSC). 
    As described in Afseth and Kohler, 2012 (10.1016/j.chemolab.2012.03.004).
    
    - spec = a + spec_mean*b + e
    
    - spec_corr = (spec - a)/b
    
    - spec_corr = (spec - a - d1*(spec) - d2*(spec**2) - ... - dn*(spec**n))/b
    
    Parameters
    ----------
    spec : ndarray
        Spectra of shape [n_spectra, n_points].
    degree : int, optional
        Degree of the polynomial model. The default is 2.
    norm : bool, optional
        Normalize the data. The default is True.

    Returns
    -------
    spec_corr : ndarray
        Corrected spectra.

    '''

    # Polynomial model
    d = np.linspace(-1, 1, np.shape(spec)[1]).reshape(-1,1)
    d = np.repeat(d,degree+1,axis=1)**np.arange(0,degree+1)
    
    # Least Squares estimation
    model = np.hstack((np.mean(spec, axis=0).reshape(-1,1), d))
    params = np.linalg.lstsq(model,spec.T,rcond=None)[0]
    
    # Baseline correction (a, d1, d2, ..., dn)
    spec_corr = spec - model[:,1:].dot(params[1:,:]).T
    
    # Normalization (b)
    if norm:
        spec_corr = spec_corr/(params[0,:].T[:,None])

    return spec_corr
