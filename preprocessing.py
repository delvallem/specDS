'''
Data pre-processing
'''

import numpy as np
import matplotlib.pyplot as plt

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
    if isinstance(data, dict):
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
    
    print('Spectra concatenated. Labels created.')
    
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


def cut_outer(wn, spec, init, final):
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
    
    mask = ((wn >= init) & (wn <= final))
    
    wn = wn[mask]
    spec = spec[:,mask]
    
    print(f'Selected from {init} cm-1 to {final} cm-1.')
    
    return wn, spec


def cut_inner(wn, spec, init, final):    
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
    
    mask = ((wn < init) | (wn > final))
    
    wn = wn[mask]
    spec = spec[:,mask]
    
    print(f'Removed from {init} cm-1 to {final} cm-1.')
    
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
    
    print('Vector normalization applied.')
        
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
    
    print('Min-max normalization applied.')
    
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
    
    print('Standard Normal Variate applied.')
    
    return spec   
 

def emsc(spec, degree=2, norm=True):
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
    params = np.linalg.lstsq(model, spec.T, rcond=None)[0]
    
    # Baseline correction (a, d1, d2, ..., dn)
    spec_corr = spec - model[:,1:].dot(params[1:,:]).T
    
    # Normalization (b)
    if norm:
        spec_corr = spec_corr/(params[0,:].T[:,None])
        
    print('EMSC applied.')

    return spec_corr


def meancenter(spec, orientation):
    '''
    Mean center rows or columns.

    Parameters
    ----------
    spec : ndarray
        Spectra of shape [n_spectra, n_points].
        
    orientation : str
        Spectra of shape [n_spectra, n_points].

    Returns
    -------
    spec : ndarray
        Mean centered spectra of same shape.

    '''
    
    if orientation == 'row':
        spec = spec - np.mean(spec, axis=1)[:,None]
        print('Mean centered (rows).')
        
    elif orientation == 'column':
        spec = spec - np.mean(spec, axis=0)[None,:]
        print('Mean centered (columns).')
        
    else:
        print('Invalid orientation! \nSelect "row" or "column" orientation.')

    return spec


def offset(spec):
    '''
    Remove spectra offset.

    Parameters
    ----------
    spec : ndarray
        Spectra of shape [n_spectra, n_points].

    Returns
    -------
    spec : ndarray
        Spectra without offset.

    '''
    spec = spec - np.min(spec, axis=1)[:,None]
    
    return spec


def quality(spec, wn, 
            signal=[1620,1690], noise=[1800,1900], 
            threshold=False, label=False):
    '''
    Quality test based on the Signal-to-Noise Ratio (SNR).
    Optional: remove bad quality spectra based on input threshold.

    Parameters
    ----------
    spec : ndarray
        Spectra of shape [n_spectra, n_points].
    wn : ndarray 
        Wavenumber of shape [n_points].
    signal : list of int, optional
        Initial and end points of the signal band. \
            The default is [1620,1690] for the Amide I band.
    noise : list of int, optional
        Initial and end points of the noise band. \
            The default is [1800,1900] for the biological dead region (usually).
    threshold : float, optional
        SNRs lower than the threshold are bad quality. The default is False.
    label : list, optional
        List of labels. The default is False. Pass only if threshold.
        
    Returns
    -------
    quality_bad : boolean, optional (only if threshold)
        Array identifying outliers.         
    spec_clean : ndarray, optional (only if threshold)
        Cleaned spectra of shape [n_spectra, n_points] (bad quality removed).
    label_clean : ndarray, optional (only if threshold)
        Cleaned labels of shape [n_spectra] (bad quality removed).
            
    '''
    
    # Offset
    spec_off = offset(spec)
    
    # Signal-to-Noise Ratio
    signal_mask = ((wn >= signal[0]) & (wn <= signal[1]))
    signal_area = np.trapz(spec_off[:,signal_mask])     
    noise_mask = ((wn >= noise[0]) & (wn <= noise[1]))
    noise_area = np.trapz(spec_off[:,noise_mask])     
    snr = signal_area/noise_area
    
    # If bad quality thresholding
    if threshold:
        
        from itertools import compress
        
        # Thresholding
        quality_bad = snr < threshold        
        spec_clean = spec[np.invert(quality_bad),:]
        label_clean = list(compress(label, np.invert(quality_bad)))

        print(f'{sum(quality_bad)} bad quality spectra found' \
              f' ({np.round((sum(quality_bad)/spec.shape[0])*100, 2)}%' \
                  ' of the total spectra).')
            
        # Plot histogram with threshold line
        fig, ax = plt.subplots()
        plt.hist(snr, bins=500)
        plt.xlabel('SNR')
        plt.ylabel('Frequency')
        plt.axvline(x=threshold, color='red', linewidth=1)
        plt.xlim(0)
        
        return quality_bad, spec_clean, label_clean
    
    else:
        
        # Plot histogram
        fig, ax = plt.subplots()
        plt.hist(snr, bins=500)
        plt.xlabel('SNR')
        plt.ylabel('Frequency')
        plt.xlim(0)


def pcanoise(spec, ncomp=False, expvar=False):
    '''
    PCA Noise Reduction.
    
    Accepts a fixed PC number OR a fixed explained variance and PC will be \
        sellected according to it. Only one has to be informed.
    
    Examples:
    
    - pcanoise(spec, ncomp=10):
        - Apply a PCA model with 10 PCs.
        
    - pcanoise(spec, expvar=0.9)
        - Apply a PCA model with the number of PCs where the explained \
            variance is up to 90%.
        

    Parameters
    ----------
    spec : ndarray
        Spectra of shape [n_spectra, n_points].
    ncomp : int, optional
        Number of Principal Components. The default is False.
    expvar : float, optional
        Explained variance to reach with the PCs. The default is False.

    Returns
    -------
    spec_denoise : ndarray
        Spectra after PCA Noise Reduction.

    '''
    
    from sklearn.decomposition import PCA
    
    if ncomp and not expvar:
        
        # Fit PCA Model and calculate the parameters
        pca = PCA(n_components=ncomp)
        scores = pca.fit_transform(spec)
        variance_cumulative = np.cumsum(pca.explained_variance_ratio_)
        loadings = (pca.components_.T
                    * np.sqrt(pca.explained_variance_)) 
        
        # Spectra noise reduction
        spec_denoise = np.mean(spec, axis=0) + np.dot(scores, loadings.T)
        
        print(f'PCA Noise Reduction applied: \n' \
              f'- Selected PCs: {ncomp}. \n' \
              f'- Explained Variance: ' \
                  f'{np.round(variance_cumulative[-1]*100, 2)}%.')   

        return spec_denoise
            
    elif expvar and not ncomp:
        
        # Fit PCA Model
        pca = PCA()
        scores = pca.fit(spec)
        
        # Check the number of PCs where the explained variance is up to expvar
        variance_cumulative = np.cumsum(pca.explained_variance_ratio_)
        pcselect = (variance_cumulative <= expvar)
        
        # Calculate the parameters (only for selected PCs)
        scores = pca.transform(spec)[:,pcselect]
        loadings = (pca.components_.T[:,pcselect]
                    * np.sqrt(pca.explained_variance_)[pcselect])
        
        # Spectra noise reduction     
        spec_denoise = np.mean(spec, axis=0) + np.dot(scores, loadings.T)        
        
        print(f'PCA Noise Reduction applied: \n' \
              f'- Selected PCs: {sum(pcselect)}. \n' \
              f'- Explained Variance: ' \
                  f'{np.round(variance_cumulative[pcselect][-1]*100, 2)}%.')
        
        return spec_denoise
    
    elif expvar and ncomp:
        print('Invalid input! Two parameters informed.\n' \
              'Inform only one: number of PCs (ncomp) ' \
                      'OR explained variance (expvar).')   
            
    elif not expvar and not ncomp:
        print('Invalid input! No parameter informed.\n' \
              'Inform only one: number of PCs (ncomp) ' \
                      'OR explained variance (expvar).')            
    
    