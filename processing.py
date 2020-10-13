'''
Spectra processing
'''

import numpy as np


def band_area(wn, spec, point1, point2):
    '''
    Integrate the band area.

    Parameters
    ----------
    wn : ndarray 
        Wavenumber of shape [n_points].
    spec : ndarray
        Spectra of shape [n_spectra, n_points].
    point1 : int
        First point of the band.
    point2 : int
        Last point of the band.

    Returns
    -------
    area : ndarray
        Integrated areas of shape [n_spectra].

    '''
    
    from specDS import preprocessing as pp
    
    band = pp.cut_outer(wn, spec, point1, point2)[1]
    area = np.trapz(band)
    
    print(f'Band area integrated from {point1} to {point2} cm-1')
    
    return area


def band_area_multiple(wn, spec, points):
    '''
    Integrate multiple band areas.

    Parameters
    ----------
    wn : ndarray 
        Wavenumber of shape [n_points].
    spec : ndarray
        Spectra of shape [n_spectra, n_points].
    points : list of tuple
        List of all band points. Example:
            points = [
                (point1,point2), # First band
                (point3,point4), # Second band
                (point5,point6), # Third band
                ]           

    Returns
    -------
    area : ndarray
        Integrated areas of shape [n_spectra, n_bands].

    '''

    area = []    
    for point in points:
        area += [band_area(wn, spec, point[0], point[1])]
    area = np.array(area).T    
    
    return area

