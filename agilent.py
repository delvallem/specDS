'''
Manage Agilent FPA files
'''

import numpy as np
import glob
import os
    
def openFPA(path, mode):
    ''' 
    Open one single or mosaic FPA acquistion.
    Agilent usually saves each mosaic in an individual folder.
    
    Parameters
    ----------
    path : str
        Path of the FPA to be opened, according to the mode:
        
        If 'single' mode:
            Path of the single FILE with any extention or without it. Examples:
                - path = 'C:/FTIR/sample.bsp'
                - path = 'C:/FTIR/sample.dat'
                - path = 'C:/FTIR/sample.seq'
                - path = 'C:/FTIR/sample'        
       
        If 'mosaic' mode:
            Path of the FOLDER containing the mosaic files. Example:
                - path = 'C:/FTIR/mosaic_sample'
            
    mode : str
        'single' or 'mosaic' acquisition mode.
    
    Returns
    -------
    data : dict
        - filename (str): name of the opened file
        - fpa_size (int): size of the equipment detector 
        - spec (ndarray): spectra
            - if single: shape [n_spectra, n_points]
            - if mosaic: shape [n_tiles, n_spectra, n_points]
        - tiles_x (int): number of tiles in x direction (if 'single', tiles_x=1)
        - tiles_y (int): number of tiles in y direction (if 'single', tiles_y=1)
        - wn (ndarray): wavenumber; shape [n_points]
        
    '''
        
    if mode == 'single':
        
        # Files path
        path = os.path.splitext(path)
        bsp_path = path[0]+".bsp"
        dat_path = path[0]+".dat"
       
        # Wavenumber array
        bsp_float = np.fromfile(bsp_path, dtype='float64')
        bsp_int = np.fromfile(bsp_path, dtype='int32')
        
        wn_start = bsp_int[557]
        wn_points = bsp_int[559]
        wn_step = bsp_float[277]
        
        wn = np.arange(1, wn_points+wn_start, 1)
        wn = wn*wn_step
        wn = wn[wn_start-1:]
        
        # FPA Size
        byte_size = os.path.getsize(dat_path)
        byte_size = ((byte_size/4)-255)/len(wn)
        fpa_size = int(byte_size**0.5)
        
        # Number of tiles
        tiles_x = int(1)
        tiles_y = int(1)
        
        # Spectra (Intensities)
        spec = np.fromfile(dat_path, dtype='float32')
        spec = spec[255:]
        spec = np.reshape(spec, (len(wn),fpa_size,fpa_size))
        spec = np.flip(spec, axis=1)
        spec = np.reshape(spec, (len(wn),fpa_size*fpa_size)).T
        
        # Filename
        filename = os.path.basename(path[0])
         
        # Data dict
        data = {
            'filename': filename,
            'fpa_size': fpa_size,
            'spec': spec,
            'tiles_x': tiles_x,
            'tiles_y': tiles_y,
            'wn': wn
            }
        
        print(f'- Loaded file {filename}')

        return data
    
    elif mode == 'mosaic':
        
        # Files path
        dmt_path = glob.glob(path + "/*.dmt")
        dmd_path = glob.glob(path + "/*.dmd")
        dat_path = glob.glob(path + "/*.dat")
        
        # Sort dmd
        dmd_path.sort()
        
        # Wavenumber array
        dmt_float = np.fromfile(dmt_path[0], dtype='float64')
        dmt_int = np.fromfile(dmt_path[0], dtype='int32')
        
        wn_start = dmt_int[557]
        wn_points = dmt_int[559]
        wn_step = dmt_float[277]
        
        wn = np.arange(1, wn_points+wn_start, 1)
        wn = wn*wn_step
        wn = wn[wn_start-1:]
        
        # FPA size
        byte_size = os.path.getsize(dmd_path[0])
        byte_size = ((byte_size/4)-255)/len(wn)
        fpa_size = int(byte_size**0.5)
        
        # Number of tiles
        tiles_x = int(np.asarray(dmd_path[-1][-13:-9]))+1
        tiles_y = int(np.asarray(dmd_path[-1][-8:-4]))+1
        
        # Spectra (intensities)
        spec = np.empty([tiles_x*tiles_y, fpa_size*fpa_size, len(wn)])
        i=0
        for dmd_file in dmd_path:
            dmd_file = np.fromfile(dmd_file, dtype='float32')
            dmd_file = dmd_file[255:]
            dmd_file = np.reshape(dmd_file, (len(wn),fpa_size,fpa_size))
            dmd_file = np.flip(dmd_file, axis=1)
            dmd_file = np.reshape(dmd_file, (len(wn),fpa_size*fpa_size)).T
            spec[i,:,:] = dmd_file
            i+=1
        
        # Filename
        filename = os.path.basename(dmt_path[0]).replace('.dmt', '')

        # Data dict
        data = {
            'filename': filename,
            'fpa_size': fpa_size,
            'spec': spec,
            'tiles_x': tiles_x,
            'tiles_y': tiles_y,
            'wn': wn            
            }
        
        print(f'- Loaded file {filename}')
        
        return data
    
    else:
        print('Invalid Mode! \nSelect "single" or "mosaic" acquisition.')


def openFPA_multiple(folder, mode):
    '''
    Open multiple single or mosaic FPA acquistions.
    
    Parameters
    ----------
    folder : str
        Path of the FOLDER containg multiple single files or mosaic subfolders.
       
        Examples:
            
        If 'single' mode:
            path = 'C:/FTIR/samples'
            
            "samples" folder containing:
                - sample1.bsp 
                - sample1.dat
                - sample1.seq
                - sample2.bsp 
                - sample2.dat
                - sample2.seq
                - so on
                    
                                
        If 'mosaic' mode:
            path = 'C:/FTIR/mosaic_samples'
            
            "mosaic_samples" folder containing subfolders:
                - C:/FTIR/mosaic_samples/mosaic1
                - C:/FTIR/mosaic_samples/mosaic2
                - so on
            Each subfolder with their respectives mosaic files.

    mode : str
        'single' or 'mosaic' acquisition mode.

    Returns
    -------
    data : list of dict
        One dictionary for each single or mosaic image.

    '''
    
    if mode == 'single':
        
        # Get all bsp files inside the folder
        files = glob.glob(folder + "/*.bsp")
        
        # Open all single files
        data = []
        for file in files:
            data += [openFPA(file, mode)]
            
        return data
    
    elif mode == 'mosaic':
    
        # Get all folder files and subfolders
        files = glob.glob(folder + "/*")
        
        # Remove non subfolders files; remove subfolders without .dmt
        for file in files:
            isdmt = False
            if not os.path.isdir(file):
                files.remove(file)
            else:
                for subfiles in glob.glob(file+"/*"):
                    if os.path.splitext(subfiles)[1] == '.dmt':
                        isdmt = True
                        break
                if not isdmt:
                    files.remove(file)
        
        # Open all mosaic files
        data = []
        for file in files:
            data += [openFPA(file, mode)]
        
        return data

    else:
        print('Invalid Mode! \nSelect "single" or "mosaic" acquisition.')


