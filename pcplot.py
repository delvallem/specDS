'''
Useful Principal Component Analysis (PCA) plottings
'''

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats


class PCPlot:

    def __init__(self, spec, ncomp, label=False):
        '''
        Calculate useful PCA parameters.

        Parameters
        ----------
        spec : ndarray
            Spectra of shape [n_spectra, n_points].
        label : list of str, optional
            Labels of shape [n_sepctra]. The default is False.
        ncomp : int
            Number of Principal Components.

        Returns
        -------
        None.

        '''
        
        self.spec = spec
        self.ncomp = ncomp
        self.label = label
        
        self.pca = PCA(n_components=ncomp)
        self.scores = self.pca.fit_transform(spec)
        self.variance_ratio = self.pca.explained_variance_ratio_
        self.variance_total = np.sum(self.variance_ratio)
        self.variance_cumulative = np.cumsum(self.variance_ratio)
        self.loadings = (
            (self.pca.components_.T)
            *(np.sqrt(self.pca.explained_variance_))
            )
        
        
    def labeling(self):
        '''
        Create legend and colors for scatter plottings.

        Returns
        -------
        legends : list of Line2D
            Legend for each unique label.
        colors : list of str
            HEX color for each label.

        '''
        
        # Create a color list based on Tableau Colors
        color_list =  [
            "#4e79a7",  # blue
            "#f28e2b",  # orange
            "#e15759",  # red
            "#59a14f",  # green
            "#b07aa1",  # purple
            "#edc948",  # yellow
            "#76b7b2",  # cyan
            "#ff9da7",  # pink
            "#9c755f",  # brown
            "#08315e",  # dark blue
            "#730607",  # dark red
            "#0b5701",  # dark green            
            "#54043e",  # dark purple
            "#edc948",  # dark yellow
            "#008c81",  # dark cyan
            "#612808",  # dark brown            
            "#8bb4e0",  # light blue
            "#ffc185",  # light orange
            "#bdebb7",  # light green
            "#f7e294",  # light yellow
            "#b0d9d6",  # light cyan
            "#d6b19c",  # light brown
            "#bab0ac",  # grey
            "#bab0ac",  # dark grey            
            ]
        
        # Get unique labels
        label_set = list(set(self.label))
        legends = []
        colors = [None]*len(self.label)
        
        # Create legends and colors
        for i in range(len(label_set)):
            legends += [Line2D(
                [0],
                [0],
                marker = 'o',
                color = 'w',
                label = label_set[i],
                markerfacecolor = color_list[i],
                markersize = 12
                )]
            for j in range(len(self.label)):
                if self.label[j] == label_set[i]:
                    colors[j] = color_list[i]

        return legends, colors
    
    
    def score(self, pcx, pcy, conf=False):
        '''
        Scores plot (PCx vs PCy). 
        If conf, plot Hotelling T2 confidence ellipse.

        Parameters
        ----------
        pcx : int
            Which PC will be selected to the x axis.
        pcy : int
            Which PC will be selected to the y axis.
        conf : float, optional
            Ellipse confidence interval (from 0 to 1). The default is False.

        Returns
        -------
        None.

        '''
        
        # Scores Plot
        if self.label:
            legends, colors = self.labeling()
        fig, ax = plt.subplots()
        plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
        plt.axvline(x=0, color='k', linestyle='--', linewidth=1)
        if self.label:
            plt.scatter(
                self.scores[:,pcx-1],
                self.scores[:,pcy-1],
                c=colors,
                s=50,
                edgecolors='w',
                alpha=0.5
                )  
            plt.legend(handles=legends)
        else:
            plt.scatter(
                self.scores[:,pcx-1],
                self.scores[:,pcy-1],

                s=50,
                edgecolors='w',
                alpha=0.5
                )              
        plt.xlabel(f"PC-{pcx} ({np.round(self.variance_ratio[pcx-1]*100,2)}%)")
        plt.ylabel(f"PC-{pcy} ({np.round(self.variance_ratio[pcy-1]*100,2)}%)")
        plt.title('PC Scores')
        
        # Hotelling T2 confidence ellipse
        if conf:            
            theta = np.concatenate(
                (np.linspace(-np.pi, np.pi, 50),
                 np.linspace(np.pi, -np.pi, 50))
                )
            circle = np.array((np.cos(theta), np.sin(theta)))
            sigma = np.cov(
                np.array(
                    (self.scores[:, pcx-1],
                     self.scores[:, pcy-1])
                    )
                )
            pdf = np.sqrt(stats.chi2.ppf(conf, df=2))
            
            ellipse = circle.T.dot(np.linalg.cholesky(sigma)*pdf)
            xmax, ymax = np.max(ellipse[:, 0]), np.max(ellipse[:, 1])
            t = np.linspace(0, 2*np.pi, 100)
            
            plt.plot(xmax*np.cos(t), ymax*np.sin(t), color='red')
            

    def expvar(self):
        '''
        Explained Variance plot (cumulative and individual variances).

        Returns
        -------
        None.

        '''

        fig, ax = plt.subplots()
        plt.xlabel('Number of PCs')
        plt.ylabel('Explained Variance (%)')
        plt.title('PC Explained Variance')
        plt.plot(
            np.arange(1, self.ncomp+1, 1.0),
            self.variance_cumulative*100,
            '-o',
            color='red',
            label='Cumulative'
            )
        plt.bar(
            x=range(1, self.ncomp+1),
            height=self.variance_ratio*100,
            tick_label=np.arange(1, self.ncomp+1, 1.0).astype(int),
            label="Individual"
            )
        plt.ylim(0, 100)
        plt.legend()
        
        
    def loading(self, wn, pc):
        '''
        Loadings plot.

        Parameters
        ----------
        wn : ndarray 
            Wavenumber of shape [n_points].        
        pc : int
            Which PC will be selected to the loading plot.

        Returns
        -------
        None.

        '''
        
        # Loadings plot
        fig, ax = plt.subplots()
        plt.xlabel('Wavenumber ($\mathrm{cm^{-1}}$)')
        plt.ylabel('Loading (a.u.)')
        plt.title(f'PC-{pc} Loadings')
        plt.xlim(np.ceil(wn.max()), np.floor(wn.min()))
        plt.plot(
            wn,
            self.loadings[:,pc-1]
            )
        plt.grid(False) 
        
    
    def hotelling(self, conf, reduced=False, clean=False, show=True):
        '''
        Hotelling's T-squared vs Q residuals Plot. 
        
        Outlier detection and removal.

        Parameters
        ----------
        conf : float
            Confidence interval (from 0 to 1).
        reduced : boolean, optional
            The default is False: regular plot with dashed confidence lines. 
            If True: normalize axes by the confidence and remove dashed lines.
        clean : boolean, optional
            The default is False: do not return outliers or modified spectra. 
            If True: return outliers array, and cleaned spectra and labels.
        show : boolean, optional
            The default is True: show the T squared vs Q residuals plot. 
            If False: do not show the plot.
            
        Returns
        -------
        outliers : boolean, optional (only if clean)
            Array identifying outliers.
        spec_clean : ndarray, optional (only if clean)
            Cleaned spectra of shape [n_spectra, n_points] (outliers removed).
        label_clean : ndarray, optional (only if clean)
            Cleaned labels of shape [n_spectra] (outliers removed).
            
        '''
        
        # Calculate Q resisuals and T-squared
        error = self.spec - np.dot(self.scores, self.loadings.T)
        qres = np.sum(error**2, axis=1)
        tsqr = np.sum((self.scores/np.std(self.scores, axis=0))**2, axis=1)
            
        # T-squared confidence interval
        pdf = stats.f.ppf(
            q=conf,
            dfn=self.ncomp,
            dfd=self.spec.shape[0]
            )        
        tsqr_conf = (pdf
                   * self.ncomp
                   * (self.spec.shape[0]-1)
                   / (self.spec.shape[0]-self.ncomp))
           
        # Q residuals confidence interval
        qres_conf = np.quantile(np.abs(qres), conf)

        # Outliers (above T-squared OR Q residuals)
        outliers = ((tsqr > tsqr_conf) | ((np.abs(qres) > qres_conf)))
            
        # Plot Hotelling T-squared vs Q resisuals
        if show:
            if self.label:
                legends, colors = self.labeling()           
            
            if reduced:
                fig, ax = plt.subplots()
                if self.label:
                    plt.scatter(
                        tsqr/tsqr_conf,
                        qres/qres_conf,
                        c=colors, 
                        s=50, 
                        edgecolors='w', 
                        alpha=0.5
                        )
                    plt.legend(handles=legends)
                else:
                    plt.scatter(
                        tsqr/tsqr_conf,
                        qres/qres_conf,
                        s=50, 
                        edgecolors='w', 
                        alpha=0.5
                        )                
                plt.xlabel("Hotelling's T$\mathrm{^{2}}$ Reduced" \
                           f" ({np.round(self.variance_total*100,2)}%)")
                plt.ylabel(f"Q Residuals Reduced" \
                           f" ({np.round(100-self.variance_total*100,2)}%)")
                plt.xlim(0)
                plt.ylim(0)
            
            else:
                fig, ax = plt.subplots()
                if self.label: 
                    plt.scatter(
                        tsqr,
                        qres,
                        c=colors, 
                        s=50, 
                        edgecolors='w', 
                        alpha=0.5
                        )
                    plt.legend(handles=legends)
                else:
                    plt.scatter(
                        tsqr,
                        qres,
                        s=50, 
                        edgecolors='w', 
                        alpha=0.5
                        )                    
                plt.axhline(y=qres_conf, color='red', linestyle='--', linewidth=1)
                plt.axvline(x=tsqr_conf, color='red', linestyle='--', linewidth=1)
                plt.xlabel("Hotelling's T$\mathrm{^{2}}$" \
                           f" ({np.round(self.variance_total*100,2)}%)")
                plt.ylabel(f"Q Residuals" \
                           f" ({np.round(100-self.variance_total*100,2)}%)")
     
        print(f'{sum(outliers)} outliers found' \
              f' ({np.round((sum(outliers)/self.spec.shape[0])*100, 2)}%' \
                  ' of the total spectra).')
        
        # Clean spec and label outliers    
        if clean:
            from itertools import compress
            
            spec_clean = self.spec[~outliers,:]
            if self.label:
                label_clean = list(compress(self.label, ~outliers))
                return outliers, spec_clean, label_clean
            else:
                return outliers, spec_clean