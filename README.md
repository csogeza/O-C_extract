# Data repository for the code and results of the Csörnyei et al. (2022) paper

Structure of the database:

Csornyei2021_data -- main data folder with the outputs of the paper stored in various tables:

    - AllOCFiles -- the O-C diagrams of the individual Cepheids in separate txt files

    - AllOCPlots -- individual O-C diagram figures with evolutionary fits
    
    - CepheidProperties.txt -- main output table of the work. Columns:
    
        - Name: GCVS name of the Cepheid
        
        - Period_O-C [d]: assumed period used for the calculation of the O-C diagram
        - ReferenceEp_O-C: reference epoch assumed for O-C calculation
        - Period_HJD2459591.5: the pulsation period of the Cepheid on the epoch of HJD 2459591.5 (calculated from the O-C diagram by propagating the period change into the assumed period)
        - Period change [d/100yr]: period change rate obtained from the O-C fit (NOT normalised by the period)
        - Period change error [d/100yr]: error of the period change values
        - g: Gaia eDR3 g magnitude
        - BP-RP: Gaia eDR3 BP-RP color (non-dereddened)
        - E(B-V): assumed reddening
        - E(B-V) error: error of assumed reddening
        - log10 Epsilon: logarith of the Epsilon period fluctuation parameter of the Eddington-Plakidis method
        - Parallax [mas]: Gaia eDR3 parallax (not corrected for zero point error)
        - Parallax error [mas]: error of Gaia eDR3 parallax
        - Reference: list of photometric data used for this Cepheid, see Phot_reference_list.txt for more details
        
    - F-test_results.txt -- summary table of the conducted F-test. Columns:
        - Cepheid: GCVS name
        - F_lp: F-statistics value calculated for the comparison of the linear and parabolic model
        - F_lw: F-statistics value calculated for the comparison of the linear and parabolic+wave model
        - F_pw: F-statistics value calculated for the comparison of the parabolic and parabolic+wave model
        - Ampl [days]: found amplitude of the O-C "wavelike" fluctuation signal, based on the joint parabolic+wave fit
        
    - Phot_reference_list.txt -- list of photometric references and codes used in the CepheidProperties table
    - Piszkes_photometry_all.dat -- photometric time series obtained at Piszkéstető Mountain Station for some of the Cepheids

# O-C extract
O-C diagram calculator code for periodic variable stars
v.0.1.

A python based code for the extraction of O-C diagrams used for the analyis on Classical Cepheid variable stars as described in Csörnyei et al. (2022) (https://arxiv.org/abs/2201.04748). The code itself requires a lot of by-hand-tuning, and is not compatible with the up-to-date version of python at the moment, due to the changes in the subprocess module. This repository presently serves only as a reference for the analysis conducted for the article, however the code itself will be patched up in the near future. If you plan to run this code in its present from, then the following packages should be installed:

- Python 3.6.5

- Astropy

- Pandas

- scipy

- PyAstronomy

- numpy

- matplotlib
