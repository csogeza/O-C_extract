import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from pygam import LinearGAM, s, f

# To adjust a GAM model for difference in amplitude,
# offset and phase
def adjustedGAM(x, gmf, A, ph, B):
    return gmf.predict(x+ph)*A + B

# To get the index of maximum in list
def get_index_of_max(data):
    return list(data).index(max(data))

# To get the index of minimum in list
def get_index_of_min(data):
    return list(data).index(min(data))

# To calculate a phase curve for a given period and T0
def get_phase_curve(data,T0,P,typ = 'N'):
    ph = np.zeros((data.shape[0],1))
    for i in range (data.shape[0]):
        ph_temp = (data[i]-T0)/P-int((data[i]-T0)/P)
        if typ == 'N':
            if ph_temp < 0:
                ph[i] = ph_temp + 1
            else:
                ph[i] = ph_temp
        else:
            if ph_temp < -0.5:
                ph[i] = ph_temp + 1
            elif ph_temp > 0.5:
                ph[i] = ph_temp - 1
            else:
                ph[i] = ph_temp
    return ph

# To extend the phase curve into the -1:2 range
def phase_curve_extender(ph, mags):
    extended_phase = np.zeros((2*ph.shape[0],1))
    extended_phase[:ph.shape[0]] = ph.copy()
    extended_phase[ph.shape[0]:] = ph.copy() - 1
    
    extended_mags = np.append(mags, mags)
    
    extended_phase[extended_phase < -0.5] += 2
    
    return extended_phase, extended_mags

# Normalize the phase curve to the -1:1 range
def normalizer(data):
    X_std = (data - min(data)) / (max(data) - min(data))
    return X_std * 2 - 1
	
# Find the period, fit with GAM, throw outliers, then refit
def GAMfitter(indir, dat_st, T0 = None):
    fname = [i for i in os.listdir(indir) if dat_st in i]
    data = np.loadtxt(indir + fname[0])
    frequency = np.linspace(1e-3,0.5,int(1e6)) # A range for frequencies (2 to 1000 day periods)
    power = LombScargle(data[:,0], data[:,1]).power(frequency=frequency) # Get spectrum
    ind = get_index_of_max(power)  # Best frequency
    
    if T0 is None:  # If we have no preset T0, try to get a minimum
        phs = get_phase_curve(data[:,0], data[0,0], 1/frequency[ind])  
        ext_phs, ext_mags = phase_curve_extender(phs, data[:,1])
        gam = LinearGAM(n_splines=30).gridsearch(ext_phs, ext_mags) # Fit a GAM

        XX = gam.generate_X_grid(term=0, n=500)
        fit = gam.predict(XX)  # This is the fit on the grid
        minimal_val = max(fit) # Maximum magnitude (minimal brightness)
        min_ind = get_index_of_min(abs(data[:,1] - minimal_val))
        T0 = data[min_ind,0]
        
    phs = get_phase_curve(data[:,0], T0, 1/frequency[ind])
    ext_phs, ext_mags = phase_curve_extender(phs, data[:,1])
    gam = LinearGAM(n_splines=30).gridsearch(ext_phs, ext_mags)
    
    pred_int_vls = gam.prediction_intervals(phs, width=.85)
    cond = (data[:,1] > pred_int_vls[:,0]) & (data[:,1] < pred_int_vls[:,1])
    
    filtered_data = data[cond]
    
    power_f = LombScargle(data[:,0], data[:,1]).power(frequency=frequency)
    ind_f = get_index_of_max(power_f)
    phs_f = get_phase_curve(filtered_data[:,0], T0, 1/frequency[ind_f])
    ext_phs, ext_mags = phase_curve_extender(phs_f, filtered_data[:,1])
    gam_f = LinearGAM(n_splines=30).gridsearch(ext_phs, ext_mags)
    
    return filtered_data, gam_f, frequency[ind_f], T0

# Plot the resulting fits
def plot_results(orig_data, filt_data, gam, F0, label = 'None'):
    V_phs = get_phase_curve(orig_data[:,0], T0, 1/F0)
    V_filt_phs = get_phase_curve(filt_data[:,0], T0, 1/F0)
    
    plt.scatter(V_phs, orig_data[:,1])
    plt.scatter(V_filt_phs, filt_data[:,1], label = label)
    plt.gca().invert_yaxis()
    XX = gam.generate_X_grid(term=0, n=500)
    plt.plot(XX, gam.predict(XX), color='royalblue', ls = '--', lw = 0.7, label = 'GAM model')
    plt.plot(XX, gam.prediction_intervals(XX, width=.85), color='b', ls='--')
    if label != 'None':
        plt.legend()