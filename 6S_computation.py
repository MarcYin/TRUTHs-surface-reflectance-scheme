import numpy as np
from total_gases import gas_trans

def get_solar(wl, solar, month = 1, jday = 1):
    '''
    Get solar irradiance given wavelength for a specific date

    Parameters
    ----------
    wl : 1D array
        solar irradiance for wavelengths (micro) 
    solar: 1D array
        solar LUT for wavelength starting at 0.25 micro
    month: scaler
        month of year
    day: scaler
        day of month
    Returns
    -------
    swl: 1D array
        solar irradiance for wl
    '''
    
    
    wl = np.atleast_1d(wl)
    
    pas = 0.0025
    iwl = ((wl - 0.250) / pas + 1.5).astype(int)
    swl = solar[iwl-1]

    if month <=2:
        j = 31 * (month - 1) + jday
    elif month >8:
        j = 31 * (month - 1) - ((month - 2) / 2) - 2 + jday
    else:
        j = 31 * (month - 1) - ((month - 1) / 2) - 2 + jday

    om = (0.9856 * (j - 4)) * np.pi / 180.
    dsol = 1. / ((1. - 0.01673 * np.cos(om))**2)
    swl = dsol * swl
    
    return swl

def compute_6s_coefs(wv_start, wv_end, bandpass, uw, uo3, sza, vza, alt, luta_ind, gas_full_tables, us62_atmosphere_profile, solar, roray, romix, sast, t_dwn_up):

    step = 0.0025
    
    wldis = 0.35, 0.40, 0.4125, 0.4425, 0.47, 0.4875, 0.515, 0.55, 0.59, 0.6325, 0.67, 0.695, 0.76, 0.86, 1.24, 1.535, 1.65, 1.95, 2.25, 3.75
    
    wldis = np.array(wldis)

    iinf = (wv_start-.25)/step+1.5
    isup = (wv_end  -.25)/step+1.5        
    
    wl = (np.arange(int(iinf), int(isup) + 1)  - 1) * step + .25
    
    swl = get_solar(wl, solar)
    sbor = bandpass

    coefs = sbor * step * swl

    tgtot      = gas_trans(wl, alt, uw, uo3, sza, vza, gas_full_tables.copy(), us62_atmosphere_profile.copy())
    tgtot_half = gas_trans(wl, alt, uw / 2, uo3, sza, vza, gas_full_tables.copy(), us62_atmosphere_profile.copy())
    
    tgtot_except_water = tgtot[1] * tgtot[2] * tgtot[3] * tgtot[4] * tgtot[5] * tgtot[6]
    
    tgp1 = tgtot_except_water             
    tgp2 = tgtot_half[0] * tgtot_except_water
    tgtot = tgtot[0] * tgtot_except_water
    
    mask = (wl[:, None] <= wldis[None,])
    
    lsup = np.maximum(np.argmax(mask, axis=1), 1)
    linf = lsup - 1
    
    coef = np.log(wldis[lsup] / wldis[linf])
    wlinf = wldis[linf]

    alphar = np.log(roray[luta_ind, lsup] / roray[luta_ind, linf]) / coef
    betar = roray[luta_ind, linf] / (wlinf**alphar)
    rorayl=betar * wl**alphar

    alphar = np.log(romix[luta_ind, lsup] / romix[luta_ind, linf]) / coef
    betar = romix[luta_ind, linf] / (wlinf**alphar)
    romixl=betar * wl**alphar


    alphar = np.log(sast[luta_ind, lsup] / sast[luta_ind, linf]) / coef
    betar = sast[luta_ind, linf] / (wlinf**alphar)
    sastl = betar * wl**alphar

    alphar = np.log(t_dwn_up[luta_ind, lsup] / t_dwn_up[luta_ind, linf]) / coef
    betar = t_dwn_up[luta_ind, linf] / (wlinf**alphar)
    t_dwn_upl = betar * wl**alphar

    ratm1 = (romixl - rorayl) * tgtot + rorayl * tgp1
    ratm3 =  romixl * tgp1
    ratm2 = (romixl - rorayl) * tgp2  + rorayl * tgp1
    
    
    ratm1 = np.sum(ratm1 * coefs) / coefs.sum()
    ratm2 = np.sum(ratm2 * coefs) / coefs.sum()
    ratm3 = np.sum(ratm3 * coefs) / coefs.sum()
    tgtot = (tgtot * coefs).sum() / coefs.sum()
    
    return np.array([ratm1, ratm2, ratm3]), tgtot, sastl, t_dwn_upl

def test(sza = 10, vza = 15, luta_ind = 100):
    # read in 6S LUT
    f = np.load('atmospheric_transmittance_LUT.npz')
    gas_full_tables = np.array(f.f.gas_full_tables)
    us62_atmosphere_profile = np.array(f.f.us62_atmosphere_profile)
    solar = f.f.solar
    
    # use solar and view angle for LUTA and LUTB filename
    # read in LUT A version 3
    fname = '/home/users/marcyin/nceo_ard/mix_000_V_1013_TRUTHSA_v0.3/LUT_TRUTHS_000_1013_%04.1f_%04.1f'%(sza, vza)
    with open(fname, 'r') as f:
        txt = f.read()
    txt = txt.split('\n')

    raa_aot2  = txt[11::6]
    t_dwn_up = np.array([i.split() for i in txt[12::6]]).astype(float)
    sast     = np.array([i.split() for i in txt[13::6]]).astype(float)
    romix    = np.array([i.split() for i in txt[14::6]]).astype(float)
    roray    = np.array([i.split() for i in txt[15::6]]).astype(float)
    roaero   = np.array([i.split() for i in txt[16::6]]).astype(float)
    
    # read in LUT B version 2
    fname = '/home/users/kjpearson/LUT_TRUTHS_test/mix_000_V_1013_TRUTHSB_v0.2/LUT_TRUTHSB_000_1013_%04.1f_%04.1f'%(sza, vza)
    with open(fname, 'r') as f:
        txt = f.read()
    txt = txt.split('\n')
    raa_aot1  = txt[11::6]
    atmo_path = np.array([i.split() for i in txt[12::6]]).astype(float)
    tgas      = np.array([i.split() for i in txt[13::6]]).astype(float)
    
    # I have found the index for LUTB is half of LUTA
    # which caused by the reduced resolution of LUTB
    lutb_ind = int(luta_ind / 2)
    
    # set up the wavelength for TRUTHs
    # ranging from 0.35 to 2.5 micro
    wv = np.arange(0.350, 2.50 + 0.01, 0.01)
    
    # test altidude 0
    alt = 0.0
    # set water vapour and ozone to match
    # LUTA and LUTB
    uw, uo3 = 2.5, 0.35
    
    print(raa_aot2[luta_ind], 'sza: ', sza, 'vza: ', vza)
    
    ratms = []
    tgas1 = []
    
    # set the bands corresponding to LUTB
    # wv +-0.005
    step = 0.0025
    for i in range(len(wv)):    
        wv_start, wv_end = wv[i] - 0.005, wv[i] + 0.005
        iinf = (wv_start-.25)/step+1.5
        isup = (wv_end  -.25)/step+1.5        
        
        
        bandpass = np.ones(int(isup) - int(iinf) + 1)
        
        # 6S internal change the first and last bandpass to 0.5
        # when mono bandpass used
        bandpass[0] = 0.5
        bandpass[-1] = 0.5
        
        # compute the atmosphere path reflectance, total gas absorption transmittance
        # spherical albedo, total scattering transmittance for up and down
        ratm, tgtot, sastl, t_dwn_upl = compute_6s_coefs(wv_start, wv_end, bandpass, uw, uo3, sza, vza, alt, luta_ind, gas_full_tables, us62_atmosphere_profile, solar, roray, romix, sast, t_dwn_up)
        
        # get atmosphere path reflectance, total gas absorption transmittance
        # for comparison with LUTB
        
        ratms.append(ratm)
        tgas1.append(tgtot)
    ratms = np.array(ratms)
    tgas1 = np.array(tgas1)
    
    gas_diff = tgas[0][:len(wv)] - tgas1
    ratm_diff = ratms[:, 1] - atmo_path[lutb_ind][:len(wv)]
    
    print('Gas absorption transmittance mean absolute difference: ', abs(gas_diff).mean())
    print('Atmospheric path reflectance mean absolute difference: ',abs(ratm_diff).mean())
    
    # plot the comparison
    from matplotlib import gridspec
    import pylab as plt
    plt.rc('font', size=22) 
    fig = plt.figure(figsize=(24,8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax0.plot(wv[:len(wv)], tgas[0][:len(wv)], '-+', lw=1, label = '6S transmittance')
    ax0.plot(wv[:len(wv)], tgas1, '--s', lw=1, ms=3, mew=0.5, mfc='none', label = 'Python code transmittance')
    ax0.plot(wv[:len(wv)], tgas[0][:len(wv)] - tgas1, '-', ms=3, lw=2, label = 'Difference')
    # ax0.legend()

    ax0.plot(wv[:len(wv)], atmo_path[lutb_ind][:len(wv)], '-o',lw=1, ms=4, mew=0.5, mfc='none',label = 'LUTB path reflectance')
    ax0.plot(wv[:len(wv)], ratms[:, 0], '-', label = 'Python code path reflectance 1')
    ax0.plot(wv[:len(wv)], ratms[:, 1], '-+', label = 'Python code path reflectance 2')
    ax0.plot(wv[:len(wv)], ratms[:, 2], '-', label = 'Python code path reflectance 3')
    ax0.plot(wv[:len(wv)], ratms[:, 1] - atmo_path[lutb_ind][:len(wv)], '-', ms=3, lw=2, label = 'Difference')
    

    ax1.plot(atmo_path[lutb_ind][:len(wv)], ratms[:, 1], 'o', ms=5)
    ax1.set_xlabel('6S path reflectance')
    ax1.set_ylabel('python code path reflectance')
    # plt.xlim(0.2, 1.)

    ax1.plot(tgas[0][:len(wv)], tgas1, 'o')
    ax1.set_xlabel('6S transmittance')
    ax1.set_ylabel('python code transmittance')
    
    ax0.legend(loc=10)
    
    title = ' '.join([raa_aot2[luta_ind], 'sza:', str(sza), 'vza:', str(vza)])
    fig.suptitle(title, fontsize="x-large")

if __name__ == '__main__':
    # test at arange of sun and view angles
    # with some different aot and raa 
    luta_inds = np.arange(0, 240, 20).reshape(4,3)
    for i, sza in enumerate(np.arange(0, 80, 20)):
        for j, vza in enumerate(np.arange(0, 60, 20)):
            test(sza, vza, luta_ind=luta_inds[i, j]) 