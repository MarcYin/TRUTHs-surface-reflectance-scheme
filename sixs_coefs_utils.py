import numpy as np
from numba import jit, njit

@njit()
def representative_bands_interpolation_jac(a, b, d_ax, d_bx, c, i, w):

    '''
    Interpolation of 20 representative bands to target wavelength based
    
    Parameters
    ----------
    a: 3D array 
        val[lsup] right end of parameters within the wavelength interval
    b: 3D array  
        val[linf] left end of parameters within the wavelength interval
    c: 1D array
        np.log(wldis[lsup] / wldis[linf])
    i: 1D array
        wlinf left end of the wavelength interval
    w: 1D array
        wl wavelength of the bandpass
    d_ax: 3D array
        jacobian of right end val to input parameters
    d_bx: 3D array
        jacobian of left end val to input parameters
    
    Returns
    -------
    out: 3D array
        interpolated val for different wavelengths(w) 
    d_out: 3D array
        jacobian of interpolated val for different wavelengths(w) 
    '''
        
    c = np.expand_dims(np.expand_dims(c, axis=1), axis=2)
    i = np.expand_dims(np.expand_dims(i, axis=1), axis=2)
    w = np.expand_dims(np.expand_dims(w, axis=1), axis=2)
    
#     i = np.expand_dims(i, axis=(1,2))
#     w = np.expand_dims(w, axis=(1,2))
    
#     c = c1[:, None, None]
#     i = i1[:, None, None]
#     w = w1[:, None, None]
    
    alpha = np.log(a/b)/c
    
    part1 = 1 / i**alpha * w**alpha
    
    out = b * part1
    
    log_diff = np.log(i) - np.log(w)
    
    ca = c * a
    
    d_out = part1 * (ca * d_bx + (a * d_bx - b * d_ax) * (log_diff)) / ca 
    
    return out, d_out

@njit()
def compute_tgtot_except_water(iv, v, tco3, gases_u, gases_up, gases_uu, gases_a, xmus, xmuv):
    '''
    compute the gas absorption except water
    
    Parameters
    ----------
    iv: 1D array
        6S equivalent wavenumber
    v: 1D array
        6S equivalent wavelength
    tco3: 1D array
        total column of ozone
    gases_u: 3D array
        gases absorption coefficients
    gases_up: 3D array
        gases absorption coefficients
    gases_uu: 2D array
        gases absorption coefficients
    gases_a: 3D array
        gases absorption coefficients
    xmus: 1D array
        cosine of sun zenith angle
    xmuv: 1D array
        cosine of sun zenith angle
    
    return
    ------
    tt_other * tt_ozone: 2D array
        total gas absorption except water vapour
    
    '''
    tt_other = compuate_other_gas_trans(gases_u, gases_up, gases_uu, gases_a, xmus, xmuv)
    tt_ozone = compute_atmo_trans_ozone(tco3, iv, v, gases_u[:, :, 3], gases_up[:, :, 3], gases_uu[:, 3], gases_a[:,3], xmus, xmuv)
    return tt_other * tt_ozone

@njit()
def compute_tg_total_and_jac(iv, v, tcwv, gases_u, gases_up, gases_uu, gases_a, xmus, xmuv, tgtot_except_water):
    
    '''
    compute total gas absorption and jacobian
    
    Parameters
    ----------
    iv: 1D array
        6S equivalent wavenumber
    v: 1D array
        6S equivalent wavelength
    tcwv: 1D array
        total column of water vapour
    gases_u: 3D array
        gases absorption coefficients
    gases_up: 3D array
        gases absorption coefficients
    gases_uu: 2D array
        gases absorption coefficients
    gases_a: 3D array
        gases absorption coefficients
    xmus: 1D array
        cosine of sun zenith angle
    xmuv: 1D array
        cosine of sun zenith angle
    tgtot_except_water: 2D array
        total gas absorption except water vapour
    
    return
    ------
    tgp1: 2D array
        total gas absorption except water vapour
    tgp2: 2D array
        total gas absorption with half of water vapour
    tgtot: 2D array
        total gas absorption with full water vapour
    tgp2_jac: 2D array
        jacobian of total gas absorption with half of water vapour to water vapour
    tgtot_jac: 2D array
        jacobian of total gas absorption with full water vapour to water vapour
    '''
        
    tg_water,      tg_water_jac      = compute_atmo_trans_wv(tcwv,   gases_u[:, :, 0], gases_up[:, :, 0], gases_a[:,0], xmus, xmuv)
    tg_water_half, tg_water_half_jac = compute_atmo_trans_wv(tcwv/2, gases_u[:, :, 0], gases_up[:, :, 0], gases_a[:,0], xmus, xmuv)

    
    tgp1  = tgtot_except_water             
    tgp2  = tg_water_half * tgtot_except_water
    tgtot = tg_water  * tgtot_except_water
    
    tgp2_jac  = tg_water_half_jac / 2 * tgtot_except_water
    tgtot_jac = tg_water_jac * tgtot_except_water
    
    return np.asarray(tgp1).astype(np.float32) , np.asarray(tgp2).astype(np.float32) , np.asarray(tgtot).astype(np.float32) , np.asarray(tgp2_jac).astype(np.float32), np.asarray(tgtot_jac).astype(np.float32)

#     return tgp1, tgp2, tgtot, tgp2_jac, tgtot_jac


def run_emus(inps, scale, mix):
    
    '''
    Run emulators for inputs with different aerosol mixture
    
    Parameters
    ----------
    inps: 2D array
        atmospheric parameters inputs ['PRES', 'SZA', 'VZA', 'RAZ', 'AOT', 'WV', 'O3']
    scale: 1D array
        scales for emulation outputs
    mix: scalar int
        aerosol mixture type
    
    Return
    ------
    romix_val: 3D array
        path reflectance of mixed aerosol and Rayleigh scattering for 20 representative bands
    roray_val: 3D array
        path reflectance of pure Rayleigh scattering for 20 representative bands
    sast_val: 3D array
        spherical albedo of atmosphere for 20 representative bands
    t_tot_val: 3D array
        total scattering transmittance of atmosphere for 20 representative bands
    
    romix_adj: 3D array
        jacobian of path reflectance of mixed aerosol and Rayleigh scattering for 20 representative bands to aot and tcwv
    roray_adj: 3D array
        jacobian of path reflectance of pure Rayleigh scattering for 20 representative bands to aot and tcwv
    sast_adj: 3D array
        jacobian of spherical albedo of atmosphere for 20 representative bands to aot and tcwv
    t_tot_adj: 3D array
        jacobian of total scattering transmittance of atmosphere for 20 representative bands to aot and tcwv
        
    '''
    
    t_tot_val, t_tot_adj = predict_select_jac(inps, t_down_up_arrModel[mix], cal_jac=True, selected_jac=[4, 5]) 
    t_tot_val = t_tot_val.T[:, :, None]
    t_tot_adj = t_tot_adj.T

    
    romix_val, romix_adj  = predict_select_jac(inps, romix_arrModel[mix], cal_jac=True, selected_jac=[4, 5])
    romix_val = romix_val.T[:, :, None]
    romix_adj = romix_adj.T
    
    
    roray_val, roray_adj = predict_select_jac(inps, roray_arrModel[mix], cal_jac=True, selected_jac=[4, 5])
    roray_val = roray_val.T[:, :, None]
    roray_adj = roray_adj.T
    
    sast_val, sast_adj = predict_select_jac(inps, sast_arrModel[mix], cal_jac=True, selected_jac=[4, 5])
    sast_val = sast_val.T[:, :, None]
    sast_adj = sast_adj.T
    
    t_tot_val = t_tot_val * scale[0][:, None, None]
    romix_val = romix_val * scale[1][:, None, None]
    roray_val = roray_val * scale[2][:, None, None]
    sast_val  = sast_val  * scale[3][:, None, None]

    
    t_tot_adj = t_tot_adj * scale[0][:, None, None]
    romix_adj = romix_adj * scale[1][:, None, None]
    roray_adj = roray_adj * scale[2][:, None, None]
    sast_adj  = sast_adj  * scale[3][:, None, None]

    return [np.asarray(romix_val).astype(np.float32) , np.asarray(roray_val).astype(np.float32) , np.asarray(sast_val).astype(np.float32) , np.asarray(t_tot_val).astype(np.float32),  
            np.asarray(romix_adj).astype(np.float32) , np.asarray(roray_adj).astype(np.float32) , np.asarray(sast_adj).astype(np.float32) , np.asarray(t_tot_adj).astype(np.float32) ]

    return [romix_val.block_until_ready() , roray_val.block_until_ready() , sast_val.block_until_ready() , t_tot_val.block_until_ready(),  
            romix_adj.block_until_ready() , roray_adj.block_until_ready() , sast_adj.block_until_ready() , t_tot_adj.block_until_ready() ]
    
@njit()
def compute_coefs(linf, lsup, coef, wlinf, wl, coefs, tgp1, tgp2, tgtot, romix_val, roray_val, sast_val, t_tot_val, tgp2_jac, tgtot_jac, romix_adj, roray_adj, sast_adj, t_tot_adj):
    '''
    Compute 6S coefficients for Lambertian atmospheric correction for a bandpass function
    
    Parameters
    ----------
    linf: scalar int 
        lower limit of wavelength in band pass function 
    lsup: scalar int
        upper limit of wavelength in band pass function 
    coef: 1D array
        np.log(wldis[lsup] / wldis[linf])
    wlinf: 1D array
        left end of the wavelength interval
    wl: 1D array
        wl wavelength of the bandpass
    coefs: 1D array
        band pass weights multiply solar irradiance
    tgp1: 2D array
        total gas absorption except water vapour
    tgp2: 2D array
        total gas absorption with half of water vapour
    tgtot: 2D array
        total gas absorption with full water vapour
    romix_val: 3D array
        path reflectance of mixed aerosol and Rayleigh scattering for 20 representative bands
    roray_val: 3D array
        path reflectance of pure Rayleigh scattering for 20 representative bands
    sast_val: 3D array
        spherical albedo of atmosphere for 20 representative bands
    t_tot_val: 3D array
        total scattering transmittance of atmosphere for 20 representative bands
    tgp2_jac: 2D array
        jacobian of total gas absorption with half of water vapour to water vapour
    tgtot_jac: 2D array
        jacobian of total gas absorption with full water vapour to water vapour
    romix_adj: 3D array
        jacobian of path reflectance of mixed aerosol and Rayleigh scattering for 20 representative bands to aot and tcwv
    roray_adj: 3D array
        jacobian of path reflectance of pure Rayleigh scattering for 20 representative bands to aot and tcwv
    sast_adj: 3D array
        jacobian of spherical albedo of atmosphere for 20 representative bands to aot and tcwv
    t_tot_adj: 3D array
        jacobian of total scattering transmittance of atmosphere for 20 representative bands to aot and tcwv    
    
    Return
    ------
    tgtot: 1D array
        total gas absorption with full water vapour
    ratm2: 2D array
        atmospheric path reflectance
    t_totl: 2D array
        atmospheric total scattering transmittance
    sastl: 2D array
        atmospheric spherical albedo
    tgtot_jac: 2D array
        jacobian of total gas absorption with full water vapour to aot and tcwv
    ratm2_jac: 2D array
        jacobian of atmospheric path reflectance to aot and tcwv
    t_tot_adj: 2D array
        jacobian of atmospheric spherical albedo to aot and tcwv
    sast_adj: 2D array
        jacobian of atmospheric spherical albedo to aot and tcwv
    '''
    sastl,   sastl_adj = representative_bands_interpolation_jac(sast_val [lsup],  sast_val[linf], sast_adj [lsup],  sast_adj[linf], coef, wlinf, wl)
    t_totl, t_totl_adj = representative_bands_interpolation_jac(t_tot_val[lsup], t_tot_val[linf], t_tot_adj[lsup], t_tot_adj[linf], coef, wlinf, wl)
            
    romixl, romixl_adj = representative_bands_interpolation_jac(romix_val[lsup], romix_val[linf], romix_adj[lsup], romix_adj[linf], coef, wlinf, wl)
    rorayl, rorayl_adj = representative_bands_interpolation_jac(roray_val[lsup], roray_val[linf], roray_adj[lsup], roray_adj[linf], coef, wlinf, wl)
    
    
    tgp1 = np.expand_dims(tgp1.T, axis=2)
    tgp2 = np.expand_dims(tgp2.T, axis=2)
    tgtot = np.expand_dims(tgtot.T, axis=2)
    tgtot_jac = np.expand_dims(tgtot_jac.T, axis=1)
    
    ratm2 = (romixl - rorayl) * tgp2  + rorayl * tgp1
    
    temp = np.zeros_like(romixl_adj)
    temp[:, :, 1] = tgp2_jac.T * 8
    ratm2_adj = (romixl_adj - rorayl_adj) * tgp2 + rorayl_adj * tgp1 + (romixl - rorayl) * temp
    
    coefs = np.expand_dims(np.expand_dims(coefs, axis=1), axis=2)
    
    
    ratm2 = np.sum(ratm2 * coefs, axis=0) / coefs.sum()
    tgtot = np.sum(tgtot * coefs, axis=0) / coefs.sum()
    sastl = np.sum(sastl * coefs, axis=0) / coefs.sum()
    t_totl = np.sum(t_totl * coefs, axis=0) / coefs.sum()
    
    
    ratm2_jac = np.sum(ratm2_adj * coefs, axis=0) / coefs.sum()
    tgtot_jac = np.sum(tgtot_jac * coefs, axis=0) / coefs.sum() * 8
    temp = np.zeros_like(ratm2_jac)
    
    temp[:, 1] = tgtot_jac
    tgtot_jac = temp
    
    
    ratm2_jac = np.sum(ratm2_adj  * coefs, axis=0) / coefs.sum()
    sast_adj  = np.sum(sastl_adj  * coefs, axis=0) / coefs.sum()
    t_tot_adj = np.sum(t_totl_adj * coefs, axis=0) / coefs.sum()
    
    return tgtot, ratm2, t_totl, sastl, tgtot_jac, ratm2_jac, t_tot_adj, sast_adj
    
def get_toa(R, tgtot, ratm2, t_totl, sastl, tgtot_jac, ratm2_jac, t_tot_adj, sast_adj):

    '''
    Compute 6S Lambertian TOA reflectance
    
    Parameters
    ----------
    R: scalar float
        surface reflectance
    tgtot: 1D array
        total gas absorption with full water vapour
    ratm2: 2D array
        atmospheric path reflectance
    t_totl: 2D array
        atmospheric total scattering transmittance
    sastl: 2D array
        atmospheric spherical albedo
    tgtot_jac: 2D array
        jacobian of total gas absorption with full water vapour to aot and tcwv
    ratm2_jac: 2D array
        jacobian of atmospheric path reflectance to aot and tcwv
    t_tot_adj: 2D array
        jacobian of atmospheric spherical albedo to aot and tcwv
    sast_adj: 2D array
        jacobian of atmospheric spherical albedo to aot and tcwv
        
    Return
    ------
    
    toa: 2D array
        TOA reflectance
    dJ: 2D array
        jacobian of TOA reflectance to aot and tcwv
    '''
    part2 = R / (1 - R * sastl)
    part3 =  ratm2 + part2 * t_totl
    toa =  tgtot * part3
    
    dJ = part3 * tgtot_jac + (part2**2 * t_totl * sast_adj + part2 * t_tot_adj + ratm2_jac) * tgtot[:, None]

    return toa, dJ

@njit()
def compuate_other_gas_trans(gases_u, gases_up, gases_uu, gases_a, xmus, xmuv):
    
    '''
    Compute total atmosphere transmittance of other gases given 
    cos(sza), cos(vza) with altidude adjusted atmosphere.
    
    Parameters
    ----------
    gases_u: 3D array
        gases absorption coefficients
    gases_up: 3D array
        gases absorption coefficients
    gases_uu: 2D array
        gases absorption coefficients
    gases_a: 3D array
        gases absorption coefficients
    xmus: 1D array
        cosine of sun zenith angle
    xmuv: 1D array
        cosine of sun zenith angle
    
    Returns
    -------
    tt_other_gases: 1D array
        total gas absorption transmittance for other gases except tcwv and ozone for all wavelengths
    '''
    
    xmus = np.expand_dims(xmus, axis=1)
    xmuv = np.expand_dims(xmuv, axis=1)
    
    gas_inds = [1, 2, 4, 5, 6]
    tt_other_gases = np.ones(gases_u.shape[:2])
    accu   = 1.e-10
    
    for gas_ind in gas_inds:
        
        u, up, uu, a = gases_u[:, :, gas_ind], gases_up[:, :, gas_ind], gases_uu[:, gas_ind], gases_a[:, gas_ind]
        
        upl  = u
        uppl = up

        atest = a[:, 1]

        atest = a[:, 1]
        mask1 = (a[:, 1] == 0) & (a[:, 0] == 0)

        atest[mask1] = 1

        ut  = u  / xmus +  upl / xmuv
        upt = up / xmus + uppl / xmuv
        utt = ut

        mask = (ut == 0) & (upt == 0)
        
#         utt[mask] = 1.
        utt = np.where(mask, 1, utt)
        
        uptt = upt 
#         uptt[mask] = 1.
        uptt = np.where(mask, 1, uptt)

        _a = np.expand_dims(a, axis=0)
        _atest = np.expand_dims(atest, axis=0)
        
        tn = _a[:, :, 1] * upt / (2 * utt)

        tt = 1 + 4 * (_a[:, :, 0] / _atest) * ((ut * ut) / uptt)
        y = -tn * (np.sqrt(tt) - 1)

        tt_other = np.exp(y)

        
        tt_other = np.where(tt_other>accu, tt_other, 0)
        tt_other_gases *= tt_other
        
    return tt_other_gases

@njit()
def compute_atmo_trans_ozone(tco3, iv, v, u, up, uu, a, xmus, xmuv):
    '''
    Compute total atmosphere transmittance of Ozone given 
    total column of Ozone (cm·atm), cos(sza), cos(vza)
    with altidude adjusted atmosphere.
    
    Parameters
    ----------
    tco3: 1D array
        total column of Ozone for whole atmosphere (cm·atm)
    xmus: 1D array
        Cosine of solar zenith angle
    xmuv: 1D array
        Cosine of view zenith angle
    u: 1D array
        integral of u for Ozone for all wavelengths
    up: 1D array
        integral of up for Ozone for all wavelengths
    a: 3D array
        LUT coefficients for Ozone at all wavelengths
        (NO. of wavelengths, 8 LUT coefficients)
    Returns
    -------
    ttwava: 1D array
        total ozone transmittance for all wavelengths
    '''
    
    co3  = np.array([4.50e-03, 8.00e-03, 1.07e-02, 1.10e-02, 1.27e-02, 1.71e-02,
                    2.00e-02, 2.45e-02, 3.07e-02, 3.84e-02, 4.78e-02, 5.67e-02,
                    6.54e-02, 7.62e-02, 9.15e-02, 1.00e-01, 1.09e-01, 1.20e-01,
                    1.28e-01, 1.12e-01, 1.11e-01, 1.16e-01, 1.19e-01, 1.13e-01,
                    1.03e-01, 9.24e-02, 8.28e-02, 7.57e-02, 7.07e-02, 6.58e-02,
                    5.56e-02, 4.77e-02, 4.06e-02, 3.87e-02, 3.82e-02, 2.94e-02,
                    2.09e-02, 1.80e-02, 1.91e-02, 1.66e-02, 1.17e-02, 7.70e-03,
                    6.10e-03, 8.50e-03, 6.10e-03, 3.70e-03, 3.20e-03, 3.10e-03,
                    2.55e-03, 1.98e-03, 1.40e-03, 8.25e-04, 2.50e-04, 0.      ,
                    0.      , 0.      , 5.65e-04, 2.04e-03, 7.35e-03, 2.03e-02,
                    4.98e-02, 1.18e-01, 2.46e-01, 5.18e-01, 1.02e+00, 1.95e+00,
                    3.79e+00, 6.65e+00, 1.24e+01, 2.20e+01, 3.67e+01, 5.95e+01,
                    8.50e+01, 1.26e+02, 1.68e+02, 2.06e+02, 2.42e+02, 2.71e+02,
                    2.91e+02, 3.02e+02, 3.03e+02, 2.94e+02, 2.77e+02, 2.54e+02,
                    2.26e+02, 1.96e+02, 1.68e+02, 1.44e+02, 1.17e+02, 9.75e+01,
                    7.65e+01, 6.04e+01, 4.62e+01, 3.46e+01, 2.52e+01, 2.00e+01,
                    1.57e+01, 1.20e+01, 1.00e+01, 8.80e+00, 8.30e+00, 8.60e+00])
    
    uo3us = 0.344
    
    gas_ind = 3
    uo3 = np.atleast_1d(tco3)
    
    rat = tco3/uo3us
    
    
#     uu = uu[:, None] * rat[:, None]
#     u  = u  * rat[:, None]
#     up = up * rat[:, None]
    
    uu = np.expand_dims(uu, axis=1)
    rat = np.expand_dims(rat, axis=1)
    
    xmus = np.expand_dims(xmus, axis=1)
    xmuv = np.expand_dims(xmuv, axis=1)
    
    uu = uu * rat
    u  = u  * rat
    up = up * rat

    
    ttco3 = np.ones((len(tco3), iv.shape[0]))

    uupl = uu
    uut = uu / xmus + uupl / xmuv 
    
    
    mask = (iv <= 23400) & (iv >= 13000)
    xi = (v[mask] - 13000.) / 200. + 1.
    n  = (xi + 1.001).astype(np.int_)
    xd = xi - n
    
    ako3 = co3[n-1] + xd * (co3[n-1] - co3[n-2])    
    
#     test3 = ako3[None, :] * uut
    ako3 = np.expand_dims(ako3, axis=0)
    test3 = ako3 * uut
    test3 = np.where(test3 > 86.00, 86.0, test3)

    ttco3[:, mask] = np.exp(-1 * test3)
    
    
    mask = (iv >= 27500) & (iv <= 50000)
    xi=(v[mask] - 27500.) / 500. + 57.
    n = (xi + 1.001).astype(np.int_)
    
    xd = xi - n
    ako3 = co3[n-1] + xd * (co3[n-1] - co3[n-2])
    
    ako3 = np.expand_dims(ako3, axis=0)
    test3 = ako3 * uut
    
    test3 = np.where(test3 > 86.00, 86.0, test3)
    
    ttco3[:, mask] = np.exp(-1 * test3)
    
    
    # mask = (iv >= 13000) & (iv <= 50000) 
    # mask = (~mask) | ((iv > 23400) & (iv < 27500))

    # mask = mask & (iv > 3020)
    # ttco3[:, mask] = 1

    
    mask = (iv >= 2350) * (iv <=3020) 
    if mask.sum() > 0:
        accu   = 1.e-10    
        upl  = u[:, mask]
        uppl = up[:, mask]


        atest = a[mask, 1]
        atest = a[mask, 1]

#         atest[(a[mask, 1] == 0) & (a[mask, 0] == 0)] = 1
        mask0 = (a[mask, 1] == 0) & (a[mask, 0] == 0)
        atest = np.where(mask0, 1, atest)

        ut  = u [:, mask] / xmus +  upl / xmuv
        upt = up[:, mask] / xmus + uppl / xmuv
        utt = ut

#         utt[(ut == 0) & (upt == 0)] = 1.
        mask1 = (ut == 0) & (upt == 0)
        utt = np.where(mask1, 1, utt)
        

        tn = a[mask, 1] * upt / (2 * utt)

        uptt = upt 
#         uptt[(ut == 0) & (upt == 0)] = 1.
        mask2 = (ut == 0) & (upt == 0)
        uptt = np.where(mask2, 1, uptt)

        tt = 1 + 4 * (a[mask, 0] / atest) * ((ut * ut) / uptt)
        y = -tn * (np.sqrt(tt) - 1)

        test3 = np.exp(y)
        test3 = np.where(test3>accu, test3, 0)
        ttco3[:, mask] = test3
    return ttco3

@njit()
def compute_atmo_trans_wv(tcwv, u, up, a, xmus, xmuv):
    
    '''
    Compute total atmosphere transmittance of water vapour 
    given total column of water vapour (g/cm^2), cos(sza), cos(vza)
    with altidude adjusted atmosphere.
    
    Parameters
    ----------
    tcwv: 1D array
        total column of water vapour for whole atmosphere (g/cm^2)
    u: 1D array
        integral of u for water vapour for all wavelengths
    up: 1D array
        integral of up for water vapour for all wavelengths
    a: 3D array
        LUT coefficients for Ozone at all wavelengths
        (NO. of wavelengths, 8 LUT coefficients)
    xmus: 1D array
        Cosine of solar zenith angle
    xmuv: 1D array
        Cosine of view zenith angle
    Returns
    -------
    ttwava: 2D array
        total water vapour transmittance for all wavelengths
    ttwava_jac: 2D array
        jacobian of total water vapour transmittance for all wavelengths
    '''
    
    xmus = np.expand_dims(xmus, axis=1)
    xmuv = np.expand_dims(xmuv, axis=1)
    
    uwus  = 1.424
        
    atest = a[:, 1]

    atest = a[:, 1]
    mask = (a[:, 1] == 0) & (a[:, 0] == 0)
#     atest[mask] = 1
    atest = np.where(mask, 1, atest)
    
#     x,c,u,p,s,v,a,b = tcwv[:, None], uwus, u, up,xmus, xmuv, a[None,:, 0], atest
    tcwv = np.expand_dims(tcwv, axis=1)
    a = np.expand_dims(a, axis=0)
    x,c,u,p,s,v,a,b = tcwv, uwus, u, up,xmus, xmuv, a[:,:, 0], atest
    
        
    part_1 = c*s*v
    part_2 = b*p*part_1 # bcpsv
    part_3 = a*u*x*(s + v) # aux(s+v)
    part_4 = part_3 * u # au^2x(s+v)
    part_5 = np.sqrt(part_4 / part_2 + 1)
    ttwava = np.exp(-part_3/(part_1 * part_5))

    ttwava_jac = -1 * a * b * p * u * part_5 * (s + v) * (part_4 + 2 * part_2) * ttwava / 2 / (part_4 + part_2)**2
    return ttwava, ttwava_jac


@njit()
def get_iv(wv, step=0.0025):
    '''
    Get 6S equivalent wavelength and wavenumber
    
    Parameters
    ----------
        wv: wavelength in micro
    
    Return:
        iv: wavenumber index
        v: wavenumber
    '''
    wv = np.atleast_1d(wv)
    
    ivli = np.array([2500,5060,7620,10180,12740,15300, 17860])
    
    wlinf, wlsup = wv, wv
    iinf = (wlinf - 0.25) / step + 1.5
    isup = (wlsup - 0.25) / step + 1.5
    
    iinf = iinf.astype(np.int_)
    isup = isup.astype(np.int_)
    
    
    wv =(isup - 1) * step + .25
    
    v = 1.0e+04 / wv
    iv = ( v / 5.).astype(np.int_)
    iv = (iv * 5.).astype(np.int_)
    return iv, v



@njit()
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
    iwl = ((wl - 0.250) / pas + 1.5).astype(np.int_)
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


def interp_u_up_uu(presure_arr, us, ups, uus):
    
    '''
    Interpolate the gas absorption coefs based on pressure array (pressue * 100)

    Parameters
    ----------
    presure_arr : 1D array
        pressure multiply by 100
    us: 3D array
        gas absorption u for 7 gases at elevation 300-1013 at 0.1Pa gaps
    ups: 3D array
        gas absorption up for 7 gases at elevation 300-1013 at 0.1Pa gaps 
    uus: 2D array
        gas absorption up for 7 gases at elevation 300-1013 at 0.1Pa gaps 
    Returns
    -------
    gases_u: 3D array 
        gas absorption u for 7 gases at target pressure
    gases_up: 3D array
        gas absorption up for 7 gases at target pressure
    gases_uu: 2D array
        gas absorption uu for 7 gases at target pressure
    '''
    
    
    presure_arr_down = np.floor(presure_arr).astype(int).ravel()
    presure_arr_up   = np.ceil( presure_arr).astype(int).ravel()
    mask = (presure_arr_down != presure_arr_up)
    
    gases_u  = np.zeros(mask.shape + us.shape[1:])
    gases_up = np.zeros(mask.shape + ups.shape[1:])
    gases_uu = np.zeros(mask.shape + uus.shape[1:])
    
    if mask.sum() > 0:
        presure_arr_inds_down = presure_arr_down[mask] - 3000
        presure_arr_inds_up   = presure_arr_up[mask]   - 3000

        
        gases_u_down, gases_up_down, gases_uu_down = us[presure_arr_inds_down], ups[presure_arr_inds_down], uus[presure_arr_inds_down]
        gases_u_up, gases_up_up, gases_uu_up       = us[presure_arr_inds_up],   ups[presure_arr_inds_up],   uus[presure_arr_inds_up]
    
        scale          = (presure_arr - presure_arr_down) / (presure_arr_up - presure_arr_down)
        gases_u[mask]  =  gases_u_down + scale[:, None, None] * (gases_u_up - gases_u_down)
        gases_up[mask] = gases_up_down + scale[:, None, None] * (gases_up_up - gases_up_down)
        gases_uu[mask] = gases_uu_down + scale[:, None]       * (gases_uu_up - gases_uu_down)
        
    presure_arr_inds = presure_arr_down[~mask] - 3000
    gases_u [~mask] =  us[presure_arr_inds]
    gases_up[~mask] = ups[presure_arr_inds]
    gases_uu[~mask] = uus[presure_arr_inds]
    
    return gases_u, gases_up, gases_uu

def load_emus_and_scale(emu_path):
    '''
    Load emulators and the scale factors of outputs for all different areosol mixture types

    Parameters
    ----------
    emu_path: str
        directory contains all the emulator files
    
    Returns
    -------
    t_down_up_arrModel: List of nD arrays
        total scattering transmittance NN layers 
    romix_arrModel: List of nD arrays
        Rayleigh + aerosol path reflectance NN layers 
    roray_arrModel: List of nD arrays
        Rayleigh path reflectance NN layers  
    sast_arrModel: List of nD arrays
        spherical albedo NN layers 
    scales: 3D array
        scales for turn NN outputs to the original units of parameters
    '''
    
    f = np.load(emu_path + '/t_down_up_arrModel.npz', allow_pickle=True)
    t_down_up_arrModel = f.f.t_down_up_arrModel
    t_down_up_scale = f.f.scale
    
    f = np.load(emu_path + '/romix_arrModel.npz', allow_pickle=True)
    romix_arrModel = f.f.romix_arrModel
    romix_scale = f.f.scale
    
    f = np.load(emu_path + '/roray_arrModel.npz', allow_pickle=True)
    roray_arrModel = f.f.roray_arrModel
    roray_scale = f.f.scale
    
    f = np.load(emu_path + '/sast_arrModel.npz', allow_pickle=True)
    sast_arrModel = f.f.sast_arrModel
    sast_scale = f.f.scale
    
    scales = np.array([t_down_up_scale, romix_scale, roray_scale, sast_scale]).transpose(1, 0, 2)
    return t_down_up_arrModel, romix_arrModel, roray_arrModel, sast_arrModel, scales

# emu_path = '/home/users/marcyin/TRUTHs/6S_emus/'
# t_down_up_arrModel, romix_arrModel, roray_arrModel, sast_arrModel, scales = load_emus_and_scale(emu_path)