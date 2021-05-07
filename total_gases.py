import numpy as np
# from numba import jit
# @jit('Tuple((float64[:,:], float64, float64))(float64, float64[:,:])',nopython=True)
def adj_altitude(altitude, us62_atmosphere_profile):
    '''
    Adjust the US62 atmospheric profile based on the input altidude (km) 
    in case target not at sea level. 
    
    Given the altitude of the target in kilometers as input, we transform the
    original atmospheric profile (Pressure, Temperature, Water Vapor, Ozone) 
    so that first level of the new profile is the one at the target altitude. 
    We also compute the new integrated content in water vapor and ozone, that
    are used as outputs or in computations when the user chooses to enter a
    specific amount of Ozone and Water Vapor.
    
    This code is translated from original 6SV2.1 pressure.f

    Parameters
    ----------
    altitude : altidude (km) of the target
        Max altidude is 99.99 kms
        
    us62_atmosphere_profile : US62 atmosphere profile look up table with shape of (5, 34). 
        
        The atmosphere has been divided into 34 layers 
        with 5 variables for each layer:
                    z -- altidude
                    p -- pressure of atmosphere at altitude z
                    t -- temperature of atmosphere at altitude z 
                    wh --  water vapor at altitude z
                    wo -- ozone at altidude z
    Returns
    -------
    adjusted_us62 : ndarray with shape of (5, 34)
        The updated US 62 atmosphere corresponding to input altidude
    uw : new integrated content in water vapor 
    uo3 : new integrated content in ozone
        
    '''
    
    # 5 variables for US62 atmosphere
    # at see level pressure
    z = us62_atmosphere_profile[0]
    p = us62_atmosphere_profile[1]
    t = us62_atmosphere_profile[2]
    wh = us62_atmosphere_profile[3]
    wo = us62_atmosphere_profile[4]
    
    # max altitude is 99.99 kms
    # for elevated target
    if altitude >= 100:
        altitude = 99.99
        
    xps = altitude # unit km
    # get the layer index based on altitude
    for i in range(100):
        if z[i] > xps:
            break

    isup = i
    iinf = i - 1
    # log interpolation of pressure based altitude
    xa = (z[isup] - z[iinf]) / np.log(p[isup] / p[iinf])
    xb =  z[isup] - xa * np.log(p[isup])
    ps = np.exp((xps - xb) / xa)
    
    # linearly interpolate t, water vapour and ozone based on altitude
    # to get layer 0 values
    xalt = xps
    xtemp = (t[isup] - t[iinf]) / (z[isup] - z[iinf])
    xtemp = xtemp * (xalt - z[iinf]) + t[iinf]
    xwo = (wo[isup] - wo[iinf]) / (z[isup] - z[iinf])
    xwo = xwo * (xalt - z[iinf]) + wo[iinf]
    xwh = (wh[isup] - wh[iinf]) / (z[isup] - z[iinf])
    xwh = xwh * (xalt - z[iinf]) + wh[iinf]
    
    z[0]  = xalt
    p[0]  = ps
    t[0]  = xtemp
    wh[0] = xwh
    wo[0] = xwo
    
    # updated atmosphere layer 1-(33-inf)
    # are copied from original profile
    z[1 : 33 - iinf] = z[1 + iinf: 33]
    p[1 : 33 - iinf] = p[1 + iinf: 33]
    t[1 : 33 - iinf] = t[1 + iinf: 33]
    wh[1 : 33 - iinf] = wh[1 + iinf: 33]
    wo[1 : 33 - iinf] = wo[1 + iinf: 33]
    
    # layer (33 - iinf - 1) to 33 are linearly 
    # interpolated from the last two layers from the original 
    # profile
    l = 33 - iinf - 1
    inds = np.arange(l, 34)
    z[l:] = (z[33] - z[l]) * (inds - l) / (33 - l) + z[l]
    p[l:] = (p[33] - p[l]) * (inds - l) / (33 - l) + p[l]
    t[l:] = (t[33] - t[l]) * (inds - l) / (33 - l) + t[l]
    wh[l:] = (wh[33] - wh[l]) * (inds - l) / (33 - l) + wh[l]
    wo[l:] = (wo[33] - wo[l]) * (inds - l) / (33 - l) + wo[l]

    # compute the column integral of water vapour and ozone
    uw  = 0.
    uo3 = 0
    g   = 98.1
    air = 0.028964 / 0.0224
    ro3 = 0.048 / 0.0224

    rmwh = np.zeros(34)
    rmo3 = np.zeros(34)

    roair = air * 273.16 * p[:33] / (1013.25 * t[:33])
    rmwh[:33] = wh[:33] / (roair * 1000.)
    rmo3[:33] = wo[:33] / (roair * 1000.)

    ds = (p[0:32] - p[1:33]) / p[0]
    uw = (rmwh[1:33] + rmwh[0:32]) / 2. * ds
    uo3 = (rmo3[1:33] + rmo3[0:32]) / 2. * ds
    uw = uw.sum()
    uo3 = uo3.sum()

    uw = uw * p[0] * 100. / g
    uo3 = uo3 * p[0] * 100. / g
    uo3 = uo3 * 1000 / ro3
    
    
    # return the updated profile as a 2D array
    us62_atmosphere_profile[0] = z
    us62_atmosphere_profile[1] = p
    us62_atmosphere_profile[2] = t
    us62_atmosphere_profile[3] = wh
    us62_atmosphere_profile[4] = wo
    
    return us62_atmosphere_profile, uw, uo3

# @jit('Tuple((float64[:,:], float64, float64))(float64, float64[:,:])',nopython=True)
def adj_pressure(pressure, us62_atmosphere_profile):
    '''
    Adjust the US62 atmospheric profile based on the input altidude (km) 
    in case target not at sea level. 
    
    Given the pressure of the target in mb as input, we transform the
    original atmospheric profile (Altidude, Pressure, Temperature, Water Vapor, Ozone) 
    so that first level of the new profile is the one at the target pressure. 
    We also compute the new integrated content in water vapor and ozone, that
    are used as outputs or in computations when the user chooses to enter a
    specific amount of Ozone and Water Vapor.
    
    This code is translated from original 6SV2.1 pressure.f

    Parameters
    ----------
    pressure : pressure (mb) of the target
        
    us62_atmosphere_profile : US62 atmosphere profile look up table with shape of (5, 34). 
        
        The atmosphere has been divided into 34 layers 
        with 5 variables for each layer:
                    z -- altidude
                    p -- pressure of atmosphere at altitude z
                    t -- temperature of atmosphere at altitude z 
                    wh --  water vapor at altitude z
                    wo -- ozone at altidude z
    Returns
    -------
    adjusted_us62 : ndarray with shape of (5, 34)
        The updated US 62 atmosphere corresponding to input altidude
    uw : new integrated content in water vapor 
    uo3 : new integrated content in ozone
        
    '''
    
    # 5 variables for US62 atmosphere
    # at see level pressure
    z = us62_atmosphere_profile[0]
    p = us62_atmosphere_profile[1]
    t = us62_atmosphere_profile[2]
    wh = us62_atmosphere_profile[3]
    wo = us62_atmosphere_profile[4]
    
    # max altitude is 99.99 kms
    # for elevated target
    if pressure > 1013:
        dps = pressure - p[0]

        for i in range(9):
            if p[i] > dps:
                p[i] = p[i] + dps
    else:
        ps = pressure # unit mb
        # get the layer index based on altitude
        for i in range(100):
            if p[i] < pressure:
                break
        isup = i
        iinf = i - 1

        # log interpolation to get altitude
        xa = (z[isup] - z[iinf]) / np.log(p[isup] / p[iinf])
        xb =  z[isup] - xa * np.log(p[isup])
        xalt = np.log(ps) * xa + xb
    
        # linearly interpolate t, water vapour and ozone based on altitude
        # to get layer 0 values
        xtemp = (t[isup] - t[iinf]) / (z[isup] - z[iinf])
        xtemp = xtemp * (xalt - z[iinf]) + t[iinf]
        xwo = (wo[isup] - wo[iinf]) / (z[isup] - z[iinf])
        xwo = xwo * (xalt - z[iinf]) + wo[iinf]
        xwh = (wh[isup] - wh[iinf]) / (z[isup] - z[iinf])
        xwh = xwh * (xalt - z[iinf]) + wh[iinf]
    
        z[0]  = xalt
        p[0]  = ps
        t[0]  = xtemp
        wh[0] = xwh
        wo[0] = xwo

        # updated atmosphere layer 1-(33-inf)
        # are copied from original profile
        z[1 : 33 - iinf] = z[1 + iinf: 33]
        p[1 : 33 - iinf] = p[1 + iinf: 33]
        t[1 : 33 - iinf] = t[1 + iinf: 33]
        wh[1 : 33 - iinf] = wh[1 + iinf: 33]
        wo[1 : 33 - iinf] = wo[1 + iinf: 33]

        
        # layer (33 - iinf - 1) to 33 are linearly 
        # interpolated from the last two layers from the original 
        # profile
        l = 33 - iinf - 1
        inds = np.arange(l, 34)
        z[l:] = (z[33] - z[l]) * (inds - l) / (33 - l) + z[l]
        p[l:] = (p[33] - p[l]) * (inds - l) / (33 - l) + p[l]
        t[l:] = (t[33] - t[l]) * (inds - l) / (33 - l) + t[l]
        wh[l:] = (wh[33] - wh[l]) * (inds - l) / (33 - l) + wh[l]
        wo[l:] = (wo[33] - wo[l]) * (inds - l) / (33 - l) + wo[l]

    # compute the column integral of water vapour and ozone
    uw  = 0.
    uo3 = 0
    g   = 98.1
    air = 0.028964 / 0.0224
    ro3 = 0.048 / 0.0224

    rmwh = np.zeros(34)
    rmo3 = np.zeros(34)

    roair = air * 273.16 * p[:33] / (1013.25 * t[:33])
    rmwh[:33] = wh[:33] / (roair * 1000.)
    rmo3[:33] = wo[:33] / (roair * 1000.)

    ds = (p[0:32] - p[1:33]) / p[0]
    uw = (rmwh[1:33] + rmwh[0:32]) / 2. * ds
    uo3 = (rmo3[1:33] + rmo3[0:32]) / 2. * ds
    uw = uw.sum()
    uo3 = uo3.sum()

    uw = uw * p[0] * 100. / g
    uo3 = uo3 * p[0] * 100. / g
    uo3 = uo3 * 1000 / ro3


    # return the updated profile as a 2D array
    us62_atmosphere_profile[0] = z
    us62_atmosphere_profile[1] = p
    us62_atmosphere_profile[2] = t
    us62_atmosphere_profile[3] = wh
    us62_atmosphere_profile[4] = wo

    return us62_atmosphere_profile, uw, uo3


def get_a(wv, gas_full_tables, step=0.0025):
    '''
    Get LUT for gas transmittance computation
    Parameters
    ----------
    wv : 1D array
        range from 0.3-2.5
    gas_full_tables: 4D array
        (7 gases, 6 wavelength ranges, 256 values for each wavelength range, 8 LUT coefficients)
    
    Returns
    -------
    a: 3D array
       LUT coefficients for all wavelengths
       (NO. of wavelengths, 7 gases, 8 LUT coefficients)
    '''
    wv = np.atleast_1d(wv)
    
    ivli = np.array([2500,5060,7620,10180,12740,15300, 17860])
    
    wlinf, wlsup = wv, wv
    iinf = (wlinf - 0.25) / step + 1.5
    isup = (wlsup - 0.25) / step + 1.5

    iinf = np.array(iinf).astype(int)
    isup = np.array(isup).astype(int)
    
    wv =(isup - 1) * step + .25
    
    v = 1.0e+04 / wv
    iv = ( v / 5.).astype(int)
    iv = (iv * 5.).astype(int)
    id = (((iv - 2500) / 10) / 256).astype(int)
    mask = id < 6
    
    inu = ((iv[mask] - ivli[id[mask]]).astype(int) / 10).astype(int)
    
    a = np.zeros((7, len(wv), 8))
    a[:, mask] = gas_full_tables[:, id[mask], inu]
    
    return a.transpose(1, 0, 2)

def get_uu(a, us62_atmosphere_profile):
    
    '''
    Compute column integral of uu, u, up
    Parameters
    ----------
    a: 3D array
        LUT coefficients for all wavelengths
        (NO. of wavelengths, 7 gases, 8 LUT coefficients)
    us62_atmosphere_profile: 2D array
       The updated US 62 atmosphere corresponding to input altidude    
       
    Returns
    -------
    uu: 1D array
        integral of uu for 7 gases
    u: 2D array
        integral of u for 7 gases for all wavelengths
    up: 2D array
        integral of up for 7 gases for all wavelengths
    '''
    
    z = us62_atmosphere_profile[0]
    p = us62_atmosphere_profile[1]
    t = us62_atmosphere_profile[2]
    wh = us62_atmosphere_profile[3]
    wo = us62_atmosphere_profile[4]
    
    ds=0.
    te=0.
    roair=0.

    air   = 1.2930357142857143
    roco2 = 1.9642857142857142
    rmo2  = 1.4285714285714286
    rmo3  = 2.142857142857143
    rmn2o = 1.9642857142857142
    rmch4 = 0.7142857142857143
    rmco  = 1.25

    p0 = 1013.25
    g  = 98.1
    t0 = 250.

    ds  = (p[0:32] - p[1:33]) / p[0]
    ds2 = (p[0:32]**2 - p[1:33]**2) / (2. * p[0] * p0)  

    roair = air * 273.16 * p[:-1] / (1013.25 * t[:-1])
    tp    = (t[:-1] + t[1:]) / 2
    te    = tp - t0
    te2   = te * te
    
    phi = np.exp(a[:, :, [2]] * te + a[:, :, [3]] * te2)
    psi = np.exp(a[:, :, [4]] * te + a[:, :, [5]] * te2)
    
    
    rm = np.zeros((7, 33))
    rm[0] = wh[:-1] / (roair * 1000.)
    rm[1] = np.array([3.3e-04 * roco2 / air] * 33)
    rm[2] = np.array([0.20947 * rmo2 / air] * 33)
    rm[3] = wo[:-1] / (roair * 1000.)
    rm[4] = np.array([3.1e-07 * rmn2o / air] * 33)
    rm[5] = np.array([1.72e-06 * rmch4 / air] * 33)
    rm[6] = np.array([1.00e-09 * rmco /air] * 33)

    r2 = rm * phi
    r3 = rm * psi

    uu = ((rm[:,  1:33] + rm[:,  0:32]) / 2.) * ds[None,]  * p[0] * 100. / g     
    u  = ((r2[:,:,1:33] + r2[:,:,0:32]) / 2.) * ds[None,]  * p[0] * 100. / g     
    up = ((r3[:,:,1:33] + r3[:,:,0:32]) / 2.) * ds2[None,] * p[0] * 100. / g    
    
    uu[1] = 1000. * uu[1] / roco2
    uu[3] = 1000. * uu[3] / rmo3  
    uu[4] = 1000. * uu[4] / rmn2o
    uu[5] = 1000. * uu[5] / rmch4
    uu[6] = 1000. * uu[6] / rmco
    
    return np.sum(uu, axis = 1), np.sum(u, axis = 2), np.sum(up, axis = 2)

def atmo_trans_wv(uw, xmus, xmuv, u, up, a):
    
    '''
    Compute total atmosphere transmittance of water vapour for 
    given total column of water vapour (g/cm^2), cos(sza), cos(vza)
    with altidude adjusted atmosphere.
    
    Parameters
    ----------
    uw: 1D array
        total column of water vapour for whole atmosphere (g/cm^2)
    xmus: scaller
        Cosine of solar zenith angle
    xmuv: scaller
        Cosine of view zenith angle
    u: 1D array
        integral of u for water vapour gases for all wavelengths
    up: 1D array
        integral of up for water vapour gases for all wavelengths
    a: 2D array
        LUT coefficients for water vapour at all wavelengths
        (NO. of wavelengths, 8 LUT coefficients)
    Returns
    -------
    ttwava: 1D array
        total water vapour transmittance for all wavelengths
    '''
    
    # this is problematic
    # should this be uwus, uo3us
    # at target elevation?
    
    uw = np.atleast_1d(uw)
    u  = np.atleast_2d(u)
    up = np.atleast_2d(up)
    
    uwus  = 1.424
    
    accu   = 1.e-10

    rat = uw / uwus

    u  =  u  * rat[:, None]
    up =  up * rat[:, None]
    
    upl  = u
    uppl = up
    
    atest = a[:, 1]
    mask = (a[:, 1] == 0) & (a[:, 0] == 0)
    atest[mask] = 1

    ut  = u / xmus + upl / xmuv
    upt = up / xmus + uppl / xmuv
    utt = ut
    
    mask = (ut == 0) & (upt == 0)
    utt[mask] = 1.
    uptt = upt 
    uptt[mask] = 1.

    y = -a[:, 0] * ut / np.sqrt(1 + (a[:, 0] / atest) * (ut * ut / uptt))
    ttwava = np.exp(y)
    
    ttwava = np.where(ttwava>accu, ttwava, 0)

    return ttwava

def atmo_trans_other_gases(xmus, xmuv, u, up, a):
    '''
    Compute total atmosphere transmittance of gases except for 
    water vapour and Ozone for given cos(sza), cos(vza) with 
    altidude adjusted atmosphere.
    
    Parameters
    ----------
    xmus: scaller
        Cosine of solar zenith angle
    xmuv: scaller
        Cosine of view zenith angle
    u: 2D array
        integral of u for other gases for all wavelengths
    up: 2D array
        integral of up for other gases for all wavelengths
    a: 3D array
        LUT coefficients for other gases at all wavelengths
        (NO. of wavelengths, 5 gases, 8 LUT coefficients)
    Returns
    -------
    ttwava: 1D array
        total water vapour transmittance for all wavelengths
    '''      
    accu   = 1.e-10
        
    upl  = u
    uppl = up

    atest = a[:, :, 1]
    
    atest = a[:, :, 1]
    mask = (a[:, :, 1] == 0) & (a[:, :, 0] == 0)
    
    atest[mask] = 1

    ut  = u / xmus + upl / xmuv
    upt = up / xmus + uppl / xmuv
    utt = ut
    
    mask = (ut == 0) & (upt == 0)
    utt[mask] = 1.
    
    tn = a[:, :, 1] * upt / (2 * utt)
    
    uptt = upt 
    uptt[mask] = 1.
    
    tt = 1 + 4 * (a[:, :, 0] / atest) * ((ut * ut) / uptt)
    y = -tn * (np.sqrt(tt) - 1)

    tt_other = np.exp(y)
    
    tt_other = np.where(tt_other>accu, tt_other, 0)

    return tt_other

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

    iinf = np.array(iinf).astype(int)
    isup = np.array(isup).astype(int)
    
    wv =(isup - 1) * step + .25
    
    v = 1.0e+04 / wv
    iv = ( v / 5.).astype(int)
    iv = (iv * 5.).astype(int)
    return iv, v

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

    iinf = np.array(iinf).astype(int)
    isup = np.array(isup).astype(int)
    
    wv =(isup - 1) * step + .25
    
    v = 1.0e+04 / wv
    iv = ( v / 5.).astype(int)
    iv = (iv * 5.).astype(int)
    return iv, v

def atmo_trans_ozone(uo3, iv, v, xmus, xmuv, uu, u, up, a):
    '''
    Compute total atmosphere transmittance of Ozone given 
    total column of Ozone (cm·atm), cos(sza), cos(vza)
    with altidude adjusted atmosphere.
    
    Parameters
    ----------
    uo3: 1D array
        total column of Ozone for whole atmosphere (cm·atm)
    xmus: scaller
        Cosine of solar zenith angle
    xmuv: scaller
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
        total water vapour transmittance for all wavelengths
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

    uo3 = np.atleast_1d(uo3)
    u  = np.atleast_2d(u)
    up = np.atleast_2d(up)
    
    rat = uo3/uo3us
    uu = uu * rat[:, None]
    u  = u  * rat[:, None]
    up = up * rat[:, None]
    
    
    ttco3 = np.zeros((len(uo3), iv.shape[0]))
    uupl = uu
    uut = uu / xmus + uupl / xmuv

    mask = (iv <= 23400) & (iv >= 13000)
    
    xi = (v[mask] - 13000.) / 200. + 1.
    n  = (xi + 1.001).astype(int)
    xd = xi - n
    
    ako3 = co3[n-1] + xd * (co3[n-1] - co3[n-2])    
    test3 = ako3 * uut
    test3 = np.where(test3 > 86.00, 86.0, test3)

    ttco3[:, mask] = np.exp(-1 * test3)

    mask = (iv >= 27500) & (iv <= 50000)
    xi=(v[mask] - 27500.) / 500. + 57.
    n = (xi + 1.001).astype(int)
    
    xd = xi - n
    ako3 = co3[n-1] + xd * (co3[n-1] - co3[n-2])
    
    test3 = ako3 * uut
    
    test3 = np.where(test3 > 86.00, 86.0, test3)
    
    ttco3[:, mask] = np.exp(-1 * test3)
    
    mask = (iv >= 13000) & (iv <= 50000) 
    mask = (~mask) | ((iv > 23400) & (iv < 27500))

    accu   = 1.e-10    
    
    upl  = u[:, mask]
    uppl = up[:, mask]
    
    atest = a[mask, 1]
    
    atest = a[mask, 1]
    atest[(a[mask, 1] == 0) & (a[mask, 0] == 0)] = 1

    ut  = u[:, mask] / xmus + upl / xmuv
    upt = up[:, mask] / xmus + uppl / xmuv
    utt = ut
    
    utt[(ut == 0) & (upt == 0)] = 1.
    
    tn = a[mask, 1] * upt / (2 * utt)
    
    uptt = upt 
    uptt[(ut == 0) & (upt == 0)] = 1.
    
    tt = 1 + 4 * (a[mask, 0] / atest) * ((ut * ut) / uptt)
    y = -tn * (np.sqrt(tt) - 1)

    test3 = np.exp(y)
    test3 = np.where(test3>accu, test3, 0)
    ttco3[:, mask] = test3
    
    return ttco3

def gas_trans(wv, alt, uw, uo3, sza, vza, gas_full_tables, us62_atmosphere_profile):
    
    '''
    if alt smaller or equal than 0, then it means input alt is altitude in kms
    if alt larger than 0, it means input alt is pressure in mb
    '''
    if alt <= 0:
        alt = alt * -1
        us62_atmosphere_profile, us_uw, us_uo3 = adj_altitude(alt, us62_atmosphere_profile)
    else:
        pressure = alt
        us62_atmosphere_profile, us_uw, us_uo3 = adj_pressure(pressure, us62_atmosphere_profile)
        
    a_array = get_a(wv, gas_full_tables)
    uu, u, up = get_uu(a_array, us62_atmosphere_profile)
    
    xmus = np.cos(np.deg2rad(sza))
    xmuv = np.cos(np.deg2rad(vza))

    ttwava = atmo_trans_wv(uw, xmus, xmuv, u[:, 0], up[:, 0], a_array[:, 0])
    tt_other = atmo_trans_other_gases(xmus, xmuv, u[:, [1, 2, 4, 5, 6]], up[:, [1, 2, 4, 5, 6]], a_array[:, [1, 2, 4, 5, 6]])
    iv, v = get_iv(wv, step=0.0025)
    ttco3 = atmo_trans_ozone(uo3, iv, v, xmus, xmuv,  uu[3], u[:, 3], up[:, 3], a_array[:, 3])
    
    tt_total = np.ones((7, len(wv)))
    
    tt_total[0] = ttwava
    tt_total[1] = tt_other[:, 0]
    tt_total[2] = tt_other[:, 1]
    tt_total[3] = ttco3
    tt_total[4] = tt_other[:, 2]
    tt_total[5] = tt_other[:, 3]
    tt_total[6] = tt_other[:, 4]
    
    return tt_total

if __name__ == '__main__':
    wv = np.arange(0.3, 2.5, 0.001)
    alt = 0.0
    uw, uo3 = 2.5, 0.35
    sza, vza = 15, 35
    f = np.load('atmospheric_transmittance_LUT.npz')
    gas_full_tables = np.array(f.f.gas_full_tables)
    us62_atmosphere_profile = np.array(f.f.us62_atmosphere_profile)
    tt_total = gas_trans(wv, alt, uw, uo3, sza, vza, gas_full_tables, us62_atmosphere_profile)
    