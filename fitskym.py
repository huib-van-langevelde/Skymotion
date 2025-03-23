#!/usr/bin/env python3

#from corner.arviz_corner import xarray_var_iter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize
import emcee
import corner
import astropy.units as u
from  astropy.coordinates import SkyCoord, ICRS, GCRS
from astropy.time import Time
import yaml
import argparse as ap
import time as utime
import skyfield
from skyfield.api import Loader

C2 = True
MASpDEG = 3.6e6
#DAYSpYR = 365.24217 #replaced to Julian date definition
DAYSpYR = 365.25

Version = '1.95; works in 1,2 star case, but not for error floors'

'''
Bayesian and Max Likelihood fitter motions on the sky
Facilitating switching between different functions 
and switching various fit parameters on and off

Program runs in 4 stages
- Generate data
- Find fit by minimisation of -log likelihood
- Bayesian posteriors with emcee
- Plot residuals

Control by external files using yaml
skym_tru_[root].yaml for generating data or supplying true values
skym_par_[root].yaml for controling fits and bayes
skym_data_[root].yaml for data

can also generate skym_post_root.yaml to store traces

Produces output
[ttag]_skym_out.txt
and possibly [ttag]_fig*[type].pdf plots

Data are in JD and sky coordinates
User supplied values are in mas, mas/yr, yr, ra, dec in degrees
errors in mas
data are in ra, dec degrees on the sky
dates are in JD (but the skyfield are in MJD)
And t0 is the date of the first obs
binT0 is in days wrt to t0

Can 'fit' a curve to the data
And 'model' nuisance paramter. 
So Bayes model may use different functions and parameters

NOTE: Modelling nuisance parameters is BROKEN
At this time only one lprob_gen is employed
ALSO switching fixed parameters is untested.


TBD and Open ends
- check reference time is consistently used
- change ellips definition to use ellipticity
- check all definitions
- Try on real data
- Need a way to estimate initials
- Commandline mode switching, debug level, output
'''

def GetArgs():
    '''
    Get the command line arguments to reading the data
    get the root string  -r
    There are 4 stages that can be run independently or chained.
    Default is to do only bayes, other must be switched explicitly
    
    do generate, dofit,plot residuals -g, -f, -x
    save plots -s
    debug -db
    dump traces -d
    '''
    parser = ap.ArgumentParser(description='Run Bayesian sky motion inference, data generation and fitting.')
    #    parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #               help='an integer for the accumulator')
    parser.add_argument('-g','--dogenerate', action = 'store_true',
                   help='generate mock data')
    parser.add_argument('-f','--dofit', action = 'store_true',
                   help='fit minimising likelihood')
    parser.add_argument('-nb','--skipbayes', action = 'store_true',
                   help='run bayesian inference')
    parser.add_argument('-s','--saveplots', action = 'store_true',
                   help='write plots to file')
    parser.add_argument('-t','--readtruth', action = 'store_true',
                   help='read truth from file')
    parser.add_argument('-x','--residualmotion', action = 'store',
                   choices = ['prlxpm','pm','full'],
                   help='plot residual of various kinds')
    parser.add_argument('-d','--dumpsamples', action = 'store_true',
                   help='dump/use samples for residual plot')      
    parser.add_argument('-c','--cornerdump', action = 'store_true',
                   help='recreate corner plot from dumped data')                         
    parser.add_argument('-db','--debuglevel',type=int,default=0)
    parser.add_argument('-r','--fileroot',type=str,default='sky5st')
    args = parser.parse_args()
    return args

#------- Functions for 5par parallax fitting
def pparlx(t,x0=90.,y0=30.,pmx=0.,pmy=0.,pi=1,t0=24445.):
    #Paul's original parallax using astropy and a way to divide out aberration
    def pm_and_px(SC, epoch):
        # Apply the proper motion. This changes obstime to the value of new_obstime
        PM = SC.apply_space_motion(new_obstime = epoch)
        # Convert to geocentric (applies parallax and aberration)
        GC = PM.transform_to(GCRS)
        # Copy the SkyCoord, but without the distance, so the parallax remains when transforming back
        GC_new = SkyCoord(GC.ra, GC.dec, frame='gcrs', obstime = GC.obstime)
        # Convert back to ICRS, which removes the aberration, but keeps the parallax
        return (GC_new.transform_to(ICRS))
        
    #print('Check:',t[0],x0,y0)
    '''
    # Barnard's Star, with Gaia DR3 coordinates and epoch
    BS = SkyCoord(269.44850252543836 * u.deg, 4.739420051112412 * u.deg,
                frame = 'icrs',
                obstime = Time(2016.0, format='decimalyear'),
                distance = (546.975939730948 * u.mas).to(u.pc, u.parallax()),
                pm_ra_cosdec =      -801.551 * u.mas / u.year,
                pm_dec = 10362.394 * u.mas / u.year);
    '''

    try:
        skyc = SkyCoord(x0 * u.deg, y0 * u.deg,
                    frame = 'icrs',
                    obstime = Time(t0, format='jd'),
                    distance = (pi * u.mas).to(u.pc, u.parallax()),
                    pm_ra_cosdec = pmx * u.mas / u.year,
                    pm_dec = pmy * u.mas / u.year);
        #print('Astro:',skyc.obstime,skyc)
# Process all times at once
        epochs = Time(t, format='jd')
        pos = pm_and_px_vectorized(skyc, epochs)
        
        # Extract positions all at once
        xpos = pos.ra.degree
        ypos = pos.dec.degree
        
        return xpos, ypos
    except:
        print('Exception on parlx')
        return np.full(len(t),-np.Inf),np.full(len(t),-np.Inf)

def skyfpbin(tobs,x0=90.,y0=30.,pmx=0.,pmy=0.,pi=1,binP=1.,bina=0.,bine=0,binT0=0.,binom=0.,binbigOm=0.,bini=0.,t0=24445.):
    #wrapper around skyfield 
    #developed as rwful_skyf
    if (debug >= 2): print('skyfpbin:',locals())
    tskyf0 = t0 - 2400000.5
    tskyf = tobs[0]['t'] - 2400000.5
    orb_a = bina
    orb_T_0 = binT0 -2400000.5 + t0
    orb_P = binP
    orb_e = bine
    orb_i = bini
    orb_omega = binom
    orb_Omega = binbigOm
    #Huib took out the division here, do later
    pm_alphacosd_deg = (pmx / MASpDEG) / DAYSpYR    # Converting proper motion from milliarcsec/yr to degree/day
    pm_delta_deg = (pmy / MASpDEG) / DAYSpYR        # Converting proper motion from milliarcsec/yr to degree/day
    orb_a_deg = orb_a / MASpDEG                     # Converting orbit size from milliarcsec to degree
    parallax_deg = pi / MASpDEG            # Converting parallax from milliarcsec to degree
    x_obs, y_obs = orbital_motion(tskyf, orb_T_0, orb_P, orb_e, orb_i, orb_omega, orb_Omega, orb_a_deg)
    #print('H1:',x_obs,y_obs)
    
    #print('H2:',frac_alpha,frac_delta)
    #Huib this is weird! The original had cosd here...
    #predict_ra = alpha_0 * np.cos(np.radians(delta)) + (pm_alpha_deg * (t - t_0(t))) + frac_alpha * parallax_deg + x_obs
    
    tmp_dec = y0 + (pm_delta_deg * (tskyf - tskyf0))
    tmp_ra =  x0 + (pm_alphacosd_deg * (tskyf - tskyf0))/np.cos(tmp_dec*np.pi/180)
    
    frac_alpha, frac_delta = frac_parallax(tskyf, tmp_ra, tmp_dec)
    #predict_ra = tmp_ra + frac_alpha * parallax_deg + x_obs
    #But Paul says you need this to agree with astropy:
    predict_ra = tmp_ra + (frac_alpha * parallax_deg + x_obs)/np.cos(tmp_dec*np.pi/180)
    predict_dec = tmp_dec + frac_delta * parallax_deg + y_obs
    return predict_ra, predict_dec
    
def skyfc2(tobs,x0=90.,y0=30.,pmx=0.,pmy=0.,pi=1,binP=1.,bina=0.,bine=0,binT0=0.,
        binom=0.,binbigOm=0.,bini=0.,mrat=0.9,t0=24445.):
    '''
    This is a version that generates images for both stars
    the input tobs = list with data for 2 stars,
    but the return is one list with xi, yi appended
    '''
    
    if (debug >= 2): print('skyfpbin:',locals())
    tskyf0 = t0 - 2400000.5
    tskyf = tobs[0]['t'] - 2400000.5
    tskys= tobs[1]['t'] - 2400000.5
    orb_a = bina
    orb_T_0 = binT0 -2400000.5 + t0
    orb_P = binP
    orb_e = bine
    orb_i = bini
    orb_omega = binom
    orb_Omega = binbigOm
    #Huib took out the division here, do later
    pm_alphacosd_deg = (pmx / MASpDEG) / DAYSpYR    # Converting proper motion from milliarcsec/yr to degree/day
    pm_delta_deg = (pmy / MASpDEG) / DAYSpYR        # Converting proper motion from milliarcsec/yr to degree/day
    orb_a_deg = orb_a / MASpDEG                     # Converting orbit size from milliarcsec to degree
    parallax_deg = pi / MASpDEG                     # Converting parallax from milliarcsec to degree
    x_obs, y_obs = orbital_motion(tskyf, orb_T_0, orb_P, orb_e, orb_i, orb_omega, orb_Omega, orb_a_deg)
    #Huib takes a guess this will work
    xsec_obs, ysec_obs = orbital_motion(tskys, orb_T_0, orb_P, orb_e, orb_i, orb_omega, orb_Omega, -1*mrat*orb_a_deg)
    #print('H1:',x_obs,y_obs)
    
    #print('H2:',frac_alpha,frac_delta)
    #Huib this is weird! The original had cosd here...
    #predict_ra = alpha_0 * np.cos(np.radians(delta)) + (pm_alpha_deg * (t - t_0(t))) + frac_alpha * parallax_deg + x_obs
    
    tmp_dec = y0 + (pm_delta_deg * (tskyf - tskyf0))
    tmp_ra =  x0 + (pm_alphacosd_deg * (tskyf - tskyf0))/np.cos(tmp_dec*np.pi/180)
    tmp_decsec = y0 + (pm_delta_deg * (tskys - tskyf0))
    tmp_rasec =  x0 + (pm_alphacosd_deg * (tskys - tskyf0))/np.cos(tmp_decsec*np.pi/180)
    
    frac_alpha, frac_delta = frac_parallax(tskyf, tmp_ra, tmp_dec)
    fsec_alpha, fsec_delta = frac_parallax(tskys, tmp_rasec, tmp_decsec)
    #predict_ra = tmp_ra + frac_alpha * parallax_deg + x_obs
    #But Paul says you need this to agree with astropy:
    fin_ra = tmp_ra + (frac_alpha * parallax_deg + x_obs)/np.cos(tmp_dec*np.pi/180)
    fin_dec = tmp_dec + frac_delta * parallax_deg + y_obs
    fis_ra = tmp_rasec + (fsec_alpha * parallax_deg + xsec_obs)/np.cos(tmp_decsec*np.pi/180)
    fis_dec = tmp_decsec + fsec_delta * parallax_deg + ysec_obs
    
    return np.concatenate((fin_ra,fis_ra)), np.concatenate((fin_dec,fis_dec))
        
    
    
def earth_position(t):
    """
    From CygX1 collab
    Calculates Earth position (X,Y,Z) at time t using Python package Skyfield (http://rhodesmill.org/skyfield/).    
    Parameters
    ----------
    t: Time
       an array in MJD format and Barycentric Dynamical Time (TDB) scale.  
    Returns
    -------
    earth_pos: Earth positions (X,Y,Z)
               in the form an array with shape [3 * N] where N is length of t, values in unit of au.
    """
    #print("load")
    earth = planets['earth']
    times = ts.tdb_jd(t+2400000.5)
    earth_pos = earth.at(times).position.au
    return earth_pos


def t_0(t):
    """
    From CygX1 collab, I think we are not using this
    Calculates the midpoint time between first and last observations.
    Note that the value is rounded using "np.floor" for convenience.
    Parameters
    t: Time
       an array (preferred to be in MJD format and Barycentric Dynamical Time (TDB) scale).   
    Returns
    t_midpoint: A single value representing the midpoint time.
    """
    return np.floor(t.min()+((t.max() - t.min())/2.0))


def frac_parallax(t, alpha, delta):
    """
    From CygX1 collab
    Calculates fractions of parallax projected on RA and Dec axes.
    Parameters
    t: Time
       an array in MJD format and Barycentric Dynamical Time (TDB) scale.  
    alpha: Observation RA, in degrees
    delta: Observation Dec, in degrees
    Returns
    f_alpha, f_delta: Ra and Dec Parallax fractions respectively, at time t (unitless fractions). 
    """
    alpha_radians = np.radians(alpha)
    delta_radians = np.radians(delta)
    X, Y, Z = earth_position(t)
    f_alpha = ((X * np.sin(alpha_radians)) - (Y * np.cos(alpha_radians)))   
    f_delta = (X * np.cos(alpha_radians) * np.sin(delta_radians)) + (Y * np.sin(alpha_radians) * np.sin(delta_radians)) - (Z * np.cos(delta_radians))
    return f_alpha, f_delta


def eccentric_anomaly(t, orb_T_0, orb_P, orb_e):
    """
    From CygX1 collab
    Calculates eccentric anomaly for the orbit at time t.
    
    The functional form for eccentric anomaly needs to be solved numerically. 
    However, the function converges fast, thus with a good starting point, 
    a simple iterative method with only a few iterations is sufficient to reach
    accurate values.
    
    Parameters
    ----------
    t: time
       an array in MJD format and Barycentric Dynamical Time (TDB) scale.  
    orb_T_0: Time of superior conjunction (when the BH is behind the star), in unit of days.
    orb_P: Orbital period, in unit of days.
    orb_e: Orbital eccentricity
    
    Returns
    -------
    E_obs: Eccentric anomaly at time t as an array, in radians.

    """
    M = 2 * np.pi * (t - orb_T_0) / orb_P
    
    #print('mmmm',M,t,orb_T_0,orb_P) 
    
    E_obs = M + (orb_e * np.sin(M)) + ((orb_e**2) * np.sin(2 * M)/M)
    

    
    for solve_iteration in range(10):
        M0 = E_obs - (orb_e * np.sin(E_obs))
        E1 = E_obs + ((M - M0) / (1 - (orb_e * np.cos(E_obs))))
        E_obs = E1
    return E_obs


def true_anomaly(t, orb_T_0, orb_P, orb_e):
    """
    From CygX1 collab
    Calculates true anomaly for the orbit at time t
    Parameters:
    t: time
       an array in MJD format and Barycentric Dynamical Time (TDB) scale.  
    orb_T_0: Time of superior conjunction (when the BH is behind the star), in unit of days.
    orb_P: Orbital period, in unit of days.
    orb_e: Orbital eccentricity
    Returns
    theta_obs: True anomaly at time t as an array, in radians.
    """
    E_obs = eccentric_anomaly(t, orb_T_0, orb_P, orb_e)
    tan_theta2 = np.sqrt( (1 + orb_e) / (1 - orb_e)) * np.tan(E_obs / 2.0) # This is tan(theta/2), where theta is the true anomaly
    orbphase = (E_obs / 2.0) % (2 * np.pi)
    quadrant = (orbphase < (np.pi / 2.0)) | (orbphase > (3.0 * np.pi / 2.0))
    theta_obs = np.ndarray(len(E_obs))
    for obs_orbphase in range(len(orbphase)):
        if quadrant[obs_orbphase]:
            theta_obs[obs_orbphase] = (2 * np.arctan(tan_theta2[obs_orbphase]))
        else:
            theta_obs[obs_orbphase] = (2 * (np.arctan(tan_theta2[obs_orbphase]) + np.pi) )
    return theta_obs


def orbital_motion(t, orb_T_0, orb_P, orb_e, orb_i, orb_omega, orb_Omega, orb_a):
    """
    From CygX1 collab
    Calculates projected components of the orbital motion on the sky, x for in RA, y for in Dec
    Parameters
    t: time
       an array in MJD format and Barycentric Dynamical Time (TDB) scale.  
    orb_T_0: Time of superior conjunction (when the BH is behind the star), in unit of days.
    orb_P: Orbital period, in unit of days.
    orb_e: Orbital eccentricity, unitless.
    orb_i: Orbital inclination to the LOS, in degrees.
    orb_omega: The argument of periastron, in degrees.
    orb_Omega: The longitude of the ascending node, in degrees.
    orb_a: The orbit size in milliarcseconds on the sky, recommended to be in degrees.
    Returns:
    x_obs, y_obs: Components of the orbital motion on the sky, in RA and Dec directions respectively, in the same units as orb_a.
    """
    E_obs = eccentric_anomaly(t, orb_T_0, orb_P, orb_e)
    theta_obs = true_anomaly(t, orb_T_0, orb_P, orb_e)
    orb_omega_rad = orb_omega * np.pi/180.0
    orb_Omega_rad = orb_Omega * np.pi/180.0
    orb_i_rad = orb_i * np.pi/180.0
    x_obs = orb_a * (1 - orb_e * np.cos(E_obs)) * ((np.cos(theta_obs + orb_omega_rad) * np.sin(orb_Omega_rad)) + (np.sin(theta_obs + orb_omega_rad) * np.cos(orb_Omega_rad) * np.cos(orb_i_rad)))
    y_obs = orb_a * (1 - orb_e * np.cos(E_obs)) * ((np.cos(theta_obs + orb_omega_rad) * np.cos(orb_Omega_rad)) - (np.sin(theta_obs + orb_omega_rad) * np.sin(orb_Omega_rad) * np.cos(orb_i_rad)))
    return x_obs, y_obs    
        
def skym7(tobs,x0=0,y0=0,pmx=0,pmy=0,rad=0,per=365.,tno=1,t0=58400.):
    #simple sky model for fast evaluation, basic function
    t = tobs[0]['t']
    if debug > 0: print('func skymm',t[0],'...',t[-1],x0,y0,pmx,pmy,rad,per,t0,tno)
    pht = 2*np.pi*(t - tno-t0)/(per*DAYSpYR)
    y = y0+(rad*np.cos(pht)+(t-t0)*pmy/DAYSpYR)/MASpDEG
    x = x0+(rad*np.sin(pht)+(t-t0)*pmx/DAYSpYR)/MASpDEG
    if debug > 4: print(x[0],'...',x[-1],'|',y[0],'...',y[-1])
    #if rad <0.:
    #    return np.full(len(t),-np.Inf),np.full(len(t),-np.Inf)
    return x,y

def skyfprlx(tobs,x0=90.,y0=30.,pmx=0.,pmy=0.,pi=1,t0=24445.):
    #Wrapper for CygX1 collab methods
    #This is a bit of a shortcut, just setting the orbit parameters to zero values
    t = tobs[0]['t']
    tskyf0 = t0 - 2400000.5
    tskyf = t - 2400000.5
    orb_a = 0.
    orb_T_0 = 0.
    orb_P = 365.25
    orb_e = 0.
    orb_i = 0.
    orb_omega = 0.
    orb_Omega = 0.
    #changed this after discussing with Paul
    pm_alpha_deg = (pmx / MASpDEG) / DAYSpYR     # Converting proper motion from milliarcsec/yr to degree/day
    pm_delta_deg = (pmy / MASpDEG) / DAYSpYR    # Converting proper motion from milliarcsec/yr to degree/day
    orb_a_deg = orb_a / MASpDEG                    # Converting orbit size from milliarcsec to degree
    parallax_deg = pi / MASpDEG            # Converting parallax from milliarcsec to degree
    x_obs, y_obs = orbital_motion(tskyf, orb_T_0, orb_P, orb_e, orb_i, orb_omega, orb_Omega, orb_a_deg)
    #Huib this is weird! The original had cosd here...
    #predict_ra = alpha_0 * np.cos(np.radians(delta)) + (pm_alpha_deg * (t - t_0(t))) + frac_alpha * parallax_deg + x_obs
    tmp_ra =  x0 + (pm_alpha_deg * (tskyf - tskyf0)) /np.cos(np.deg2rad(y0))
    tmp_dec = y0 + (pm_delta_deg * (tskyf - tskyf0))
    frac_alpha, frac_delta = frac_parallax(tskyf, tmp_ra, tmp_dec)
    #changed after discussing with Paul
    predict_ra = tmp_ra + (frac_alpha * parallax_deg + x_obs)/ np.cos(np.deg2rad(y0))
    predict_dec = tmp_dec + frac_delta * parallax_deg + y_obs
    #print('H3:',predict_ra, predict_dec)
    return predict_ra, predict_dec
    
def genf_erf(obs,**tpar):
    '''
    Wraps the function in usef_fits adding the noise and 
    makes a  difference between things to fit (least squares) and model (Bayes)
    As an error floor is the only thing we support in nuisance, this can be generic for now
    This function changes obs in place
    '''
    def c2unpack(obs,x,y):
        n1=obs[0]['nobs']
        if len(obs)>1:
            n2=obs[1]['nobs']
        for i,star in enumerate(obs):
            if i==0:
                star['x']=x[:n1]
                star['y']=y[:n1]
            else:
                star['x']=x[-n2:]
                star['y']=y[-n2:]
        return
    
    
    if debug > 4: print('func skymef:',tpar)

    #the low level function needs a tobs structure, which is the same
    #but returns x, y    
    if 'erf' in tpar:
        xpar = tpar.copy()
        del xpar['erf']
        tobs = obs
        xj, yj = usef_fits(tobs,**xpar)
        c2unpack(obs,xj,yj)
        for star in obs:
            star['y'] = star['y'] +tpar['erf']*np.random.normal(0.,1.,len(star['y']))/MASpDEG
            star['x'] = star['x'] +tpar['erf']*np.random.normal(0.,1.,len(star['x']))/MASpDEG
    else:
        tobs = obs
        xj, yj = usef_fits(tobs,**tpar)
        c2unpack(obs,xj,yj)
        if debug> 6: print('calling ',usef_fits.__name__)
    return

def lmd_generf(theta,t,x,y,xerr,yerr,touse,t0=2445.):
    '''
    THIS ROUTINE IS NOT USED?
    #simple sky model, log likelihood with error floor model for emcee
    print('YOU ARE TRYING TO FIND ERROR FLOOR, LIKELY BROKEN')
    '''
    dpar = dict(zip(thefun['tofit']+thefun['tomod'],theta))
    fpar = dict(zip(thefun['tofit'],theta[0:len(thefun['tofit'])]))

    xmod, ymod = usef_fits(t,**fpar,**touse,t0=t0)
    sigy2 = (yerr)**2 + (dpar['erf']/3600e3)**2
    sigx2 = (xerr)**2 + (dpar['erf']/3600e3)**2
#    llh = -0.5*np.sum( (y-ymod)**2/sigy2 + (x-xmod)**2/sigx2 + np.sqrt(0.5)*np.pi*np.log(sigy2 + sigx2) )
    llh = -0.5*np.sum( (y-ymod)**2/sigy2 + (x-xmod)**2/sigx2 + 2.*np.log(2*np.pi)+np.log(sigy2*sigx2))
    return llh
    
def lft_gen(theta, obs, touse, t0=2445.):
    """Calculate log likelihood for model fitting.
    
    Args:
        theta (array-like): Parameter values in order of thefun['tofit']
        obs (list): List of observation dictionaries
        touse (dict): Fixed parameters to use
        t0 (float): Reference time
            
    Returns:
        float: Log likelihood value
    """
    # Create parameter dictionary from theta array
    dpar = dict(zip(thefun['tofit'], theta))
    xmod, ymod = usef_fits(obs, **dpar, **touse, t0=t0)
        
    # Handle single or multiple objects
    if len(obs) > 1:
        y = np.concatenate((obs[0]['y'], obs[1]['y']))
        x = np.concatenate((obs[0]['x'], obs[1]['x']))
        yerr = np.concatenate((obs[0]['yr'], obs[1]['yr']))
        xerr = np.concatenate((obs[0]['xr'], obs[1]['xr']))
    else:
        y=obs[0]['y']
        x=obs[0]['x']
        yerr=obs[0]['yr']
        xerr=obs[0]['xr']
    llh = -0.5*np.sum(((y-ymod)/(2*yerr))**2 + ((x-xmod)/(2*xerr))**2 +np.log(xerr**2 + yerr**2))
    return llh
    
def lprob_gen(theta, obs ,touse,t0=2445. ):
    lp = vec_prios(theta)
    llh = lft_gen(theta, obs,touse,t0)
    if not np.isfinite(lp):
        return -np.inf
    return lp + llh     
    
def vec_prios(theta):
    #Set Gaussian/flat priors based on input data
    def gaussian(x, mu, sig):
        return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

    cpar = dict(zip(thefun['tofit']+thefun['tomod'],theta))
    #print('P',theta)
    pars = thefun['pars']
    #print(pars)
    lpriosum=0
    for p in thefun['tofit']+thefun['tomod']:
        if 'min' in pars[p]:
            if (pars[p]['min'] < cpar[p] and pars[p]['max'] > cpar[p]):
                lpriosum+=0.
            else:
                lpriosum+=-np.inf
        elif 'sig' in pars[p]:
            lpriosum+=-np.log(gaussian(cpar[p],pars[p]['ini'],pars[p]['sig']))
        else:
            break
    return lpriosum 

def report_fit(fit,fsel,outp):
    #format fit to output record
    outp.write("Maximum likelihood estimates:\n")
    for par in thefun['tofit']:
        outp.write("{} = {:.3f}\n".format(par,fit[par]))
    outp.write("\n")
    return
    
def report_resi(fsel,obs,fits):
    #reports on residuals after fit
    fpar = {}
    fpar.update(dict(zip(thefun['tofit'],[fits[key] for key in thefun['tofit']] )))
    fpar.update(dict(zip(thefun['touse'],[thefun['pars'][key]['ini'] for key in thefun['touse']] )))
    fpar.update({'t0':obs[0]['t'][0]})
    #now update nuisance par with val
    for key in thefun['tofit']:
        if 'nuisa' in thefun['pars'][key]:
            fpar[key]=thefun['pars'][key]['nuisa']
        
        #trupar = [ttru, *[truth[key] for key in (thefun['tofit']+thefun['touse'])],tobs[0]]
    if debug >5: print('fpar:',fpar)
    xpred,ypred = usef_fits(obs,**fpar)
    outp.write('Estimates from residuals:')
    for i, star in enumerate(obs):
        chix=np.sqrt(np.sum(((star['x']-seloutobs(i,star['nps'],xpred))/star['xr'])**2))
        chiy=np.sqrt(np.sum(((star['y']-seloutobs(i,star['nps'],ypred))/star['yr'])**2))
        rmsres=np.sqrt(np.sum((star['x']-seloutobs(i,star['nps'],xpred))**2 + 
             (star['y']-seloutobs(i,star['nps'],ypred))**2)/(2*len(star['x'])))*3600e3
        outp.write("star {}; rms residual: {:.3f} mas\n".format(i,rmsres))
        outp.write("chi square (x,y): {:.3f} {:.3f}\n".format(chix,chiy))
    
def report_truth(soln,outp):
    #report on the truth
    outp.write("The truth was:\n")
    for name, val in truth.items():
        outp.write("{:6} = {:6}\n".format(name,val))
    outp.write("\n")
    return

def report_init(init,func,args,outp):
    outp.write("The args for initiation: {}\n".format(args))
    nv = len(init[0])
    strf = "Init {:2} "
    for iv in range(nv):
        strf += "{:6.3f} "
    strf += ": {:9.5f}\n"
    for idim in range(len(init)):
        tmp = func(init[idim],*args)
        outp.write(strf.format(idim,*init[idim],tmp))    
    return

def check_prio(pos,fun):
    inbound = True
    for vec in pos:
        for (ip,par) in enumerate(fun['tofit']):
            #print('hhuh',ip,par,vec,fun)
            if 'min' in fun['pars'][par]:
                if (vec[ip] < fun['pars'][par]['min'] or vec[ip] > fun['pars'][par]['max']):
                    print('Makes no sense for {}: {} < {} < {}'.format(
                        par,fun['pars'][par]['min'],vec[ip],fun['pars'][par]['max']))
                    inbound = False
            elif 'sig' in fun['pars'][par]:
                if (np.abs(vec[ip]-fun['pars'][par]['ini']) > 5*fun['pars'][par]['sig']):
                    inbound = False
            else:
                break
    if (inbound): print('----- Priors make sense')
    return inbound

def openyaml(filesel):
    #load parameters for specific function
    filename = filesel+'.yaml'
    
    try:
        yhandl = open(filename)
    except IOError: 
        print("Error: File does not appear to exist.",filename)
        exit()

    fun = yaml.load(yhandl, Loader=yaml.FullLoader)
    return fun
    
def gendata(func,truth,Nst):
    obs = [{'nobs':truth['N']}]
    if Nst>1: obs.append({'nobs':truth['M']})
    errpow = truth['errpow']
    obs[0]['t'] = np.sort(truth['tref']+truth['dtobs'] * np.random.rand(truth['N']))
    #Huib shortcut, second star is ssen for M recent epochs
    if Nst>1: obs[1]['t'] = obs[0]['t'][-truth['M']:]
    if debug > 5:
        print('Adding some times for source B')
        print(obs[1]['t'])

    #The func requires a ref day, which is t[0]
    #largs = [t, *[truth[p] for p in thefun['tofit']], *[thefun['pars'][p]['ini'] for p in thefun['touse']], t[0]]

    tpar = {}
    tpar.update(dict(zip(thefun['tofit'],[truth[key] for key in thefun['tofit']] )))
    tpar.update(dict(zip(thefun['touse'],[truth[key] for key in thefun['touse']] )))
    tpar.update(dict(zip(thefun['tomod'],[truth[key] for key in thefun['tomod']] )))
    tpar.update({'t0':obs[0]['t'][0]})
    if debug > 2: print('tpar:',tpar)

    func(obs,**tpar)  
    
    if debug > 4: print('Data generated',obs )

    for i in range(Nst):
        obs[i]['y'] += errpow * np.random.normal(0.,1.,obs[i]['nobs'])/3600e3
        obs[i]['x'] += errpow * np.random.normal(0.,1.,obs[i]['nobs'])/3600e3
        obs[i]['xr'] =  np.zeros(obs[i]['nobs'])+errpow/3600e3
        obs[i]['yr'] =  np.zeros(obs[i]['nobs'])+errpow/3600e3
 
    if debug > 3:
        print('Genertated:',obs)
 
    return obs
    
def dictdump(funpars):
    outstr = ''
    for key,val in funpars.items():
        if key == 'pars':
            outstr += 'pars:\n'
            for key2, val2 in val.items():
                if (funpars['pars'][key2]['dofit']):
                    if 'min' in funpars['pars'][key2]:
                        outstr += '{:6}: ini={:9}, min={:9}, max={:9}, dofit=True\n'.format(key2,
                            funpars['pars'][key2]['ini'],funpars['pars'][key2]['min'],funpars['pars'][key2]['max'])
                    elif 'sig' in funpars['pars'][key2]:
                        outstr += '{:6}: ini={:9}, ± sig={:9}, dofit=True\n'.format(key2,
                            funpars['pars'][key2]['ini'],funpars['pars'][key2]['sig'])
                    else:
                        break
                else:
                    outstr += '{:6}: ini={:9}, dofit=False\n'.format(key2,
                        funpars['pars'][key2]['ini'])
        else:
            outstr += '{} = {}\n'.format(key,val)
    return outstr

def seloutobs(istar,nobs,xboth):
    #split the output from the model in two stars (or one)
    if istar==0: return xboth[:nobs]
    elif istar==1: return xboth[-nobs:]
    else: print('This many stars cannot be handled')
    return -1


def plot_traces(samples,labels):
    fig, axes = plt.subplots(npar, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()

    if (npar>1):
        for i in range(npar):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        if plotInteract:
            plt.show()
        else:
            plt.savefig(nttag+'_fig2trace.pdf')
    else:
        ax = axes
        ax.plot(samples[:, :, 0], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        axes.set_xlabel("step number")
        if plotInteract:
            plt.show()
        else:
            plt.savefig(nttag+'_fig2trace.pdf')
            
def plot_skym(obs,fits={},truth={},samples=[],name='fig0data', 
          submod=None, connect=False):
    '''
    Creates visualization plots for astrometric data and model fits.
    
    This function generates interactive or static plots showing astrometric positions,
    model fits, and optionally truth values or MCMC samples. It can also show residuals
    after removing specific model components.

    Parameters
    ----------
    obs : list of dict
        List of observation dictionaries, one dict for each object being fit.
        Each dictionary contains:
        - 't': observation times
        - 'x': RA coordinates
        - 'y': Dec coordinates
        - 'xr': RA uncertainties
        - 'yr': Dec uncertainties
        - 'nps': number of points for this object
    fits : dict, optional
        Dictionary of best-fit parameters for the model. If provided, will plot
        the model fit alongside the data.
    truth : dict, optional
        Dictionary of true parameter values. If provided, will plot the true
        model alongside the data.
    samples : list, optional
        MCMC samples for plotting parameter uncertainties. If provided, will plot
        multiple model realizations to show the uncertainty range.
    name : str, optional
        Root name for saving the figure (default: 'fig0data')
    submod : str, optional
        Type of model component to subtract from data. Options are:
        - 'pm': subtract proper motion
        - 'prlxpm': subtract parallax and proper motion
        - 'full': subtract full model
    connect : bool, optional
        If True, connects data points with lines (default: False)
        
    Notes
    -----
    - The function creates a figure with multiple panels showing different aspects
      of the astrometric solution
    - Coordinates are plotted relative to a reference position computed from the
      mean position of the first object
    - If multiple objects are present, they are plotted in different colors
    - Model fits and uncertainties (if provided) are overplotted on the data
    '''
    def dref2kas(ymean):
        """Convert decimal degrees to DMS format with 100mas precision.
        Returns (value, string) tuple where string is in format '±DDo MM'SS".mas'
        """
        # Handle negative values
        sign = '-' if ymean < 0 else ' '
        ymean = abs(ymean)
        
        # Extract degrees, arcmin, arcsec, milliarcsec
        ymdeg = int(ymean)
        remainder = ymean - ymdeg
        ymmin = int(remainder * 60)
        remainder = remainder * 60 - ymmin
        ymsec = int(remainder * 60)
        ymkas = int((remainder * 60 - ymsec) * 100)
        
        # Calculate reference value and format string
        yref = ymdeg + ymmin/60 + ymsec/3600 + ymkas/3600e2
        if sign == '-': 
            yref *= -1
        
        yrefstr = f"{sign}{ymdeg}o{ymmin}'{ymsec}\".{ymkas}"
        return yref, yrefstr

    def rref2das(xmean):
        """Convert decimal degrees to HMS format with 100mas precision.
        Returns (value, string) tuple where string is in format 'HHhMM'SS.d's
        Note: Input is in degrees but output reference is in hours (divided by 15)
        """
        # Convert degrees to hours
        xmean = xmean/15.
        
        # Extract hours, minutes, seconds, deciseconds
        xmhrs = int(xmean)
        remainder = xmean - xmhrs
        xmmin = int(remainder * 60)
        remainder = remainder * 60 - xmmin
        xmsec = int(remainder * 60)
        xmdas = int((remainder * 60 - xmsec) * 10)  # Truncate to deciseconds
        
        # Calculate reference value (in degrees) and format string
        xref = 15. * (xmhrs + xmmin/60. + xmsec/3600. + xmdas/36000.)
        xrefstr = f"{xmhrs:02d}h{xmmin:02d}m{xmsec:02d}.{xmdas:01d}s"
        return xref, xrefstr
        
    def setpar(pars,thefun,thet0):
        
        par={}
        par.update(dict(zip(thefun['tofit'],[pars[key] for key in thefun['tofit']] )))
        par.update(dict(zip(thefun['touse'],[thefun['pars'][key]['ini'] for key in thefun['touse']] )))
        par.update({'t0':thet0})
        #Huib ugly this is hetting obs from local scope
        #trupar = [ttru, *[pars[key] for key in (thefun['tofit']+thefun['touse'])],tobs[0]]
        if debug >5: print('par:',par)
        for key in thefun['tofit']:
            if 'nuisa' in thefun['pars'][key]:
                tpar[key]=thefun['pars'][key]['nuisa']
            
        return par
        
    def setpari(pars,thefun,thet0):
        #set par dictionary for model function
        par = {}
        par.update(dict(zip(thefun['tofit'],[pars[i] for i in range(len(thefun['tofit']))] )))
        par.update(dict(zip(thefun['touse'],[thefun['pars'][key]['ini'] for key in thefun['touse']] )))
        par.update({'t0':thet0})
        if debug >5: print('par:',par)
        for key in thefun['tofit']:
            if 'nuisa' in thefun['pars'][key]:
                tpar[key]=thefun['pars'][key]['nuisa']
        return par

    def estsub(x,y,fits,submod,tobs,t0):
        #work out various subtraction modes input are x,y coords
        #returns a new one
        if submod == 'pm':
            # this needs t concatenated
            if len(tobs) > 1:
                t = np.concatenate((tobs[0]['t'],tobs[1]['t']))
            else:
                t = obs[0]['t']
            estpmx = fits['pmx']/(MASpDEG*DAYSpYR)
            estpmy = fits['pmy']/(MASpDEG*DAYSpYR)
            x = x-estpmx*(t-t0)
            y = y-estpmy*(t-t0)
        elif submod == 'prlxpm':
        #run model for bina = 0
            if ('bina' in fits.keys()):
                #print('setting bina = 0')
                fits['bina']= 0.
            fitpar = setpar(fits,thefun,t0)                
            xmod,ymod = usef_fits(tobs, **fitpar)
            x -= xmod
            y -= ymod
        elif submod == 'full':
            fitpar = setpar(fits,thefun,t0)
            xmod,ymod = usef_fits(tobs, **fitpar)
            print('lent ',len(xmod),len(x))
            x -= xmod
            y -= ymod
        return x,y
        
    

    doSample = len(samples)>0

    # Validate input and extract observation data
    if not obs or not isinstance(obs, list) or not obs[0]:
        raise ValueError("obs must be a non-empty list containing at least one observation dictionary")
    
    try:
        t0 = obs[0]['t'][0]  # Reference time from first observation
        
        # Combine coordinates if we have multiple objects, otherwise use single object data
        if len(obs) > 1:
            xobs = np.concatenate((obs[0]['x'], obs[1]['x']))  # Concatenate first two objects
            yobs = np.concatenate((obs[0]['y'], obs[1]['y']))
        else:
            xobs = obs[0]['x']
            yobs = obs[0]['y']
    except (KeyError, AttributeError, IndexError) as e:
        raise ValueError("Invalid observation data format. Each observation must contain 't', 'x', and 'y' arrays") from e

    #print('Sure? ',xobs,yobs,tobs)
    #if submod and (not (fits or doSample)): print('Does not make sense to ')
    ntgrid = 200
    if truth or fits or doSample:
        #we thus need a fine grid of timestamps
        #implicit the 1st star is longest, use these times
        t_span = obs[0]['t'][-1] - obs[0]['t'][0]
        tgrid = np.linspace(obs[0]['t'][0] - 0.1*t_span, obs[0]['t'][-1] + 0.1*t_span, ntgrid)
        tgrobs = [{'t': tgrid}] * 2

        if truth:
            xtru, ytru = usef_fits(tgrobs, **setpar(truth, thefun, t0))
        
        if fits:
            fitpar = setpar(fits, thefun, t0)
            xfit, yfit = usef_fits(tgrobs, **fitpar)
            xpred, ypred = usef_fits(obs, **fitpar)

            if submod:  # Apply model component subtraction
                xobs, yobs = estsub(xobs, yobs, fits, submod, obs, t0)
                xpred, ypred = estsub(xpred, ypred, fits, submod, obs, t0)
                xfit, yfit = estsub(xfit, yfit, fits, submod, tgrobs, t0)
                if truth:
                    xtru, ytru = estsub(xtru, ytru, fits, submod, tgrobs, t0)

        if doSample:  # Generate ensemble predictions from MCMC samples
            xens, yens = [], []
            for ind in np.random.randint(len(samples), size=100):
                xensi, yensi = usef_fits(tgrobs, **setpari(samples[ind], thefun, t0))
                if submod:
                    xensi, yensi = estsub(xensi, yensi, setpari(samples[ind], thefun, t0), submod, tgrid, tobs[0])
                xens.append(xensi)
                yens.append(yensi)

    # Calculate reference coordinates for plotting
    yref, yrefstr = dref2kas((obs[0]['y'][0] + obs[0]['y'][-1])/2.)
    xref, xrefstr = rref2das((obs[0]['x'][0] + obs[0]['x'][-1])/2.)
    
    if doSample: print('-----Plotting a selection from samples')
    if truth: print('-----Plotting the truth as well')
    if submod: print('-----Plotting residuals after a: {}'.format(submod))
    print('-----Plotting with respect to {},{}'.format(xrefstr,yrefstr))

    font = {'size': 8}
    matplotlib.rc('font', **font)
    fig=plt.figure()
    grid = plt.GridSpec(2,5)
    #1st subplot, biggest plot with trajectory on sky --------------------------------
    ax1 = fig.add_subplot(grid[:2,:3])
    ax1.invert_xaxis()
    ratio=1/np.cos(np.pi*obs[0]['y'][0]/180.) # cos(d) for center
    ax1.set_aspect(ratio, 'datalim')

    osym = ['navy','firebrick']
    psym = ['.y','.y']
    colr=['r','b']
    tcol=['green','orange']
    fcol=['blue','red']
    ecol=['lightsteelblue','mistyrose']
    for i, star in enumerate(obs):
        ax1.errorbar(3600e3*(seloutobs(i,obs[i]['nps'],xobs)-xref), 
            3600e3*(seloutobs(i,obs[i]['nps'],yobs)-yref), 
            yerr=star['yr']*3600e3, xerr=star['xr']*3600e3, fmt='.',color=osym[i], capsize=0)
        if truth:
            ax1.plot(3600e3*(seloutobs(i,ntgrid,xtru)-xref),
                     3600e3*(seloutobs(i,ntgrid,ytru)-yref),tcol[i])
        if fits: 
            ax1.plot(3600e3*(seloutobs(i,ntgrid,xfit)-xref),
                 3600e3*(seloutobs(i,ntgrid,yfit)-yref),fcol[i])
            ax1.plot(3600e3*(seloutobs(i,star['nps'],xpred)-xref),
                 3600e3*(seloutobs(i,star['nps'],ypred)-yref),psym[i])
        if doSample:
            for isamp in range(len(xens)):
                ax1.plot(3600e3*(seloutobs(i,ntgrid,xens[isamp])-xref),
                    3600e3*(seloutobs(i,ntgrid,yens[isamp])-yref), ecol[i], alpha=0.05)
            
        if connect:
            ax1.plot(3600e3*(seloutobs(i,star['nps'],xobs)-xref), 
                     3600e3*(seloutobs(i,star['nps'],yobs)-yref), '-k')
            ax1.plot(3600e3*(seloutobs(i,star['nps'],xobs)-xref)[0],
                     3600e3*(seloutobs(i,star['nps'],yobs)-yref)[0], 'bo')
            ax1.plot(3600e3*(seloutobs(i,star['nps'],xobs)-xref)[-1],
                     3600e3*(seloutobs(i,star['nps'],yobs)-yref)[-1], 'ro')
    
    
    ax1.set_xlabel("x [mas wrt {}]".format(xrefstr))
    ax1.set_ylabel("y [mas wrt {}]".format(yrefstr))

    #2nd plot, right top RA versus time--------------------------------------------
    ax2 = fig.add_subplot(grid[0,3:])
    for i, star in enumerate(obs):
        ax2.errorbar(star['t'],3600e3*(seloutobs(i,obs[i]['nps'],xobs)-xref),yerr=star['xr'],
            fmt='.',color=osym[i])
        if truth:
            ax2.plot(tgrobs[i]['t'],3600e3*(seloutobs(i,ntgrid,xtru)-xref),tcol[i])
        if fits: 
            ax2.plot(obs[i]['t'],3600e3*(seloutobs(i,star['nps'],xpred)-xref),psym[i])
            ax2.plot(tgrobs[i]['t'],3600e3*(seloutobs(i,ntgrid,xfit)-xref),fcol[i])
    if doSample:
        for i,star in enumerate(obs):
            for isamp in range(len(xens)):
                ax2.plot(tgrid,3600e3*(seloutobs(i,ntgrid,xens[isamp])-xref), 
                    ecol[i], alpha=0.05)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel("x [mas]")
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    #3nd plot, right bottom Dec versus time-------------------------------------------
    ax3 = fig.add_subplot(grid[1,3:])
    for i, star in enumerate(obs):
        ax3.errorbar(star['t'],3600e3*(seloutobs(i,obs[i]['nps'],yobs)-yref),yerr=star['yr'],
            fmt='.',color=osym[i])
        if truth:
            ax3.plot(tgrobs[i]['t'],3600e3*(seloutobs(i,ntgrid,ytru)-yref),tcol[i])
        if fits: 
            ax3.plot(tgrobs[i]['t'],3600e3*(seloutobs(i,ntgrid,yfit)-yref),fcol[i])
            ax3.plot(obs[i]['t'],3600e3*(seloutobs(i,star['nps'],ypred)-yref),psym[i])
        if doSample:
            for isamp in range(len(xens)):
                 ax3.plot(tgrid,3600e3*(seloutobs(i,ntgrid,yens[isamp])-yref), 
                     ecol[i], alpha=0.05)
    ax3.tick_params(axis='both', which='major', labelsize=6)
    ax3.set_xlabel("t [mj]")
    ax3.set_ylabel("y [mas]")
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()

    if plotInteract:
        plt.show()
    else:
        plt.savefig(nttag+'_'+name+'.pdf')
    return

def plot_corners(flat_samples,labels,truth={}):

    ptruths = {}
    if truth: 
        ptruths = [truth[par] for par in (thefun['tofit']+thefun['tomod'])]

    
    if debug: print('Check',flat_samples.shape, labels, ptruths)
    passtru = None
    if truth: passtru = ptruths
    fig = corner.corner(flat_samples, labels=labels, truths=passtru,
            show_titles=True, label_kwargs=dict(fontsize=6), title_kwargs=dict(fontsize=6) )
    #print(fig.get_size_inches())
    #plt.rcParams.update({'font.size': 6})
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=6)
    fig.set_size_inches(8,9.3)
    
    #fig.rc('figure', figsize=(2.0, 1.0))
    if plotInteract:
        plt.show()
    else:
        plt.savefig(nttag+'_fig3corner.pdf')
    return

def steptiming(times,stagetext,init=False,outstr=None):
    times.append(utime.process_time())
    if init:
        print('-----timing {}'.format(stagetext))
    else:
        if (debug or not outp):
            print('-----timing at {:22}: {:12.3f}, elapse: {:12.3f}'.format(stagetext,times[-1]-times[0],times[-1]-times[-2]))
        else:
            outstr.write('-----timing at {:22}: {:12.3f}, elapse: {:12.3f}\n'.format(stagetext,times[-1]-times[0],times[-1]-times[-2]))
    return

def timetag():
    nowstr=str(Time(Time.now(),format='fits', out_subfmt='longdate_hms'))
    return nowstr[4:6]+nowstr[7:9]+nowstr[10:12]+'-'+nowstr[13:15]+nowstr[16:18]
    
def check_pars(funfile):
    '''
    There are nparfunc parameters for the func
    There could be 1 errorfloor par
    There are nparfit for the fit =< nparfunc
    and possibly nparbay=nparfit or nparfit+1 for Bayes
    the npar = nparbay
    '''
    thefun = openyaml(funfile)
    #print('Dump:',thefun)
        
    tofit2 = []
    for par in thefun['pars'].keys():
        if thefun['pars'][par]['dofit']:
            tofit2.append(par)
    touse2 = []
    tomod2 = []
    for par in thefun['pars'].keys():
        if (not thefun['pars'][par]['dofit']):
            if 'domod' in thefun['pars'][par]:
                if thefun['pars'][par]['domod']:
                    tomod2.append(par)
            else:
                touse2.append(par)

    if 'tofit' in thefun:
        npar2 = len(thefun['tofit'])+len(tomod2)
        npar3 = len(thefun['pars'].keys())
        if (thefun['npar'] != npar2 or 
            len(thefun['tofit']) != len(tofit2) or
            len(thefun['tomod']) != len(tomod2) ):
            print('bayes',thefun['npar'],"=",npar2,' of total',npar3)
            print('to fit',thefun['tofit'],'=',tofit2)
            print('used fixed',thefun['touse'],'=',touse2)
            print('model',thefun['tomod'],'=',tomod2)
            print("This is inconsistent")
            exit()
    else:
        thefun['npar']=len(tofit2)+len(tomod2)
        thefun['tofit']=tofit2
        thefun['touse']=touse2
        thefun['tomod']=tomod2
        
    if 'erfmod' in thefun: 
        print('----Inserting an error floor erf',thefun['erfmod'])
    else:
        print('----no erf set')
        
    if thefun['tomod']:
        print('-**-Note: modeling nuisance parameter is BROKEN')
    if thefun['touse']:
        print('-**-Note: fixing model parameters is UNTESTED')
        
    return thefun
    
    
# Cache for Skyfield data
_skyfield_cache = {}
def get_skyfield_data():
    global _skyfield_cache
    if not _skyfield_cache:
        loader = Loader('Skyfield-data')
        _skyfield_cache['planets'] = loader('de438.bsp')
        _skyfield_cache['timescale'] = loader.timescale()
    return _skyfield_cache['planets'], _skyfield_cache['timescale']
# Initialize Skyfield data
planets, ts = get_skyfield_data()

#------ Main body ---------------------------------------------------------------------------------------
# data for generation
times = []
steptiming(times,'Initialising',init=True,outstr=None)

opts = GetArgs()

debug = opts.debuglevel
root = opts.fileroot
plotInteract = not opts.saveplots
doBayes = not  opts.skipbayes
knowTruth = opts.dogenerate or opts.readtruth
plotResidual = opts.residualmotion
dumpSamples = opts.dumpsamples
cornerDump = opts.cornerdump

nttag = timetag()+'_'+root
#nttag = str(Time(Time.now(),format='fits', out_subfmt='longdate_hms'))[4:-7]+'_'+root

print('---Using {} and {}'.format(root,nttag))

#------ Generate some data --------
np.random.seed(1543)
#There are some families of functions. Derived from a model description:

ffits = {'skyfprlx':skyfprlx,
        'skyfpbin':skyfpbin,
        'pparlx':pparlx,
        'sky7':skym7,
        'skyfc2':skyfc2}

funfile = 'skym_par_'+root
datafile = 'skym_data_'+root+'.yaml'
if opts.dogenerate: datafile = nttag + '_data.yaml'
fitfile = nttag + '_post.yaml'
#read-only version has generic name
if not doBayes: fitfile = 'skym_post_'+root+'.yaml'

outfile = nttag+'_out'+'.txt'
#set a different name for runs with no major work
if opts.skipbayes: outfile = nttag+'_out_nb'+'.txt'
outp = open(outfile,'w',encoding='ascii',errors='ignore')
outp.write('Code {} run at {}\n'.format(__file__,str(Time(Time.now()))))
outp.write('Version: {}\n'.format(Version))
outp.write('Switches:'+str(opts)+'\n')
outp.write('This is processing based on input in {}\n'.format(funfile))

thefun = check_pars(funfile)
Nst = 1
if 'Nst' in thefun: Nst = thefun['Nst']
fsel = thefun['fun']
npar = thefun['npar']

usef_fits = ffits[fsel]

outp.write('Using the codes ffits:{}, fmods:{}, lhfits:{}, lhmods:{}, lpfuncs:{}\n'.
    format(usef_fits.__name__,genf_erf.__name__,
    lft_gen.__name__,genf_erf.__name__,
    lprob_gen.__name__))

outp.write('Using parameters:')
outp.write(dictdump(thefun))    

if knowTruth:
    #can be used for generating data and/or plot truth
    print('---Read truths from:',fsel)
    truthfile = 'skym_tru_'+root
    truth = openyaml(truthfile)
    errpow = truth['errpow']
    if truth['fun']==fsel:
        report_truth(truth,outp)
    else:
        raise RuntimeError("Inconsistent functions")

if opts.dogenerate:
    #Generate data and dump to file
    steptiming(times,'Generating data',outstr=outp)
    #Generate data set    
    print('---Generate data for:',fsel)
    #Huibhier dit gaat per ongeluk goed
    obs = gendata(genf_erf,truth,Nst)
    #Huib fix writing data
    tmpdata = []
    for star in obs:
        tmpdata.append({'nps':star['nobs'],'t':star['t'].tolist(), 'x':star['x'].tolist(), 'y':star['y'].tolist(), 
            'xr':star['xr'].tolist(), 'yr':star['yr'].tolist()})
    if debug > 3:
        print('Convert for yaml',tmpdata )
        print('Loooks like',yaml.dump(tmpdata))

    datafile = 'skym_data_'+root+'.yaml'
    yaml.dump(tmpdata,open(datafile,'w'))
    #tmpdata = { 't':t.tolist(), 'x':x.tolist(), 'y':y.tolist(), 'xr':xerr.tolist(), 'yerr':yerr.tolist()}
    print('writing data to yaml')
    outp.write('Using the data:\n')
    outp.write(yaml.dump(tmpdata))
    outp.write("\n")
    
    #Write data to file
    #datadump=open(datafile,"w")
    #yaml.dump(obs,datadump)
    steptiming(times,'Writing data',outstr=outp)
    
#if doFit or doBayes or plotResidual:
if True:
    #always read data
    #more work; read-back data
    steptiming(times,'Reading data',outstr=outp)
    #Huib read data
    if debug > 3: print('Reading this:',datafile)

    datasource = open(datafile,"r")
    print('---Use data in:',datafile)
    thedata = yaml.load(datasource, Loader=yaml.FullLoader)
    
    if debug > 3: print(thedata)

    obs = []
    for star in thedata:
        obs.append({'nps':star['nps'],
            't':np.array(star['t']),
            'x':np.array(star['x']),
            'y':np.array(star['y']),
            'xr':np.array(star['xr']),
            'yr':np.array(star['yr'])})

    # Maximum likelihood first
    np.random.seed(1203)
    outp.write('Using seed2 {}\n'.format(np.random.random()))

    if knowTruth:
        plot_skym(obs,truth=truth)
    else:
        plot_skym(obs)

if opts.dofit:
    #Do a fit with minimising llh
    steptiming(times,'Initialising minimization',outstr=outp)
    
    nll = lambda *args: -lft_gen(*args)
    if debug > 1: print('Using initials from par specs')
    initial = [thefun['pars'][x]['ini'] for x in thefun['tofit']]
    #initial = np.array(inip + 1.0 * np.random.randn(npar))

    upar = dict(zip(thefun['touse'],[thefun['pars'][x]['ini'] for x in thefun['touse']]))
    selargs = (obs,upar, obs[0]['t'][0])
    #selargs = (t, x, y, xerr, yerr,upar, t[0])
    if debug > 1: print('Sel arguments F: {}\n'.format(selargs))
    
    print('---Minimise')
    if debug > 1: print('Initial: {}\n'.format(initial))
    if debug > 1: print('Args: {}\n'.format(selargs))

    steptiming(times,'Starting minimization',outstr=outp)


    #Huibhier dit is moeilijk
    soln = minimize(nll, initial, selargs, method = 'Nelder-Mead', options={'maxiter':5000})
    steptiming(times,'Finishing minimization',outstr=outp)
    if (soln.success): 
        print("-----Succesful minimisation")
    else:
        print("-----Minimisation failed")
    mfit=dict(zip(thefun['tofit'],soln.x))
        
    report_fit(mfit,fsel,outp)
    outp.write(str(soln)+'\n')
    if knowTruth:
        plot_skym(obs,fits=mfit,truth=truth,name='fig1fit')
    else:
        plot_skym(obs,fits=mfit,name='fig1fit')

if doBayes:
    steptiming(times,'Initialising Bayes',outstr=outp)
    nwalker = thefun['nwalker']
    nlength = thefun['nlength']
    print('---Go Bayes')
    upar = dict(zip(thefun['touse'],[thefun['pars'][x]['ini'] for x in thefun['touse']]))
    selargs = (obs,upar, obs[0]['t'][0])
    #selargs = (t, x, y, xerr, yerr,upar, t[0])
    #selargs = (t, x, y, xerr, yerr, *[thefun['pars'][x]['ini'] for x in thefun['touse']], t[0])

    if opts.dofit:
        pos = [mfit[par] for par in thefun['tofit']]
        for xname in thefun['tomod']:
            pos.append(thefun['pars'][xname]['ini'])
        #print('fun',thefun['tofit'],thefun['tomod'])
        pos = pos + 1e-4 * np.random.randn(nwalker, npar)
    else:
        inip = [thefun['pars'][xname]['ini'] for xname in thefun['tofit']]
        for xname in thefun['tomod']:
            inip.append(thefun['pars'][xname]['ini'])
        pos = inip + 1e-4 * np.random.randn(nwalker, npar)

    check_prio(pos,thefun)

    if debug: print('Shape walkers',nwalker, npar, pos.shape)
    if debug: print('Sel arguments B: {}\n'.format(selargs))
    report_init(pos,lprob_gen,selargs,outp)

    sampler = emcee.EnsembleSampler(
        nwalker, npar, lprob_gen, args=selargs)
    steptiming(times,'Starting MCMC',outstr=outp)
    
    #sampler.run_mcmc(pos, nlength, progress=plotInteract);
    sampler.run_mcmc(pos, nlength, progress=True);
    steptiming(times,'Ending MCMC',outstr=outp)

    plot_traces(sampler,thefun['tofit']+thefun['tomod'])
    try:
        tau = sampler.get_autocorr_time()
        outp.write('tau: {}\n'.format(tau))
        print("-----Got tau consistently",tau)
    except:
        print("-----Failed at tau, ohh well")

    ndisc = int(nlength*thefun['fdisc'])
    #Huib, kludge
    fthin = 15
    outp.write('discarding {} and thinning by {}\n'.format(ndisc,fthin))
    flat_samples = sampler.get_chain(discard=ndisc, thin=fthin, flat=True)
    outp.write('shape of samples {}\n'.format(flat_samples.shape))
    
    tmpout = {}
    for i,par in enumerate(thefun['tofit']+thefun['tomod']):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}\n"
        txt = txt.format(mcmc[1], q[0], q[1], par)
        outp.write(txt)
        tmpout.update({par:{'fit':float(mcmc[1]),'minus':float(q[0]),'plus':float(q[1])}})
        #display(Math(txt))y
    tmpout.update({'tref':float(obs[0]['t'][0]),'dtobs':float(obs[0]['t'][-1]-obs[0]['t'][0])})
    outp.write('Inferred parameters:\n')
    outp.write(yaml.dump(tmpout))

    if (dumpSamples or plotResidual):
        print('---writing out traces')
        tmpout.update({'traces':flat_samples.tolist()})
    #outp.write(yaml.dump(tmpout))

    fitdump=open(fitfile,"w")
    yaml.dump(dict(tmpout),fitdump)
    
if plotResidual or cornerDump:
    try:
        fitsource = open(fitfile,"r")
    except IOError: 
        print("Error: File does not appear to exist.",fitfile)
        exit()

    print('---Use fits in:',fitfile)
    thefit = yaml.load(fitsource, Loader=yaml.FullLoader)
    if not 'traces' in thefit:
        print('----These fits did not save samples!')
        dumpSamples = False
        cornerDump = False
    else:
        flat_samples = np.array(thefit['traces'])


if doBayes or cornerDump:
    if knowTruth:
        plot_corners(flat_samples,thefun['tofit']+thefun['tomod'],truth)
    else:
        plot_corners(flat_samples,thefun['tofit']+thefun['tomod'])
    
    if knowTruth:
        plot_skym(obs,samples=flat_samples,truth=truth,name='fig4ens')
        #plot_ensemble(fsel,thefun,t,x,y,xerr,yerr,t[0],(t[-1]-t[0]),flat_samples,truth)
    else:
        plot_skym(obs,samples=flat_samples,name='fig4ens')
        #plot_ensemble(fsel,thefun,t,x,y,xerr,yerr,t[0],(t[-1]-t[0]),flat_samples)   
     
        
if plotResidual:

    bfit = {}
    for par in thefun['tofit']:
        bfit.update({par:thefit[par]['fit']})
    report_resi(fsel,obs,bfit)
    
    if dumpSamples:
        if knowTruth:
            plot_skym(obs,fits=bfit,truth=truth,samples=thefit['traces'],
                name='fig5subpm',submod=opts.residualmotion,connect=True)
        else:
            plot_skym(obs,fits=bfit,samples=thefit['traces'],name='fig5subpm',
                submod=opts.residualmotion,connect=True)
    else:
        if knowTruth:
            plot_skym(obs,fits=bfit,truth=truth,name='fig5subpm',
                submod=opts.residualmotion,connect=True)
        else:
            plot_skym(obs,fits=bfit,name='fig5subpm',
                submod=opts.residualmotion,connect=True)
steptiming(times,'Finishing',outstr=outp)
outp.close()
