#!/usr/bin/env python3

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

MASpDEG = 3.6e6
#DAYSpYR = 365.24217 #replaced to Julian date definition
DAYSpYR = 365.25

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
                pm_ra_cosdec =  	-801.551 * u.mas / u.year,
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

def skyfpbin(t,x0=90.,y0=30.,pmx=0.,pmy=0.,pi=1,binP=1.,bina=0.,bine=0,binT0=0.,binom=0.,binbigOm=0.,bini=0.,t0=24445.):
    #wrapper around skyfield 
    #developed as rwful_skyf
    if (debug >= 2): print('skyfpbin:',locals())
    tskyf0 = t0 - 2400000.5
    tskyf = t - 2400000.5
    orb_a = bina
    orb_T_0 = binT0 -2400000.5 + t0
    orb_P = binP
    orb_e = bine
    orb_i = bini
    orb_omega = binom
    orb_Omega = binbigOm
    #Huib took out the division here, do later
    pm_alphacosd_deg = (pmx / MASpDEG) / DAYSpYR    # Converting proper motion from milliarcsec/yr to degree/day
    pm_delta_deg = (pmy / MASpDEG) / DAYSpYR    # Converting proper motion from milliarcsec/yr to degree/day
    orb_a_deg = orb_a / MASpDEG                    # Converting orbit size from milliarcsec to degree
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
        
def skym7(t,x0=0,y0=0,pmx=0,pmy=0,rad=0,per=365.,tno=1,t0=58400.):
    #simple sky model for fast evaluation, basic function
    if debug > 0: print('func skymm',t[0],'...',t[-1],x0,y0,pmx,pmy,rad,per,t0,tno)
    pht = 2*np.pi*(t - tno-t0)/(per*DAYSpYR)
    y = y0+(rad*np.cos(pht)+(t-t0)*pmy/DAYSpYR)/MASpDEG
    x = x0+(rad*np.sin(pht)+(t-t0)*pmx/DAYSpYR)/MASpDEG
    if debug > 4: print(x[0],'...',x[-1],'|',y[0],'...',y[-1])
    #if rad <0.:
    #    return np.full(len(t),-np.Inf),np.full(len(t),-np.Inf)
    return x,y

def skyfprlx(t,x0=90.,y0=30.,pmx=0.,pmy=0.,pi=1,t0=24445.):
    #Wrapper for CygX1 collab methods
    #This is a bit of a shortcut, just setting the orbit parameters to zero values
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
    
def genf_erf(**tpar):
    #As an error floor is the only thing we support in nuisance, this can be generic for now
    if debug > 4: print('func skymef:',tpar)

    if 'erf' in tpar:
        xpar = tpar.copy()
        del xpar['erf']
        x,y = usef_fits(**xpar)
        y = y+tpar['erf']*np.random.normal(0.,1.,len(y))/MASpDEG
        x = x+tpar['erf']*np.random.normal(0.,1.,len(x))/MASpDEG
    else:
        x,y = usef_fits(**tpar)
        #print('HIER',x,y)

    if debug > 4: print(x[0],'...',x[-1],'|',y[0],'...',y[-1])

    return x,y

def lmd_generf(theta,t,x,y,xerr,yerr,touse,t0=2445.):
    #simple sky model, log likelihood with error floor model for emcee
    dpar = dict(zip(thefun['tofit']+thefun['tomod'],theta))
    fpar = dict(zip(thefun['tofit'],theta[0:len(thefun['tofit'])]))

    xmod, ymod = usef_fits(t,**fpar,**touse,t0=t0)
    sigy2 = (yerr)**2 + (dpar['erf']/3600e3)**2
    sigx2 = (xerr)**2 + (dpar['erf']/3600e3)**2
#    llh = -0.5*np.sum( (y-ymod)**2/sigy2 + (x-xmod)**2/sigx2 + np.sqrt(0.5)*np.pi*np.log(sigy2 + sigx2) )
    llh = -0.5*np.sum( (y-ymod)**2/sigy2 + (x-xmod)**2/sigx2 + 2.*np.log(2*np.pi)+np.log(sigy2*sigx2))
    return llh
    
def lft_gen(theta,t,x,y,xerr,yerr,touse,t0=2445.):
    #simple sky model, log likelihood pure function for fitting
    dpar = dict(zip(thefun['tofit'],theta))
    #upar = dict(zip(thefun['touse'],touse))
    xmod, ymod = usef_fits(t,**dpar,**upar,t0=t0)
    llh = -0.5*np.sum(((y-ymod)/(2*yerr))**2 + ((x-xmod)/(2*xerr))**2 +np.log(xerr**2 + yerr**2))
    return llh
    
def lprob_gen(theta, t, x, y, xerr, yerr,touse,t0=2445. ):
    lp = vec_prios(theta)
    llh = lft_gen(theta,t,x,y,xerr,yerr,touse,t0)
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
    
def report_resi(fsel,t,x,y,xerr,yerr,fits):
    #reports on residuals after fit
    fpar = {'t':t}
    fpar.update(dict(zip(thefun['tofit'],[fits[key] for key in thefun['tofit']] )))
    fpar.update(dict(zip(thefun['touse'],[thefun['pars'][key]['ini'] for key in thefun['touse']] )))
    fpar.update({'t0':t[0]})
    #now update nuisance par with val
    for key in thefun['tofit']:
        if 'nuisa' in thefun['pars'][key]:
            fpar[key]=thefun['pars'][key]['nuisa']
        
        #trupar = [ttru, *[truth[key] for key in (thefun['tofit']+thefun['touse'])],tobs[0]]
    if debug >5: print('fpar:',fpar)
    xpred,ypred = usef_fits(**fpar)
    outp.write('Estimates from residuals:')
    chix=np.sqrt(np.sum(((x-xpred)/xerr)**2))
    chiy=np.sqrt(np.sum(((y-ypred)/yerr)**2))
    rmsres=np.sqrt(np.sum((x-xpred)**2 + (y-ypred)**2)/(2*len(x)))*3600e3
    outp.write("rms residual: {:.3f} mas\n".format(rmsres))
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
    

    
def gendata(func,truth):
    nps = truth['N']
    errpow = truth['errpow']
    t = np.sort(truth['tref']+truth['dtobs'] * np.random.rand(nps))
    ynoi = errpow * np.random.normal(0.,1.,nps)
    xnoi = errpow * np.random.normal(0.,1.,nps)

    #The func requires a ref day, which is t[0]
    largs = [t, *[truth[p] for p in thefun['tofit']], *[thefun['pars'][p]['ini'] for p in thefun['touse']], t[0]]

    tpar = {'t':t}
    tpar.update(dict(zip(thefun['tofit'],[truth[key] for key in thefun['tofit']] )))
    tpar.update(dict(zip(thefun['touse'],[truth[key] for key in thefun['touse']] )))
    tpar.update(dict(zip(thefun['tomod'],[truth[key] for key in thefun['tomod']] )))
    tpar.update({'t0':t[0]})
    if debug >5: print('tpar:',tpar)
    x,y = func(**tpar)
    
    y += ynoi/3600e3
    x += xnoi/3600e3
    yerr = np.zeros(nps)+errpow/3600e3
    xerr = np.zeros(nps)+errpow/3600e3
    if debug > 3:
        print('Genertated:',t,x,y,xerr,yerr)
    return t,x,y,xerr,yerr
    
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
            
def plot_skym(tobs,xobs,yobs,xerr,yerr,fits={},truth={},samples=[],name='fig0data', submod=None, connect=False):
    '''
    Plots everything interactively or hardcopy, inputs:
    - tobs, xobs, yobs, xerr, yerr: the data
    - fits: parameters of best fit
    - truth: parameters of a-priori truth
    - samples: instead of fit, use ensemble of traces
    - name: create a figure with this root
    - submod: plot data and fit after removing specific model contributions
    - connect: connect data points
    
    Depends on fsel & thefun in global scope
    '''
    def dref2kas(ymean):
        #find ref coord to nearest 100mas
        neg = False
        sign = ' '
        if (ymean < 0.):
            neg=True
            sign='-'
            ymean=abs(ymean)
        ymdeg = int(ymean)
        ymean = ymean - ymdeg
        ymmin = int(ymean*60.)
        ymean = ymean*60 - ymmin
        ymsec = int(ymean*60.)
        ymean = ymean*60. - ymsec
        ymkas = int(ymean*100.)
        ymean = ymean*100 - ymkas
        yref = ymdeg+ymmin/60.+ymsec/3600.+ymkas/3600e2
        if neg: yref*=-1.
        yrefstr = '{}{}o{}\'{}\".{}'.format(sign,ymdeg,ymmin,ymsec,ymkas)
        return yref,yrefstr
        
    def rref2das(xmean):
        #find ref coord to nearest 100mas
        xmean = xmean/15.
        xmhrs = int(xmean)
        xmean = xmean - xmhrs
        xmmin = int(xmean*60.)
        xmean = xmean*60 - xmmin
        xmsec = int(xmean*60.)
        xmean = xmean*60. - xmsec
        xmdas = int(xmean*10.)
        xmean = xmean*100 - xmdas
        xref = 15.*(xmhrs+xmmin/60.+xmsec/3600.+xmdas/3600e1)
        xrefstr = '{}h{}m{}.{}s'.format(xmhrs,xmmin,xmsec,xmdas)
        return xref,xrefstr
        
    def setpar(tgrid,pars,thefun):
        #set par dictionary for model function
        par = {'t':tgrid}
        par.update(dict(zip(thefun['tofit'],[pars[key] for key in thefun['tofit']] )))
        par.update(dict(zip(thefun['touse'],[thefun['pars'][key]['ini'] for key in thefun['touse']] )))
        par.update({'t0':tobs[0]})
        #trupar = [ttru, *[pars[key] for key in (thefun['tofit']+thefun['touse'])],tobs[0]]
        if debug >5: print('par:',par)
        for key in thefun['tofit']:
            if 'nuisa' in thefun['pars'][key]:
                tpar[key]=thefun['pars'][key]['nuisa']
        return par
        
    def setpari(tgrid,pars,thefun):
        #set par dictionary for model function
        par = {'t':tgrid}
        par.update(dict(zip(thefun['tofit'],[pars[i] for i in range(len(thefun['tofit']))] )))
        par.update(dict(zip(thefun['touse'],[thefun['pars'][key]['ini'] for key in thefun['touse']] )))
        par.update({'t0':tobs[0]})
        if debug >5: print('par:',par)
        for key in thefun['tofit']:
            if 'nuisa' in thefun['pars'][key]:
                tpar[key]=thefun['pars'][key]['nuisa']
        return par

    def estsub(x,y,fits,submod,t,tref) :   
        if submod == 'pm':
            estpmx = fits['pmx']/(MASpDEG*DAYSpYR)
            estpmy = fits['pmy']/(MASpDEG*DAYSpYR)
            #print('voor:',xobs[0],xobs[-1])
            x = x-estpmx*(t-tref)
            #print('na:',xobs[0],xobs[-1])
            y = y-estpmy*(t-tref)
        elif submod == 'prlxpm':
        #run model for bina = 0
            if ('bina' in fits.keys()):
                #print('setting bina = 0')
                fits['bina']= 0.
            fitpar = setpar(t,fits,thefun)                
            xmod,ymod = usef_fits(**fitpar)
            x -= xmod
            y -= ymod
        elif submod == 'full':
            fitpar = setpar(t,fits,thefun)                
            xmod,ymod = usef_fits(**fitpar)
            x -= xmod
            y -= ymod
        return x, y

    doSample = len(samples)>0   

    #if submod and (not (fits or doSample)): print('Does not make sense to ')

    if truth or fits or doSample:
        #we thus need a fine grid of timestamps
        tgrid=np.linspace(tobs[0]-0.1*(tobs[-1]-tobs[0]),tobs[-1]+0.1*(tobs[-1]-tobs[0]),200)
    if truth:
        trupar = setpar(tgrid,truth,thefun)
        xtru,ytru = usef_fits(**trupar)
    if fits:
        fitpar = setpar(tgrid,fits,thefun)
        xfit,yfit = usef_fits(**fitpar)
        fitpar['t']=tobs
        xpred,ypred = usef_fits(**fitpar)
        
    if submod:
        xobs,yobs = estsub(xobs,yobs,fits,submod,tobs,tobs[0])
        xpred, ypred = estsub(xpred,ypred,fits,submod,tobs,tobs[0])
        xfit,yfit = estsub(xfit,yfit,fits,submod,tgrid,tobs[0])
        if truth:
            xtru,ytru = estsub(xtru,ytru,fits,submod,tgrid,tobs[0])
        
    if doSample:
        inds = np.random.randint(len(samples), size=100)
        xens = []; yens = []
        
        for ind in inds:
            sample = samples[ind]
            enspar = setpari(tgrid,samples[ind],thefun)
            xensi,yensi=usef_fits(**enspar)
            if (submod):
                xensi,yensi = estsub(xensi,yensi,enspar,submod,tgrid,tobs[0])        
            xens.append(xensi)
            yens.append(yensi)

    yref,yrefstr = dref2kas((yobs[0]+yobs[-1])/2.)
    xref,xrefstr = rref2das((xobs[0]+xobs[-1])/2.)
    
    if doSample: print('-----Plotting a selection from samples')
    if truth: print('-----Plotting the truth as well')
    if submod: print('-----Plotting residuals after a: {}'.format(submod))
    print('-----Plotting with respect to {},{}'.format(xrefstr,yrefstr))

    font = {'size'   : 8}
    matplotlib.rc('font', **font)
    fig=plt.figure()
    grid = plt.GridSpec(2,5)
    ax1 = fig.add_subplot(grid[:2,:3])
    ax1.invert_xaxis()
    ratio=1/np.cos(np.pi*yobs[0]/180.) # cos(d) for center
    ax1.set_aspect(ratio, 'datalim')

    ax1.errorbar(3600e3*(xobs-xref), 3600e3*(yobs-yref), yerr=yerr*3600e3, xerr=xerr*3600e3, fmt=".k", capsize=0)
    if connect:
        ax1.plot(3600e3*(xobs-xref), 3600e3*(yobs-yref), '-k')
        ax1.plot(3600e3*(xobs-xref)[0], 3600e3*(yobs-yref)[0], 'bo')
        ax1.plot(3600e3*(xobs-xref)[-1], 3600e3*(yobs-yref)[-1], 'ro')
    if truth: ax1.plot(3600e3*(xtru-xref),3600e3*(ytru-yref),'r')
    if fits: ax1.plot(3600e3*(xfit-xref),3600e3*(yfit-yref),'g-')
    if fits: ax1.plot(3600e3*(xpred-xref),3600e3*(ypred-yref),'go')
    if doSample:
        for isamp in range(len(xens)):
            ax1.plot(3600e3*(xens[isamp]-xref),3600e3*(yens[isamp]-yref), "grey", alpha=0.05)

    ax1.set_xlabel("x [mas wrt {}]".format(xrefstr))
    ax1.set_ylabel("y [mas wrt {}]".format(yrefstr))

    ax2 = fig.add_subplot(grid[0,3:])
    ax2.errorbar(tobs,3600e3*(xobs-xref),yerr=xerr,fmt=".k")

    if truth: ax2.plot(tgrid,3600e3*(xtru-xref),'r')
    if fits: ax2.plot(tgrid,3600e3*(xfit-xref),'g-')
    if fits: ax2.plot(tobs,3600e3*(xpred-xref),'go')
    if doSample:
        for isamp in range(len(xens)):
            ax2.plot(tgrid,3600e3*(xens[isamp]-xref), "grey", alpha=0.05)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel("x [deg]")
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    ax3 = fig.add_subplot(grid[1,3:])
    ax3.errorbar(tobs,3600e3*(yobs-yref),yerr=yerr,fmt=".k")
    if truth: ax3.plot(tgrid,3600e3*(ytru-yref),'r')
    if fits: ax3.plot(tgrid,3600e3*(yfit-yref),'g-')
    if fits: ax3.plot(tobs,3600e3*(ypred-yref),'go')
    if doSample:
        for isamp in range(len(xens)):
            ax3.plot(tgrid,3600e3*(yens[isamp]-yref), "grey", alpha=0.05)
    ax3.tick_params(axis='both', which='major', labelsize=6)
    ax3.set_xlabel("t [mj]")
    ax3.set_ylabel("y [deg]")
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
        print('erf',thefun['erfmod'])
    else:
        print('----no erf set')
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

#The first are the simple descriptions, used for displaying the result, root for the others
ffits = {'skyfprlx':skyfprlx,
         'skyfpbin':skyfpbin,
         'pparlx':pparlx,
         'sky7':skym7}

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
outp.write('Switches:'+str(opts)+'\n')
outp.write('This is processing based on input in {}\n'.format(funfile))

thefun = check_pars(funfile)
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
    t,x,y,xerr,yerr = gendata(genf_erf,truth)
    tmpdata = { 't':t.tolist(), 'x':x.tolist(), 'y':y.tolist(), 'xerr':xerr.tolist(), 'yerr':yerr.tolist()}
    outp.write('Using the data:\n')
    outp.write(yaml.dump(tmpdata))
    outp.write("\n")
    
    #Write data to file
    datadump=open(datafile,"w")
    yaml.dump(tmpdata,datadump)
    steptiming(times,'Writing data',outstr=outp)
    
#if doFit or doBayes or plotResidual:
if True:
    #always read data
    #more work; read-back data
    steptiming(times,'Reading data',outstr=outp)
    #Huib read data

    datasource = open(datafile,"r")
    print('---Use data in:',datafile)
    thedata = yaml.load(datasource, Loader=yaml.FullLoader)
    
    t = np.array(thedata['t'])
    x = np.array(thedata['x'])
    y = np.array(thedata['y'])
    xerr = np.array(thedata['xerr'])
    yerr = np.array(thedata['yerr'])

    # Maximum likelihood first
    np.random.seed(1203)
    outp.write('Using seed2 {}\n'.format(np.random.random()))

if knowTruth:
    plot_skym(t,x,y,xerr,yerr,truth=truth)
else:
    plot_skym(t,x,y,xerr,yerr)

if opts.dofit:
    #Do a fit with minimising llh
    steptiming(times,'Initialising minimization',outstr=outp)
    nll = lambda *args: -lft_gen(*args)
    if debug > 1: print('Using initials from par specs')
    initial = [thefun['pars'][x]['ini'] for x in thefun['tofit']]
    #initial = np.array(inip + 1.0 * np.random.randn(npar))

    upar = dict(zip(thefun['touse'],[thefun['pars'][x]['ini'] for x in thefun['touse']]))
    selargs = (t, x, y, xerr, yerr,upar, t[0])
    #selargs = (t, x, y, xerr, yerr, *[thefun['pars'][x]['ini'] for x in thefun['touse']], t[0])
    if debug > 1: print('Sel arguments F: {}\n'.format(selargs))
    
    print('---Minimise')
    if debug > 1: print('Initial: {}\n'.format(initial))
    if debug > 1: print('Args: {}\n'.format(selargs))

    steptiming(times,'Starting minimization',outstr=outp)

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
        plot_skym(t,x,y,xerr,yerr,fits=mfit,truth=truth,name='fig1fit')
    else:
        plot_skym(t,x,y,xerr,yerr,fits=mfit,name='fig1fit')

if doBayes:
    steptiming(times,'Initialising Bayes',outstr=outp)
    nwalker = thefun['nwalker']
    nlength = thefun['nlength']
    print('---Go Bayes')
    upar = dict(zip(thefun['touse'],[thefun['pars'][x]['ini'] for x in thefun['touse']]))
    selargs = (t, x, y, xerr, yerr,upar, t[0])
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
    tmpout.update({'tref':float(t[0]),'dtobs':float(t[-1]-t[0])})
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
        plot_skym(t,x,y,xerr,yerr,samples=flat_samples,truth=truth,name='fig4ens')
        #plot_ensemble(fsel,thefun,t,x,y,xerr,yerr,t[0],(t[-1]-t[0]),flat_samples,truth)
    else:
        plot_skym(t,x,y,xerr,yerr,samples=flat_samples,name='fig4ens')
        #plot_ensemble(fsel,thefun,t,x,y,xerr,yerr,t[0],(t[-1]-t[0]),flat_samples)   
     
        
if plotResidual:

    bfit = {}
    for par in thefun['tofit']:
        bfit.update({par:thefit[par]['fit']})
    report_resi(fsel,t,x,y,xerr,yerr,bfit)
    
    if dumpSamples:
        if knowTruth:
            plot_skym(t,x,y,xerr,yerr,fits=bfit,truth=truth,samples=thefit['traces'],name='fig5subpm',submod=opts.residualmotion,connect=True)
        else:
            plot_skym(t,x,y,xerr,yerr,fits=bfit,samples=thefit['traces'],name='fig5subpm',submod=opts.residualmotion,connect=True)
    else:
        if knowTruth:
            plot_skym(t,x,y,xerr,yerr,fits=bfit,truth=truth,name='fig5subpm',submod=opts.residualmotion,connect=True)
        else:
            plot_skym(t,x,y,xerr,yerr,fits=bfit,name='fig5subpm',submod=opts.residualmotion,connect=True)
steptiming(times,'Finishing',outstr=outp)
outp.close()

