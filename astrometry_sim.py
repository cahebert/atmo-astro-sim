import os
import time
import pickle

import galsim
import numpy as np
from scipy.optimize import bisect
from astropy.utils.console import ProgressBar

import sys
sys.path.append('/Users/clairealice/Documents/repos/psf-weather-station')
import psfws

import astropy.io.fits as fits

# utilities
def lodToDol(lst):
    keys = lst[0].keys()
    out = {}
    for k in keys:
        out[k] = []
    for l in lst:
        for k in keys:
            out[k].append(l[k])
    return out

def loadExpInfo(rng):
    exp = np.load('./Exposures_altaz.npy')
    exp_id = np.random.default_rng(rng.raw()).choice(len(exp))
    mjd, sigmaPsf, skyBright, filter_id, alt, az = exp[exp_id]
    filterName = ['g', 'r', 'i', 'z'][int(filter_id)]
    return filterName, sigmaPsf, alt, az, mjd

def vkSeeing(r0_500, wavelength, L0):
    kolm_seeing = galsim.Kolmogorov(r0_500=r0_500, lam=wavelength).fwhm
    r0 = r0_500 * (wavelength/500)**1.2
    arg = 1. - 2.183*(r0/L0)**0.356
    factor = np.sqrt(arg) if arg > 0.0 else 0.0
    return kolm_seeing*factor

def seeingResid(r0_500, wavelength, L0, targetSeeing):
    return vkSeeing(r0_500, wavelength, L0) - targetSeeing

def compute_r0_500(wavelength, L0, targetSeeing):
    """Returns r0_500 to use to get target seeing."""
    r0_500_max = min(1.0, L0*(1./2.183)**(-0.356)*(wavelength/500.)**1.2)
    r0_500_min = 0.01
    return bisect(seeingResid, r0_500_min, r0_500_max, args=(wavelength, L0, targetSeeing))

def genAtmSummary(rng):
    wlen_dict = dict(g=480.03, r=622.20, i=754.06, z=868.21)
    filterName, sigmaPsf, alt, az, mjd = loadExpInfo(rng)
    wavelength = wlen_dict[filterName]
    targetFWHM = sigmaPsf * 2.38  # (convert back to FWHM from Daniel's conversion)
    
    # Draw L0 from truncated log normal
    gd = galsim.GaussianDeviate(rng)
    L0 = 0
    while L0 < 10.0 or L0 > 100:
        L0 = np.exp(gd() * 0.6 + np.log(25.0))
    
    # Compute r0_500 that yields targetFWHM
    r0_500 = compute_r0_500(wavelength, L0, targetFWHM)
    airmass = 1/np.cos((90-alt)*np.pi/180)

    return {
        'MJD':mjd,
        'filterName':filterName,
        'wavelength':wavelength,
        'targetFWHM':targetFWHM,
        'L0':L0,
        'r0_500':r0_500,
        'alt':alt,
        'az':az,
        'airmass':airmass
    }

# generate realization parameters
def genAtmKwargs(rng, atmSummary, args):
    ws = psfws.ParameterGenerator(seed=args.atmSeed)
    pt = ws.draw_datapoint()
    params = ws.get_parameters(pt, nl=6, location='com', skycoord=True,
                               alt=atmSummary['alt'], az=atmSummary['az'])
    params['h'] = [p - ws.h0 for p in params['h']]
    params['h'][0] += args.glHeight
    params['phi'] = [p * galsim.degrees for p in params['phi']]

    # Broadcast outer scale
    L0 = [atmSummary['L0']]*6
    
    atmKwargs = dict(
            r0_500=atmSummary['r0_500'],
            L0=L0,
            speed=list(params['speed']),
            direction=params['phi'],
            altitude=params['h'],
            r0_weights=params['j'],
            screen_size=args.screen_size,
            screen_scale=args.screen_scale,
            rng=rng
        )

    return atmKwargs


if __name__ == '__main__':
    from argparse import ArgumentParser
    from multiprocessing import Pool
    parser = ArgumentParser()
    parser.add_argument('--atmSeed', type=int, default=1)
    parser.add_argument('--psfSeed', type=int, default=2)
    parser.add_argument('--screen_size', type=float, default=819.2)
    parser.add_argument('--screen_scale', type=float, default=0.1)
    parser.add_argument('--outdir', type=str, default='output')
    parser.add_argument('--nPhot', type=int, default=1e6)
    parser.add_argument('--outfile', type=str, default='out.pkl')
    parser.add_argument('--nPool', type=int, default=10)
    parser.add_argument('--exptime', type=int, default=30.)
    parser.add_argument('--glHeight', type=float, default=0.4, help="GL height")
    args = parser.parse_args()
    
    # Generate random atmospheric input statistics
    atmRng = galsim.BaseDeviate(args.atmSeed)
    atmSummary = genAtmSummary(atmRng)
    atmKwargs = genAtmKwargs(atmRng, atmSummary, args)

    atm = galsim.Atmosphere(**atmKwargs)
    aper = galsim.Aperture(
        diam=8.36, obscuration=0.61,
        lam=atmSummary['wavelength'], 
        screen_list=atm
    )

    r0 = atmSummary['r0_500'] * (atmSummary['wavelength']/500)**1.2
    kcrit = 0.2
    kmax = kcrit / r0
    print("instantiating")
    atm.instantiate(kmax=kmax, check='phot')
    print("done")

    hdu = fits.open('./stars.fits')
    header = hdu[1].header
    stars = hdu[1].data
    hdu.close()

    psfRng = galsim.BaseDeviate(args.psfSeed)
    ud = galsim.UniformDeviate(psfRng)
    
    keep = [stars['MAG_AUTO_G']<40]
    ras = stars['ALPHAWIN_J2000'][keep]
    decs = stars['DELTAWIN_J2000'][keep]
    
    grid = np.load('./Simulated_grid.npy')
    rasTest = grid[:,1] 
    decsTest = grid[:,0] 

    psfIsTest = [0]*len(ras) + [1]*len(rasTest)

    psfPhotSeeds = np.empty(len(psfIsTest), dtype=float)
    ud.generate(psfPhotSeeds)
    psfPhotSeeds *= 2**20
    psfPhotSeeds = psfPhotSeeds.astype(np.int64)

    # convert to position on focal plane
    all_ras = np.concatenate([ras, rasTest])
    mean_ras = np.mean(all_ras)
    all_decs = np.concatenate([decs, decsTest])
    mean_decs = np.mean(all_decs)
    thxs = all_ras - mean_ras
    thys = all_decs - mean_decs

    def f(aaaa):
        thx, thy, flagTest, seed = aaaa
        rng = galsim.BaseDeviate(int(seed))
        theta = (thx*galsim.degrees, thy*galsim.degrees)
        psf = atm.makePSF(atmSummary['wavelength'], aper=aper, exptime=args.exptime, theta=theta)
        psf = galsim.Convolve(psf, galsim.Gaussian(fwhm=0.35))
        img = psf.drawImage(nx=50, ny=50, scale=0.2, method='phot', rng=rng, n_photons=args.nPhot)
        mom = galsim.hsm.FindAdaptiveMom(img)
        
        return {
            'isTest':flagTest,
            'ra':thx + mean_ras,
            'dec':thy + mean_decs,
            'seed':seed,
            'dx':mom.moments_centroid.x,
            'dy':mom.moments_centroid.y
        }

    output = []
    with Pool(args.nPool) as pool:
        with ProgressBar(len(psfIsTest)) as bar:
            for o in pool.imap_unordered(f, zip(thxs, thys, psfIsTest, psfPhotSeeds)):
                output.append(o)
                bar.update()

    output = lodToDol(output)
    output['args'] = args
    output['atmSummary'] = atmSummary

    fullpath = os.path.join(args.outdir, args.outfile)
    with open(fullpath, 'wb') as f:
        pickle.dump(output, f)
