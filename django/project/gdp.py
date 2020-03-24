
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
from matplotlib.gridspec import GridSpec
#from astropy.table import Table

def z_n(phase, n=2, norm=1):
    '''Z^2_n statistics, a` la Buccheri+03, A&A, 128, 245, eq. 2.
    
    http://adsabs.harvard.edu/abs/1983A%26A...128..245B
    
    Parameters
    ----------
    phase : array of floats
        The phases of the events
    n : int, default 2
        The ``n`` in $Z^2_n$.

    Other Parameters
    ----------------
    norm : float or array of floats
        A normalization factor that gets multiplied as a weight.

    Returns
    -------
    z2_n : float
        The Z^2_n statistics of the events.
    '''
    nbin = len(phase)

    if nbin == 0:
        return 0

    norm = np.asarray(norm)
    if norm.size == 1:
        total_norm = nbin * norm
    else:
        total_norm = np.sum(norm)
    phase = phase * 2 * np.pi
    return 2 / total_norm * \
        np.sum([np.sum(np.cos(k * phase) * norm) ** 2 +
                np.sum(np.sin(k * phase) * norm) ** 2
                for k in range(1, n + 1)])
    

def get_z2n_from_profile(profile, n=2):
    phase = np.arange(0, 1, 1 / len(profile))
    return z_n(phase, norm=profile, n=n)


def get_Htest_from_profile(profile, nmax=20):
    """Calculate the H-test 
    
    http://adsabs.harvard.edu/abs/1989A&A...221..180D
    https://arxiv.org/pdf/1005.4867.pdf
    """
    phase = np.arange(0, 1, 1 / len(profile))
    z_values = np.zeros(nmax)
    for n in range(1, nmax + 1):
        z_values[n - 1] = z_n(phase, n=n, norm=profile) - 4 * n + n
    
    return np.max(z_values)


def gaussian_periodic(x, x0, amp, width):
    '''Approximates a Gaussian periodic function by summing the contributions in the phase
    range 0--1 with those in the phase range -1--0 and 1--2'''
    # Phase is always between 0 and 1
    phase = x - np.floor(x)
    lc = np.zeros_like(x)
    for shift in [-1, 0, 1]:
        lc += amp * np.exp(-(phase + shift - x0)**2 / width ** 2)
        
    return lc


def generate_profile(total_phase, ngauss=None, ph0=0.1, amp=1, width=0.05):
    if ngauss is None:
        ngauss = random.randint(1, 3)
    lc = np.zeros_like(total_phase)
    for i in range(ngauss):
        if i > 0:
            ph0 = random.uniform(0, 1)
            amp = random.uniform(0.1 * amp, amp)
            width = random.uniform(0.01, 0.2)
        lc += gaussian_periodic(total_phase, ph0, amp, width)
    
    return lc

# MI INTERESSA QUESTO!!!!!!!!!!!!

import inspect
from scipy.ndimage import gaussian_filter
def generate_dispersed_profile(pulse_freq=None, start_freq=1400, bandwidth=258, 
                               nchan=128, dm=None, amp=None, width=None, 
                               ph0=None, nbin=128, noise_level=None):

    local_vars = locals()
    
    info = type('info', (object,), {})
    for a, v in local_vars.items():
        setattr(info, a, v)
        
    if ph0 is None:
        info.ph0 = random.uniform(0, 1)         
    if amp is None:
        info.amp = random.uniform(0.1, 1)
    if width is None:
        info.width = random.uniform(0.01, 0.2)
    if noise_level is None:
        info.noise_level = random.uniform(0, 1)
    if dm is None:
        info.dm = random.uniform(0, 500)
    if pulse_freq is None:
        info.pulse_freq = random.uniform(50, 750)

    phases = np.arange(0, 1, 1/info.nbin)
    prof = generate_profile(phases, 2, info.ph0, info.amp, info.width) / info.nchan
    allprofs = np.tile(prof, (info.nchan, 1))
    allprofs += np.random.normal(allprofs, info.noise_level / info.nchan)
    profile = np.sum(allprofs, axis=0)
    dfreq = info.bandwidth / info.nchan
    stop_freq = info.start_freq + info.bandwidth
    ref_delay = 4.5e3 * info.dm * stop_freq**(-2)
    for i, p in enumerate(allprofs):
        # DM delay
        chan_freq = info.start_freq + i * dfreq
        delay = 4.5e3 * info.dm * chan_freq ** (-2) - ref_delay
        # DM broadening inside each channel
        broadening = 8.3 * info.dm * dfreq * (chan_freq / 1000) ** (-3) * 1e-6
        window = broadening * info.pulse_freq // info.nbin
        if window > 1:
            p = gaussian_filter(p, window)
            
        dph = int(np.rint(delay * info.pulse_freq))
        allprofs[i, :] = np.roll(p, dph) 
    info.dedisp_profile = profile
    info.allprofs = allprofs
    info.disp_profile = np.sum(allprofs, axis=0)

    info.disp_z2 = get_z2n_from_profile(info.disp_profile, n=2)
    info.disp_z6 = get_z2n_from_profile(info.disp_profile, n=6)
    info.disp_z12 = get_z2n_from_profile(info.disp_profile, n=12)
    info.disp_z20 = get_z2n_from_profile(info.disp_profile, n=20)
    info.disp_H = get_Htest_from_profile(info.disp_profile)
    
    info.dedisp_z2 = get_z2n_from_profile(info.dedisp_profile, n=2)
    info.dedisp_z6 = get_z2n_from_profile(info.dedisp_profile, n=6)
    info.dedisp_z12 = get_z2n_from_profile(info.dedisp_profile, n=12)
    info.dedisp_z20 = get_z2n_from_profile(info.dedisp_profile, n=20)
    info.dedisp_H = get_Htest_from_profile(info.dedisp_profile)
    return info
    
    
    
#from astropy.table import Table    

def print_info_to_table(info):
    info_dict = {}
    for key, value in dict(info.__dict__).items():
        if not key.startswith("__"):
            info_dict[key] = [value]   
    return info_dict

#table = Table(info_dict)




def plot_profile(info):
    profile = info.dedisp_profile
    disp_profile = info.disp_profile
    allprofs = info.allprofs
    start_freq = info.start_freq
    bandwidth = info.bandwidth
    pulse_freq = info.pulse_freq

    plt.figure(figsize=(15, 15))
    gs = GridSpec(3, 1, height_ratios=(1, 2, 1), hspace=0)
    aximg = plt.subplot(gs[1])
    axprof = plt.subplot(gs[0], sharex=aximg)
    axdisp = plt.subplot(gs[2], sharex=aximg)

    phases = np.arange(0, 1, 1/len(profile))
    aximg.imshow(allprofs, aspect="auto", origin="lower",
                 extent=(0, 1, start_freq, start_freq + bandwidth))
    axprof.plot(phases, profile, drawstyle='steps')
    axdisp.plot(phases, disp_profile, drawstyle='steps')
    aximg.set_xlim((0, 1))
    aximg.set_ylabel("Frequency (MHz)")
    axprof.set_ylabel("Dedispersed profile")
    axdisp.set_ylabel("Dispersed profile")
    axdisp.set_xlabel("Phase")
    aximg.text(0.5, 0.5, 
               "DM={:.1f}\n$\\nu={:.1f}$ Hz\nH-test={:.1f} -> {:.1f}".format(info.dm, 
                                                                   info.pulse_freq,
                                                                   info.disp_H,
                                                                   info.dedisp_H), 
               horizontalalignment='center',
               verticalalignment='center', transform=aximg.transAxes, 
               fontsize=40, color='white')
    
    
    
    
    