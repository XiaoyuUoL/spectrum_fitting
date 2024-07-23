import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_csv(csv_file, x_label):
    '''
    read spectrum data based on CSV file
    two options of 'x_label': 'wl' for wavelength; 'wn' for wavenumber
    '''
    fin = open(csv_file, newline='')
    reader = csv.DictReader(fin)
    wl = []
    wn = []
    epsilon = []
    for row in reader:
        wl.append(float(row['Wavelength']))
        wn.append(float(row['Wavenumber']))
        epsilon.append(float(row['Epsilon']))
    fin.close()

    if x_label.lower() == 'wn':
        wl = [w for _,w in sorted(zip(wn, wl))]
        epsilon = [e for _,e in sorted(zip(wn, epsilon))]
        wn.sort()
    elif x_label.lower() == 'wl':
        wl = [w for _,w in sorted(zip(wl, wn))]
        epsilon = [e for _,e in sorted(zip(wl, epsilon))]
        wl.sort()
    else:
        print('read_csv: only "wl" or "wn" for "x_label"')
        exit()

    wl = np.array(wl, dtype=float)
    wn = np.array(wn, dtype=float)
    epsilon = np.array(epsilon, dtype=float)

    return wl, wn, epsilon

def fc_spect(e, e0, omega, s, sigma0, dsigma, weight):
    '''
    calculate vibrational spectrum
    e0: ground state energy; omega: effective mode frequency; s: HR
    sigma = sigma0 + dsigma * n: broaden
    '''

    def gaussian(x, x0, sigma, weight):
        '''
        multi-centre-weighted Gaussian broaden
        '''
        xNum = len(x)
        result = np.zeros(xNum, dtype=float)
        for i in np.arange(xNum):
            dx = x0 - x[i]
            result[i] = np.sum(weight / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * (dx / sigma) ** 2.))

        return result

    freqs = [e0]
    sigmas = [sigma0]
    # franck-condon overlap
    fcs = [np.exp(-s) * weight]
    for i in np.arange(1, 5):
        freqs.append(e0 + i * omega)
        sigmas.append(sigma0 + i * dsigma)
        fc = fcs[-1]
        fc *= s / i
        fcs.append(fc)
    freqs = np.array(freqs)
    sigmas = np.array(sigmas)
    fcs = np.array(fcs)

    return gaussian(e, freqs, sigmas, fcs)

def fc_fitting(x, y0, x0):
    '''
    fit vibrational peaks using one effective mode
    '''

    initial = [x0, 1400.0, 0.5, 1000.0, 100.0, 1.0]
    bound_min = [x0 - 10000.0,    0.0, 0.0,    50.0,     0.0, 0.8]
    bound_max = [x0 + 10000.0, 2000.0, 3.0, 10000.0, 10000.0, 1.2]
    popt,pcov = curve_fit(fc_spect, x, y0, p0=initial,
        bounds=(bound_min, bound_max))

    y = fc_spect(x, *popt)

    error = np.trapz(np.sqrt((y - y0) ** 2), x)

    return popt, pcov, y, error

# the index of refraction for different solvent
# ref: https://macro.lsu.edu/HowTo/solvents/Refractive%20Index.htm;
# ref: https://www.engineeringtoolbox.com/refractive-index-d_1264.html
ref_idx = {'ch': 1.43, # ethylene glycol
           'et': 1.36, # ethanol
           'eg': 1.43, # cyclohexane
           'me': 1.33, # methanol
           'hex': 1.37, # hexane
           'hep': 1.39, # heptane
           'ds': 1.48} # dimethyl sulfoxide

fin = open('systems.dat')
lines = fin.readlines()
fin.close()

fout = open('results/fitting.dat', 'w')
fout.writelines('   system       e0        omega     s      sigma0    dsigma     I      error    f0        fsol\n')
for line in lines:
    data = line.split()
    mol,sol = data[0].split('_')[:2]
    system = '{}_{}'.format(mol, sol)
    abs_file = 'exp_data/{}'.format(data[0])

    e0_guess = float(data[1])
    wn_max = float(data[2])
    wl,wn,epsilon0 = read_csv(abs_file, 'wn')
    for i,x in enumerate(wn):
        if x > wn_max:
            break

    wl = wl[:i]
    wn = wn[:i]
    epsilon0 = epsilon0[:i]
    factor = np.trapz(epsilon0, wn)

    popt,pcov,epsilon,error = fc_fitting(wn, epsilon0 / factor, e0_guess)
    epsilon *= factor
    popt[5] *= factor
    # oscillator strength calculation
    f0 = 4.32e-9 * popt[5]
    fsol = 4.32e-9 * ref_idx[sol] * popt[5]

    # print results of fitting parameters
    form = '{:>10}{:12.3f}{:10.3f}{:7.3f}{:10.3f}{:10.3f}{:10.3e}{:7.3f}{:10.3e}{:10.3e}\n'
    fout.writelines(form.format(system, *popt, error, f0, fsol))

    # save experimental and fitting spectrum data
    np.savetxt('results/data/{}_exp.dat'.format(system),
               np.array([wn, epsilon0]).T)
    np.savetxt('results/data/{}_fit.dat'.format(system),
               np.array([wn, epsilon]).T)

    # plot experimental and fitting spectrum
    plt.xlabel('Wavenumber / cm$^{-1}$')
    plt.ylabel('Intensity')
    plt.ylim(0.00, max(max(epsilon), max(epsilon0)) * 1.05)
    plt.title('Fitting spectrum of {} (error {:.3f})'.format(system, error))
    plt.plot(wn, epsilon, 'r', label='fit')
    plt.plot(wn, epsilon0, 'k--', label='exp.')
    plt.legend()
    #plt.show()
    # save figure
    plt.savefig('results/figures/{}.png'.format(system), format='png')
    plt.close()
fout.close()