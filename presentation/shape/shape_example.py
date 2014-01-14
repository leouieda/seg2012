"""
Generate functions and calculate their l2 norm misfit and shape of anomaly
"""
import numpy as np
from matplotlib import pyplot as pl

def gauss(x, scale, mean, std):
    return scale*np.exp(-(x - mean)**2/(2.*std**2))

def soa(f1, f2):
    alpha = np.sum(f1*f2)/(np.linalg.norm(f1)**2)
    return np.linalg.norm(alpha*f1 - f2)

def rms(f1, f2):
    return np.linalg.norm(f1 - f2)/np.linalg.norm(f1)

def plot(x, f1, f2):
    pl.figure(figsize=(4, 3.5))
    pl.subplots_adjust(top=0.80)
    pl.title('shape-of-anomaly = %0.2f\n$\\ell_2$-norm data-misfit = %0.2f'
        % (soa(f1, f2), rms(f1, f2)))
    pl.plot(x, f1, '-b', linewidth=3)
    pl.plot(x, f2, '-r', linewidth=3)
    ax = pl.gca()
    #ax.set_xticks([])
    ax.set_yticks([0, 0.5, 1])
    pl.ylim(0, 1.05)

if __name__ == '__main__':
    x = np.linspace(-10, 10, 200)
    f = gauss(x, 1, 0, 5)
    f1 = gauss(x, 0.48, 0, 5)
    f2 = 0.5*(gauss(x, 0.5, -7, 1) + gauss(x, 1.2, 0, 3) + gauss(x, 0.5, 7, 1))
    plot(x, f, f1)
    pl.savefig('shape-example-low-soa.png', dpi=600)
    plot(x, f, f2)
    pl.savefig('shape-example-high-soa.png', dpi=600)
    pl.show()


