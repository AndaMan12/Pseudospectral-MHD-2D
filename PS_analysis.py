import subprocess
import sys
import time
import argparse

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    import h5py
except:
    install('h5py')
    import h5py
try:
    from numba import jit
except:
    install('numba')
    time.sleep(3)
    from numba import jit

try:
    from tqdm import tqdm
except:
    install('tqdm')
    time.sleep(3)
    from tqdm import tqdm

try:
    import numpy as np
except:
    install('numpy')
    import numpy as np

try:
    from scipy.fft import fft2, ifft2
except:
    install('scipy')
    from scipy.fft import fft2, ifft2

try:
    from matplotlib import animation
    from matplotlib.animation import PillowWriter
    import matplotlib.pyplot as plt
except:
    install('matplotlib')
    from matplotlib import animation
    from matplotlib.animation import PillowWriter
    import matplotlib.pyplot as plt
###############################################################################################################################

parser = argparse.ArgumentParser(description="Input to the code")
parser.add_argument('-name', type=str, help='test name', default="test_nova")
parser.add_argument('-T', type=float, help='time to integrate over', default=100)
parser.add_argument('-ts', type=float, help='time step', default= 4e-4)
args = parser.parse_args()
test_name = args.name
###############################################################################################################################

f = h5py.File('sim_data' + test_name + '.hdf5', 'r')

w_dat = np.array(f['w'])
#j_dat = np.array(f['j'])

f.close()

w_k = fft2(w_dat, axes=(-2, -1))
#j_k = fft2(j_dat, axes=(-2, -1))


###############################################################################################################################

Nframes = w_dat.shape[0]
N = w_dat.shape[1]
L = 5  
x = np.linspace(0, L, N)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, x)

kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
k = np.array(np.meshgrid(kx, ky, indexing='ij'), dtype=np.float32)
k_perp = np.array(np.meshgrid(-ky, kx, indexing='ij'), dtype=np.float32)
k2 = np.sum(k * k, axis=0, dtype=np.float32)


def curl_fft(V):
    Vx_fft = fft2(V[0])
    Vy_fft = fft2(V[1])
    curl = -1j * (k_perp[1] * Vy_fft + k_perp[0] * Vx_fft)
    return curl


def u_biotSavaart(w_k):
    a = 1j * k_perp * w_k
    return ifft2(np.divide(a, k2, out=np.zeros_like(a), where=k2 != 0)).real


def B_biotSavaart(j_k):
    a = 1j * k_perp * j_k
    return ifft2(np.divide(a, k2, out=np.zeros_like(a), where=k2 != 0)).real

#############################################################################################################################

Nsteps = int(args.T / args.ts)
t = np.linspace(0.0, Nsteps * args.ts, Nframes)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
writervideo = animation.FFMpegWriter(fps = Nframes // args.T)

u = np.array([u_biotSavaart(w_k[i,:,:]) for i in tqdm(range(Nframes))])
speed = np.sqrt(u[:, 0, :, :]**2 + u[:, 1, :, :]**2) 
lw_u = speed / np.max(speed) # linewidth for visualisation
"""
B = np.array([u_biotSavaart(j_k[i,:,:]) for i in tqdm(range(Nframes))])
B_strength = np.sqrt(B[:, 0, :, :]**2 + B[:, 1, :, :]**2) 
lw_B = B_strength / np.max(B_strength) # linewidth for visualisation
gc.collect()
"""

speed_k = np.fft.fftshift(fft2(speed, axes=(-2, -1)))
power_spectrum_speed = np.abs(speed_k)**2

k_mag = np.sqrt(k2)
k_bins = np.arange(0.5, np.max(k_mag), 1.0)
k_val = 0.5 * (k_bins[1:] + k_bins[:-1])


# Function to animate velocity field
def animate_u(i):
    ax.clear()    
    ax.streamplot(X, Y, u[i, 0, :, :], u[i, 1, :, :], color='k', linewidth=lw_u[i])
    ax.text(190, 20, 't={:.0f}'.format(t[i]), bbox=dict(boxstyle="round", ec='white', fc='white'))
    ax.set_xticks([])
    ax.set_yticks([])
    return fig,

# Function to animate magnetic field
def animate_B(i):
    ax.clear()
    ax.streamplot(X, Y, B[i, 0, :, :], B[i, 1, :, :], color='k', linewidth=lw_B[i])
    ax.text(190, 20, 't={:.0f}'.format(t[i]), bbox=dict(boxstyle="round", ec='white', fc='white'))
    ax.set_xticks([])
    ax.set_yticks([])
    return fig,

# function to animate speed power spectrum
def animate_PS_speed (i):
    power_spectrum_speed_binned = np.histogram(k_mag, bins=k_bins, weights=power_spectrum_speed[i])[0] / np.histogram(k_mag, bins=k_bins)[0]
    ax.clear()
    ax.loglog(k_val, power_spectrum_speed_binned)
    ax.set_title('t={:.0f}'.format(t[i]))


# Create and save the animations
"""
ani_u = animation.FuncAnimation(fig, animate_u, frames=Nframes, interval=10)
ani_u.save('pseudoSpectraMHD_' + test_name + '_velocity.mp4', writer=writervideo, dpi=200)

ani_B = animation.FuncAnimation(fig, animate_B, frames=Nframes, interval=10)
ani_B.save('pseudoSpectraMHD_' + test_name + '_magnetic_field.mp4', writer=writervideo, dpi=200)
"""

ani_u_PS = animation.FuncAnimation(fig, animate_PS_speed, frames=Nframes, interval=10)
ani_u_PS.save('pseudoSpectraMHD_' + test_name + '_speed_PS.mp4', writer=writervideo, dpi=200)

############################################################################################################################




































