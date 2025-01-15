########################################################### Trying to get all required packages #########################################################

import subprocess
import sys

subprocess.run(["sudo", "apt", "install", "-y", 'ffmpeg'], check=True)

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

#################################### Collecting input parameters to run the instance (given through terminal)#############################################
parser = argparse.ArgumentParser(description="Input to the code")
parser.add_argument('-name', type=str, help='test name', default="test_nova")
parser.add_argument('-n', type=int, help='Grid points = 2^n', default=8)
parser.add_argument('-w', type=float, help='Vorticity scale', default=1.0)
parser.add_argument('-j', type=float, help='J scale', default=10.0)
parser.add_argument('-BJscale', type=float, help='Boundary B or J scale', default = 10)
parser.add_argument('-T', type=float, help='time to integrate over', default=100)
parser.add_argument('-ts', type=float, help='time step', default= 4e-4)
parser.add_argument('-method', type=str, help='Integration method', default='IF')
parser.add_argument('-BJ', type=str, help='method for J/B B.C.', default='J')
parser.add_argument('-eta', type=float, help='mag diffusivity', default=1e-2)
parser.add_argument('-nu', type=float, help='viscosity', default=1e-2)
parser.add_argument('-eps', type=float, help='penalisation factor', default=5e-4)
args = parser.parse_args()
test_name = args.name  # Make sure this is defined or replace with your simulation name
BJ_choice = args.BJ
####################################### Grid initialisation ####################################################################################################

N = 2 ** args.n  # 2**8 = 256
L = 5  # 2*np.pi #16*np.pi
x = np.linspace(0, L, N)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, x)
center_x, center_y = L / 2, L / 2
rad_in = L / 5
rad_out = rad_in + 4*dx
distance_from_centre = (X - center_x) ** 2 + (Y - center_y) ** 2
mask = distance_from_centre >= rad_in ** 2

nu = args.nu  # viscosity
eta = args.eta  # magnetic diffusivity
epsilon = args.eps  # porosity

# The time step definition
h = args.ts

T = args.T
Nsteps = int(T / h)
dframes = h * 50  # time step to output
Nframes = int(T / dframes)  # frames to the output
nframes = Nsteps // Nframes

# The Fourier variables
w_k = np.empty((N, N), dtype=np.complex64)
j_k = np.empty((N, N), dtype=np.complex64)

kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
k = np.array(np.meshgrid(kx, ky, indexing='ij'), dtype=np.float32)
k_perp = np.array(np.meshgrid(-ky, kx, indexing='ij'), dtype=np.float32)
k2 = np.sum(k * k, axis=0, dtype=np.float32)
kmax_dealias = kx.max() * 2.0 / 3.0  # The Nyquist mode
# Dealising matrix
dealias = np.array((np.abs(k[0]) < kmax_dealias) * (np.abs(k[1]) < kmax_dealias), dtype=bool)


##################################################### Initial conditions for w_k and j_k from spectrum ##########################################################
def initialize_field_fourier_space(kmag, spectrum_function):
    # Assign spectrum-based amplitudes
    amplitude = spectrum_function(kmag)

    # Generate random phases
    phases = np.random.uniform(0, 2 * np.pi, kmag.shape)

    # Combine amplitude and phase to get the Fourier components
    fourier_space_field = amplitude * np.exp(1j * phases)

    # Ensure the reality condition for the inverse FFT
    nx, ny = kmag.shape

    for i in range(nx):
        for j in range(ny):
            if i == 0 or 2 * i == nx:
                if j > ny // 2:
                    fourier_space_field[i, j] = np.conj(fourier_space_field[i, ny - j])
            else:
                if j > ny // 2:
                    fourier_space_field[i, j] = np.conj(fourier_space_field[nx - i, ny - j])
                elif j == 0 or 2 * j == ny:
                    fourier_space_field[i, j] = np.conj(fourier_space_field[nx - i, j])

    return ifft2(fourier_space_field).real


# Energy spectrum
def init_spectrum(kmag):
    g = 0.98
    k0 = 0.75 * np.sqrt(2) * np.pi
    return kmag / (g + (kmag / k0) ** 2)


######################################################### General Utility functions for grad and laplacian #################################################################

@jit(nopython=True)
def laplacian_2d(field, dx):
    laplacian = np.zeros_like(field)
    dx2 = dx ** 2

    # Computing Laplacian for interior points
    laplacian[1:-1, 1:-1] = (field[:-2, 1:-1] + field[2:, 1:-1] + field[1:-1, :-2] + field[1:-1, 2:] - 4 * field[1:-1,1:-1]) / dx2

    # Assuming reflective boundary conditions for simplicity
    # Top and bottom
    laplacian[0, 1:-1] = laplacian[1, 1:-1]
    laplacian[-1, 1:-1] = laplacian[-2, 1:-1]
    # Left and right
    laplacian[1:-1, 0] = laplacian[1:-1, 1]
    laplacian[1:-1, -1] = laplacian[1:-1, -2]
    # Corners
    laplacian[0, 0] = laplacian[1, 1]
    laplacian[-1, -1] = laplacian[-2, -2]
    laplacian[0, -1] = laplacian[1, -2]
    laplacian[-1, 0] = laplacian[-2, 1]

    return laplacian


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


############################################################## Initial conditions #########################################################################

w0 = args.w
temp_w = initialize_field_fourier_space(np.sqrt(k2), init_spectrum)  
temp_w = w0 * temp_w / np.max(temp_w)
temp_wk = fft2(temp_w)  # FT initial condition
u_init = u_biotSavaart(temp_wk)

w_k[:] = curl_fft(u_init)
ww = ifft2(w_k[:]).real 
Noperator_w_k = w_k.copy()  # auxiliary array

j0 = args.j
jj = initialize_field_fourier_space(np.sqrt(k2), init_spectrum) 
jj = j0 * jj / np.max(jj)
j_k[:] = fft2(jj)  # FT initial condition
Noperator_j_k = j_k.copy()  # auxiliary array

################################################### Solid boundary values of u0 and B0 ##################################################################

u0 = np.zeros_like(u_init)

def BJ_method (BJ_choice):
    annular_mask = (distance_from_centre >= rad_in ** 2) & (distance_from_centre <= rad_out ** 2)
    if BJ_choice == "B":
        B0_x = annular_mask * (Y - center_y) / rad_in
        B0_y = annular_mask * (-X + center_x) / rad_in
        #B_mag = np.sqrt(B0_x ** 2 + B0_y ** 2)
        #B0_x = annular_mask * np.divide(B0_x, B_mag, out=np.zeros_like(B0_x), where=B_mag != 0)
        #B0_y = annular_mask * np.divide(B0_y, B_mag, out=np.zeros_like(B0_y), where=B_mag != 0)
        B0 = args.BJscale * np.array([B0_x, B0_y])       
        return B0
        
    elif BJ_choice == "J":
        j0 = args.BJscale * annular_mask * np.ones_like(u_init[0])
        return j0
        
    else:
        print("Not a valid BC choice for BJ")
        return None

##################################################### Linear and Non-Linear operators ###################################################################

Loperator_w_k = -nu * k2
Loperator_j_k = -eta * k2


def Noperator_func_w_B(w_k, j_k):
    # Inverse fourier transform to real space
    w = ifft2(w_k).real
    j = ifft2(j_k).real

    # Gradients in done in fft space
    grad_w = ifft2(w_k * k).real  # gradient(w, dx)
    grad_j = ifft2(j_k * k).real  # gradient(j, dx)

    # Biot Savaart's law to find u and B
    u = u_biotSavaart(w_k)
    B = B_biotSavaart(j_k)

    # Non linear terms for the w_k evolution
    B_dot_grad_j = B[0, :, :] * grad_j[0, :, :] + B[1, :, :] * grad_j[1, :, :]
    u_dot_grad_w = u[0, :, :] * grad_w[0, :, :] + u[1, :, :] * grad_w[1, :, :]
    mask_w = 1 / epsilon * curl_fft(mask[None, :, :] * (u - u0))

    # Non linear terms for the j_k evolution
    B_cross_u = np.cross(B, u, axis=0)
    lap_B_cross_u_fft = fft2(laplacian_2d(B_cross_u, dx))
    
    mask_BJ = np.zeros_like(w)
    
    if BJ_choice == "B":
        mask_BJ = 1 / epsilon * curl_fft(mask[None, :, :] * (B - BJ_method(BJ_choice)))
    elif BJ_choice == "J":
        mask_BJ = 1 / epsilon * fft2(mask * (j - BJ_method(BJ_choice)))
    else:
        print ("Not a valid BC choice for BJ")
        return None
    
    Noperator_func_w = fft2(B_dot_grad_j - u_dot_grad_w) - mask_w
    Noperator_func_j = lap_B_cross_u_fft - mask_BJ

    return (Noperator_func_w, Noperator_func_j)


######################################################### Time integral methods #############################################################################

method = args.method
# Defining the time marching operators arrays

Tlinear_w_k = 0
Tnon_w_k = 0
Tlinear_j_k = 0
Tnon_j_k = 0

if method == 'IMEX':
    Tlinear_w_k = 1.0 / (1.0 - h * Loperator_w_k)
    Tnon_w_k = dealias * h / (1.0 - h * Loperator_w_k)
    Tlinear_j_k = 1.0 / (1.0 - h * Loperator_j_k)
    Tnon_j_k = dealias * h / (1.0 - h * Loperator_j_k)
elif method == 'IF':
    Tlinear_w_k = np.exp(h * Loperator_w_k)
    Tnon_w_k = dealias * h * Tlinear_w_k
    Tlinear_j_k = np.exp(h * Loperator_j_k)
    Tnon_j_k = dealias * h * Tlinear_j_k
elif method == 'ETD':
    Tlinear_w_k = np.exp(h * Loperator_w_k)
    Tlinear_j_k = np.exp(h * Loperator_j_k)


    def myexp(x):
        if x == 1:
            return 1.0
        else:
            return (x - 1.0) / np.log(x)


    vmyexp = np.vectorize(myexp)  # vectorize myexp (could be jitted)
    Tnon_w_k = dealias * h * vmyexp(Tlinear_w_k)
    Tnon_j_k = dealias * h * vmyexp(Tlinear_j_k)
else:
    print('ERROR: Undefined Integrator')

########################################## time evolution loop #################################################################################################

with h5py.File('sim_data' + test_name + '.hdf5', 'w') as f:
    max_shape = (None,) + temp_w.shape  # None indicates an extendable dimension
    w_dataset = f.create_dataset('w', shape=(0,) + temp_w.shape, maxshape=max_shape)
    j_dataset = f.create_dataset('j', shape=(0,) + temp_w.shape, maxshape=max_shape)
    
    frame_counter = 0
    for i in tqdm(range(1, Nsteps)):
        # calculate the nonlinear operator (with dealising)
        Noperator_w_k, Noperator_j_k = Noperator_func_w_B(w_k, j_k)
        # updating in time
        w_k, j_k = w_k * Tlinear_w_k + Noperator_w_k * Tnon_w_k, j_k * Tlinear_j_k + Noperator_j_k * Tnon_j_k 
        # IFT to next step
        ww = ifft2(w_k).real
        jj = ifft2(j_k).real

        if (i % nframes) == 0:
            # Resize the dataset to accommodate the new data
            w_dataset.resize(frame_counter + 1, axis=0)
            j_dataset.resize(frame_counter + 1, axis=0)

            # Append the new frames
            w_dataset[frame_counter] = ww
            j_dataset[frame_counter] = jj

            frame_counter += 1
            f.flush()



############################################################# Movie making ##################################################################################

with h5py.File('sim_data' + test_name + '.hdf5', 'r') as f:
    # Access the datasets
    w_dataset = f['w']
    j_dataset = f['j']
    Nframes = w_dataset.shape[0]  # Assuming the first dimension is time/frames
    t = np.linspace(0.0, Nsteps * h, Nframes)

    # Setup for animation
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    writervideo = animation.FFMpegWriter(fps = Nframes // T)


    # Function to animate vorticity
    def animate_w(i):
        ax.clear()
        ax.imshow(w_dataset[i], cmap='RdBu_r')
        ax.text(190, 20, 't={:.0f}'.format(t[i]), bbox=dict(boxstyle="round", ec='white', fc='white'))
        ax.set_xticks([])
        ax.set_yticks([])
        return fig,


    # Function to animate current density
    def animate_j(i):
        ax.clear()
        ax.imshow(j_dataset[i], cmap='RdBu_r')
        ax.text(190, 20, 't={:.0f}'.format(t[i]), bbox=dict(boxstyle="round", ec='white', fc='white'))
        ax.set_xticks([])
        ax.set_yticks([])
        return fig,


    # Create and save the animations
    ani_w = animation.FuncAnimation(fig, animate_w, frames=Nframes, interval=10)
    ani_w.save('pseudoSpectraMHD_' + test_name + '_vorticity.mp4', writer=writervideo, dpi=200)

    ani_j = animation.FuncAnimation(fig, animate_j, frames=Nframes, interval=10)
    ani_j.save('pseudoSpectraMHD_' + test_name + '_current_density.mp4', writer=writervideo, dpi=200)

############################################################# GAME OVER ####################################################################################
