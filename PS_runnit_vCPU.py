########################################################### Trying to get all required packages #########################################################

import subprocess
import sys
import time

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import argparse
except:
    install('argparse')
    import argparse

try:
    import h5py
except:
    install('h5py')
    import h5py
    
try:
    from tqdm import tqdm
except:
    install('tqdm')
    time.sleep(2)
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
    

#################################### Collecting input parameters to run the instance (given through terminal)#############################################
parser = argparse.ArgumentParser(description="Input to the code")
parser.add_argument('-name', type=str, help='test name', default = "test_nova")
parser.add_argument('-n', type=int, help='Grid points = 2^n', default = 8)
parser.add_argument('-w', type=float, help='Vorticity scale', default = 1.0)
parser.add_argument('-j', type=float, help='J scale', default = 10.0)
parser.add_argument('-T', type=float, help='time to integrate over', default = 100)
parser.add_argument('-method', type=str, help='Integration method', default = 'IF')
parser.add_argument('-eta', type=float, help='mag diffusivity', default = 1e-3)
parser.add_argument('-nu', type=float, help='viscosity', default = 1e-3)
parser.add_argument('-CFL', type=float, help='CFL number', default = 0.1)
args = parser.parse_args()

##########################################################################################################################################################

N = 2**args.n  # 2**8 = 256
L = 2*np.pi
x = np.linspace(0, L, N)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, x)

nu = args.nu  # viscosity
eta = args.eta # magnetic diffusivity
#epsilon = 5e-4 #porosity (example usage if volume penalisation is enabled)

# The Fourier variables
w_k = np.empty((N, N), dtype=np.complex64)
j_k = np.empty((N, N), dtype=np.complex64)

kx = np.fft.fftfreq(N, d=dx)*2*np.pi
ky = np.fft.fftfreq(N, d=dx)*2*np.pi
k = np.array(np.meshgrid(kx, ky, indexing='ij'), dtype=np.float32)
k_perp = np.array(np.meshgrid(-ky, kx, indexing='ij'), dtype=np.float32)
k2 = np.sum(k*k, axis=0, dtype=np.float32)
kmax_dealias = kx.max() * 2.0/3.0  # The Nyquist mode

# Dealising matrix
dealias = np.array((np.abs(k[0]) < kmax_dealias) * (np.abs(k[1]) < kmax_dealias), dtype=bool)

############################################################# Initial conditions for w_k and j_k from spectrum ##########################################################
def initialize_field_fourier_space(kmag, spectrum_function):
    # Assign spectrum-based amplitudes
    amplitude = spectrum_function(kmag)
    
    # Generate random phases
    phases = np.random.uniform(0, 2*np.pi, kmag.shape)
    
    # Combine amplitude and phase to get the Fourier components
    fourier_space_field = amplitude * np.exp(1j * phases)
    
    # Ensure the reality condition for the inverse FFT
    nx, ny = kmag.shape
    for i in range(nx):
        for j in range(ny):
            if i == 0 or 2*i == nx:
                if j > ny // 2:
                    fourier_space_field[i, j] = np.conj(fourier_space_field[i, ny-j])
            else:
                if j > ny // 2:
                    fourier_space_field[i, j] = np.conj(fourier_space_field[nx-i, ny-j])
                elif j == 0 or 2*j == ny:
                    fourier_space_field[i, j] = np.conj(fourier_space_field[nx-i, j])
    
    return ifft2(fourier_space_field).real

# Energy spectrum
def init_spectrum(kmag):
    g = 0.98
    k0 = 0.75 * np.sqrt(2) * np.pi
    return kmag / (g + (kmag / k0)**2)

######################################################### General Utility functions for grad and laplacian #######################################################################
def gradient(data, h):  # WARNING! This is not being used in this code!!
    derivative_grid = np.zeros(shape=(2,) + data.shape)
    # Doing x derivative
    derivative_grid[0, 1:-1, :] = (data[2:, :] - data[:-2, :]) / (2*h)  # interior
    derivative_grid[0, 0, :]    = (data[1, :] - data[0, :]) / (h)       # left edge
    derivative_grid[0, -1, :]   = (data[-1, :] - data[-2, :]) / (h)     # right edge
    
    # Doing y derivative
    derivative_grid[1, :, 1:-1] = (data[:, 2:] - data[:, :-2]) / (2*h)  # interior
    derivative_grid[1, :, 0]    = (data[:, 1] - data[:, 0]) / (h)       # bottom edge
    derivative_grid[1, :, -1]   = (data[:, -1] - data[:, -2]) / (h)     # top edge

    return derivative_grid

def laplacian_2d(field, dx):
    laplacian = np.zeros_like(field)
    nx, ny = field.shape
    
    for i in range(nx):
        for j in range(ny):
            # Interior points
            if 1 <= i < nx - 1 and 1 <= j < ny - 1:
                laplacian[i, j] = (field[i-1, j] + field[i+1, j] + field[i, j-1] + field[i, j+1] - 4*field[i, j]) / dx**2
            else:
                # Edges and corners - one-sided difference
                dx2 = dx**2
                # Top and bottom rows (without corners)
                if i == 0 and 1 <= j < ny - 1:
                    laplacian[i, j] = (2*field[i, j] - 5*field[i+1, j] + 4*field[i+2, j] - field[i+3, j]) / dx2
                elif i == nx - 1 and 1 <= j < ny - 1:
                    laplacian[i, j] = (2*field[i, j] - 5*field[i-1, j] + 4*field[i-2, j] - field[i-3, j]) / dx2
                
                # Left and right columns (without corners)
                if j == 0 and 1 <= i < nx - 1:
                    laplacian[i, j] = (2*field[i, j] - 5*field[i, j+1] + 4*field[i, j+2] - field[i, j+3]) / dx2
                elif j == ny - 1 and 1 <= i < nx - 1:
                    laplacian[i, j] = (2*field[i, j] - 5*field[i, j-1] + 4*field[i, j-2] - field[i, j-3]) / dx2
                
                # Corners
                if i == 0 and j == 0:
                    laplacian[i, j] = (2*field[i, j] - 5*field[i+1, j+1] + 4*field[i+2, j+2] - field[i+3, j+3]) / dx2
                elif i == 0 and j == ny - 1:
                    laplacian[i, j] = (2*field[i, j] - 5*field[i+1, j-1] + 4*field[i+2, j-2] - field[i+3, j-3]) / dx2
                elif i == nx - 1 and j == 0:
                    laplacian[i, j] = (2*field[i, j] - 5*field[i-1, j+1] + 4*field[i-2, j+2] - field[i-3, j+3]) / dx2
                elif i == nx - 1 and j == ny - 1:
                    laplacian[i, j] = (2*field[i, j] - 5*field[i-1, j-1] + 4*field[i-2, j-2] - field[i-3, j-3]) / dx2

    return laplacian

def u_biotSavaart(w_k):
    a = 1j * k_perp * w_k
    return ifft2(np.divide(a, k2, out=np.zeros_like(a), where=k2!=0)).real

def B_biotSavaart(j_k):
    a = 1j * k_perp * j_k
    return ifft2(np.divide(a, k2, out=np.zeros_like(a), where=k2!=0)).real


############################################################## Initial conditions #########################################################################

"""
Set your appropriate initial condition here. For my project on MHD turbulence, I started with a random field with
a pre-set energy spectrum.
"""

w0 = args.w
temp_w = initialize_field_fourier_space(np.sqrt(k2), init_spectrum)
temp_w = w0 * temp_w / np.max(temp_w)
temp_wk = fft2(temp_w)  # FT initial condition
u_init = u_biotSavaart(temp_wk)

# The time step definition
log10h = int(np.log10(args.CFL / (np.max(u_init[0]) / dx + np.max(u_init[1]) / dx))) - 1 
h = 10**log10h
print("Time step = {}".format(h))

T = args.T
Nsteps = int(T / h)
dframes = h * 10  # time step to output
Nframes = int(T / dframes)  # frames to the output
nframes = Nsteps // Nframes

# The array of outputs (w: vorticity, j: current density)
w = np.empty((Nframes, N, N), dtype=np.float32)
j = np.empty((Nframes, N, N), dtype=np.float32)

w[0] = temp_w
w_k[:] = temp_wk
Noperator_w_k = w_k.copy()  # auxiliary array
ww = w[0].copy()  # auxiliary array

j0 = args.j
j[0] = initialize_field_fourier_space(np.sqrt(k2), init_spectrum)
j[0] = j0 * j[0] / np.max(j[0])
j_k[:] = fft2(j[0])  # FT initial condition
Noperator_j_k = j_k.copy()  # auxiliary array
jj = j[0].copy()  # auxiliary array


############################################## Utility Functions for computing non-linearoperators ############################################################

"""
The masking/ volume penalisation feature has been tested but needs more thorough checks.
"""

# Example: Circular mask #########################################################################
# Below is an example mask that is 0 inside a circle of diameter = 3/4 of the simulation box,
# and 1 outside it. The circle is centered at (L/2, L/2).
# Uncomment the relevant lines in the Noperator_func_w_B to apply a volume penalisation 
# or boundary condition if desired.

# radius = 0.75 * 0.5 * L  # => diameter = 0.75 * L, so radius = 0.375 * L
# center_x = L/2
# center_y = L/2
# dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

# Inside circle => 0, outside => 1
# mask = np.where(dist <= radius, 0.0, 1.0)
# mask = mask.astype(np.float32)
##################################################################################################
Loperator_w_k = -nu * k2
Loperator_j_k = -eta * k2

def Noperator_func_w_B(w_k, j_k):
    # Inverse fourier transform to real space
    w = ifft2(w_k).real
    j = ifft2(j_k).real

    # Gradients in done in fft space
    grad_w = ifft2(w_k * k).real
    grad_j = ifft2(j_k * k).real
    
    # Biot Savaart's law to find u and B
    u = u_biotSavaart(w_k)
    B = B_biotSavaart(j_k)

    # Non linear terms for the w_k evolution
    B_dot_grad_j = B[0,:,:] * grad_j[0,:,:] + B[1,:,:] * grad_j[1,:,:]
    u_dot_grad_w = u[0,:,:] * grad_w[0,:,:] + u[1,:,:] * grad_w[1,:,:]
    
    # Example volume penalization for w or j can be applied here, e.g.:
    # epsilon = 1e-4 # epsilon needs to be a quite small number: so that the values of w and j outside the box is driven to 0
    # mask_w = -1/epsilon * mask * w
    # mask_j = -1/epsilon * mask * j

    # Non linear terms for the j_k evolution
    B_cross_u = np.cross(B, u, axis=0)
    lap_B_cross_u = laplacian_2d(B_cross_u, dx)
    
    # Add the mask penalization if desired:
    # Noperator_func_w = fft2(mask_w + B_dot_grad_j - u_dot_grad_w)
    # Noperator_func_j = fft2(mask_j + lap_B_cross_u)

    # By default, no mask usage:
    Noperator_func_w = fft2(B_dot_grad_j - u_dot_grad_w)
    Noperator_func_j = fft2(lap_B_cross_u)

    return (Noperator_func_w, Noperator_func_j)

    
######################################################### Time integral methods #######################################################################################

method = args.method

Tlinear_w_k = 0
Tnon_w_k = 0
Tlinear_j_k = 0
Tnon_j_k = 0

if method == 'IMEX':
    Tlinear_w_k = 1.0 / (1.0 - h * Loperator_w_k)
    Tnon_w_k    = dealias * h / (1.0 - h * Loperator_w_k)
    Tlinear_j_k = 1.0 / (1.0 - h * Loperator_j_k)
    Tnon_j_k    = dealias * h / (1.0 - h * Loperator_j_k)

elif method == 'IF':
    Tlinear_w_k = np.exp(h * Loperator_w_k)
    Tnon_w_k    = dealias * h * Tlinear_w_k
    Tlinear_j_k = np.exp(h * Loperator_j_k)
    Tnon_j_k    = dealias * h * Tlinear_j_k

elif method == 'ETD':
    Tlinear_w_k = np.exp(h * Loperator_w_k)
    Tlinear_j_k = np.exp(h * Loperator_j_k)
    def myexp(x):
        if x == 1:
            return 1.0
        else:
            return (x - 1.0) / np.log(x)
    vmyexp = np.vectorize(myexp)
    Tnon_w_k = dealias * h * vmyexp(Tlinear_w_k)
    Tnon_j_k = dealias * h * vmyexp(Tlinear_j_k)
else:
    print('ERROR: Undefined Integrator')


############################################### Actually running simulation and saving ##############################################################################
test_name = args.name
with h5py.File('sim_data' + test_name + '.hdf5', 'w') as f:
    max_shape = (None,) + temp_w.shape  # None indicates an extendable dimension
    w_dataset = f.create_dataset('w', shape=(0,) + temp_w.shape, maxshape=max_shape)
    j_dataset = f.create_dataset('j', shape=(0,) + temp_w.shape, maxshape=max_shape)
    
    frame_counter = 0
    for i in tqdm(range(1, Nsteps)):
        # calculate the nonlinear operator (with dealising)
        Noperator_w_k, Noperator_j_k = Noperator_func_w_B(w_k, j_k)
        
        # updating in time
        w_k = w_k * Tlinear_w_k + Noperator_w_k * Tnon_w_k
        j_k = j_k * Tlinear_j_k + Noperator_j_k * Tnon_j_k
        
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

############################################################# GAME OVER ####################################################################################
