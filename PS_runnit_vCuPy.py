#!/usr/bin/env python3

########################################################### Trying to get all required packages #########################################################
import subprocess
import sys
import time
subprocess.run(["sudo", "apt", "install", "-y", 'ffmpeg'], check=True)

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

#---------------- GPU-specific imports ----------------
try:
    import cupy as cp
    from cupy.fft import fft2, ifft2
except:
    install('cupy-cuda11x')  # or another appropriate version, e.g. cupy-cuda12x
    import cupy as cp
    from cupy.fft import fft2, ifft2

# The lines below (for matplotlib) are optional if plotting is desired:
try:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.animation import PillowWriter
except:
    install('matplotlib')
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.animation import PillowWriter

#################################### Collecting input parameters to run the instance (given through terminal)#############################################
parser = argparse.ArgumentParser(description="Input to the code (GPU version)")
parser.add_argument('-name', type=str, help='test name', default="test_nova")
parser.add_argument('-n', type=int, help='Grid points = 2**n', default=8)
parser.add_argument('-w', type=float, help='Vorticity scale', default=1.0)
parser.add_argument('-j', type=float, help='J scale', default=10.0)
parser.add_argument('-T', type=float, help='time to integrate over', default=100)
parser.add_argument('-method', type=str, help='Integration method', default='IF')
parser.add_argument('-eta', type=float, help='mag diffusivity', default=1e-3)
parser.add_argument('-nu', type=float, help='viscosity', default=1e-3)
parser.add_argument('-CFL', type=float, help='CFL number', default=0.1)
args = parser.parse_args()

##########################################################################################################################################################

N = 2**args.n  # 2**8 = 256
L = 2*cp.pi
x = cp.linspace(0, L, N)
dx = x[1] - x[0]
X, Y = cp.meshgrid(x, x)

nu = args.nu   # viscosity
eta = args.eta # magnetic diffusivity
#epsilon = 5e-4 # porosity (example usage if volume penalisation is enabled)

# The Fourier variables
w_k = cp.empty((N, N), dtype=cp.complex64)
j_k = cp.empty((N, N), dtype=cp.complex64)

kx = cp.fft.fftfreq(N, d=dx)*2*cp.pi
ky = cp.fft.fftfreq(N, d=dx)*2*cp.pi
k = cp.array(cp.meshgrid(kx, ky, indexing='ij'), dtype=cp.float32)
k_perp = cp.array(cp.meshgrid(-ky, kx, indexing='ij'), dtype=cp.float32)
k2 = cp.sum(k*k, axis=0, dtype=cp.float32)
kmax_dealias = kx.max() * 2.0 / 3.0  # The "2/3 rule" for dealiasing

# Dealising matrix
dealias = ((cp.abs(k[0]) < kmax_dealias) & (cp.abs(k[1]) < kmax_dealias)).astype(bool)

############################################################# Initial conditions for w_k and j_k from spectrum ##########################################################
def initialize_field_fourier_space(kmag, spectrum_function):
    # Assign spectrum-based amplitudes
    amplitude = spectrum_function(kmag)
    
    # Generate random phases
    phases = cp.random.uniform(0, 2*cp.pi, kmag.shape)
    
    # Combine amplitude and phase to get the Fourier components
    fourier_space_field = amplitude * cp.exp(1j * phases)

    # Ensure the reality condition for the inverse FFT
    nx, ny = kmag.shape
    for i in range(nx):
        for j in range(ny):
            if i == 0 or 2*i == nx:
                if j > ny // 2:
                    fourier_space_field[i, j] = cp.conj(fourier_space_field[i, ny-j])
            else:
                if j > ny // 2:
                    fourier_space_field[i, j] = cp.conj(fourier_space_field[nx-i, ny-j])
                elif j == 0 or 2*j == ny:
                    fourier_space_field[i, j] = cp.conj(fourier_space_field[nx-i, j])
    
    return cp.real(ifft2(fourier_space_field))

# Energy spectrum
def init_spectrum(kmag):
    g = 0.98
    k0 = 0.75 * cp.sqrt(2) * cp.pi
    return kmag / (g + (kmag / k0)**2)

######################################################### General Utility functions for grad and laplacian #######################################################################
def gradient(data, h):  # WARNING! This is not being used in this code!!
    derivative_grid = cp.zeros((2,) + data.shape, dtype=data.dtype)

    # Doing x derivative
    derivative_grid[0, 1:-1, :] = (data[2:, :] - data[:-2, :]) / (2*h)
    derivative_grid[0, 0, :]    = (data[1, :] - data[0, :]) / h
    derivative_grid[0, -1, :]   = (data[-1, :] - data[-2, :]) / h
    
    # Doing y derivative
    derivative_grid[1, :, 1:-1] = (data[:, 2:] - data[:, :-2]) / (2*h)
    derivative_grid[1, :, 0]    = (data[:, 1] - data[:, 0]) / h
    derivative_grid[1, :, -1]   = (data[:, -1] - data[:, -2]) / h

    return derivative_grid

def laplacian_2d(field, dx):
    laplacian = cp.zeros_like(field)
    nx, ny = field.shape

    for i in range(nx):
        for j in range(ny):
            # Interior points
            if 1 <= i < nx - 1 and 1 <= j < ny - 1:
                laplacian[i, j] = (field[i-1, j] + field[i+1, j] + field[i, j-1] + field[i, j+1] - 4*field[i, j]) / (dx**2)
            else:
                # Edges and corners - one-sided difference
                dx2 = dx**2
                if i == 0 and 1 <= j < ny - 1:
                    laplacian[i, j] = (2*field[i, j] - 5*field[i+1, j] + 4*field[i+2, j] - field[i+3, j]) / dx2
                elif i == nx - 1 and 1 <= j < ny - 1:
                    laplacian[i, j] = (2*field[i, j] - 5*field[i-1, j] + 4*field[i-2, j] - field[i-3, j]) / dx2

                if j == 0 and 1 <= i < nx - 1:
                    laplacian[i, j] = (2*field[i, j] - 5*field[i, j+1] + 4*field[i, j+2] - field[i, j+3]) / dx2
                elif j == ny - 1 and 1 <= i < nx - 1:
                    laplacian[i, j] = (2*field[i, j] - 5*field[i, j-1] + 4*field[i, j-2] - field[i, j-3]) / dx2

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
    return cp.real(ifft2(cp.divide(a, k2, out=cp.zeros_like(a), where=(k2!=0))))

def B_biotSavaart(j_k):
    a = 1j * k_perp * j_k
    return cp.real(ifft2(cp.divide(a, k2, out=cp.zeros_like(a), where=(k2!=0))))

############################################################## Initial conditions #########################################################################
w0 = args.w
temp_w = initialize_field_fourier_space(cp.sqrt(k2), init_spectrum)
temp_w = w0 * temp_w / cp.max(temp_w)
temp_wk = fft2(temp_w)  # FT initial condition
u_init = u_biotSavaart(temp_wk)

# Time step definition (CFL-based)
cfl_est = args.CFL / ((cp.max(u_init[0]) / dx + cp.max(u_init[1]) / dx).get())
log10h = int(cp.log10(cfl_est).get()) - 1
h = 10**log10h
print("Time step = {}".format(h))

T = args.T
Nsteps = int(T / h)
dframes = h * 10
Nframes = int(T / dframes)
nframes = Nsteps // Nframes

# Arrays of outputs (w: vorticity, j: current density)
w = cp.empty((Nframes, N, N), dtype=cp.float32)
j = cp.empty((Nframes, N, N), dtype=cp.float32)

w[0] = temp_w
w_k[:] = temp_wk
Noperator_w_k = w_k.copy()  # auxiliary
ww = w[0].copy()

j0 = args.j
tmp_j = initialize_field_fourier_space(cp.sqrt(k2), init_spectrum)
tmp_j = j0 * tmp_j / cp.max(tmp_j)
j[0] = tmp_j
j_k[:] = fft2(tmp_j)
Noperator_j_k = j_k.copy()
jj = j[0].copy()

######################################## Utility Functions for computing non-linearoperators ############################################################
# Optional Volume Penalization feature if you want to do your simulations in any wonky geometrical region!
# Make a mask with 1's in the regions you want to keep all quantities damped to 0 and 0's in actualsim. region.

# Example: Circular mask #########################################################################
# Below is an example mask that is 0 inside a circle of diameter = 3/4 of the simulation box,
# and 1 outside it. The circle is centered at (L/2, L/2).
# Uncomment the relevant lines in the Noperator_func_w_B to apply a volume penalisation 
# or boundary condition if desired.

# radius = 0.75 * 0.5 * L  # => diameter = 0.75 * L => radius = 0.375 * L
# center_x = L / 2
# center_y = L / 2
# dist = cp.sqrt((X - center_x)**2 + (Y - center_y)**2)

# Inside circle => 0, outside => 1
# mask = cp.where(dist <= radius, 0.0, 1.0).astype(cp.float32)
#################################################################################################

Loperator_w_k = -nu * k2
Loperator_j_k = -eta * k2

def cross2D(B, u):
    # cross2D(B, u) = B_x u_y - B_y u_x (scalar in 2D)
    return B[0] * u[1] - B[1] * u[0]

def Noperator_func_w_B(w_k, j_k):
    # Inverse fourier transform to real space
    w_real = cp.real(ifft2(w_k))
    j_real = cp.real(ifft2(j_k))

    # Gradients in fft space
    grad_w = cp.real(ifft2(w_k * k))
    grad_j = cp.real(ifft2(j_k * k))

    # Biot-Savaart's law to find u and B
    u = u_biotSavaart(w_k)
    B = B_biotSavaart(j_k)

    # Nonlinear terms for the w_k evolution
    B_dot_grad_j = B[0] * grad_j[0] + B[1] * grad_j[1]
    u_dot_grad_w = u[0] * grad_w[0] + u[1] * grad_w[1]

    # Example volume penalization for w or j can be applied here, e.g.:
    # epsilon = 1e-4
    # mask_w = -1 / epsilon * mask * w_real
    # mask_j = -1 / epsilon * mask * j_real

    # Nonlinear terms for the j_k evolution
    B_cross_u = cross2D(B, u)
    lap_B_cross_u = laplacian_2d(B_cross_u, dx)

    # If using mask penalization, you might do:
    # Noperator_w = fft2(mask_w + B_dot_grad_j - u_dot_grad_w)
    # Noperator_j = fft2(mask_j + lap_B_cross_u)

    # Default: no mask usage
    Noperator_w = fft2(B_dot_grad_j - u_dot_grad_w)
    Noperator_j = fft2(lap_B_cross_u)

    return (Noperator_w, Noperator_j)

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
    Tlinear_w_k = cp.exp(h * Loperator_w_k)
    Tnon_w_k    = dealias * h * Tlinear_w_k
    Tlinear_j_k = cp.exp(h * Loperator_j_k)
    Tnon_j_k    = dealias * h * Tlinear_j_k

elif method == 'ETD':
    Tlinear_w_k = cp.exp(h * Loperator_w_k)
    Tlinear_j_k = cp.exp(h * Loperator_j_k)

    def myexp(x):
        # if x == 1 => 1.0; else => (x - 1.0)/log(x)
        return cp.where(cp.isclose(x, 1.0), 1.0, (x - 1.0) / cp.log(x))

    Tnon_w_k = dealias * h * myexp(Tlinear_w_k)
    Tnon_j_k = dealias * h * myexp(Tlinear_j_k)
else:
    print('ERROR: Undefined Integrator method')

############################################### Actually running simulation and saving ##############################################################################
test_name = args.name
with h5py.File('sim_data' + test_name + '.hdf5', 'w') as f:
    max_shape = (None,) + (N, N)
    w_dataset = f.create_dataset('w', shape=(0, N, N), maxshape=max_shape)
    j_dataset = f.create_dataset('j', shape=(0, N, N), maxshape=max_shape)
    
    frame_counter = 0
    for i in tqdm(range(1, Nsteps)):
        # calculate the nonlinear operator (with dealising)
        Noperator_w_k, Noperator_j_k = Noperator_func_w_B(w_k, j_k)
        
        # updating in time
        w_k = w_k * Tlinear_w_k + Noperator_w_k * Tnon_w_k
        j_k = j_k * Tlinear_j_k + Noperator_j_k * Tnon_j_k

        # IFT to next step in real space
        ww = cp.real(ifft2(w_k))
        jj = cp.real(ifft2(j_k))

        # Save frames every nframes
        if (i % nframes) == 0:
            # Resize the dataset to accommodate new data
            w_dataset.resize(frame_counter + 1, axis=0)
            j_dataset.resize(frame_counter + 1, axis=0)

            # Transfer GPU data to CPU for HDF5 I/O
            w_dataset[frame_counter] = cp.asnumpy(ww)
            j_dataset[frame_counter] = cp.asnumpy(jj)

            frame_counter += 1
            f.flush()

############################################################# GAME OVER ####################################################################################
