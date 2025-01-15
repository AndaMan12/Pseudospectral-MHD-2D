from scipy.fft import fft2, ifft2
import numpy as np
import matplotlib.pyplot as plt
####################################### Grid initialisation ####################################################################################################

N =  2**8 
L = 5 #2*np.pi #16*np.pi
x = np.linspace(0,L,N)
dx = x[1]-x[0]
X, Y = np.meshgrid(x, x)

kx = np.fft.fftfreq(N, d=dx)*2*np.pi
ky = np.fft.fftfreq(N, d=dx)*2*np.pi
k = np.array(np.meshgrid(kx , ky ,indexing ='ij'), dtype=np.float32)
k_perp = np.array(np.meshgrid(-ky , kx ,indexing ='ij'), dtype=np.float32)
k2 = np.sum(k*k,axis=0, dtype=np.float32)


######################################################## Laplacian finite diff ###################################################################################

def laplacian_2d(field, dx):
    
    laplacian = np.zeros_like(field)
    nx, ny = field.shape
    
    for i in range(nx):
        for j in range(ny):
            # Interior points
            if 1 <= i < nx - 1 and 1 <= j < ny - 1:
                laplacian[i, j] = (field[i-1, j] + field[i+1, j] + field[i, j-1] + field[i, j+1] - 4 * field[i, j]) / dx**2
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
    
test_field = np.sin(X*Y) 
  
plt.imshow(laplacian_2d(test_field, dx))
plt.show()
plt.imshow(ifft2(-k2 * fft2(test_field)).real)
plt.show()    
    
    
    
    
    
    
