import qutip as qt
import numpy as np
import scipy.sparse as sp 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def site_index(Ly: int, m: int, n: int):
    """This function maps 2D coordinates to a 1D axis"
    Parameters:

    -Ly: grid size on the transversal axis (Y axis)
    -m,n: coordinates of the current position on the grid 
    """
    return m * Ly + n

def minkowsky_H(Lx: int, Ly: int, t0=1.0):
    """
    This function builds the Minkowksky Hamiltonian defined in the paper 'Synthetic Unruh effect in cold atoms' .
    It takes the grid size (Lx and Ly) and the tunneling parameter t0 as inputs. 
    It returns the Hamiltonian as a QuTip matrix and as a SciPy C00 matrix.
    """
    data_M = []
    rows_M = []
    cols_M = []

    N_sites = Lx * Ly
    for m in range(Lx):
        for n in range(Ly):
            i = site_index(Ly, m, n)

            # If we're not at the edge of the lattice, perform a jump in X from (m,n) to (m+1,n)
            if m < Lx - 1:
                j_x = site_index(Ly, m + 1, n)
                
                # Jump coefficient from paper
                val = -t0 * np.exp(1j * np.pi / 2 * (m - n))
                
                # c_j_x^\dagger * c_i
                data_M.append(val)
                rows_M.append(j_x)
                cols_M.append(i)
                
                # H.c. (Hermitian conjugate) 
                data_M.append(val.conjugate())
                rows_M.append(i)
                cols_M.append(j_x)

            # If we're not at the edge of the lattice, perfrom a jump in Y from (m,n) to (m,n+1)
            if n < Ly - 1:
                j_y = site_index(Ly, m, n + 1) 
                
                # Jump coefficient from paper
                val = -t0 * np.exp(1j * np.pi / 2 * (m - n))

                # c_j_y^\dagger * c_i
                data_M.append(val)
                rows_M.append(j_y)
                cols_M.append(i)
                
                # H.c. (Hermitian conjugate)
                data_M.append(val.conjugate())
                rows_M.append(i)
                cols_M.append(j_y)

    #Construct the sparse matrix in C00 format using SciPy
    H_M_coo = sp.coo_matrix((data_M, (rows_M, cols_M)), shape=(N_sites, N_sites))

    #Create the Qobj in qutip
    H_M_2D = qt.Qobj(H_M_coo, dims=[[N_sites], [N_sites]])

    return H_M_2D, H_M_coo

def rindler_H(Lx: int, Ly: int, c=2.0):
    """
    This function builds the Rindler Hamiltonian defined in the paper 'Synthetic Unruh effect in cold atoms'
    It takes the grid size (Lx and Ly) as inputs. The tunneling parameter (t_r) is defined as 'c/Lx', where 'c' is a
    constant that controls the strenght of the tunneling. 
    It returns the Hamiltonian as a QuTip matrix and as a SciPy C00 matrix. 
    """
    data_R = []
    rows_R = []
    cols_R = []

    N_sites= Lx*Ly
    t_r= c/Lx
    for m in range(Lx):
        for n in range(Ly):
            i = site_index(Ly, m, n)
            #Since an event horizon in created at m=0, we want to put this value in the middle, 
            # because we want the horizon in the center of the lattice
            #Therefore, we redefine the coordinates as:
            m_coord = m - (Lx // 2)
            # If we're not at the edge of the lattice, perform a jump in X from (m,n) to (m+1,n)
            if m < Lx - 1:
                j_x = site_index(Ly, m + 1, n)
                
                # Jump coefficient from paper
                val = - (m_coord+1/2)*t_r * np.exp(1j * np.pi / 2 * (m - n))
                
                # c_j_x^\dagger * c_i
                data_R.append(val)
                rows_R.append(j_x)
                cols_R.append(i)
                
                # H.c. (Hermitian conjugate) 
                data_R.append(val.conjugate())
                rows_R.append(i)
                cols_R.append(j_x)

            # If we're not at the edge of the lattice, perfrom a jump in Y from (m,n) to (m,n+1)
            if n < Ly - 1:
                j_y = site_index(Ly, m, n + 1) 
                
                # Jump coefficient from Eq. (29)
                val = -m_coord*t_r * np.exp(1j * np.pi / 2 * (m - n))

                # c_j_y^\dagger * c_i
                data_R.append(val)
                rows_R.append(j_y)
                cols_R.append(i)
                
                # H.c. (Hermitian conjugate)
                data_R.append(val.conjugate())
                rows_R.append(i)
                cols_R.append(j_y)

    #Construct the sparse matrix in C00 format using SciPy
    H_R_coo = sp.coo_matrix((data_R, (rows_R, cols_R)), shape=(N_sites, N_sites))

    #Create the Qobj in qutip
    H_R_2D = qt.Qobj(H_R_coo, dims=[[N_sites], [N_sites]])
        
    return H_R_2D, H_R_coo

def get_eigenstates(H):
    """
    This function takes a QuTip Hamiltonian as imput and returns its eigenstates and eigenenergies.
    """
    energies=H.eigenstates()[0]
    states=H.eigenstates()[1]

    return energies, states

def gaussian_wavepacket(Lx: int, Ly: int, sigma=1.0, k_x=2.0):
    """
    This function creates a gaussian wavepacket located at the left of the event horizon. 
    It takes the grid size (Lx and Ly) as inputs. Sigma is the width of the gaussian and k_x the wavevector of the plane
    wave in the x direction.
    It returns the normalized wave.
    """
    m_center = Lx // 4      # Start on the left
    n_center = Ly // 2      # Centered in Y

    N_sites= Lx*Ly
    psi0_vec = np.zeros(N_sites, dtype=complex)

    for m in range(Lx):
        for n in range(Ly):
            idx = site_index(Ly, m, n)
            # Coordinates relative to packet center
            dm = m - m_center
            dn = n - n_center
            
            # Gaussian amplitude * Plane wave
            amplitude = np.exp(-(dm**2 + dn**2) / (2 * sigma**2))
            phase = np.exp(1j * k_x * m)
            psi0_vec[idx] = amplitude * phase

    # Normalize
    psi0_vec /= np.linalg.norm(psi0_vec)

    return psi0_vec

def gaussian_event_horizon(Lx: int, Ly: int, times: list, H_M, H_R, savefig: bool):
    """
    This function plots the time evolution of the gaussian wavepacket based on the Minkowsky and Rindler Hamiltonians.
    Defined as: psi(t) = sum( c_n * exp(-i*En*t) * |n> ), in matrix notation: V_M @ (c_M * phases)
    """
    #get eigenstates and eigenenergies of both Hamiltonians
    energiesM, statesM=get_eigenstates(H_M)
    energiesR, statesR=get_eigenstates(H_R)
    #generate the gaussian wavepacket
    psi0_vec=gaussian_wavepacket(Lx, Ly)
    #create a transformation matrix 
    V_M = np.hstack([s.full() for s in statesM])
    V_R = np.hstack([s.full() for s in statesR])
    #project ket(V) into bra(psi0) to get the coefficients
    c_M = V_M.conj().T @ psi0_vec
    c_R = V_R.conj().T @ psi0_vec

    #-------------------begin plot----------------------------------
    fig, axes = plt.subplots(2, len(times), figsize=(15, 10))

    # Coordinates for plotting
    x_axis = np.arange(Lx) - (Lx//2)
    y_axis = np.arange(Ly) - (Ly//2)

    for t_idx, t in enumerate(times):
        # --- Minkowski time Evolution ---
        psi_t_M = V_M @ (c_M * np.exp(-1j * energiesM * t))
        dens_M = np.abs(psi_t_M.flatten().reshape(Lx, Ly)).T
        
        # --- Rindler time Evolution ---
        psi_t_R = V_R @ (c_R * np.exp(-1j * energiesR * t))
        dens_R = np.abs(psi_t_R.flatten().reshape(Lx, Ly)).T
        
        # --- Minkowsky subplot ---
        ax = axes[0, t_idx]
        ax.imshow(dens_M, origin='lower', extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]], 
                cmap='inferno', vmin=0, vmax=0.05, aspect='auto')
        ax.axvline(0, color='cyan', linestyle='--') # Horizon
        ax.set_title(f"Minkowski (t={t})")

        # --- Rindler subplot ---
        ax = axes[1, t_idx]
        ax.imshow(dens_R, origin='lower', extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]], 
                cmap='inferno', vmin=0, vmax=0.05, aspect='auto')
        ax.axvline(0, color='cyan', linestyle='--', linewidth=2) # Horizon
        ax.set_title(f"Rindler (t={t})")

    plt.suptitle(f"Collision with the Horizon ({Lx}x{Ly} Lattice)", fontsize=14)
    plt.tight_layout()
    if savefig:
        plt.savefig('gaussian_wave.pdf')
    plt.show()

def event_horizon_animation(Lx: int, Ly: int, times: list, H_M, H_R, verbose: bool, saveanim: bool, frames=60):
    """
    This function creates an animation (a video) showing the propagation of the wavepacket towards the event horizon
    It returns the animation. In order to display it, set verbose=True. In order to save it, set saveanim=True.
    """

    #define time interval
    t_max = max(times)
    dt = t_max / frames

    #create initial gaussian state
    psi0 = gaussian_wavepacket(Lx, Ly)

    #get eigenstates
    energiesM, statesM=get_eigenstates(H_M)
    energiesR, statesR=get_eigenstates(H_R)

    #create matrix with the eigenstates
    V_M = np.hstack([s.full() for s in statesM])
    V_R = np.hstack([s.full() for s in statesR])

    #project initial state onto the eigenbases -----> c_n = <n|psi0>
    c_M = V_M.conj().T @ psi0
    c_R = V_R.conj().T @ psi0

    #to store frames
    evolution_M = []
    evolution_R = []

    for i in range(frames):
        #time 
        t = i * dt
        #evolve: psi(t) = sum( c_n * exp(-iEn*t) * |n> )
        psi_M_t = V_M @ (c_M * np.exp(-1j * energiesM * t))
        psi_R_t = V_R @ (c_R * np.exp(-1j * energiesR * t))
        #store density |psi|^2
        dens_M = np.abs(psi_M_t.flatten().reshape(Lx, Ly)).T
        dens_R = np.abs(psi_R_t.flatten().reshape(Lx, Ly)).T
        #append
        evolution_M.append(dens_M)
        evolution_R.append(dens_R)

    #animation plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.close() # Prevent double display

    #initial frame
    extent = [-(Lx//2), Lx//2, -(Ly//2), Ly//2]
    img1 = ax1.imshow(evolution_M[0], origin='lower', extent=extent, cmap='inferno', vmin=0, vmax=0.05)
    img2 = ax2.imshow(evolution_R[0], origin='lower', extent=extent, cmap='inferno', vmin=0, vmax=0.05)

    #styling
    for ax in [ax1, ax2]:
        ax.axvline(0, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7) # Horizon
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    ax1.set_title("Minkowski")
    ax2.set_title("Rindler")

    #animation function
    def update(frame):
        img1.set_data(evolution_M[frame])
        img2.set_data(evolution_R[frame])
        return img1, img2

    #create the animation
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    
    #display animation if we want
    if verbose:
        HTML(anim.to_jshtml())
    #save animation if we want
    if saveanim:
        anim.save('wave_animation.gif', writer='imagemagick', fps=30)

    return anim

def thermal_atmosphere(Lx: int, Ly: int, H_M, H_R):
    """
    This function computes the vaccum occupation factor for the Minkowsky and Rindler Hamiltonians. 
    Then, it computes the difference and reshapes the array as a 2D heatmap resembling the grid.
    """
    energiesM, statesM=get_eigenstates(H_M)
    energiesR, statesR=get_eigenstates(H_R)

    # Minkowski vaccum occupation factor
    # sum the probability densities of all occupied negative-energy Minkowksky states
    indices_occ_M = np.where(energiesM < 0)[0]
    V_M = np.hstack([statesM[i].full() for i in indices_occ_M])
    n_Minkowski = np.sum(np.abs(V_M)**2, axis=1)

    # Rindler vaccum occupation factor
    # sum the probability densities of all occupied negative-energy Rindler states
    indices_occ_R = np.where(energiesR < 0)[0]
    V_R = np.hstack([statesR[i].full() for i in indices_occ_R])
    n_Rindler = np.sum(np.abs(V_R)**2, axis=1)

    # Amount of excess particles (Minkowsky Vacuum - Rindler vaccum)
    atmosphere_1D = n_Minkowski - n_Rindler
    #Reshape as a heatmap to "see" the event horizon
    atmosphere_2D = atmosphere_1D.reshape(Lx, Ly).T

    return atmosphere_1D, atmosphere_2D

def plot_atmosphere_1D(atmosphere_2D, Lx: int, Ly: int, savefig: bool):
    """
    Plots the cross section of the thermal atmosphere heatmap in log scale
    """
    # Extract the central row for the cross-section (this is like having Ly=1)
    mid_y = Ly // 2
    profile = atmosphere_2D[mid_y, :]
    x_axis = np.arange(Lx) - (Lx // 2) #center x axis
    
    #calculate limits for consistent plotting
    epsilon = 1e-15
    data_abs = np.abs(profile) + epsilon
    peak_val = np.max(data_abs)
    min_val = peak_val * 1e-5
    
    plt.figure(figsize=(8, 5))

    plt.plot(x_axis, data_abs, 'o-', color='darkorange', 
             markersize=5, linewidth=2, label='Particle Density')

    plt.yscale('log')
    plt.title("Thermal Decay Profile (Cross-Section)", fontsize=12)
    plt.xlabel("Distance from Horizon ($x$)", fontsize=10)
    plt.ylabel("Particle Density (Log Scale)", fontsize=10)
    plt.axvline(0, color='black', linestyle='--', linewidth=2, label='Event Horizon')

    # Grid and Ticks
    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.grid(True, which="minor", ls=":", alpha=0.3)
    
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    plt.ylim(min_val, peak_val * 5)
    plt.xlim(-(Lx // 2), Lx // 2)
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig('thermal_gradient_1D.pdf')
    plt.show()

def plot_atmosphere_2D(atmosphere_2D, Lx: int, Ly: int, savefig: bool):
    """
    Plots the thermal atmosphere heatmap in log scale
    """
    # Avoid log(0)
    epsilon = 1e-15
    atmosphere_pos = np.abs(atmosphere_2D) + epsilon
    
    # Limits
    peak_val = np.max(atmosphere_pos)
    min_val = peak_val * 1e-5

    plt.figure(figsize=(8, 5))

    im = plt.imshow(atmosphere_pos, origin='lower', cmap='inferno', 
            extent=[-(Lx//2), Lx//2, -(Ly//2), Ly//2],
            norm=LogNorm(vmin=min_val, vmax=peak_val), 
            aspect='auto') 

    plt.title("Thermal Atmosphere Visualization", fontsize=12)
    plt.xlabel("Distance from Horizon ($x$)", fontsize=10)
    plt.ylabel("Transverse Position ($y$)", fontsize=10)
    plt.axvline(0, color='black', linestyle='--', linewidth=2, label='Event Horizon')

    # Zoom
    plt.xlim(-15, 15)

    cbar = plt.colorbar(im, format=ticker.LogFormatterMathtext())
    cbar.set_label("Excess Particle Density", fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    if savefig:
        plt.savefig('thermal_gradient_2D.pdf')
    plt.show()