import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# -----------------------
# Set Parameters and define grid characteristics
# -----------------------
nx, ny = 100, 100         # grid points
Lx, Ly = 1.0, 1.0       # domain size
dx, dy = Lx/(nx-1), Ly/(ny-1) # grid spacing
nt = 200                # time steps
dt = 0.001              # time step size
nu = 0.005               # viscosity
rho = 1.0               # density

# -----------------------
# Define the unknown variables with initial conditions
# -----------------------
u = np.zeros((ny, nx))  # x-velocity
v = np.zeros((ny, nx))  # y-velocity
p = np.zeros((ny, nx))  # pressure
T = np.full((ny, nx), 280.0)  # temperature tracer 280 K

# -----------------------
# Example: model three ice patches of different temperatures
# to see how the gradient in the temperature field
# evolves over time
# -----------------------
num_patches = 3
patch_temps = [265.0, 240.0, 250.0]
ice_patch = [np.zeros((ny, nx), dtype=bool) for _ in range(num_patches)]
cx, cy = nx//2, ny//2
radius = 8

# Example patch centers (customize as needed)
patch_centers = [
    (cx, cy),           # Center
    (cx - nx//4, cy - ny//4),   # Top-left
    (cx + nx//4, cy + ny//4)    # Bottom-right
]

# Set ice patch masks
for idx, (px, py) in enumerate(patch_centers):
    for x in range(nx):
        for y in range(ny):
            if (x - px)**2 + (y - py)**2 < radius**2:
                ice_patch[idx][y, x] = True

# Set initial temperatures for each patch
for idx, mask in enumerate(ice_patch):
    T[mask] = patch_temps[idx]

# Combine all patches into one mask for plotting and boundary conditions
# ice_patch_mask = np.any(ice_patch, axis=0)
ice_patch = np.any(ice_patch, axis=0)

# -----------------------
# Implement functions to calculate Poisson-pressure to ensure
# that the velocity field stays divergence free: nabla*v = 0 
# -----------------------
# delta(p) = b
# -----------------------
def build_up_pressure(p, u, v, dx, dy, dt, rho):
    b = np.zeros_like(p)
    b[1:-1, 1:-1] = rho * (
        (1/dt) * ((u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx) +
                  (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy))
        - ((u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx))**2
        - 2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2*dy) *
               (v[1:-1, 2:] - v[1:-1, :-2]) / (2*dx))
        - ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy))**2
    )
    return b

def pressure_poisson(p, b, dx, dy):
    pn = np.empty_like(p)
    for _ in range(50):
        pn[:] = p
        p[1:-1, 1:-1] = (
            ((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
             (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
            (2*(dx**2 + dy**2))
            - ( dx**2 * dy**2 / (2*(dx**2 + dy**2))) * b[1:-1, 1:-1]
        )
        # BCs: p=0 at boundaries
        p[:, 0] = p[:, -1] = 0
        p[0, :] = p[-1, :] = 0
    return p

# -----------------------
# Time-stepping loop
# -----------------------
def simulate():  
    global u, v, p, T
    for n in range(nt):
        un, vn, Tn = u.copy(), v.copy(), T.copy()
        b = build_up_pressure(p, u, v, dx, dy, dt, rho)
        p = pressure_poisson(p, b, dx, dy)

        # Momentum equations (Navier–Stokes)
        # ---------------------
        # the unknowns will be updated with the explicit Euler method
        # ---------------------

        # x-Komponent of the velocity vector
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt/dx * (un[1:-1, 1:-1] - un[1:-1, :-2])
            - vn[1:-1, 1:-1] * dt/dy * (un[1:-1, 1:-1] - un[:-2, 1:-1])
            - dt/(2*rho*dx) * (p[1:-1, 2:] - p[1:-1, :-2])
            + nu * (dt/dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2])
                    + dt/dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]))
        )

        # y-Komponent of the velocity vector
        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt/dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2])
            - vn[1:-1, 1:-1] * dt/dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1])
            - dt/(2*rho*dy) * (p[2:, 1:-1] - p[:-2, 1:-1])
            + nu * (dt/dx**2 * (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2])
                    + dt/dy**2 * (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1]))
        )

        # Temperature advection
        T[1:-1, 1:-1] = (
            Tn[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt/dx * (Tn[1:-1, 1:-1] - Tn[1:-1, :-2])
            - vn[1:-1, 1:-1] * dt/dy * (Tn[1:-1, 1:-1] - Tn[:-2, 1:-1])
            + nu * (dt/dx**2 * (Tn[1:-1, 2:] - 2*Tn[1:-1, 1:-1] + Tn[1:-1, :-2])
                    + dt/dy**2 * (Tn[2:, 1:-1] - 2*Tn[1:-1, 1:-1] + Tn[:-2, 1:-1]))
        )

        #for idx, mask in enumerate(ice_patch):
        #    T[mask] = patch_temps[idx]

        # Boundary conditions: walls (no-slip condition)
        u[0, :], u[-1, :], u[:, 0], u[:, -1] = 0, 0, 0, 0
        v[0, :], v[-1, :], v[:, 0], v[:, -1] = 0, 0, 0, 0

        # Ice patch solid boundary condition (Dirichlet)
        # change to ice_patch_mask
        u[ice_patch] = 0.0 # no flow in x-direction
        v[ice_patch] = 0.0 # no flow in y-direction

# -----------------------
# Plot the domain 
# -----------------------
fig, ax = plt.subplots()
def animate(i):
    simulate()
    ax.clear()
    ax.set_title(f"Step {i}")
    # change to ice_patch_mask
    T_plot = np.ma.masked_where(ice_patch, T) # mask the ice patch temperature profile
    ax.imshow(T_plot, cmap='coolwarm', origin='lower', extent=[0,Lx,0,Ly])
    ax.quiver(np.linspace(0,Lx,nx), np.linspace(0,Ly,ny), u, v)

ani = animation.FuncAnimation(fig, animate, frames=50, interval=50)
plt.show()