import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# -----------------------
# 1. Parameters
# -----------------------
nx, ny = 41, 41         # grid points
Lx, Ly = 1.0, 1.0       # domain size (m)
dx, dy = Lx/(nx-1), Ly/(ny-1)
nt = 200                # time steps
dt = 0.001              # time step size
nu = 0.01               # viscosity
rho = 1.0               # density

# -----------------------
# 2. Initial conditions
# -----------------------
u = np.zeros((ny, nx))  # x-velocity
v = np.zeros((ny, nx))  # y-velocity
p = np.zeros((ny, nx))  # pressure
T = np.full((ny, nx), 280.0)  # temperature tracer 280 K

# Example: cold patch of ice 250 K, slightly out of centre
ice_patch = np.zeros((ny, nx), dtype=bool)
cx, cy = nx//2-2, ny//2-2
radius = 8
for i in range(nx):
    for j in range(ny):
        if (i-cx)**2 + (j-cy)**2 < radius**2:
            ice_patch[j, i] = True

# -----------------------
# 3. Helper functions
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
            - dx**2 * dy**2 / (2*(dx**2 + dy**2)) * b[1:-1, 1:-1]
        )
        # BCs: p=0 at boundaries
        p[:, 0] = p[:, -1] = 0
        p[0, :] = p[-1, :] = 0
    return p

# -----------------------
# 4. Time-stepping loop
# -----------------------
def simulate():
    global u, v, p, T
    for n in range(nt):
        un, vn, Tn = u.copy(), v.copy(), T.copy()
        b = build_up_pressure(p, u, v, dx, dy, dt, rho)
        p = pressure_poisson(p, b, dx, dy)

        # Momentum equations (Navier–Stokes)
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt/dx * (un[1:-1, 1:-1] - un[1:-1, :-2])
            - vn[1:-1, 1:-1] * dt/dy * (un[1:-1, 1:-1] - un[:-2, 1:-1])
            - dt/(2*rho*dx) * (p[1:-1, 2:] - p[1:-1, :-2])
            + nu * (dt/dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2])
                    + dt/dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]))
        )

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

        # Boundary conditions: walls
        u[0, :], u[-1, :], u[:, 0], u[:, -1] = 0, 0, 0, 0
        v[0, :], v[-1, :], v[:, 0], v[:, -1] = 0, 0, 0, 0

        # Ice patch solid boundary condition
        T[ice_patch] = 250.0 # Dirichlet BC 
        u[ice_patch] = 0.0 # no flow in x-direction
        v[ice_patch] = 0.0 # no flow in y-direction

# -----------------------
# 5. Visualization
# -----------------------
fig, ax = plt.subplots()
def animate(i):
    simulate()
    ax.clear()
    ax.set_title(f"Step {i}")
    T_plot = np.ma.masked_where(ice_patch, T) # masks the ice patch temperature profile
    ax.imshow(T_plot, cmap='coolwarm', origin='lower', extent=[0,Lx,0,Ly])
    ax.quiver(np.linspace(0,Lx,nx), np.linspace(0,Ly,ny), u, v)

ani = animation.FuncAnimation(fig, animate, frames=50, interval=50)
plt.show()