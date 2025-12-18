import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. LATTICE INITIALIZATION
# ============================================================
def initialize_lattice(L, init_type="random", rng=None):
    """
    Creates an L x L array of spins (each spin is +1 or -1).

    init_type:
        "random" -> each spin is chosen randomly with equal chance.
        "up"     -> all spins +1 (fully magnetized).
        "down"   -> all spins -1.
    """
    if rng is None:
        rng = np.random.default_rng()

    if init_type == "random":
        # Randomly assign +1 or -1 to each lattice site
        spins = rng.choice([-1, 1], size=(L, L))
    elif init_type == "up":
        spins = np.ones((L, L), dtype=int)
    elif init_type == "down":
        spins = -np.ones((L, L), dtype=int)
    else:
        raise ValueError("init_type must be 'random', 'up', or 'down'.")
    return spins


# ============================================================
# 2. ENERGY CALCULATION USING ISING HAMILTONIAN
# ============================================================
def compute_total_energy(spins, J=1.0, h=0.0):
    """
    Computes the total energy of the system using the Ising Hamiltonian:

        H = -J sum(s_i * s_neighbor) - h sum(s_i)

    Only RIGHT and DOWN neighbors are used to avoid double counting.
    Periodic boundary conditions are applied via modulo indexing.
    """
    L = spins.shape[0]
    E = 0.0

    for i in range(L):
        for j in range(L):

            S = spins[i, j]

            # RIGHT neighbor (wrap around using modulo)
            S_right = spins[i, (j + 1) % L]

            # DOWN neighbor
            S_down = spins[(i + 1) % L, j]

            # Add the pair interactions
            E -= J * S * (S_right + S_down)

    # Add contribution from external field term
    M = np.sum(spins)
    E -= h * M

    return E


# ============================================================
# 3. METROPOLIS SINGLE SPIN FLIP
# ============================================================
def metropolis_single_flip(spins, beta, J=1.0, h=0.0, rng=None):
    """
    Attempts one spin flip using the Metropolis acceptance rule.

    Steps:
        1. Pick random site (i, j).
        2. Calculate dE = energy change if s_ij -> -s_ij.
        3. If dE <= 0: accept (energy decreases).
        4. If dE > 0: accept with probability exp(-beta * dE).
        5. If accepted, flip the spin. Otherwise do nothing.
    """
    if rng is None:
        rng = np.random.default_rng()

    L = spins.shape[0]

    # Pick random spin
    i = rng.integers(0, L)
    j = rng.integers(0, L)
    s = spins[i, j]

    # Identify the 4 nearest neighbors (with periodic boundary conditions)
    up    = spins[(i - 1) % L, j]
    down  = spins[(i + 1) % L, j]
    left  = spins[i, (j - 1) % L]
    right = spins[i, (j + 1) % L]

    neighbor_sum = up + down + left + right

    # Formula for the energy change of flipping a single spin:
    # dE = 2 * J * s * sum(neighbors) + 2 * h * s
    dE = 2.0 * J * s * neighbor_sum + 2.0 * h * s

    # Metropolis acceptance rule
    if dE <= 0:
        # Flip is favorable -> always accept
        spins[i, j] = -s
        return dE, True
    else:
        # Flip is unfavorable -> accept probabilistically
        if rng.random() < np.exp(-beta * dE):
            spins[i, j] = -s
            return dE, True
        else:
            return 0.0, False


# ============================================================
# 4. METROPOLIS SWEEP = L^2 SPIN-FLIP ATTEMPTS
# ============================================================
def metropolis_sweep(spins, beta, J=1.0, h=0.0, rng=None):
    """
    Performs one Monte Carlo sweep.
    A sweep = one attempted flip per spin (L^2 total attempts).
    """
    if rng is None:
        rng = np.random.default_rng()

    L = spins.shape[0]
    N = L * L

    total_dE = 0.0
    n_accept = 0

    # Try flipping each spin once (on average)
    for _ in range(N):
        dE, accepted = metropolis_single_flip(spins, beta, J=J, h=h, rng=rng)
        total_dE += dE
        if accepted:
            n_accept += 1

    return total_dE, n_accept


# ============================================================
# 5-7. RUN SIMULATION AT ONE TEMPERATURE: EQUILIBRATION + MEASUREMENT
# ============================================================
def run_ising_at_temperature(L,
                             T,
                             n_eq_sweeps=500,
                             n_meas_sweeps=1000,
                             J=1.0,
                             h=0.0,
                             init_type="random",
                             measure_interval=1,
                             rng=None):
    """
    Runs the Ising simulation at a fixed temperature T.

    Process:
        (A) Initialize lattice.
        (B) Equilibrate -> discard these sweeps.
        (C) Perform measurement sweeps -> record E, M, E^2, M^2, |M|.
        (D) Compute averages: e, m, Cv, Chi.
    """
    if rng is None:
        rng = np.random.default_rng()

    beta = 1.0 / T
    N = L * L

    # Step A: Initialize lattice
    spins = initialize_lattice(L, init_type=init_type, rng=rng)

    # Step B: Equilibration (no measurements)
    for _ in range(n_eq_sweeps):
        metropolis_sweep(spins, beta, J=J, h=h, rng=rng)

    # Variables to accumulate measurements
    E_sum = 0.0
    E2_sum = 0.0
    M_sum = 0.0
    M2_sum = 0.0
    M_abs_sum = 0.0
    n_meas = 0

    # Step C: Measurement sweeps
    for sweep in range(n_meas_sweeps):
        metropolis_sweep(spins, beta, J=J, h=h, rng=rng)

        # Only record every "measure_interval" sweeps
        if sweep % measure_interval == 0:

            # Compute energy and magnetization of current lattice
            E = compute_total_energy(spins, J=J, h=h)
            M = np.sum(spins)

            E_sum += E
            E2_sum += E * E
            M_sum += M
            M2_sum += M * M
            M_abs_sum += abs(M)

            n_meas += 1

    # Step D: Compute averages
    E_mean = E_sum / n_meas
    E2_mean = E2_sum / n_meas
    M_mean = M_sum / n_meas
    M2_mean = M2_sum / n_meas
    M_abs_mean = M_abs_sum / n_meas

    # Convert to per-spin quantities
    e_mean = E_mean / N
    m_mean = M_mean / N
    m_abs_mean = M_abs_mean / N

    # Heat capacity per spin (fluctuation formula)
    Cv = (E2_mean - E_mean**2) / (N * T**2)

    # Susceptibility per spin
    Chi = (M2_mean - M_mean**2) / (N * T)

    return {
        "T": T,
        "E_mean": E_mean,
        "E2_mean": E2_mean,
        "M_mean": M_mean,
        "M2_mean": M2_mean,
        "M_abs_mean": M_abs_mean,
        "e_mean": e_mean,
        "m_mean": m_mean,
        "m_abs_mean": m_abs_mean,
        "Cv": Cv,
        "Chi": Chi,
        "final_spins": spins.copy()  # store final configuration
    }


# ============================================================
# 8-9. TEMPERATURE SWEEP
# ============================================================
def temperature_sweep(L,
                      T_list,
                      n_eq_sweeps=500,
                      n_meas_sweeps=1000,
                      J=1.0,
                      h=0.0,
                      init_type="random",
                      measure_interval=1,
                      rng=None):
    """
    Runs the Ising simulation for every temperature in T_list.

    Returns a dictionary of observables vs temperature,
    which can be plotted or saved.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Lists to accumulate results
    T_values = []
    e_values = []
    m_values = []
    m_abs_values = []
    Cv_values = []
    Chi_values = []
    final_configs = []

    for T in T_list:
        print(f"Running T = {T:.3f} ...")

        res = run_ising_at_temperature(
            L=L,
            T=T,
            n_eq_sweeps=n_eq_sweeps,
            n_meas_sweeps=n_meas_sweeps,
            J=J,
            h=h,
            init_type=init_type,
            measure_interval=measure_interval,
            rng=rng
        )

        # Append results from this temperature
        T_values.append(res["T"])
        e_values.append(res["e_mean"])
        m_values.append(res["m_mean"])
        m_abs_values.append(res["m_abs_mean"])
        Cv_values.append(res["Cv"])
        Chi_values.append(res["Chi"])
        final_configs.append(res["final_spins"])

    # Return everything as a structured dictionary
    return {
        "T": np.array(T_values),
        "e_mean": np.array(e_values),
        "m_mean": np.array(m_values),
        "m_abs_mean": np.array(m_abs_values),
        "Cv": np.array(Cv_values),
        "Chi": np.array(Chi_values),
        "final_configs": final_configs
    }


# ============================================================
# 10. PLOTTING OBSERVABLES
# ============================================================
def plot_observables(data):
    """
    Creates 4 plots:
        - Energy per spin vs T
        - Magnetization vs T
        - Heat capacity vs T
        - Susceptibility vs T
    """
    T = data["T"]
    e = data["e_mean"]
    m = data["m_mean"]
    m_abs = data["m_abs_mean"]
    Cv = data["Cv"]
    Chi = data["Chi"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(T, e, marker='o')
    ax1.set_xlabel("Temperature T")
    ax1.set_ylabel("Energy per spin e")
    ax1.set_title("Energy vs. Temperature")

    ax2.plot(T, m, marker='o', label="m")
    ax2.plot(T, m_abs, marker='s', label="|m|")
    ax2.set_xlabel("Temperature T")
    ax2.set_ylabel("Magnetization per spin")
    ax2.set_title("Magnetization vs. Temperature")
    ax2.legend()

    ax3.plot(T, Cv, marker='o')
    ax3.set_xlabel("Temperature T")
    ax3.set_ylabel("Heat Capacity Cv")
    ax3.set_title("Heat Capacity vs. Temperature")

    ax4.plot(T, Chi, marker='o')
    ax4.set_xlabel("Temperature T")
    ax4.set_ylabel("Susceptibility Chi")
    ax4.set_title("Susceptibility vs. Temperature")

    plt.tight_layout()
    plt.show()


# ============================================================
# 11. VISUALIZE SPIN CONFIGURATION
# ============================================================
def visualize_spins(spins, ax=None, title=None):
    """
    Displays a 2D spin configuration using a red/blue colormap.
    +1 -> red
    -1 -> blue

    If ax is provided, draw into that axes (for subplots) and
    DO NOT call plt.show() here. If ax is None, create a new
    figure and show it immediately.
    """
    own_fig = False   # track whether we created the figure

    if ax is None:
        fig, ax = plt.subplots()
        own_fig = True

    ax.imshow(spins, cmap="bwr", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    # Only show if we created our own figure
    if own_fig:
        plt.show()


# ============================================================
# MAIN EXAMPLE (RUN WHEN FILE IS EXECUTED DIRECTLY)
# ============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    # ---------- CHOOSE ONE MODE BY COMMENTING/UNCOMMENTING ----------
    # more sweeps to reach equilibrium
    # more measurements -> smoother statistics
    # measure every 10 sweeps

    # FAST MODE: quick runs, good for debugging & sanity checks
    #L = 20
    #T_list = np.linspace(1.5, 3.5, 7)  # fewer temperatures
    #n_eq_sweeps = 500
    #n_meas_sweeps = 2000
    #measure_interval = 5

    # MEDIUM MODE: nicer curves, still reasonable runtime
    L = 32
    T_list = np.linspace(1.5, 3.5, 17)  # focus around Tc ~ 2.27
    n_eq_sweeps = 2000
    n_meas_sweeps = 5000
    measure_interval = 10

    # -------------------------------------------------------------

    data = temperature_sweep(
        L=L,
        T_list=T_list,
        n_eq_sweeps=n_eq_sweeps,
        n_meas_sweeps=n_meas_sweeps,
        init_type="random",
        measure_interval=measure_interval,
        rng=rng
    )

    # Plot observables
    plot_observables(data)

    # Visualize sample configurations (low, mid, high T)
    configs = data["final_configs"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    visualize_spins(configs[0], ax=axes[0], title=f"T = {T_list[0]:.2f}")
    visualize_spins(configs[len(T_list)//2], ax=axes[1], title=f"T = {T_list[len(T_list)//2]:.2f}")
    visualize_spins(configs[-1], ax=axes[2], title=f"T = {T_list[-1]:.2f}")

    plt.tight_layout()
    plt.show()
