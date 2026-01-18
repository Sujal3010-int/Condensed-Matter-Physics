import numpy as np
import matplotlib.pyplot as plt

E0 = -5.0
beta = -1.0
bin_w= 0.2
chain_len = [5, 10, 20, 50, 200]

for N in chain_len:

    L = np.zeros((N, N))
    np.fill_diagonal(L, E0)
    for i in range(N - 1):
        L[i, i + 1] = beta
        L[i + 1, i] = beta

    energies = np.linalg.eigvalsh(L)
    energies = np.repeat(energies, 2)

    energies_sor = np.sort(energies)
    Ef = energies_sor[len(energies_sor) // 2]

    bins = np.arange(energies.min() - bin_w, energies.max() + bin_w, bin_w)
    counts, edges = np.histogram(energies, bins=bins)

    dos = counts / bin_w
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(6, 4))
    plt.bar(centers, dos, width=bin_w,
            edgecolor='b', alpha=0.75)
    plt.axvline(Ef, color='r', linestyle='--',
                linewidth=1.5, label="Fermi level")

    plt.xlabel("Energy (eV)")
    plt.ylabel("Density of states")
    plt.title(f"DOS vs energy plot (N = {N})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


for N in chain_len:
    L_0= np.zeros((N, N))
    np.fill_diagonal(L_0, E0)
    for i in range(N - 1):
        L_0[i, i + 1] = beta
        L_0[i + 1, i] = beta

    energies = np.linalg.eigvalsh(L_0)
    energies = np.sort(energies)
    n_occ = N // 2

    E_min = energies[0]
    Homo = energies[n_occ - 1]
    Lumo = energies[n_occ]
    E_max = energies[-1]
    sites = np.arange(1, N + 1)

    plt.figure(figsize=(6, 4))

    plt.plot(sites, [E_min]*N, label=f"Min energy ({E_min:.2f} eV)")
    plt.plot(sites, [Homo]*N, label=f"HOMO ({Homo:.2f} eV)")
    plt.plot(sites, [Lumo]*N, label=f"LUMO ({Lumo:.2f} eV)")
    plt.plot(sites, [E_max]*N, label=f"Max energy ({E_max:.2f} eV)")

    plt.xlabel("Sites of atom")
    plt.ylabel("Energy eigenvalue (eV)")
    plt.title(f"Energy Eigenstates vs site of atoms (N = {N})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()