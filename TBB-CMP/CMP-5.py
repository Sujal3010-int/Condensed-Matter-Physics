import numpy as np
import matplotlib.pyplot as plt


def ladder_bands(k, t, tp):
    E_plus = -2*t*np.cos(k) + tp
    E_minus = -2*t*np.cos(k) - tp
    return E_plus, E_minus

def ladder_with_diagonal(k, t, tp, tpp):
    off_diag = tp + 2*tpp*np.cos(k)
    E_plus = -2*t*np.cos(k) + off_diag
    E_minus = -2*t*np.cos(k) - off_diag
    return E_plus, E_minus

def plot_ladder():
    k = np.linspace(-np.pi, np.pi, 500)
    t = 1.0
    tp_values = [0.2, 1.0, 2.0]

    plt.figure(figsize=(8,6))
    for tp in tp_values:
        E1, E2 = ladder_bands(k, t, tp)
        plt.plot(k, E1, label=f"E+ (t'={tp})")
        plt.plot(k, E2, linestyle='--')

    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Two-leg ladder band structure")
    plt.xlabel("k")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid()
    plt.show()


def plot_ladder_diagonal():
    k = np.linspace(-np.pi, np.pi, 500)
    t = 1.0
    tp = 1.5
    tpp_values = [0.0, 0.3, -0.3]

    plt.figure(figsize=(8,6))
    for tpp in tpp_values:
        E1, E2 = ladder_with_diagonal(k, t, tp, tpp)
        plt.plot(k, E1, label=f't\"={tpp}')
        plt.plot(k, E2, linestyle='--')

    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Effect of diagonal hopping")
    plt.xlabel("k")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid()
    plt.show()



def square_energy(KX, KY, t):
    return -2*t*(np.cos(KX) + np.cos(KY))

def square_energy_diag(KX, KY, t, tpp):
    return -2*t*(np.cos(KX) + np.cos(KY)) - 4*tpp*np.cos(KX)*np.cos(KY)


def get_fermi_energy(E, filling):

    E_flat = np.sort(E.flatten())
    total_states = len(E_flat)

    index = int(filling * total_states / 2)

    index = max(0, min(index, total_states - 1))

    return E_flat[index]


def plot_square_fermi(t):
    N = 300  # higher resolution (better FS)
    kx = np.linspace(-np.pi, np.pi, N)
    ky = np.linspace(-np.pi, np.pi, N)
    KX, KY = np.meshgrid(kx, ky)

    E = square_energy(KX, KY, t)

    fillings = np.arange(0.5, 2.01, 0.25)

    plt.figure(figsize=(8,6))

    for n in fillings:
        EF = get_fermi_energy(E, n)
        cs = plt.contour(KX, KY, E, levels=[EF])
        plt.clabel(cs, inline=True, fontsize=8, fmt={EF: f"{n:.2f}e"})

    plt.title("Fermi surface (square lattice)")
    plt.xlabel("$k_x$")
    plt.ylabel("$k_y$")
    plt.grid()
    plt.show()


def plot_square_fermi_diag(t):
    N = 300
    kx = np.linspace(-np.pi, np.pi, N)
    ky = np.linspace(-np.pi, np.pi, N)
    KX, KY = np.meshgrid(kx, ky)

    fillings = np.arange(0.5, 2.01, 0.25)
    tpp_values = [-1, 1]

    for tpp in tpp_values:
        E = square_energy_diag(KX, KY, t, tpp)

        plt.figure(figsize=(8,6))

        for n in fillings:
            EF = get_fermi_energy(E, n)
            cs = plt.contour(KX, KY, E, levels=[EF])
            plt.clabel(cs, inline=True, fontsize=8)

        plt.title(f"Fermi surface with t'' = {tpp}")
        plt.xlabel("$k_x$")
        plt.ylabel("$k_y$")
        plt.grid()
        plt.show()



if __name__ == "__main__":

    plot_ladder()
    plot_ladder_diagonal()

    t = -2  # eV
    plot_square_fermi(t)
    plot_square_fermi_diag(t)
