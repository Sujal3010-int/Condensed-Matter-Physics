import numpy as np
import matplotlib.pyplot as plt

# Use natural units: ħ²/2m = 1
hbar2_2m = 1.0

# lattice constant
a = 1.0

# reciprocal lattice vector
G0 = 2*np.pi/a

N = 3

# V(x) = V1 cos(Gx) + V2 cos(2Gx)

V = {
    1: 0.5,
   -1: 0.5,
    2: 0.2,
   -2: 0.2
}


def build_hamiltonian(k, N, V, G0):

    size = 2*N + 1
    H = np.zeros((size, size))

    G_list = np.arange(-N, N+1) * G0

    for i in range(size):
        G = G_list[i]
        H[i,i] = hbar2_2m * (k + G)**2

    for i in range(size):
        for j in range(size):

            Gdiff = int(round((G_list[i] - G_list[j])/G0))

            if Gdiff in V:
                H[i,j] += V[Gdiff]

    return H

k_vals = np.linspace(-2*G0, 2*G0, 500)

bands = []

for k in k_vals:

    H = build_hamiltonian(k, N, V, G0)

    eigenvalues = np.linalg.eigvalsh(H)

    bands.append(eigenvalues)

bands = np.array(bands)

plt.figure(figsize=(8,6))

for i in range(bands.shape[1]):
    plt.plot(k_vals, bands[:,i])

for n in range(-3,4):
    plt.axvline(n*G0/2, linestyle="--")

plt.title("Band Structure")
plt.xlabel("k")
plt.ylabel("Energy")
plt.grid()
plt.show()


dk = k_vals[1] - k_vals[0]

slope = np.gradient(bands[:,0], dk)

index_center = np.argmin(np.abs(k_vals))
index_edge   = np.argmin(np.abs(k_vals - G0/2))

print("\nSlope at BZ center k=0:", slope[index_center])
print("Slope at BZ edge k=G/2:", slope[index_edge])


VG_values = np.linspace(0.1, 2.0, 20)
gaps = []

k_edge = G0/2

for VG in VG_values:

    V_test = {
        1: VG,
       -1: VG
    }

    H = build_hamiltonian(k_edge, 1, V_test, G0)

    eigenvalues = np.linalg.eigvalsh(H)

    gap = eigenvalues[1] - eigenvalues[0]

    gaps.append(gap)


plt.figure()

plt.plot(VG_values, gaps)

plt.title("Band Gap vs Fourier Harmonic Strength")
plt.xlabel("Fourier component V_G")
plt.ylabel("Band Gap Δ")
plt.grid()

plt.show()


print("\nBand gap proportionality check:")

for i in range(5):
    print("V_G =", round(VG_values[i],3),
          " Gap =", round(gaps[i],3))
"""Output:
Slope at BZ center k=0: 0.05030157689174012
Slope at BZ edge k=G/2: -0.40834497594372693

Band gap proportionality check:
V_G = 0.1  Gap = 0.2
V_G = 0.2  Gap = 0.4
V_G = 0.3  Gap = 0.6
V_G = 0.4  Gap = 0.8
V_G = 0.5  Gap = 1.0
"""