import numpy as np
import matplotlib.pyplot as plt

a = 1 #h/2m takein in natural units
m_vals = np.array([-3,-2,-1,0,1,2,3])

k_vals = np.linspace(-6*np.pi/a, 6*np.pi/a, 400)


def eigenvalues(k, periodicity):

    if periodicity == "a":
        G = 2*np.pi*m_vals/a
    else:                         
        G = 2*np.pi*m_vals/(2*a)  

    H = np.diag((k + G)**2)

    E, _ = np.linalg.eigh(H)
    return E


E_a = []
E_2a = []

for k in k_vals:
    E_a.append(eigenvalues(k, "a"))
    E_2a.append(eigenvalues(k, "2a"))

E_a = np.array(E_a)
E_2a = np.array(E_2a)




plt.figure()
for i in range(len(m_vals)):
    plt.plot(k_vals, E_a[:, i])
plt.title(" E(k) for periodicity a")
plt.xlabel("k")
plt.ylabel("E")
plt.show()


plt.figure()
for i in range(len(m_vals)):
    plt.plot(k_vals, E_2a[:, i])
plt.title(" E(k) for periodicity 2a")
plt.xlabel("k")
plt.ylabel("E")
plt.show()
