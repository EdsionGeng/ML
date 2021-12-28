import numpy as np

A = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        if (i == j):
            A[i, j] = 2
        if (abs(i - j) == 1):
            A[i, j] = A[j, i] = -1
b = np.ones((100, 1))
print('Conjugate Gradient X:')
x = np.zeros((100, 1))
r = b - np.dot(A, x)
p = r
for i in range(100):
    r1 = r
    a = np.dot(r.T, r) / np.dot(p.T, np.dot(A, p))
    x = x + a * p
    r = b - np.dot(A, x)
    q = np.linalg.norm(np.dot(A, x) - b) / np.linalg.norm(b)
    if q < 10 ** -6:
        break
    else:
        beta = np.linalg.norm(r) ** 2 / np.linalg.norm(r1) ** 2
        p = r + beta * p

print(x)
print("done Conjugate Gradient!")
