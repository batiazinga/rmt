import numpy as np

from wishart.lib import scm

np.random.seed(42)

p1 = 490
p2 = 10
p = p1 + p2  # dimension
n = 5000  # sample size
c = p / n

# generate gaussian data
# the correlation matrix has p2 eigenvalues with high multiplicity
data1 = np.random.randn(p1, n)
data2 = np.sqrt(2.0) * np.random.randn(p2, n)
data = np.concatenate([data1, data2], axis=0)

# eigen vectors of the Wishart matrix
m = scm(data)
_, eigen_vectors = np.linalg.eigh(m)


# correlation of eigenspace 2
original_eigen_vectors = np.identity(p2)
new_eigen_vectors = eigen_vectors[p1:, p1:]
corr = 0.0
for i in range(p2):
    for j in range(p2):
        corr += np.dot(new_eigen_vectors[:, i], original_eigen_vectors[:, j]) ** 2
corr /= p2
print(corr)
