import scipy.io
import pandas as pd
import numpy as np

# Load the QM7 dataset (.mat file)
mat_data = scipy.io.loadmat('qm7.mat')

# Inspect the keys in the dataset
print(mat_data.keys())

# Extract relevant data
X = mat_data['X']  # Coulom matrices: X (7165 x 3 x 3)
T = mat_data['T']  # Atomisation energies: T (7165)
Z = mat_data['Z']  # Atomic charge: Z (7165)
R = mat_data['R']  # Cartesian coordinates: R (7165 x 3)

# Flatten the Coulomb matrix for each molecule
X_flattened = X.reshape(X.shape[0], -1)

# Convert atomic charges and coordinates into separate columns
Z_df = pd.DataFrame(Z, columns=[f'Z_atom_{i}' for i in range(Z.shape[1])])
R_df = pd.DataFrame(R.reshape(R.shape[0], -1), columns=[
    f'{coord}_atom_{i}' for i in range(R.shape[1]) for coord in ['x', 'y', 'z']
])

# Add labels (atomization energies) and metadata
df = pd.DataFrame(X_flattened, columns=[f'Coulomb_{i}' for i in range(X_flattened.shape[1])])
df['Atomization_Energy'] = T[0]

# Add atomic charges and Cartesian coordinates
df = pd.concat([df, Z_df, R_df], axis=1)

# Display the DataFrame
print(df.columns)
print()

print(df.head())
print()


