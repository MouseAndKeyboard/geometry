import numpy as np
import matplotlib.pyplot as plt
import math


class kVector():
    def __init__(self, oneVector):
        self.vector_dim = oneVector.shape[0] # dimension of the vector
        self.k = 1 # Always start with a 1-vector
        
        self.dim = math.comb(self.vector_dim, self.k) # dimension of the k-vector space

        self.coefficients = dict()  # coefficients of the basis vectors

        for i in range(self.vector_dim):
            self.coefficients[(i,)] = oneVector[i]

    def set_coefficient(self, indices, value):
        """Set the coefficient for the basis vector formed by wedging the vectors with the given indices."""
        
        # check if the indices are valid
        if len(indices) != self.k:
            raise ValueError(f"Indices must be of length {self.k=}.")

        for index in indices:
            if index < 0 or index >= self.vector_dim:
                raise ValueError(f"Index {index} is out of range.")
            
        self.coefficients[tuple(sorted(indices))] = value

    def get_coefficient(self, indices):
        """Get the coefficient for the basis vector formed by wedging the vectors with the given indices."""
        return self.coefficients.get(tuple(sorted(indices)), 0)
    
    def merge(self, left, right):
        """Merge two sorted lists and return the number of inversions and the sorted list."""
        merged = []
        inversions = 0
        i = 0
        j = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
                inversions += len(left) - i

        merged += left[i:]
        merged += right[j:]

        return inversions, merged

    def wedge(self, other):

        if self.vector_dim != other.vector_dim:
            raise ValueError(f"Vectors must have the same dimension.")
        
        new_k = self.k + other.k

        new_k_vector = kVector(np.zeros(self.vector_dim))
        new_k_vector.k = new_k
        new_k_vector.coefficients = dict()

        for indices1, coefficient1 in self.coefficients.items():
            for indices2, coefficient2 in other.coefficients.items():
                if len(set(indices1).intersection(set(indices2))) != 0:
                    continue # skip if the indices are not disjoint (i.e. the wedge product is 0)

                inversions, new_indices = self.merge(indices1, indices2)
                new_indices = tuple(new_indices)

                if inversions % 2 == 1:
                    new_coefficient = -coefficient1 * coefficient2
                else:
                    new_coefficient = coefficient1 * coefficient2

                new_k_vector.set_coefficient(new_indices,
                                             new_k_vector.get_coefficient(new_indices) + new_coefficient)
        
        return new_k_vector
    
    def __neg__(self):
        new_k_vector = kVector(np.zeros(self.vector_dim))
        new_k_vector.k = self.k
        new_k_vector.coefficients = dict()

        for indices, coefficient in self.coefficients.items():
            new_k_vector.set_coefficient(indices, -coefficient)
        
        return new_k_vector
    
    def add(self, other):
        # check if the vectors have the same dimension
        if self.vector_dim != other.vector_dim:
            raise ValueError(f"Vectors must have the same dimension.")
        
        # check if the vectors have the same k
        if self.k != other.k:
            raise ValueError(f"Vectors must have the same k.")
        
        new_k_vector = kVector(np.zeros(self.vector_dim))
        new_k_vector.k = self.k
        new_k_vector.coefficients = dict()

        for indices, coefficient in self.coefficients.items():
            new_k_vector.set_coefficient(indices, coefficient)

        for indices, coefficient in other.coefficients.items():
            new_k_vector.set_coefficient(indices, new_k_vector.get_coefficient(indices) + coefficient)

        return new_k_vector
    
    def __eq__(self, other):
        if not isinstance(other, kVector):
            # other is not a kVector
            return False
        if self.vector_dim != other.vector_dim or self.k != other.k:
            # Dimensions or k's don't match
            return False
        
        for indices, coefficient in self.coefficients.items():
            if coefficient != other.get_coefficient(indices):
                return False

        return True
    
    def __str__(self):
        s = ""
        for indices, coefficient in self.coefficients.items():
            if coefficient == 0:
                continue
            if s != "":
                if coefficient > 0:
                    s += " + "
                else:
                    s += " - "
            else:
                if coefficient < 0:
                    s += "-"

            s += f"{abs(coefficient)}*e{indices}"
        
        return f"kVector({s})"

    def __repr__(self) -> str:
        return self.__str__()
    

    def plot(self, ax=None):
        if self.vector_dim != 3:
            raise ValueError(f"Can only plot 3-dimensional vectors.")
        
        if self.k > 2:
            raise ValueError(f"Can only plot 1-vectors and 2-vectors.")
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        if self.k == 1:
            ax.quiver(0, 0, 0, self.coefficients[(0,)], self.coefficients[(1,)], self.coefficients[(2,)])

            # calculate the axis limits
            max_val = max([abs(self.coefficients[(0,)]), abs(self.coefficients[(1,)]), abs(self.coefficients[(2,)])])

            ax.set_xlim(-max_val, max_val)
            ax.set_ylim(-max_val, max_val)
            ax.set_zlim(-max_val, max_val)

            return ax

        if self.k == 2:
            # Define the coordinates of the unit square in the xy, xz, and yz planes
            unit_square_xy = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0]])
            unit_square_xz = np.array([[0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 1, 0]])
            unit_square_yz = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])

            # Scale the unit squares by the coefficients of the 2-vector
            square_xy = self.get_coefficient((0, 1)) * unit_square_xy
            square_xz = self.get_coefficient((0, 2)) * unit_square_xz
            square_yz = self.get_coefficient((1, 2)) * unit_square_yz

            # Plot the squares
            ax.plot(square_xy[0], square_xy[1], square_xy[2])
            ax.plot(square_xz[0], square_xz[1], square_xz[2])
            ax.plot(square_yz[0], square_yz[1], square_yz[2])

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            return ax