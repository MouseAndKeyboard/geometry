
import unittest
import numpy as np
from discrete_diff_geo.k_vector import kVector

class TestKVector(unittest.TestCase):
    def test_wedge(self):
        u = kVector(np.array([1, 2, 3]))
        v = kVector(np.array([4, 5, 6]))

        w = u.wedge(v)

        self.assertEqual(w.get_coefficient((0, 1)), -3)
        self.assertEqual(w.get_coefficient((0, 2)), -6)
        self.assertEqual(w.get_coefficient((1, 2)), -3)


    def test_wedge_anticommutativity(self):
        u = kVector(np.array([1, 2, 3]))
        v = kVector(np.array([4, 5, 6]))
        
        self.assertEqual(u.wedge(v), -v.wedge(u))

    def test_wedge_associativity(self):
        u = kVector(np.array([1, 2, 3]))
        v = kVector(np.array([4, 5, 6]))
        w = kVector(np.array([7, 8, 9]))

        self.assertEqual(u.wedge(v).wedge(w), u.wedge(v.wedge(w)))

    def test_wedge_distributivity(self):
        u = kVector(np.array([1, 2, 3]))
        v = kVector(np.array([4, 5, 6]))
        w = kVector(np.array([7, 8, 9]))

        self.assertEqual(u.wedge(v.add(w)), u.wedge(v).add(u.wedge(w)))
        self.assertEqual(u.add(v).wedge(w), u.wedge(w).add(v.wedge(w)))

if __name__ == "__main__":
    unittest.main()
