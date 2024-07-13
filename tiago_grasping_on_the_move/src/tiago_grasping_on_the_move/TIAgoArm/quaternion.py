import numpy as np

class Quaternion:
    """
    Defines a Quaternion class with various operations.

    Attributes:
        w (float): Real part of the quaternion.
        x (float): Imaginary part along the x-axis.
        y (float): Imaginary part along the y-axis.
        z (float): Imaginary part along the z-axis.

    Methods:
        __repr__(): Returns a string representation of the quaternion.
        norm(): Computes the norm (magnitude) of the quaternion.
        normalize(): Normalizes the quaternion.
        conjugate(): Computes the conjugate of the quaternion.
        multiply(other): Multiplies the quaternion by another quaternion.
        quaternion_to_numpy(): Converts the quaternion to a NumPy array.
    """
    def __init__(self, w=None, x=None, y=None, z=None):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}i, {self.y}j, {self.z}k)"

    def set_coeffs(self,w,x,y,z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        n = self.norm()
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def multiply(self, other):
        w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
        x = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
        y = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
        z = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
        return Quaternion(w, x, y, z)

    def quaternion_to_numpy(self):
        return np.array([self.w, self.x, self.y, self.z])

    def inverse(self):
        """Calcola l'inverso del quaternione."""
        norma_quad = self.norm()**2
        return Quaternion(self.w/norma_quad, -self.x/norma_quad, -self.y/norma_quad, -self.z/norma_quad)

    def rotate_vector(self, vector):
        """Ruota un vettore nello spazio 3D usando questo quaternione."""
        q_vector = Quaternion(0, *vector)
        q_rotated_vector = self * q_vector * self.inverse()
        return np.array([q_rotated_vector.x, q_rotated_vector.y, q_rotated_vector.z])

    def to_euler_angles(self):
        """Converte il quaternione in angoli di Eulero (tornando roll(x), pitch(y), yaw(z))."""
        # Assumendo che il quaternione sia normalizzato
        def to_deg(X,Y,Z):
            return [np.degrees(X),np.degrees(Y),np.degrees(Z)]
        
        ysqr = self.y * self.y
        
        t0 = +2.0 * (self.w * self.x + self.y * self.z)
        t1 = +1.0 - 2.0 * (self.x * self.x + ysqr)
        X = np.arctan2(t0, t1)
        
        t2 = +2.0 * (self.w * self.y - self.z * self.x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = np.arcsin(t2)
        
        t3 = +2.0 * (self.w * self.z + self.x * self.y)
        t4 = +1.0 - 2.0 * (ysqr + self.z * self.z)
        Z = np.arctan2(t3, t4)
        
        return [X, Y, Z], to_deg(X,Y,Z)