import numpy as np
from regpy.operators import Operator

class EnergyNorm:
    def __init__(self, sigma):
        self.shape=sigma.shape
        self.sigma=sigma
        
    def norm(self, x):
        return np.linalg.norm(x/self.sigma)
    
from scipy.ndimage import gaussian_filter    
class GaussianSmoothing(Operator):
    def __init__(self, domain, smoothing_scale):
        self.smoothing_scale=smoothing_scale
        super().__init__(domain, domain)
        
    def _eval(self, x, differentiate):
        return gaussian_filter(x, sigma=self.smoothing_scale)
    
    def _derivative(self, x):
        return gaussian_filter(x, sigma=self.smoothing_scale)
    
    def _adjoint(self, x):
        return gaussian_filter(x, sigma=self.smoothing_scale)