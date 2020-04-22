import numpy as np

'''
Some operator that is needed for the computation of the Richardson-Lucy algorithmm in the context of Ly-alpha forest tomography
Implements the application of the adjoint of intergral operator for computing the effective optical depth
'''

class RLHydrogen:
    def __init__(self, op):
        self.op=op
        return
    
    def _eval(self, density, argument):
        if self.op.compute_vel:
            dens=density[0:self.op.N_space]
            vel_pec=density[self.op.N_space:2*self.op.N_space]
        else:
            dens=density
            vel_pec=self.op.vel_pec
        density_baryon=self.op._find_baryonic_density(dens)
        toret=self.op._apply_kernel_adjoint(density_baryon, vel_pec, argument)
        return toret
    
class RLBar:
    def __init__(self, op):
        self.op=op
        return
    
    def _eval(self, delta, argument):
        if self.op.compute_vel:
            dens=delta[0:self.op.N_space]
            vel_pec=delta[self.op.N_space:2*self.op.N_space]
        else:
            dens=delta
            vel_pec=self.op.vel_pec
        density_hydrogen=self.op._find_neutral_hydrogen_fraction(argument)
        toret=self.op._apply_kernel_adjoint(dens, vel_pec, density_hydrogen)
        return toret


