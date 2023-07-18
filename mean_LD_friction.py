import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy import special as sp
import pandas as pd
import numba as nb

gama = 1
kT = 1
beta = 1/kT
beta_imp = 1/(kT * 2)
hw = 0.3
miu = 0.
g = 0.75
Ed = g**2/hw - 2
###############
#photon
###############
A_photon = 0.2
w_photon = 0.2
ratio = A_photon/w_photon
nbessel = 10


n_range = np.arange(-nbessel,nbessel+1)
m_range = np.arange(-nbessel,nbessel+1)
bessel_n = sp.jv(n_range, ratio)
bessel_m = sp.jv(m_range, ratio)
nm = n_range.reshape(-1,1) - m_range.reshape(1,-1)

def coeff(time):
    tmp0 = 1j**nm
    tmp1 = np.exp(1j*nm*w_photon*time)
    ## J_n * J_m * i^(n-m) * e^i((n-m)Wt)
    A = np.einsum("n,m,nm,nm->nm", bessel_n, bessel_m, tmp0, tmp1)
    A_real = np.real(A)
    return A_real

def fermi_force_fric_rand(x,miu,A_real):
    Ex = np.sqrt(2)*g*x+Ed
    Ex_dot = np.sqrt(2)*g
    
    fermi = 1/(1+np.exp((Ex - m_range*w_photon - miu)/kT))
    
    fermi_tilde = np.einsum("nm,m->", A_real, fermi)
    fermi_bar = np.einsum("nn,n->", A_real, fermi)
    fermi_bar_dot = np.einsum("nn,n,n->", A_real, fermi,(1-fermi))
    
    dum = (1/gama)*(hw/kT)*Ex_dot**2
    force = - hw*x - Ex_dot*fermi_bar
    friction = dum*fermi_bar_dot
    random = dum*fermi_bar*(1-fermi_bar)
    
    return force,friction,random,fermi_bar


def momentum_dot(x,p,rand_force,A_real):
    force,friction,random,fermi_bar = fermi_force_fric_rand(x,miu,A_real)
    p_dot = force - friction*p + rand_force
    return p_dot

def position_dot(p):
    x_dot = hw * p
    return x_dot

def rk4(x,p,tstep,A_real):
    random = fermi_force_fric_rand(x,miu,A_real)[2]
    RF_sigma = np.sqrt(2*random*kT/(hw*tstep))
    rand_force = np.random.normal(0.0,RF_sigma)

    k1_x = position_dot(p) * tstep
    k1_p = momentum_dot(x,p,rand_force,A_real) * tstep
    k2_x = position_dot(p+0.5*k1_p) *tstep
    k2_p = momentum_dot(x+0.5*k1_x, p+0.5*k1_p,rand_force,A_real) * tstep
    k3_x = position_dot(p+0.5*k2_p) * tstep
    k3_p = momentum_dot(x+0.5*k2_x, p+0.5*k2_p,rand_force,A_real) * tstep
    k4_x = position_dot(p+k3_p) * tstep
    k4_p = momentum_dot(x+k3_x, p+k3_p,rand_force,A_real) * tstep
    
    x += 1/6*(k1_x+2*(k2_x+k3_x)+k4_x)
    p += 1/6*(k1_p+2*(k2_p+k3_p)+k4_p)
    return x,p

@nb.jit
def main():
    mu,sigma = 0.0, np.sqrt(1/(beta_imp*hw))
    N_points = 100
    time_list, pop_list, momen_corr_list,Ekn_list = [],[],[],[]
    tstep = 0.025
    x = np.random.normal(mu, sigma, N_points)
    p = np.random.normal(mu, sigma, N_points)
    
    for time in np.arange(0,500,tstep):
        time_list.append(time*gama)
        fermi_pop = 0.0
        Ekn = 0.0
        for n in range(N_points):
            A_real = coeff(time)
            x[n],p[n] = rk4(x[n],p[n],tstep,A_real)
            fermi_pop += fermi_force_fric_rand(x[n],miu,A_real)[3]
            Ekn += 0.5 * hw * p[n]**2
            
        pop_list.append(fermi_pop/N_points)
        Ekn_list.append(Ekn/N_points/kT)

    time_data = pd.DataFrame(time_list,columns = ['time'])   
    time_data.to_csv('time.csv',index=False,header=False)
    population_data = pd.DataFrame(pop_list,columns = ['population'])
    population_data.to_csv('population.csv',index=False,header=False)
    Ekn_data = pd.DataFrame(Ekn_list,columns = ['kinetics'])
    Ekn_data.to_csv('kinetics.csv',index=False,header=False)

main()


