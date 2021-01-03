import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

camb_path = os.path.realpath('/home/lorenzennio/CAMB')
sys.path.insert(0,camb_path)

import camb
from camb import model, initialpower, dark_energy
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

from astropy.io import fits
wkdata= fits.open("wkdata.fits")
wkdata = np.array([wkdata[i].data for i in range(1)])[0]
wkdata = np.require(wkdata, dtype='float64', requirements='F')
plt.plot(wkdata[:,0], wkdata[:,1])
plt.xscale('log')
plt.ylim([-2,2])
plt.show()

cs2data= fits.open("datacs2.fits")
cs2data = np.array([cs2data[i].data for i in range(1)])[0]
cs2data = np.require(cs2data, dtype='float64', requirements='F')
plt.plot(cs2data[:,0], cs2data[:,1])
plt.xscale('log')
plt.ylim([-2,2])
plt.show()

#PARAMS-------------------------------------------------------------------------------------------
H0 = 67.32117
ombh2=0.0223828
omch2=0.1201075
mnu=0.0006451439
omk=0

cs20 = 0
cs21 = 1

redshifts=[0.]
kmax=1.0 #1.3464234

#0 sound speed model
quint0 = camb.CAMBparams()
quint0.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk);
quint0.DarkEnergy = dark_energy.DarkEnergyFluid();
quint0.DarkEnergy.set_params(cs2=cs20);
quint0.DarkEnergy.set_w_a_table(wkdata[:,0], wkdata[:,1]);
quint0.set_matter_power(redshifts=redshifts, kmax=kmax);

results0 = camb.get_results(quint0)
powers0 =results0.get_cmb_power_spectra(quint0, CMB_unit='muK')
rho0 = results0.get_background_densities(wkdata[:,0], vars=['de', 'tot'])
kh0, z0, pk0 = results0.get_linear_matter_power_spectrum(var1=7,var2=7)
Cl0 = powers0['total'][2:,0]

#1 sound speed model
quint1 = camb.CAMBparams()
quint1.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk);
quint1.DarkEnergy = dark_energy.DarkEnergyFluid();
quint1.DarkEnergy.set_params(cs2=cs21);
quint1.set_matter_power(redshifts=redshifts, kmax=kmax); #1.3464234)

results1 = camb.get_results(quint1)
powers1 =results1.get_cmb_power_spectra(quint1, CMB_unit='muK')
rho1 = results1.get_background_densities(wkdata[:,0], vars=['de', 'tot'])
kh1, z1, pk1 = results1.get_linear_matter_power_spectrum(var1=7,var2=7)
Cl1 = powers1['total'][2:,0]

#k-essence
kess = camb.CAMBparams()
kess.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk)
kess.DarkEnergy = dark_energy.DarkEnergyFluid()
kess.DarkEnergy.set_w_a_table(wkdata[:,0], wkdata[:,1])
kess.DarkEnergy.set_cs2_a_table(cs2data[:,0], cs2data[:,1])
kess.set_matter_power(redshifts=redshifts, kmax=kmax)

resultsk = camb.get_results(kess)
powersk =resultsk.get_cmb_power_spectra(kess, CMB_unit='muK')
rhok = resultsk.get_background_densities(wkdata[:,0], vars=['de', 'tot'])
khk, zk, pkk = resultsk.get_linear_matter_power_spectrum(var1=7,var2=7)
Clk = powersk['total'][2:,0]

#PLOTS ------------------------------------------------------------------------------------
ls = np.arange(powers0['total'].shape[0], dtype='float64')
ls = ls[2:]

#CMB power spectrum
fig, ax = plt.subplots(2,1, figsize = (6,8))
ax[0].plot(ls,Clk, 'k', label='k-ess.')
ax[0].fill_between(ls,Clk*(1+np.sqrt(2/(2*ls +1))), Clk*(1-np.sqrt(2/(2*ls +1))), alpha=0.2, color='k')
ax[0].plot(ls,Cl1, 'c--', label= '$\mathregular{c_S^2}=%.1f$' %cs21)
ax[0].fill_between(ls,Cl1*(1+np.sqrt(2/(2*ls +1))), Cl1*(1-np.sqrt(2/(2*ls +1))), alpha=0.2, color='c')
ax[0].plot(ls,Cl0, 'm-.', label= '$\mathregular{c_S^2}=%.1f$' %cs20)
ax[0].fill_between(ls,Cl0*(1+np.sqrt(2/(2*ls +1))), Cl0*(1-np.sqrt(2/(2*ls +1))), alpha=0.2, color='m')
ax[0].set_xlabel('l')
ax[0].set_ylabel('$\mathregular{l(l+1)C_l^{TT}/2\pi} \ [\mu K^2]$')
ax[0].legend()

ax[1].plot(ls,(Clk-Clk)/Clk, 'k-', label='k-ess.')
ax[1].fill_between(ls,(Clk-Clk)/Clk + np.sqrt(2)*2/(2*ls +1), (Clk-Clk)/Clk - np.sqrt(2)*2/(2*ls +1), alpha=0.2, color='k')
ax[1].plot(ls,(Cl0-Clk)/Clk, 'm-.',label= '$\mathregular{c_S^2}=%.1f$' %cs20)
ax[1].fill_between(ls,(Cl0-Clk)/Clk*(1 + np.sqrt(2+ 1)*np.sqrt(2/(2*ls +1)) ), (Cl0-Clk)/Clk*(1- np.sqrt(2 + 1)* np.sqrt(2/(2*ls +1))), alpha=0.2, color='m')
ax[1].plot(ls,(Cl1-Clk)/Clk, 'c--', label= '$\mathregular{c_S^2}=%.1f$' %cs21)
ax[1].fill_between(ls,(Cl1-Clk)/Clk*(1 + np.sqrt(2+ 1)*np.sqrt(2/(2*ls +1)) ), (Cl1-Clk)/Clk*(1- np.sqrt(2 + 1)* np.sqrt(2/(2*ls +1))), alpha=0.2, color='c')
ax[1].set_xlabel('l')
ax[1].set_ylabel('$\mathregular{\Delta C_l^{TT}/ C_l^{TT}}$')
ax[1].legend()
plt.show()

#matter power spectrum
fig, ax = plt.subplots(2,1, figsize = (6,8))
ax[0].loglog(kh0, pk0[0,:], 'k', label= 'k-ess')
ax[0].loglog(kh0, pk0[0,:], 'm-.', label= '$\mathregular{c_S^2}=%.1f$' %cs20)
ax[0].loglog(kh1, pk1[0,:], 'c--', label= '$\mathregular{c_S^2}=%.1f$' %cs21)
ax[0].set_xlabel('$k \ [ h Mpc^{-1}]$')
ax[0].set_ylabel('$P(k) \ [(h^{-1} Mpc)^3]$')
ax[0].legend()

ax[1].plot(khk, (pkk[0,:]-pkk[0,:])/pkk[0,:], 'k', label= 'k-ess.')
ax[1].plot(khk, (pk0[0,:]-pkk[0,:])/pkk[0,:], 'm-.', label= '$\mathregular{c_S^2}=%.1f$' %cs20)
ax[1].plot(khk, (pk1[0,:]-pkk[0,:])/pkk[0,:], 'c--', label= '$\mathregular{c_S^2}=%.1f$' %cs21)
ax[1].set_xlabel('$k \ [ h Mpc^{-1}]$')
ax[1].set_ylabel('$\Delta P(k)/P(k)$')
ax[1].set_ylim(bottom=-0.1)
ax[1].legend()

plt.show()
