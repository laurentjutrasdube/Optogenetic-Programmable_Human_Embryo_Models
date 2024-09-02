import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from matplotlib.animation import (FuncAnimation, writers)



# 1. Partial differential equations (PDEs)
#    modelling the gene regulatory network

def compute_edge_parameters(x, param):

    x_total = param["x_total"]
    x_light = param["x_light"]
    K1 = param["K_pSMAD1"]

    if param["micro"]: 

        # Impact of tension on WNT3 production
        r = x-x_total+205.
        r[r<0.] = 0.
        T = param["T_WNT3"]*((r/160.)**35./(1.+(r/160.)**35.)+1.)

        # Sensitivity to BMP4
        K = K1/(1.+(r/160.)**35.)+0.1
        if param["fixed_BMP4"] == 0.:
            k2 = K1*(x**25./((x_light*1.5)**25.+x**25.))+0.1
            K = np.min(np.array([K, k2]), axis=0)
            
    else:
        T = param["T_WNT3"]
        K = K1*(np.abs(x)**25./((x_light*1.5)**25.+np.abs(x)**25.))+0.1

    return K, T



def PDEs(t, var, param):
    
    # Distance from the center
    x_total, x_arr_len, x_light = param["x_total"], param["x_arr_len"], param["x_light"]
    x = np.linspace(-x_total, x_total, x_arr_len)
    if param["micro"]:   x = np.linspace(0., x_total, x_arr_len)
    
    # Variables
    var[var < 0] = 0.
    BMP4 = var[:x_arr_len]
    NOG = var[x_arr_len:2*x_arr_len]
    WNT3 = var[2*x_arr_len:3*x_arr_len]
    NODAL = var[3*x_arr_len:]
    free_BMP4 = BMP4/(1.+param["a"]*NOG)

    # Parameters
    n1 = param["n_pSMAD1"]
    n2, K2 = param["n_nYAP"], param["K_nYAP"]
    n3, K3 = param["n_bCAT"], param["K_bCAT"]

    # Edge-dependent parameters
    K1, T = compute_edge_parameters(x, param)
    
    # pSMAD1, nYAP & nß-CATENIN
    pSMAD1 = free_BMP4**n1/(free_BMP4**n1+K1**n1)
    nYAP = pSMAD1**n2/(pSMAD1**n2+K2**n2)
    if param["fixed_nYAP"] != 0.:    nYAP = param["fixed_nYAP"]
    bCAT = WNT3**n3/(WNT3**n3+K3**n3)
    
    # Production terms
    prod_BMP4 = param["p_BMP4"]*np.heaviside(x_light-np.abs(x), 1)
    prod_NOG = param["p_NOG"]*pSMAD1
    prod_WNT3 = (param["p1_WNT3"]*pSMAD1 +param["p2_WNT3"]*bCAT) *(1.-nYAP) *T
    prod_NODAL = param["p_NODAL"]*bCAT *(1.-nYAP)
    
    # Diffusion terms
    diff_BMP4, diff_NOG, diff_WNT3, diff_NODAL = np.zeros(x_arr_len), np.zeros(x_arr_len), np.zeros(x_arr_len), np.zeros(x_arr_len)
    diff_BMP4[1:-1] = BMP4[:-2]-2.*BMP4[1:-1]+BMP4[2:]
    diff_NOG[1:-1] = NOG[:-2]-2.*NOG[1:-1]+NOG[2:]
    diff_WNT3[1:-1] = WNT3[:-2]-2.*WNT3[1:-1]+WNT3[2:]
    diff_NODAL[1:-1] = NODAL[:-2]-2.*NODAL[1:-1]+NODAL[2:]
    if param["micro"]:
        diff_BMP4[1:-1] -= 0.5*(BMP4[:-2]-BMP4[2:])/x[1:-1]
        diff_NOG[1:-1] -= 0.5*(NOG[:-2]-NOG[2:])/x[1:-1]
        diff_WNT3[1:-1] -= 0.5*(WNT3[:-2]-WNT3[2:])/x[1:-1]
        diff_NODAL[1:-1] -= 0.5*(NODAL[:-2]-NODAL[2:])/x[1:-1]
    
    # Boundary conditions (regular culture)
    diff_BMP4[-1], diff_BMP4[0] = 2.*(BMP4[-2]-BMP4[-1]), 2.*(BMP4[1]-BMP4[0])
    diff_NOG[-1], diff_NOG[0] = 2.*(NOG[-2]-NOG[-1]), 2.*(NOG[1]-NOG[0])
    diff_WNT3[-1], diff_WNT3[0] = 2.*(WNT3[-2]-WNT3[-1]), 2.*(WNT3[1]-WNT3[0])
    diff_NODAL[-1], diff_NODAL[0] = 2.*(NODAL[-2]-NODAL[-1]), 2.*(NODAL[1]-NODAL[0])
    
    # PDEs
    dBMP4 = prod_BMP4 +param["D_BMP4"]*diff_BMP4 -param["l_BMP4"]*BMP4
    dNOG = prod_NOG +param["D_NOG"]*diff_NOG -param["l_NOG"]*NOG
    dWNT3 = prod_WNT3 +param["D_WNT3"]*diff_WNT3 -param["l_WNT3"]*WNT3
    dNODAL = prod_NODAL +param["D_NODAL"]*diff_NODAL -param["l_NODAL"]*NODAL
    
    # Boundary conditions (micropatterned substrate)
    if param["micro"]:    dBMP4[-1], dNOG[-1], dWNT3[-1], dNODAL[-1] = 0., 0., 0., 0.

    # If the concentration of BMP4 is fixed
    if param["fixed_BMP4"] != 0.:    dBMP4 = np.zeros(x_arr_len)
    
    return np.concatenate((dBMP4, dNOG, dWNT3, dNODAL))




# 2. Integration routine

def get_results(param):

    # Position along the micropattern
    x = np.linspace(-param["x_total"], param["x_total"], param["x_arr_len"])
    if param["micro"]:    x = np.linspace(0., param["x_total"], param["x_arr_len"])
    
    # BMP4, NOGGIN, WNT3 and NODAL concentrations
    results = integrate(param)
    BMP4 = results[:,:param["x_arr_len"]]
    NOG = results[:,param["x_arr_len"]:2*param["x_arr_len"]]
    WNT3 = results[:,2*param["x_arr_len"]:3*param["x_arr_len"]]
    NODAL = results[:,3*param["x_arr_len"]:]
    
    # Sensitivity to BMP4 (K parameter)
    K_pSMAD1, T = compute_edge_parameters(x, param)
    
    # Free BMP4
    free_BMP4 = BMP4/(1.+param["a"]*NOG)
    
    # pSMAD1
    n_pSMAD1 = param["n_pSMAD1"]
    pSMAD1 = free_BMP4**n_pSMAD1/(free_BMP4**n_pSMAD1+K_pSMAD1**n_pSMAD1)
    
    # nYAP
    n_nYAP, K_nYAP = param["n_nYAP"], param["K_nYAP"]
    nYAP = pSMAD1**n_nYAP/(pSMAD1**n_nYAP+K_nYAP**n_nYAP)
    
    # nß-CATENIN
    n_bCAT, K_bCAT = param["n_bCAT"], param["K_bCAT"]
    bCAT = WNT3**n_bCAT/(WNT3**n_bCAT+K_bCAT**n_bCAT)
    
    # pSMAD2
    n_pSMAD2, K_pSMAD2 = param["n_pSMAD2"], param["K_pSMAD2"]
    pSMAD2 = NODAL**n_pSMAD2/(NODAL**n_pSMAD2+K_pSMAD2**n_pSMAD2)
    
    # ISL1
    n_ISL1, K_ISL1 = param["n_ISL1"], param["K_ISL1"]
    ISL1 = pSMAD1**n_ISL1/(pSMAD1**n_ISL1+K_ISL1**n_ISL1)
    
    # BRA
    n_BRA, K_BRA = param["n_BRA"], param["K_BRA"]
    n2_BRA, K2_BRA = param["n2_BRA"], param["K2_BRA"]
    n3_BRA, K3_BRA = param["n3_BRA"], param["K3_BRA"]
    BRA = pSMAD2**n_BRA/(pSMAD2**n_BRA+K_BRA**n_BRA)
    BRA *= K2_BRA**n2_BRA/(ISL1**n2_BRA+K2_BRA**n2_BRA)
    BRA *= bCAT**n3_BRA/(bCAT**n3_BRA+K3_BRA**n3_BRA)
    
    # SOX2
    n_SOX2, K_SOX2 = param["n_SOX2"], param["K_SOX2"]
    n2_SOX2, K2_SOX2 = param["n2_SOX2"], param["K2_SOX2"]
    SOX2 = K_SOX2**n_SOX2/(pSMAD1**n_SOX2+K_SOX2**n_SOX2)
    SOX2 *= K2_SOX2**n2_SOX2/(BRA**n2_SOX2+K2_SOX2**n2_SOX2)
    
    # Grouping results
    morphogens = np.array([BMP4, free_BMP4, NOG, WNT3, NODAL])
    TFs = np.array([pSMAD1, nYAP, bCAT, pSMAD2])
    fates = np.array([SOX2, ISL1, BRA])

    return x, morphogens, TFs, fates



def integrate(param):
    
    # Set the integration options and parameters
    integrator = ode(PDEs)
    integrator.set_integrator('lsoda', with_jacobian=False, rtol=1e-4, nsteps=1000)
    integrator.set_f_params(param)

    # Set the time points
    times = param["times"]
    
    # Set the initial conditions
    x_arr_len = param["x_arr_len"]
    init_cond = np.zeros(4*x_arr_len)
    if param["fixed_BMP4"] != 0.:    init_cond[:x_arr_len] = param["fixed_BMP4"]
    integrator.set_initial_value(init_cond, times[0])
    results = [init_cond]
    
    # Perform the integration
    for t in times[1:]:
        integrator.integrate(t)
        if (integrator.successful):
            results.append(integrator.y)
        else:
            print("An error occurred during the integration.")
            break
           
    # Output the results
    if len(times) > 1:    return np.array(results)
    else:    return results[0]




# 3. Generating plots

def plot_morphogens(x, morphogens, file_name, x_light=0., micro=False, fixed_BMP4=False):

    fig, ax = plt.subplots(figsize=[6.35, 3.6])
    
    # BMP4, Noggin, Wnt & Nodal
    ax.plot(x, morphogens[0], c='grey', lw=5, label='BMP4$_{total}$', zorder=10)
    ax.plot(x, morphogens[1], c='lightgrey', lw=5, label='BMP4$_{free}$', zorder=11)
    ax.plot(x, morphogens[2], c='dodgerblue', lw=5, label='NOGGIN', zorder=12)
    ax.plot(x, morphogens[3], c='orange', lw=5, label='WNT3', zorder=13)
    ax.plot(x, morphogens[4], c='orchid', lw=5, label='NODAL', zorder=14)
    ax.set_xlabel('Position (µm)', fontsize=18)
    ax.set_ylabel('Concentration', fontsize=18)
    ax.set_xlim([-1000.,1000.])
    ax.set_ylim([0., 65.])
    ax.tick_params(labelsize=15)
    ax.grid()
    ax.legend(loc=2, fontsize=16, bbox_to_anchor=(1., 1.))

    # Micropatterns
    if micro:
        ax.set_xlabel('Distance from the center (µm)', fontsize=18)
        ax.set_xlim([x[0],x[-1]])
        ax.set_ylim([0., 55.])

    # Light-activated region
    if not fixed_BMP4:
        ax.fill_between([-x_light,x_light], [-1.,-1.], [80.,80.], color='gold', alpha=0.25)
    
    fig.tight_layout()
    fig.savefig('Figures/'+file_name+'_morphogens.pdf', dpi=300)



def plot_TFs(x, TFs, file_name, x_light=0., micro=False, fixed_BMP4=False):

    fig, ax = plt.subplots(figsize=[6.35, 3.6])
    
    # pSMAD1, nYAP, nß-Catenin & pSMAD2
    ax.plot(x, TFs[0], c='orangered', lw=5, label='pSMAD1', zorder=11)
    ax.plot(x, TFs[1], c='gold', lw=5, label='nYAP', zorder=12)
    ax.plot(x, TFs[2], c='mediumpurple', lw=5, label='nß-CAT', zorder=13)
    ax.plot(x, TFs[3], c='yellowgreen', lw=5, label='pSMAD2', zorder=14)
    ax.set_xlabel('Position (µm)', fontsize=18)
    ax.set_ylabel('Proportion of cells', fontsize=18)
    ax.set_xlim([-1000.,1000.])
    ax.set_ylim([-0.02,1.02])
    ax.tick_params(labelsize=15)
    ax.grid()
    ax.legend(loc=2, fontsize=16, bbox_to_anchor=(1., 1.))

    # Micropatterns
    if micro:
        ax.set_xlabel('Distance from the center (µm)', fontsize=18)
        ax.set_xlim([x[0],x[-1]])

    # Light-activated region
    if not fixed_BMP4:
        ax.fill_between([-x_light,x_light], [-1.,-1.], [2.,2.], color='gold', alpha=0.25)
    
    fig.tight_layout()
    fig.savefig('Figures/'+file_name+'_TFs.pdf', dpi=300)



def plot_fates(x, fates, file_name, x_light=0., micro=False, fixed_BMP4=False):

    fig, ax = plt.subplots(figsize=[6., 3.6])
    
    # SOX2, ISL1 & BRA
    ax.plot(x, fates[0], c='steelblue', lw=5, label='SOX2', zorder=10)
    ax.plot(x, fates[1], c='seagreen', lw=5, label='ISL1', zorder=11)
    ax.plot(x, fates[2], c='firebrick', lw=5, label='BRA', zorder=11)
    ax.set_xlabel('Position (µm)', fontsize=18)
    ax.set_ylabel('Proportion of cells', fontsize=18)
    ax.set_xlim([-1000.,1000.])
    ax.set_ylim([-0.02,1.02])
    ax.tick_params(labelsize=15)
    ax.grid()
    ax.legend(loc=2, fontsize=16, bbox_to_anchor=(1.,1.))

    # Micropatterns
    if micro:
        ax.set_xlabel('Distance from the center (µm)', fontsize=18)
        ax.set_xlim([x[0],x[-1]])

    # Light-activated region
    if not fixed_BMP4:
        ax.fill_between([-x_light,x_light], [-1.,-1.], [2.,2.], color='gold', alpha=0.25)
    
    fig.tight_layout()
    fig.savefig('Figures/'+file_name+'_fates.pdf', dpi=300)