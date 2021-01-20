#!/usr/bin/env python
# coding: utf-8

# In[1]:


# sympy
import sympy as sp
from IPython.display import display, Math
# sp.init_printing()

# numpy
import numpy as np

# scipy
from scipy import optimize

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({'legend.frameon': True,
                     'legend.framealpha': 1,
                     'legend.facecolor': 'white',
                     'axes.titlesize': 20,
                     'axes.labelsize': 16,
                     'legend.fontsize': 14,
                     'axes.edgecolor': 'black'})


# # Symbolic Derivation
# 
# ## Wind-Aware Control Law

# In[2]:


k = sp.symbols('k', real=True, positive=True)
T = sp.symbols('T', real=True, positive=True)
vGP = sp.symbols('v_{G\,P}', real=True, positive=True)
xP = sp.symbols('x_P', real=True)
vA = sp.symbols('v_A', real=True, positive=True)
vW = sp.symbols('v_W', real=True, positive=True)
kapP = sp.symbols('\\kappa_P', real=True, positive=True)
lP = sp.symbols('\\lambda_P', real=True)

# ground to air projection speed ratio
a = sp.symbols('a', real=True)
a_expr = vGP / vA / sp.cos(xP)

# wind-modulated turn rate
b = sp.symbols('b', real=True)
b_expr = vGP * kapP * sp.tan(xP)

# linearized on-track dynamics
d_etadot_d_eta = sp.symbols('\\cfrac{\\partial\\dot{\\eta}}{\\partial\\eta}')
d_etadot_d_beta = sp.symbols('\\cfrac{\\partial\\dot{\\eta}}{\\partial\\beta}')
d_betadot_d_eta = sp.symbols('\\cfrac{\\partial\\dot{\\beta}}{\\partial\\eta}')
d_betadot_d_beta = sp.symbols('\\cfrac{\\partial\\dot{\\beta}}{\\partial\\beta}')
d_etadot_d_eta_expr = -k + sp.pi/T
d_etadot_d_beta_expr = a * ( b - sp.pi / T )
d_betadot_d_eta_expr = sp.pi / T / a
d_betadot_d_beta_expr = b - sp.pi / T

# print('Jacobian:')
# display(Math('%s = %s' %(sp.latex(d_etadot_d_eta), sp.latex(d_etadot_d_eta_expr.subs({b:b_expr,a:a_expr})))))
# display(Math('%s = %s' %(sp.latex(d_etadot_d_beta), sp.latex(d_etadot_d_beta_expr.subs({b:b_expr,a:a_expr})))))
# display(Math('%s = %s' %(sp.latex(d_betadot_d_eta), sp.latex(d_betadot_d_eta_expr.subs({b:b_expr,a:a_expr})))))
# display(Math('%s = %s' %(sp.latex(d_betadot_d_beta), sp.latex(d_betadot_d_beta_expr.subs({b:b_expr,a:a_expr})))))
#
# print('Collect terms:')
# display(Math('%s = %s' %(sp.latex(a), sp.latex(a_expr))))
# display(Math('%s = %s' %(sp.latex(b), sp.latex(b_expr))))

# print('Condensed Jacobian:')
AMatrix = sp.Matrix([[d_etadot_d_eta, d_etadot_d_beta], [d_betadot_d_eta, d_betadot_d_beta]])
AMatrix_expr = sp.Matrix([[d_etadot_d_eta_expr, d_etadot_d_beta_expr], [d_betadot_d_eta_expr, d_betadot_d_beta_expr]])

# display(Math('%s = %s' %(sp.latex(AMatrix), sp.latex(AMatrix_expr))))


# In[3]:


# characteristic equation
s = sp.symbols('s')
s_m_A = sp.Matrix([[s,0],[0,s]]) - AMatrix_expr
char_eq = s_m_A.det()

# print('Characteristic equation:')
# display(expand(char_eq))

# collect coeffs
char_eq_poly = sp.poly(char_eq, s)
char_eq_coeffs = char_eq_poly.coeffs()

# display(Math('%s + (%s)s + (%s)' %(sp.latex(s**2),
#                                    sp.latex(sp.expand(char_eq_coeffs[1])),
#                                    sp.latex(sp.expand(char_eq_coeffs[2])))))

# natural frequency
# print('Natural frequency:')
omn = sp.symbols('\\omega_n')
omn_expr = sp.sqrt(sp.expand(char_eq_coeffs[2]))
# display(Math('%s = %s' %(sp.latex(omn), sp.latex(omn_expr))))

# damping ratio
# print('Damping ratio:')
zeta = sp.symbols('\\zeta')
zeta_expr = sp.simplify(sp.expand(char_eq_coeffs[1] / 2 / omn_expr))
# display(Math('%s = %s' %(sp.latex(zeta), sp.latex(zeta_expr))))


# In[4]:


# nominal gains (zero wind case)

om0 = sp.symbols('\\omega_0', positive = True, real = True)
zeta0 = sp.symbols('\\zeta_0', positive = True, real = True)

k0 = sp.symbols('k_0', positive = True, real = True)
k0_expr = 2 * zeta0 * om0
# display(Math('%s = %s' %(sp.latex(k0), sp.latex(k0_expr))))

T0 = sp.symbols('T_0', positive = True, real = True)
T0_expr = 2 * sp.pi / om0 * zeta0
# display(Math('%s = %s' %(sp.latex(T0), sp.latex(T0_expr))))

# plug in and confirm it works when wind freq = 0
# display(Math('%s(%s=%s,%s=%s,%s=0) = %s = %s' %(sp.latex(omn),sp.latex(k),sp.latex(k0),                                                 sp.latex(T),sp.latex(T0),sp.latex(b),                                                 sp.latex(omn_expr.subs({b:0, k:k0, T:T0})),                                                 sp.latex(omn_expr.subs({b:0, k:k0_expr, T:T0_expr})))))
# display(Math('%s(%s=%s,%s=%s,%s=0) = %s = %s' %(sp.latex(zeta),sp.latex(k),sp.latex(k0),                                                 sp.latex(T),sp.latex(T0),sp.latex(b),                                                 sp.latex(zeta_expr.subs({b:0, k:k0, T:T0})),                                                 sp.latex(zeta_expr.subs({b:0, k:k0_expr, T:T0_expr})))))


# ## Period and Damping in Terms of Nominal Tuning

# In[6]:


# subtitute nominal tuning

# Period
P0 = sp.symbols('P_0')

P = sp.symbols('P')
P_expr = 2 * sp.pi / omn_expr

# print('Period')

# display(Math('%s = %s' %(sp.latex(P), sp.latex(P_expr))))

P_expr_sub = P_expr.subs({k:k0_expr.subs(om0, 2 * sp.pi / P0), T:T0_expr.subs(om0, 2 * sp.pi / P0)})

# display(Math('%s = %s' %(sp.latex(P), sp.latex(P_expr_sub))))

# Damping ratio

# print('Damping Ratio:')

# display(Math('%s = %s' %(sp.latex(zeta), sp.latex(zeta_expr))))

zeta_expr_sub = zeta_expr.subs({k:k0_expr.subs(om0, 2 * sp.pi / P0), T:T0_expr.subs(om0, 2 * sp.pi / P0)})

# display(Math('%s = %s' %(sp.latex(zeta), sp.latex(zeta_expr_sub))))


# ## Wind Factor

# In[7]:


alpW = sp.symbols('\\alpha_W')

fW = sp.symbols('f_W')
fW_expr = alpW**2 * sp.sin(lP) * sp.cos(lP) / sp.sqrt(1 - alpW**2 * sp.sin(lP)**2) + alpW * sp.sin(lP)
# display(Math('%s = %s' %(sp.latex(fW), sp.latex(fW_expr))))

b_expr_fW = vA * kapP * fW
# display(Math('%s = %s = %s' %(sp.latex(b), sp.latex(b_expr_fW), sp.latex(b_expr_fW.subs(fW,fW_expr)))))


# ## Ground Speed Based Control Law
# For *straight* paths, $\kappa_P=0$

# In[8]:


# ground speed based control law

omP = sp.symbols('\\omega_P')

d_etadot_d_eta_gsp_expr = -k/a + sp.pi/T + kapP * vW * sp.sin(lP) * (a - 1)
d_etadot_d_beta_gsp_expr = a * ( b * (vGP - 2) - sp.pi / T ) + b + omP**2 * T / sp.pi * (1 - a)
d_betadot_d_eta_gsp_expr = sp.pi / T / a
d_betadot_d_beta_gsp_expr = b - sp.pi / T

# print('Jacobian:')
# display(Math('%s = %s' %(sp.latex(d_etadot_d_eta), sp.latex(d_etadot_d_eta_gsp_expr.subs({b:b_expr,a:a_expr})))))
# display(Math('%s = %s' %(sp.latex(d_etadot_d_beta), sp.latex(d_etadot_d_beta_gsp_expr.subs({b:b_expr,a:a_expr})))))
# display(Math('%s = %s' %(sp.latex(d_betadot_d_eta), sp.latex(d_betadot_d_eta_gsp_expr.subs({b:b_expr,a:a_expr})))))
# display(Math('%s = %s' %(sp.latex(d_betadot_d_beta), sp.latex(d_betadot_d_beta_gsp_expr.subs({b:b_expr,a:a_expr})))))

# print('Condensed Jacobian:')
AMatrix_gsp_expr = sp.Matrix([[d_etadot_d_eta_gsp_expr, d_etadot_d_beta_gsp_expr], [d_betadot_d_eta_gsp_expr, d_betadot_d_beta_gsp_expr]])

# display(Math('%s = %s' %(sp.latex(AMatrix), sp.latex(AMatrix_gsp_expr))))

# characteristic equation
s_m_A_gsp = sp.Matrix([[s,0],[0,s]]) - AMatrix_gsp_expr
char_eq_gsp = s_m_A_gsp.det()

# print('Characteristic equation:')
# display(expand(char_eq_gsp))

# collect coeffs -- straight paths
char_eq_gsp_poly = sp.poly(char_eq_gsp, s)
char_eq_gsp_coeffs = char_eq_gsp_poly.coeffs()

# display(Math('%s + (%s)s + (%s)' %(sp.latex(s**2),
#                                    sp.latex(sp.expand(char_eq_gsp_coeffs[1].subs({kapP:0, b:0, omP:0}))),
#                                    sp.latex(sp.expand(char_eq_gsp_coeffs[2].subs({kapP:0, b:0, omP:0}))))))

# natural frequency
# print('Natural frequency:')
omn = sp.symbols('\\omega_n')
omn_gsp_expr = sp.sqrt(sp.expand(char_eq_gsp_coeffs[2].subs({kapP:0, b:0, omP:0})))
# display(Math('%s = %s' %(sp.latex(omn), sp.latex(omn_gsp_expr))))

# damping ratio
# print('Damping ratio:')
zeta = sp.symbols('\\zeta')
zeta_gsp_expr = sp.simplify(sp.expand(char_eq_gsp_coeffs[1].subs({kapP:0, b:0, omP:0}) / 2 / omn_gsp_expr))
# display(Math('%s = %s' %(sp.latex(zeta), sp.latex(zeta_gsp_expr))))

# nominal tuning
# print('At nominal tuning:')

# period
# print('Period:')
P_gsp_expr_sub = 2 * sp.pi / omn_gsp_expr.subs({k:k0_expr.subs(om0, 2 * sp.pi / P0), T:T0_expr.subs(om0, 2 * sp.pi / P0)})
# modify... 
P_gsp_expr_sub = P0 * sp.sqrt(a)
# display(Math('%s = %s' %(sp.latex(P), sp.latex(P_gsp_expr_sub))))

# damping ratio
# print('Damping ratio:')
zeta_gsp_expr_sub = sp.simplify(zeta_gsp_expr.subs({k:k0_expr.subs(om0, 2 * sp.pi / P0), T:T0_expr.subs(om0, 2 * sp.pi / P0)}))
# modify... 
zeta_gsp_expr_sub = zeta0 / sp.sqrt(a)
# display(Math('%s = %s' %(sp.latex(zeta), sp.latex(zeta_gsp_expr_sub))))


# # Numerical Evaluation
# 
# ## Check the Solutions...

# In[9]:


# helpful functions

def evalb(kapP, lP, vA, vW):
    xP = np.arcsin(vW/vA * np.sin(lP))
    psiP = lP + xP
    vGP = np.sqrt(vA**2 + vW**2 - 2*vA*vW*np.cos(np.pi - psiP))
    return kapP * vGP * np.tan(xP)

# Check the developed Jacobian

def evalJac(k, T, kapP, lP, vA, vW):
    
    # n = 0
    # b = 0
    
    xP = np.arcsin(vW/vA * np.sin(lP))
    psiP = lP + xP
    vGP = np.sqrt(vA**2 + vW**2 - 2*vA*vW*np.cos(np.pi - psiP))
    
    a11 = -k + np.pi/T
    a12 = vGP**2 * kapP * np.tan(xP) / vA / np.cos(xP) - vGP / vA / np.cos(xP) * np.pi / T
    a21 = np.pi / T * vA * np.cos(xP) / vGP
    a22 = vGP * kapP * np.tan(xP) - np.pi / T

    return np.array([a11, a12, a21, a22])


def evalDyn(n, b, k, T, kapP, lP, vA, vW):
    
    ll = b + lP
    
    xl = np.arcsin(vW / vA * np.sin(ll))
    
    psi = b - n + lP + xl
    
    vG = np.sqrt(vA**2 + vW**2 - 2*vA*vW*np.cos(np.pi - psi))
    
    x = np.arccos((vA**2 - vW**2 + vG**2) / (2 * vA * vG))
    
    n_P = n - b + x - xl
    
    vGl = vW * np.cos(ll) + vA * np.cos(xl)
    
    e = -b * vG * T / np.pi
    
    omP = kapP * vG * np.cos(n_P) / (1 - kapP * e)
    
    psidot = k * np.sin(n) + vGl / vA / np.cos(xl) * omP
    
    vGdot = -vA*vW*np.sin(psi) / vG * psidot
    
    bdot = -b * vGdot / vG + np.pi / T * np.sin(n_P)
   
    ndot = -psidot + vGl / vA / np.cos(xl) * (bdot + omP)

    return ndot, bdot


def evalNumJac(n0, b0, k, T, kapP, lP, vA, vW):
    
    dn = 0.000001
    db = 0.000001
    twod = dn*2

    dndot_dn_p, dbdot_dn_p = evalDyn(n0 + dn, b0, k, T, kapP, lP, vA, vW)
    dndot_dn_m, dbdot_dn_m = evalDyn(n0 - dn, b0, k, T, kapP, lP, vA, vW)
    dndot_db_p, dbdot_db_p = evalDyn(n0, b0 + db, k, T, kapP, lP, vA, vW)
    dndot_db_m, dbdot_db_m = evalDyn(n0, b0 - db, k, T, kapP, lP, vA, vW)
    
    dndot_dn = (dndot_dn_p - dndot_dn_m) / twod
    dndot_db = (dndot_db_p - dndot_db_m) / twod
    dbdot_dn = (dbdot_dn_p - dbdot_dn_m) / twod
    dbdot_db = (dbdot_db_p - dbdot_db_m) / twod
    
    return np.array([dndot_dn, dndot_db, dbdot_dn, dbdot_db])


n1 = 0.0
b1 = 0.0
k1 = 0.11
T1 = 7.0
kapP1 = 1.0 / 50.0
lP1 = 0.52
vA1 = 15.0
vW1 = 10.0

ndot0, bdot0 = evalDyn(n1, b1, k1, T1, kapP1, lP1, vA1, vW1)
jacAnalytic = evalJac(k1, T1, kapP1, lP1, vA1, vW1)
jacNumerical = evalNumJac(n1, b1, k1, T1, kapP1, lP1, vA1, vW1)

# display('Equilibrium Diff. Eq.: ndot0 = %s, bdot0 = %s' %(ndot0,bdot0,))
# display('Numerical Jacobian: %s' %jacNumerical)
# display('Analytic Jacobian: %s' %jacAnalytic)


# ## Evaluate Dynamics for Wind-Aware Control Law

# In[10]:


len_lP = 301
len_alpW = 11

lP_data = np.linspace(-np.pi,np.pi,len_lP)
alpW_data = np.linspace(0,1,len_alpW)

fW_data = np.zeros([len_alpW, len_lP])
b_data = np.zeros([len_alpW, len_lP])
P_data = np.zeros([len_alpW, len_lP])
zeta_data = np.zeros([len_alpW, len_lP])

vA1 = 10
kapP1 = 1/50
P1 = 10
zeta1 = 0.7071

fW_eval =sp.lambdify((lP, alpW), fW_expr, 'numpy')
b_eval = sp.lambdify((vA, kapP, fW), b_expr_fW, 'numpy')
P_eval = sp.lambdify((b, P0, zeta0), P_expr_sub, 'numpy')
zeta_eval = sp.lambdify((b, P0, zeta0), zeta_expr_sub, 'numpy')

eps_pi_2 = 0.0001
for i in range(len_alpW):
    if alpW_data[i] == 1:
        idx_sel_geq_pi_2 = lP_data >= np.pi/2 - eps_pi_2
        fW_data[i, idx_sel_geq_pi_2] = np.nan
        
        idx_sel_leq_m_pi_2 = lP_data <= -np.pi/2 + eps_pi_2
        fW_data[i, idx_sel_leq_m_pi_2] = np.nan
        
        idx_sel = np.all([~idx_sel_geq_pi_2, ~idx_sel_leq_m_pi_2],axis=0)
        fW_data[i, idx_sel] = fW_eval(lP_data[idx_sel], alpW_data[i])
    else:
        fW_data[i,:] = fW_eval(lP_data, alpW_data[i])
        
    b_data[i,:] = b_eval(vA1, kapP1, fW_data[i,:])
    P_data[i,:] = P_eval(b_data[i,:], P1, zeta1)
    zeta_data[i, :] = zeta_eval(b_data[i,:], P1, zeta1)


# ## Plot Period and Damping w.r.t. Wind Factor

# In[11]:


plot_lw = 2
cmap = plt.cm.get_cmap('viridis', len_alpW)

fig = plt.figure(figsize=(16,4))
spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)

ax1 = fig.add_subplot(spec[:, 0])
ax2 = fig.add_subplot(spec[:, 1])
ax3 = fig.add_subplot(spec[:, 2])

ax1.set_title(r'Period')
ax1.set_xlabel(r'${\lambda_P}$ [deg]')
ax1.set_ylabel(r'$P$ [s]')
for i in range(len_alpW):
    ax1.plot(np.rad2deg(lP_data), P_data[i,:], linewidth=plot_lw, color=cmap(i))
# ax1.annotate(r'$P_0$', xy=(-80, P1), xytext=(-100, P1+2), arrowprops=dict(arrowstyle="->"), fontsize=14)

ax2.set_title(r'Damping Ratio')
ax2.set_xlabel(r'${\lambda_P}$ [deg]')
ax2.set_ylabel(r'$\zeta$')
for i in range(len_alpW):
    ax2.plot(np.rad2deg(lP_data), zeta_data[i,:], linewidth=plot_lw, color=cmap(i))

ax3.set_title(r'Wind Factor')
ax3.set_xlabel(r'${\lambda_P}$ [deg]')
ax3.set_ylabel(r'$f_W$')
for i in range(len_alpW):
    ax3.plot(np.rad2deg(lP_data), fW_data[i,:], linewidth=plot_lw, color=cmap(i))
ax3.set_xlim(np.rad2deg([lP_data[0], lP_data[-1]]))
ax3.set_ylim([-2, 2])

# plot dummy data to have stand alone colorbar
dummy_array = np.array([[0,1]])
dummy_plot = ax3.imshow(dummy_array, cmap='viridis')
dummy_plot.set_visible(False)
fig.colorbar(dummy_plot, ax=ax3, label=r'$\alpha_W$')

ax3.set_aspect('auto')
    
print('Wind-Aware Control Law (for an arbitrary curvature in wind)')
plt.show()


# ## Evaluate Dynamics for Ground Speed Based Control Law

# In[12]:


len_lP = 301
len_alpW = 11

lP_data = np.linspace(-np.pi,np.pi,len_lP)
alpW_data = np.linspace(0,1,len_alpW)

a_data = np.zeros([len_alpW, len_lP])
P_gsp_data = np.zeros([len_alpW, len_lP])
zeta_gsp_data = np.zeros([len_alpW, len_lP])

vA1 = 10
kapP1 = 1/50
P1 = 10
zeta1 = 0.7071

a_expr_expanded = alpW * sp.cos(lP) / sp.sqrt(1 - alpW**2 * sp.sin(lP)**2) + 1
a_eval = sp.lambdify((alpW, lP), a_expr_expanded, 'numpy')
P_gsp_eval = sp.lambdify((P0, a), P_gsp_expr_sub, 'numpy')
zeta_gsp_eval = sp.lambdify((zeta0, a), zeta_gsp_expr_sub, 'numpy')

eps_alpW = (alpW_data[1] - alpW_data[0]) / 2
eps_lP = (lP_data[1] - lP_data[0]) / 2
for i in range(len_alpW):
    if alpW_data[i]  > 1 - eps_alpW:
        idx_sel = np.any([np.all([lP_data > (np.pi/2 - eps_lP),lP_data < (np.pi/2 + eps_lP)],axis=0),
                          np.all([lP_data > (-np.pi/2 - eps_lP),lP_data < (-np.pi/2 + eps_lP)],axis=0)], axis=0)
        a_data[i, idx_sel] = np.nan

        inv_idx_sel = ~idx_sel
        a_data[i, inv_idx_sel] = a_eval(alpW_data[i], lP_data[inv_idx_sel])
    else:
        a_data[i,:] = a_eval(alpW_data[i], lP_data)
    
    idx_neg_a = a_data[i,:] < 0
    a_data[i,idx_neg_a] = 0
    
    inv_idx_neg_a = ~idx_neg_a
    P_gsp_data[i,inv_idx_neg_a] = P_gsp_eval(P1, a_data[i,inv_idx_neg_a])
    P_gsp_data[i,idx_neg_a] = 0
    
    idx_zeta_sel = a_data[i,:] < eps_alpW
    inv_idx_zeta_sel = ~idx_zeta_sel
    zeta_gsp_data[i,inv_idx_zeta_sel] = zeta_gsp_eval(zeta1, a_data[i,inv_idx_zeta_sel])
    zeta_gsp_data[i,idx_zeta_sel] = np.nan


# ## Plot Period and Damping w.r.t. Speed Ratio

# In[13]:


plot_lw = 2
cmap = plt.cm.get_cmap('viridis', len_alpW)

fig = plt.figure(figsize=(16,4))
spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)

ax1 = fig.add_subplot(spec[:, 0])
ax2 = fig.add_subplot(spec[:, 1])
ax3 = fig.add_subplot(spec[:, 2])

ax1.set_title(r'Period')
ax1.set_xlabel(r'${\lambda_P}$ [deg]')
ax1.set_ylabel(r'$P$ [s]')
for i in range(len_alpW):
    ax1.plot(np.rad2deg(lP_data), P_gsp_data[i,:], linewidth=plot_lw, color=cmap(i))

ax2.set_title(r'Damping Ratio')
ax2.set_xlabel(r'${\lambda_P}$ [deg]')
ax2.set_ylabel(r'$\zeta$')
for i in range(len_alpW):
    ax2.plot(np.rad2deg(lP_data), zeta_gsp_data[i,:], linewidth=plot_lw, color=cmap(i))

ax3.set_title(r'Speed Ratio')
ax3.set_xlabel(r'${\lambda_P}$ [deg]')
ax3.set_ylabel(r'$a$')
for i in range(len_alpW):
    ax3.plot(np.rad2deg(lP_data), a_data[i,:], linewidth=plot_lw, color=cmap(i))
ax3.set_xlim(np.rad2deg([lP_data[0], lP_data[-1]]))
ax3.set_ylim([-0.1, 2.1])

# plot dummy data to have stand alone colorbar
dummy_array = np.array([[0,1]])
dummy_plot = ax3.imshow(dummy_array, cmap='viridis')
dummy_plot.set_visible(False)
fig.colorbar(dummy_plot, ax=ax3, label=r'$\alpha_W$')

ax3.set_aspect('auto')

print('Ground Speed Based Control Law for Straight Paths')
plt.show()


# - - - - - - - - - - -
# Critical Wind Factor
# - - - - - - - - - - -


# function to evaluate derivative of f w.r.t. path orientation angle lP
def eval_df(l, a):
    # inputs:
    # l         path orientation angle
    # a         wind ratio
    # outputs:
    # df/dl

    sinl = np.sin(l)
    sin2l = sinl*sinl
    cosl = np.cos(l)
    a2 = a*a
    sin4l = sin2l*sin2l
    one_m_a2sin2l = 1 - a2 * sin2l
    one_m_a2sin2l_3_2 = one_m_a2sin2l**(3/2)

    if a == 1:
        return 1 - 2 * sin2l + sin4l
    else:
        return a*(a*(a2 * sin4l - 2 * sin2l + 1) + cosl * one_m_a2sin2l_3_2) / one_m_a2sin2l_3_2


# function to evaluate f
def eval_f(l, a):
    # inputs:
    # l         path orientation angle
    # a         wind ratio
    # outputs:
    # f

    sinl = np.sin(l)
    sin2l = sinl * sinl
    cosl = np.cos(l)
    a2 = a * a
    a2sin2l = a2 * sin2l

    if a2sin2l >= 1.0:
        return 2
    else:
        return a2 * sinl * cosl / np.sqrt(1 - a2sin2l) + a * sinl


# function to evaluate critical f approximation
def eval_f_crit_analytic_approx(a):
    # inputs:
    # a         wind ratio
    # outputs:
    # fcrit

    return 2*(1 - np.sqrt(1 - a))


len_data = 501

wrd1 = np.linspace(0, 0.995, len_data - 100)
wrd2 = np.linspace(0.995, 1, 101)
wind_ratio_data = np.concatenate((wrd1[:len_data - 101], wrd2))

path_ori_crit_num_sol = np.zeros(len_data)      # numerical solution of critical path orientation per wind ratio

f_crit_analytic_approx = np.zeros(len_data)     # approximation of critical f value per wind ratio
f_crit_num_sol = np.zeros(len_data)             # numerical solution of critical f value per wind ratio


# numerical soultion
verbose_sol_output = False  # enable to display each solution status and value

a_solution_failed = False
for i in range(len_data):
    sol = optimize.root(eval_df, np.pi/2*0.99, args=(wind_ratio_data[i],))
    path_ori_crit_num_sol[i] = sol.x
    if verbose_sol_output:
        print(f'{wind_ratio_data[i]:.2f}' + ':' + str(sol.success) + '; sol = ' + str(sol.x))
    if not sol.success:
        a_solution_failed = True

if a_solution_failed:
    print('ERROR: One ore more numerical solutions failed, enable verbose_sol_output and re-run cell.')

# df(a=0) = 0, define a crit orientation at 0 which follows the trend of the other solutions
path_ori_crit_num_sol[0] = np.pi/2

for i in range(len_data):
    f_crit_num_sol[i] = eval_f(path_ori_crit_num_sol[i], wind_ratio_data[i])

# approximate solution
for i in range(len_data):
    f_crit_analytic_approx[i] = eval_f_crit_analytic_approx(wind_ratio_data[i])

f_crit_analytic_approx_err = f_crit_analytic_approx - f_crit_num_sol


# plot critical wind factor num solution and approx
fig = plt.figure(figsize=(16, 4))
spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)

ax1 = fig.add_subplot(spec[:, 0])
ax2 = fig.add_subplot(spec[:, 1])
ax3 = fig.add_subplot(spec[:, 2])

ax1.plot(wind_ratio_data, np.rad2deg(path_ori_crit_num_sol), linewidth=plot_lw, color='black')
ax1.set_ylabel(r'${\lambda_P}^\star$ [deg]')
ax1.set_title(r'Critical Path Orientation')
ax1.set_xlabel(r'Wind Ratio $\alpha_W$')

ax2.plot(wind_ratio_data, f_crit_num_sol, linewidth=plot_lw, color='black', label=r'Numerical Solution')
ax2.plot(wind_ratio_data, f_crit_analytic_approx, '--', linewidth=plot_lw, color='tab:blue', label=r'Approx. Analytic Solution')
ax2.set_ylabel(r'${f_W}^\star$')
ax2.set_title(r'Critical Wind Factor')
ax2.set_xlabel(r'Wind Ratio $\alpha_W$')
ax2.legend(loc="upper left")

ax3.plot([wind_ratio_data[i] for i in [0, -1]], [0, 0], linewidth=plot_lw, color='black')
ax3.plot(wind_ratio_data, f_crit_analytic_approx_err, linewidth=plot_lw, color='tab:blue')
ax3.set_title(r'Approximation Error')
ax3.set_xlabel(r'Wind Ratio $\alpha_W$')

plt.show()