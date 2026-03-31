import numpy as np
import math as mt
from scipy.stats import norm
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
# from scipy.optimize import basinhopping
# from scipy.optimize import differential_evolution
# import nlopt

np.set_printoptions(suppress=True, precision=5) # Set precision for number visualization in output
shift = 0.05
N = 10000 #Nominal
R = np.load("R.npy")
K = np.load("K.npy")
quoted_price = np.load("quoted_price.npy")
T = np.load("T.npy")
A = np.load("A.npy")
sigma = np.load("imp_vol_bach.npy")
moneyness = np.load("moneyness.npy")

##################################
## 2.1) SABR VOLATILITY
# Parameters are with respect to an option

# p = [alpha, beta, rho, nu]

def SABR_vol_bachelier_otm(R, T, K, p):
    alpha = p[0]
    beta = p[1]
    rho = p[2]
    nu = p[3]
    E_max = 1
    
    alpha_bar = alpha* (1+1/4*alpha*beta*rho*nu*R**(1-p[1])*T)
    delta = beta*(2-beta)/(8*R**(2-2*beta))
    
    if beta == 1 :
        z = nu/alpha_bar* mt.log(K/R)
    
    else:
        z = nu/alpha_bar*((K**(1-beta)-R**(1-beta))/(1-beta)) 
    
    E = mt.sqrt(1+2*rho*z+z**2)
    
    z_minus = -rho-mt.sqrt(E_max**2-1+rho**2)
    z_plus = -rho+mt.sqrt(E_max**2-1+rho**2) 
    
    Y_minus = -mt.log((E_max + mt.sqrt(E_max**2-1+rho**2))/(1-rho))
    Y_plus = +mt.log((E_max + mt.sqrt(E_max**2-1+rho**2))/(1+rho))
    
    if z < z_minus:
        Y = Y_minus -(z_minus-z)/E_max
        theta = nu**2/(24*Y)*(-Y_minus-3*mt.sqrt(E_max**2-1+rho**2)/E_max-3*rho)+delta*alpha_bar**2/(6*Y)*(2*E_max*(z-z_minus)+(1-rho**2)*Y_minus-E_max*mt.sqrt(E_max**2-1+rho**2)-rho)
    
    elif z_minus < z < z_plus:
        Y = mt.log((mt.sqrt(1+2*rho*z+z**2)+rho+z)/(1+rho))
        theta = nu**2/24*(-1+3*(rho+z-rho*E)/(Y*E)) + alpha_bar**2*delta/6*(1-rho**2+((z+rho)*E-rho)/Y)
    
    else:
        Y = Y_plus + (z-z_plus)/E_max
        theta = nu**2/(24*Y)*(-Y_plus+3*mt.sqrt(E_max**2-1+rho**2)/E_max-3*rho)+delta*alpha_bar**2/(6*Y)*(2*E_max*(z-z_plus)+(1-rho**2)*Y_plus+E_max*mt.sqrt(E_max**2-1+rho**2)-rho)

    if theta >= 0:
        Z = 1+theta*T
    else:
        Z = (1-theta*T)**(-1)
        
    return nu*(K-R)*Z/Y


def SABR_vol_bachelier_atm(R, T, p):
    alpha = p[0]
    beta = p[1]
    rho = p[2]
    nu = p[3]
    
    delta = beta*(2-beta)/(8*R**(2-2*beta))
    alpha_bar = alpha*(1+1/4*alpha*beta*rho*nu*R**(1-beta)*T)
    theta = (2-3*rho*2)/24*nu**2+1/3*delta*alpha_bar**2
        
    if theta >= 0:
        Z = 1+ theta*T
    
    else:
        Z = (1-theta*T)**(-1)
      
    return alpha_bar * R**beta * Z


# Parameters are with respect to a volatility cube
def vol_SABR(R, K, T , p):
    vol_SABR_ = np.zeros((K.shape[0],K.shape[1]))
    
    for i in range(len(K)):
        for j in range(K.shape[1]):
            
            if K[i,j] == R[i]:
                vol_SABR_[i,j]= SABR_vol_bachelier_atm(R[i], T[i], p[i])
                
            else:
                vol_SABR_[i,j] = SABR_vol_bachelier_otm(R[i], K[i,j], T[i], p[i])
    
    return vol_SABR_


###################################
## 2.2) OPTIMIZATION FUNCTIONS
# Parameters are with respect to a smile section

def RMSE(sigma, R, T, K, p): 

    summ = 0
    
    for i in range(len(sigma)):
        
        if sigma[i] != 0:
            
            if K[i] == R:
                
                summ = summ + (SABR_vol_bachelier_atm(R, T, p)-sigma[i])**2
            
            else:
                summ = summ + (SABR_vol_bachelier_otm(R, K[i], T, p)-sigma[i])**2
    
    loss = (mt.sqrt(1/(len(sigma)-np.count_nonzero(sigma == 0))*summ))
    
    return loss 

def RMSE_vega_w(sigma, R, T, K, A, N, p): 
    summ = 0
    vegas = vega_bachelier(sigma, R, T, K, A, N)
    
    for i in range(len(sigma)):
        
        
        if sigma[i] != 0:
            weight = vegas[i]/sum(vegas)
            
            if K[i] == R:
                summ = summ + (weight*(SABR_vol_bachelier_atm(R, T, p)-sigma[i]))**2
            
            else:
                summ = summ + (weight*(SABR_vol_bachelier_otm(R, K[i], T, p)-sigma[i]))**2

    return (mt.sqrt(1/(len(sigma)-np.count_nonzero(sigma == 0))*summ))

def RMSRE(sigma, R, T, K, A, N, quoted_price, shift, p): 
    summ = 0
    epsilon = 1000
    for i in range(len(sigma)):
        
        if i <= 5:
            w = -1
        
        else:
            w = 1
        
        if sigma[i] != 0:
            price = SABR_price_bachelier(R, T, K[i], A, N, w, shift, p)
            summ = summ + ((price+epsilon)/(quoted_price[i]+epsilon)-1)**2
    
    return (mt.sqrt(1/(len(sigma)-np.count_nonzero(sigma == 0))*summ))


####################################
## 2.3) SENSITIVITY
# Parameters are with respect to a smile section

def vega_bachelier(sigma, R, T, K, A, N):
    vegas = []
    
    for i in range(len(sigma)):
        
        if i <= 5:
            w = -1
        
        else:
            w = 1
        
        if sigma[i] == 0 or T == 0: # Protection versus division for 0
            vegas.append(0)
            
            continue 
        
        d = (R - K[i])/((sigma[i]*mt.sqrt(T)))
        v = N*A*(-w**2*(R - K[i])*d/sigma[i]*norm.pdf(w*d) + mt.sqrt(T)*norm.pdf(d)*(1+d**2))
        vegas.append(v)
      
    return vegas


####################################
## 2.4) PRICING

# Parameters are with respect to an option
def SABR_price_bachelier(R, T, K, A, N, w, shift, p):
    
    if R == K:
        sigma = SABR_vol_bachelier_atm(R+shift, T, p)
        d = (R-K)/(sigma*mt.sqrt(T))
    
    else:
        sigma = SABR_vol_bachelier_otm(R+shift, K+shift, T, p)
        d = (R-K)/(sigma*mt.sqrt(T))

    return N*A*(w*((R-K)*norm.cdf(w*d))+mt.sqrt(T)*sigma*norm.pdf(d))

# Parameters are with respect to a volatility cube
def SABR_price_bachelier_cube(R, T, K, A, N, shift, p):
    price_cube = np.zeros((K.shape[0],K.shape[1]))
    
    for i in range(len(K)):
        for j in range(K.shape[1]):
            
            if j <= 5:
                w = -1
            
            else:
                w = 1
            
            if R[i] == K[i,j]:
                sigma = SABR_vol_bachelier_atm(R[i] + shift, T[i], p[i])
                d = (R[i]-K[i,j])/(sigma*mt.sqrt(T[i]))
            
            else:
                sigma = SABR_vol_bachelier_otm(R[i] + shift , K[i,j] + shift, T[i], p[i])
                d = (R[i]-K[i,j])/(sigma*mt.sqrt(T[i]))

            price_cube[i,j] = N*A[i]*(w*((R[i]-K[i,j])*norm.cdf(w*d))+mt.sqrt(T[i])*sigma*norm.pdf(d))

    return price_cube


##################################
## 2.5) SABR OPTIMIZATOR
# Parameters are with respect to a volatility cube

def SABR(sigma, R, T, K, shift, A, N, quoted_price, function, p = None, bounds = None):
    
    if p is None:
        p = np.array([0.005, 0.1, 0.0, 0.005])
    
    if bounds is None:
        bounds = [ (1e-6, 5.0), (0.0, 1.0), (-0.999, 0.999), (1e-6, 5.0)]
    
    p_optimized = []
    err = []
    succes = []
    iteration = []
    
    for i in range(len(sigma)):
        
        if function == RMSE:
            fixed_parameters = lambda p_: function(sigma[i], R[i]+shift, T[i], K[i]+shift, p_) #To lock known parameters
            start = time.time() # Save the time at that istant
            res = minimize(fixed_parameters, p, method = 'L-BFGS-B', bounds=bounds, tol = 1e-12, options= {'maxiter': 1000000})
            #res = minimize(fixed_parameters, p, method='trust-constr', bounds=bounds, options= {'maxiter': 1000000, 'gtol': 1e-12, 'xtol': 1e-12, 'barrier_tol': 1e-12 })
            #res = minimize(fixed_parameters, p, method='SLSQP', bounds=bounds, tol = 1e-12, options= {'maxiter': 1000000})
            #res = basinhopping(fixed_parameters, p, minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds, "options": {"ftol": 1e-12}})
            #res = differential_evolution(fixed_parameters, bounds, tol=1e-12, maxiter=10000)
            durata = time.time() - start # Evaluate the time it takes to minimize the function
            p_optimized.append(res.x) # Append the optimized parameters
            err.append(res.fun) # Append the final RMSE 
            succes.append(res.success) # Append True or False wether it converged or not
            iteration.append(res.nit) # Append number of iteration to converge
            
        
        elif function == RMSE_vega_w:
            fixed_parameters = lambda p_: function(sigma[i], R[i]+shift, T[i], K[i]+shift, A[i], N, p_) 
            start = time.time()
            res = minimize(fixed_parameters, p, method ='L-BFGS-B', bounds=bounds, tol = 1e-12, options= {'maxiter': 1000000})
            #res = basinhopping(fixed_parameters, p, minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds, "options": {"ftol": 1e-12}})
            #res = differential_evolution(fixed_parameters, bounds, tol=1e-12, maxiter=10000)
            durata = time.time() - start
            p_optimized.append(res.x)
            err.append(res.fun)
            succes.append(res.success)
            iteration.append(res.nit)
        
        else:
            fixed_parameters = lambda p_: function(sigma[i], R[i], T[i], K[i], A[i], N, quoted_price[i], shift, p_) 
            start = time.time()
            res = minimize(fixed_parameters, p, method='L-BFGS-B', bounds=bounds, tol = 1e-12, options= {'maxiter': 1000000})
            #res = basinhopping(fixed_parameters, p, minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds, "options": {"ftol": 1e-12}})
            #res = differential_evolution(fixed_parameters, bounds, tol=1e-12, maxiter=10000)
            durata = time.time() - start
            p_optimized.append(res.x)
            err.append(res.fun)
            succes.append(res.success)
            iteration.append(res.nit)
     
    return p_optimized, err, durata, succes, iteration


#############################
## 2.6) VALIDATION PRECISION AND PERFORMANCE

def plot_SABR_vs_quoted(SABR, market, K, L, method):
    mask = market > 0 #In order to eliminate zeros from the plot
    plt.figure()
    plt.plot(K[mask], market[mask], 'o-', label = 'Market')
    plt.plot(K, SABR, 'x--', label = 'SABR')
    plt.xlabel('Moneyness')
    
    if L == 'volatility':
        y = 'Volatility'
        title = f'Volatility Bachelier vs SABR {method}'
    
    else:
        y = 'Price'
        title = f'Price Bachelier vs SABR {method}'
        
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

SABR_vol_opt = SABR(sigma, R, T, K, shift, A, N, quoted_price, RMSE)
SABR_vega_opt = SABR(sigma, R, T, K, shift, A, N, quoted_price, RMSE_vega_w)
SABR_price_opt = SABR(sigma, R, T, K, shift, A, N, quoted_price, RMSRE)

# Plot RMSE
plt.figure()
plt.bar(['Volatility RMSE', 'Vega-Weighted RMSE', 'RMSRE'],[np.mean(SABR_vol_opt[1]), np.mean(SABR_vega_opt[1]), np.mean(SABR_price_opt[1])])
plt.ylabel('RMSE')
plt.title('Calibration Precision')
plt.grid(True)
plt.show()

# Plot Time to Convergence
plt.figure()
plt.bar(['Volatility RMSE', 'Vega-Weighted RMSE', 'RMSRE'],[np.mean(SABR_vol_opt[2]), np.mean(SABR_vega_opt[2]), np.mean(SABR_price_opt[2])])
plt.ylabel('Time to Convergence')
plt.title('Time to convergence per metodo')
plt.grid(True)
plt.show()

#Plot Successes
lista_bool = [SABR_vol_opt[3],SABR_vega_opt[3], SABR_price_opt[3]]
success_count = [sum(method_success) for method_success in lista_bool]
total = len(SABR_vol_opt[3])  # total number of smile
colori = ['green' if count/total > 0.95 else 'red' for count in success_count]

plt.figure()
plt.bar(['Volatility RMSE', 'Vega-Weighted RMSE', 'Price RMSE'], success_count, color = colori)
plt.ylabel('Number of Successes')
plt.title('Convergence Success per Method')
plt.ylim(0, total + 2)
plt.grid(True)
plt.show()   

# Plot Iteration
plt.figure()
plt.bar(['Volatility RMSE', 'Vega-Weighted RMSE', 'RMSRE'],[np.sum(SABR_vol_opt[4]), np.sum(SABR_vega_opt[4]), np.sum(SABR_price_opt[4])])
plt.ylabel('Iteration')
plt.title('Number of Iterations')
plt.grid(True)
plt.show()

SABR_RMSE_opt_vol = vol_SABR(R+shift, K+shift, T, SABR_vol_opt[0])
SABR_vega_opt_vol = vol_SABR(R+shift, K+shift, T , SABR_vega_opt[0])
SABR_RMSRE_opt_vol = vol_SABR(R+shift, K+shift, T , SABR_price_opt[0])

SABR_RMSE_opt_price = SABR_price_bachelier_cube(R, T, K, A, N, shift, SABR_vol_opt[0])
SABR_vega_opt_price = SABR_price_bachelier_cube(R, T, K, A, N, shift, SABR_vega_opt[0])
SABR_RMSRE_opt_price = SABR_price_bachelier_cube(R, T, K, A, N, shift, SABR_price_opt[0])

n = 35 # Smile section we consider

#Test volatility estimation with SABR
plot_SABR_vs_quoted(SABR_RMSE_opt_vol[n], sigma[n], moneyness, 'volatility', 'RMSE')
plot_SABR_vs_quoted(SABR_vega_opt_vol[n], sigma[n], moneyness, 'volatility', 'Vega') 
plot_SABR_vs_quoted(SABR_RMSRE_opt_vol[n], sigma[n], moneyness, 'volatility', 'RMSRE')
       
#Test price estimation with SABR
plot_SABR_vs_quoted(SABR_RMSE_opt_price[n], quoted_price[n], moneyness, 'price', 'RMSE')
plot_SABR_vs_quoted(SABR_vega_opt_price[n], quoted_price[n], moneyness, 'price', 'Vega') 
plot_SABR_vs_quoted(SABR_RMSRE_opt_price[n], quoted_price[n], moneyness, 'price', 'RMSRE') 

#######################################
## 2.7) DIFFERENT OPTIMIZATION METHOD

methods = ['L-BFGS-B', 'trust-constr', 'SLSQP']

SABR_vol_opt_t = SABR(sigma, R, T, K, shift, A, N, quoted_price, RMSE) # You have to change method in SABR() before run it
SABR_vol_opt_SL = SABR(sigma, R, T, K, shift, A, N, quoted_price, RMSE) # You have to change method in SABR() before run it


# Plot Calibration
plt.figure()
plt.bar(methods,[np.mean(SABR_vol_opt[1]), np.mean(SABR_vol_opt_t[1]), np.mean(SABR_vol_opt_SL[1])])
plt.ylabel('RMSE')
plt.title('Calibration Precision')
plt.grid(True)
plt.show()

# Plot Time to Convergence
plt.figure()
plt.bar(methods,[np.mean(SABR_vol_opt[2]), np.mean(SABR_vol_opt_t[2]), np.mean(SABR_vol_opt_SL[2])])
plt.ylabel('Time to Convergence')
plt.title('Time to convergence per metodo')
plt.grid(True)
plt.show()

#Plot Successes
lista_bool = [SABR_vol_opt[3],SABR_vol_opt_t[3], SABR_vol_opt_SL[3]]
success_count = [sum(method_success) for method_success in lista_bool]
total = len(SABR_vol_opt[3])  # total number of smile
colori = ['green' if count/total > 0.95 else 'red' for count in success_count]

plt.figure()
plt.bar(methods, success_count, color = colori)
plt.ylabel('Number of Successes')
plt.title('Convergence Success per Method')
plt.ylim(0, total + 2)
plt.grid(True)
plt.show()   

# Plot Iteration
plt.figure()
plt.bar(methods, [np.sum(SABR_vol_opt[4]), np.sum(SABR_vol_opt_t[4]), np.sum(SABR_vol_opt_SL[4])])
plt.ylabel('Iteration')
plt.title('Number of Iterations')
plt.grid(True)
plt.show()


################################
## 2.8) TESTING REDUNDANCY

def plot_redundancy(x1, x2, x3, x4, x5, K, par, lista, zoom_left = False, zoom_right = False):
    plt.figure()
    mask = x1 > 0
    plt.plot(K[mask], x1[mask], '--', label = f'{lista[0]}')
    plt.plot(K[mask], x2[mask], '--', label = f'{lista[1]}')
    plt.plot(K[mask], x3[mask], '--', label = f'{lista[2]}')
    plt.plot(K[mask], x4[mask], '--', label = f'{lista[3]}')
    plt.plot(K[mask], x5[mask], '--', label = f'{lista[4]}')

    plt.legend()
    plt.xlabel('Moneyness')
    plt.ylabel('volatility')
    plt.title('Redundancy '+ par)
    plt.grid()
    plt.show()


lista = [0.01, 0.25, 0.5, 0.75, 1.0]
risultati_alpha = []

for alpha_val in lista:    
    p0 = [alpha_val, 0.5, 0.0, 0.5]
    bounds = [
        (alpha_val, alpha_val),
        (0, 1),  
        (-0.999, 0.999),
        (1e-6, 5.0)]
    p = SABR(sigma, R, T, K, shift, A, N, quoted_price, RMSE, p=p0, bounds=bounds)[0]
    cube = vol_SABR(R + shift, K + shift, T , p)
    risultati_alpha.append(cube)

plot_redundancy(risultati_alpha[0][n], risultati_alpha[1][n], risultati_alpha[2][n], risultati_alpha[3][n], risultati_alpha[4][n], moneyness, 'alpha', lista)
    
lista = [0.0, 0.25, 0.5, 0.75, 1.0]
risultati_beta = []

for beta_val in lista:
    p0 = [0.01, beta_val, 0.0, 0.5]
    bounds = [
        (1e-6, 5.0),
        (beta_val, beta_val),
        (-0.999, 0.999),
        (1e-6, 5.0)]
    p = SABR(sigma, R, T, K, shift, A, N, quoted_price, RMSE, p=p0, bounds=bounds)[0]
    cube = vol_SABR(R + shift, K + shift, T , p)
    risultati_beta.append(cube)

plot_redundancy(risultati_beta[0][n], risultati_beta[1][n], risultati_beta[2][n], risultati_beta[3][n], risultati_beta[4][n], moneyness, 'beta', lista)

lista = [-0.5, -0.25, 0.0, 0.25, 0.5]
risultati_rho = []

for rho_val in lista:
    p0 = [0.01, 0.5, rho_val, 0.5]
    bounds = [
        (1e-6, 5.0),
        (0, 0.5),  # fisso
        (rho_val, rho_val),
        (1e-6, 5.0)]
    p = SABR(sigma, R, T, K, shift, A, N, quoted_price, RMSE, p=p0, bounds=bounds)[0]
    cube = vol_SABR(R + shift, K + shift, T , p)
    risultati_rho.append(cube)
    
plot_redundancy(risultati_rho[0][n], risultati_rho[1][n], risultati_rho[2][n], risultati_rho[3][n], risultati_rho[4][n], moneyness, 'rho', lista)

lista = [0.01, 0.25, 0.5, 0.75, 1.0]
risultati_nu = []

for nu_val in lista:
    p0 = [0.01, 0.5, 0.0, nu_val]
    bounds = [
        (1e-6, 5.0),
        (0, 1),
        (-0.999, 0.999),
        (nu_val, nu_val)]
    p = SABR(sigma, R, T, K, shift, A, N, quoted_price, RMSE, p=p0, bounds=bounds)[0]
    cube = vol_SABR(R + shift, K + shift, T , p)
    risultati_nu.append(cube)

plot_redundancy(risultati_nu[0][n], risultati_nu[1][n], risultati_nu[2][n], risultati_nu[3][n], risultati_nu[4][n], moneyness, 'nu', lista)

#################################
## NLOPT IMPLEMENTATION FOR A SMILE

# def objective_function(p, grad):
#     if grad.size > 0:
#         
#         grad[:] = 0  
#     return RMSE(sigma[35,:], R[35], T[35], K[35,:], p)  # <-- la tua funzione RMSE

# opt = nlopt.opt(nlopt.GN_CRS2_LM, 4)  # 4 è la dimensione (numero parametri: alpha, beta, rho, nu)

# # Set bounds 
# opt.set_lower_bounds([1e-6, 0.0, -0.999, 1e-6])
# opt.set_upper_bounds([5.0, 1.0, 0.999, 5.0])

# opt.set_min_objective(objective_function)

# # Setta toleranza su funzione e sui parametri
# opt.set_xtol_rel(1e-15)
# opt.set_ftol_rel(1e-15)

# # Set massimo numero di valutazioni
# opt.set_maxeval(10000)

# # Starting point iniziale
# x0 = [0.005, 0.1, 0.0, 0.005]

# x_opt = opt.optimize(x0)

# minf = opt.last_optimum_value()

# print('Ottimizzato:', x_opt)
# print('Valore funzione:', minf)

# price = []

# for i in range(len(sigma[35])):
#     if i <=5:
#         w=-1
#     else:
#         w=1
#     price.append(SABR_price_bachelier(R[35], T[35], K[35,i], A[35], N, w, shift, x_opt))

# volatility = []

# for i in range(len(sigma[35])):
#     if i <=5:
#         w=-1
#     else:
#         w=1
    
#     if R[35] == K[35,i]:
#         volatility.append(SABR_vol_bachelier_atm(R[35]+shift, T[35], x_opt))
    
#     else:
#         volatility.append( SABR_vol_bachelier_otm(R[35]+shift, K[35,i]+shift, T[35], x_opt))



# plot_SABR_vs_quoted(volatility, sigma[35], moneyness, 'volatility', 'RMSE')
# plot_SABR_vs_quoted(price, quoted_price[35], moneyness, 'price', 'RMSE')