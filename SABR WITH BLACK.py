import numpy as np
import math as mt
from scipy.stats import norm
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=5) # Set precision for number visualization in output
shift = 0.05
N = 10000 #Nominal
R = np.load("R.npy")
K = np.load("K.npy")
quoted_price = np.load("quoted_price.npy")
T = np.load("T.npy")
A = np.load("A.npy")
sigma = np.load("imp_vol_black.npy")
moneyness = np.load("moneyness.npy")

##################################
## 2.1) SABR VOLATILITY
# Parameters are with respect to an option

# p = [alpha, beta, rho, nu]

def SABR_vol_black_atm(F, T, p):

    return p[0]/F**(1-p[1])*(1+(p[0]**2*(1-p[1])**2/(24*(F)**(2-2*p[1]))+p[0]*p[1]*p[2]*p[3]/(4*(F)**(1-p[1]))+p[3]**2*(2-3*p[2]**2)/24)*T)

def SABR_vol_black_otm(F,K,T,p):
    
    try:
        z = p[3]/p[0]*(F*K)**((1-p[1])/2)*mt.log(F/K)
        under_root = 1 - 2*p[2]*z + z**2
        
        if under_root <= 0:
            
            return 1e6  # penalità se fuori dominio
        
        Y = mt.log((mt.sqrt(1-2*p[2]*z+ z**2)+z-p[2])/(1-p[2]))
        B = 1+1/24*((1-p[1])*mt.log(F/K))**2+1/1920*((1-p[1])*mt.log(F/K))**4
        A_ = 1+(p[0]**2*(1-p[1])**2/(24*(F*K)**(1-p[1]))+p[0]*p[1]*p[2]*p[3]/(4*(F*K)**((1-p[1])/2))+p[3]**2*(2-3*p[2]**2)/24)*T
        
        return  p[3]*mt.log(F/K)/Y*A_/B     
    
    except (ValueError, ZeroDivisionError):
        
        return 1e6  # penalità per far scartare la soluzione

# Parameters are with respect to a volatility cube
def vol_SABR(R, K, T , p):
    vol_SABR_ = np.zeros((K.shape[0],K.shape[1]))
    
    for i in range(len(K)):
        for j in range(K.shape[1]):
            
            if K[i,j] == R[i]:
                vol_SABR_[i,j]= SABR_vol_black_atm(R[i],T[i],p[i])
                
            else:
                vol_SABR_[i,j] = SABR_vol_black_otm(R[i],K[i,j],T[i],p[i])
    
    return vol_SABR_


###################################
## 2.2) OPTIMIZATION FUNCTIONS
# Parameters are with respect to a smile section

def RMSE(sigma, R, T, K, p): 
    summ = 0
    
    for i in range(len(sigma)):
        
        if sigma[i] != 0:
            
            if K[i] == R:
                
                summ = summ + (SABR_vol_black_atm(R, T, p)-sigma[i])**2
            
            else:
                summ = summ + (SABR_vol_black_otm(R, K[i], T, p)-sigma[i])**2
    
    return mt.sqrt(1/(len(sigma)-np.count_nonzero(sigma == 0))*summ)

def RMSE_vega_w(sigma, R, T, K, A, N, p): 
    summ = 0
    vegas = vega_black(sigma, R, T, K, A, N)
    
    for i in range(len(sigma)):
        
        
        if sigma[i] != 0:
            weight = vegas[i]/sum(vegas)
            
            if K[i] == R:
                summ = summ + (weight*(SABR_vol_black_atm(R, T, p)-sigma[i]))**2
            
            else:
                summ = summ + (weight*(SABR_vol_black_otm(R, K[i], T, p)-sigma[i]))**2

    return mt.sqrt(1/(len(sigma)-np.count_nonzero(sigma == 0))*summ)

def RMSRE(sigma, R, T, K, A, N, quoted_price, p): 
    summ = 0
    
    for i in range(len(sigma)):
        
        if i <= 5:
            w = -1
        
        else:
            w = 1
        
        if sigma[i] != 0:
            price = SABR_price_black(R, T, K[i], A, N, w, p)
            summ = summ + (price/quoted_price[i]-1)**2
    
    return mt.sqrt(1/(len(sigma)-np.count_nonzero(sigma == 0))*summ)


####################################
## 2.3) SENSITIVITY
# Parameters are with respect to a smile section

def vega_black(sigma, R, T, K, A, N):
    vegas = []
    
    for i in range(len(sigma)):
        
        if i <= 5:
            w = -1
        
        else:
            w = 1
        
        #Protezione contro divisioni per 0
        if sigma[i] == 0 or sigma[i]*np.sqrt(T) == 0:
            vegas.append(0)
            
            continue
        
        try:
            d_plus = (mt.log(R/K[i])+1/2*sigma[i]**2*T)/(sigma[i]*mt.sqrt(T))
            d_minus = (mt.log(R/K[i])-1/2*sigma[i]**2*T)/(sigma[i]*mt.sqrt(T))
            v = N*A*w*(R*norm.pdf(w*d_plus)*w*(-mt.log(R/K[i])/(sigma[i]**2*mt.sqrt(T))+1/2*mt.sqrt(T))-K[i]*norm.pdf(w*d_minus)*(-mt.log(R/K[i])/(sigma[i]**2*mt.sqrt(T))-1/2*mt.sqrt(T)))
            vegas.append(v)
        
        except (ZeroDivisionError, ValueError, OverflowError):
           vegas.append(0)
           
    return vegas


####################################
## 2.4) PRICING

# Parameters are with respect to an option
def SABR_price_black(R, T, K, A, N, w, p):
    
    if R == K:
        sigma = SABR_vol_black_atm(R, T, p)
        d_plus = (mt.log(R/K)+1/2*sigma**2*T)/(sigma*mt.sqrt(T)) 
        d_minus = (mt.log(R/K)-1/2*sigma**2*T)/(sigma*mt.sqrt(T))
    
    else:
        sigma = SABR_vol_black_otm(R, K, T, p)
        d_plus = (mt.log(R/K)+1/2*sigma**2*T)/(sigma*mt.sqrt(T)) 
        d_minus = (mt.log(R/K)-1/2*sigma**2*T)/(sigma*mt.sqrt(T))

    return w*N*A*(R*norm.cdf(w*d_plus)-K*norm.cdf(w*d_minus))

# Parameters are with respect to a volatility cube
def SABR_price_black_cube(R, T, K, A, N, p):
    price_cube = np.zeros((K.shape[0],K.shape[1]))
    
    for i in range(len(K)):
        for j in range(K.shape[1]):
            
            if j <= 5:
                w = -1
            
            else:
                w = 1
 
            price_cube[i,j] = SABR_price_black(R[i], T[i], K[i,j], A[i], N, w, p[i])

    return price_cube


##################################
## 2.5) SABR OPTIMIZATOR

# Parameters are with respect to a volatility cube
def SABR(sigma, R, T, K, shift, A, N, quoted_price, function, p = None, bounds = None):
    
    if p is None:
        p = np.array([0.01, 0.5, 0.0, 0.5])
    
    if bounds is None:
        bounds = [
        (1e-6, 5.0),     # alpha > 0
        (0.0, 1.0),      # beta in [0, 1]
        (-1, 1),         # rho in [-1, 1]
        (1e-6, 5.0)]      # nu > 0
    
    p_optimized = []
    err = []
    succes = []
    iteration = []
    
    for i in range(len(sigma)):
        
        if function == RMSE:
            fixed_parameters = lambda p_: function(sigma[i], R[i]+shift, T[i], K[i]+shift, p_) 
            start = time.time() # Save the time at that istant
            res = minimize(fixed_parameters, p, method="L-BFGS-B", bounds=bounds)
            durata = time.time() - start # Evaluate the time it takes to minimize the function
            p_optimized.append(res.x) # Append the optimized parameters
            err.append(res.fun) # Append the final RMSE 
            succes.append(res.success) # Append True or False wether it converged or not.
            iteration.append(res.nit) # Append number of iteration to converge
            
        
        elif function == RMSE_vega_w:
            fixed_parameters = lambda p_: function(sigma[i], R[i]+shift, T[i], K[i]+shift, A[i], N, p_) 
            start = time.time()
            res = minimize(fixed_parameters, p, method="L-BFGS-B", bounds=bounds)
            durata = time.time() - start
            p_optimized.append(res.x)
            err.append(res.fun)
            succes.append(res.success)
            iteration.append(res.nit)
        
        else:
            fixed_parameters = lambda p_: function(sigma[i], R[i]+shift, T[i], K[i]+shift, A[i], N, quoted_price[i], p_) 
            start = time.time()
            res = minimize(fixed_parameters, p, method="L-BFGS-B", bounds=bounds)
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
        title = f'Volatility Black vs SABR {method}'
    
    else:
        y = 'Price'
        title = f'Price Black vs SABR {method}'
        
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
plt.title('Time to convergence per method')
plt.grid(True)
plt.show()
    
#Plot Successes
lista_bool = [SABR_vol_opt[3],SABR_vega_opt[3], SABR_price_opt[3]]
success_count = [sum(method_success) for method_success in lista_bool]
total = len(SABR_vol_opt[3])  # numero totale di sezioni smile
colori = ['green' if count/total > 0.8 else 'red' for count in success_count]

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

SABR_vol_opt_vol = vol_SABR(R+shift, K+shift, T , SABR_vol_opt[0])
SABR_vega_opt_vol = vol_SABR(R+shift, K+shift, T , SABR_vega_opt[0])
SABR_price_opt_vol = vol_SABR(R+shift, K+shift, T , SABR_price_opt[0])

SABR_vol_opt_price = SABR_price_black_cube(R+shift, T, K+shift, A, N, SABR_vol_opt[0])
SABR_vega_opt_price = SABR_price_black_cube(R+shift, T, K+shift, A, N, SABR_vega_opt[0])
SABR_price_opt_price = SABR_price_black_cube(R+shift, T, K+shift, A, N, SABR_price_opt[0])


n = 35 #Smile section da considerare

#Test stima volatility con SABR
plot_SABR_vs_quoted(SABR_vol_opt_vol[n], sigma[n], moneyness, 'volatility', 'RMSE')
plot_SABR_vs_quoted(SABR_vega_opt_vol[n], sigma[n], moneyness, 'volatility', 'Vega') 
plot_SABR_vs_quoted(SABR_price_opt_vol[n], sigma[n], moneyness, 'volatility', 'RMSRE')
       
#Test stima prezzi con SABR
plot_SABR_vs_quoted(SABR_vol_opt_price[n], quoted_price[n], moneyness, 'price', 'RMSE')
plot_SABR_vs_quoted(SABR_vega_opt_price[n], quoted_price[n], moneyness, 'price', 'Vega') 
plot_SABR_vs_quoted(SABR_price_opt_price[n], quoted_price[n], moneyness, 'price', 'RMSRE') 

################################
## 2.7) TESTING REDUNDANCY

def plot_redundancy(x1, x2, x3, x4, x5, K, par, lista):
    
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
    plt.grid(True)


lista_alpha = [0.01, 0.25, 0.5, 0.75, 1.0]
risultati_alpha = []

for alpha_val in lista_alpha:    
    p0 = [alpha_val, 0.5, 0.0, 0.5]
    bounds = [
        (alpha_val, alpha_val),
        (0, 1),  
        (-0.999, 0.999),
        (1e-6, 5.0)]
    p = SABR(sigma, R, T, K, shift, A, N, quoted_price, RMSE, p=p0, bounds=bounds)[0]
    cube = vol_SABR(R+shift, K+shift, T , p)
    risultati_alpha.append(cube)
    
lista_beta = [0.0, 0.25, 0.5, 0.75, 1.0]
risultati_beta = []

for beta_val in lista_beta:
    p0 = [0.01, beta_val, 0.0, 0.5]
    bounds = [
        (1e-6, 5.0),
        (beta_val, beta_val),
        (-0.999, 0.999),
        (1e-6, 5.0)]
    p = SABR(sigma, R, T, K, shift, A, N, quoted_price, RMSE, p=p0, bounds=bounds)[0]
    cube = vol_SABR(R+shift, K+shift, T , p)
    risultati_beta.append(cube)


lista_rho = [-0.5, -0.25, 0.0, 0.25, 0.5]
risultati_rho = []

for rho_val in lista_rho:
    p0 = [0.01, 0.5, rho_val, 0.5]
    bounds = [
        (1e-6, 5.0),
        (0, 0.5),  # fisso
        (rho_val, rho_val),
        (1e-6, 5.0)]
    p = SABR(sigma, R, T, K, shift, A, N, quoted_price, RMSE, p=p0, bounds=bounds)[0]
    cube = vol_SABR(R+shift, K+shift, T , p)
    risultati_rho.append(cube)
    
lista_nu = [0.01, 0.25, 0.5, 0.75, 1.0]
risultati_nu = []

for nu_val in lista_nu:
    p0 = [0.01, 0.5, 0.0, nu_val]
    bounds = [
        (1e-6, 5.0),
        (0, 1),
        (-0.999, 0.999),
        (nu_val, nu_val)]
    p = SABR(sigma, R, T, K, shift, A, N, quoted_price, RMSE, p=p0, bounds=bounds)[0]
    cube = vol_SABR(R+shift, K+shift, T , p)
    risultati_nu.append(cube)




plot_redundancy(risultati_alpha[0][n], risultati_alpha[1][n], risultati_alpha[2][n], risultati_alpha[3][n], risultati_alpha[4][n], moneyness, 'alpha', lista_alpha)

plot_redundancy(risultati_beta[0][n], risultati_beta[1][n], risultati_beta[2][n], risultati_beta[3][n], risultati_beta[4][n], moneyness, 'beta', lista_beta)

plot_redundancy(risultati_rho[0][n], risultati_rho[1][n], risultati_rho[2][n], risultati_rho[3][n], risultati_rho[4][n], moneyness, 'rho', lista_rho)

plot_redundancy(risultati_nu[0][n], risultati_nu[1][n], risultati_nu[2][n], risultati_nu[3][n], risultati_nu[4][n], moneyness, 'nu', lista_nu)

    