import numpy as np
import math as mt
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import time
from tabulate import tabulate

np.set_printoptions(suppress=True, precision=5) # Set precision for number visualization in output
N = 10000
INDEX = np.load("INDEX.npy", allow_pickle=True)
R = np.load("R.npy")
K = np.load("K.npy")
quoted_price = np.load("quoted_price.npy")
T = np.load("T.npy")
A = np.load("A.npy")
moneyness = np.load("moneyness.npy")
A_IRR = np.load("A_IRR.npy")


###################################
## 1.1) PRICING OPTION
# Parameters are with respect to an option

def price_black(R, T, K, sigma, A, N, w):
    d_plus = (mt.log(R/K)+1/2*sigma**2*T)/(sigma*mt.sqrt(T)) 
    d_minus = (mt.log(R/K)-1/2*sigma**2*T)/(sigma*mt.sqrt(T))
    
    return w*N*A*(R*norm.cdf(w*d_plus)-K*norm.cdf(w*d_minus))

def price_bachelier(R, T, K, sigma, A, N, w): 
    d = (R-K)/(sigma*mt.sqrt(T))
    
    return N*A*(w*(R-K)*norm.cdf(w*d)+sigma*mt.sqrt(T)*norm.pdf(d))

def vega_bachelier(sigma, R, T, K, A, N, w):
    
    d = (R - K)/((sigma*mt.sqrt(T)))
    v = N*A*(-w**2*(R - K)*d/sigma*norm.pdf(w*d) + mt.sqrt(T)*norm.pdf(d)*(1+d**2))
  
    return v

# Parameters are with respect to a volatility cube
def PRICE(sigma, R, K, T, A, N, shift, B):
    
    IRR_cube = np.zeros((quoted_price.shape[0], quoted_price.shape[1]))
    for i in range(K.shape[0]):
        
        for j in range(K.shape[1]):
            vol = sigma[i,j]
            
            if vol == 0 or vol * np.sqrt(T[i]) == 0:
                IRR_cube[i,j] = 0
                
                continue 
            
            if j <= 5:
                w = -1
            
            else:
                w = +1
            
            if B == 'black':
                IRR_cube[i,j] = price_black(R[i] + shift, T[i], K[i,j] + shift, vol, A[i], N, w)
            
            else:
                IRR_cube[i,j] = price_bachelier(R[i], T[i], K[i,j], vol, A[i], N, w)
                
    return IRR_cube

#################################
## 1.2) ALGORHITMIC METHOD
# Parameters are with respect to an option

def bisection_scheme(R, T, K, A, N, w, B, quoted_price, tol = 1e-12, max_iter = 100):
    low = 1e-6
    high = 1
    count = 0
    
    for i in range(max_iter):
        
        count = count + 1
        mid = (high + low)/2
        check = False
        
        if B == 'black':
            price = price_black(R, T, K, mid, A, N, w)
            
        else:
            price = price_bachelier(R, T, K, mid, A, N, w)
            
        diff = price - quoted_price
        
        if abs(diff) < tol:
            check = True
            
            return mid, check, count
        
        elif diff > 0:
            high = mid
        
        else:
            low = mid
    
    return mid, check, count

def newton_raphson(R, K, T, quoted_price, A, N, w, initial_guess=0.05, tol=1e-12, max_iter=100):

    sigma = initial_guess
    count = 0
    
    for _ in range(max_iter):
        check = False
        count = count + 1
        price = price_bachelier(R, T, K, sigma, A, N, w)
        diff = price - quoted_price

        if abs(diff) < tol:
            check = True
            
            return sigma, count, check

        vega = vega_bachelier(sigma, R, T, K, A, N, w)
        
        if abs(vega) < 1e-12:
            print("Vega is too small; numerical trouble.")
            
            return 1e-6, check
            

        sigma_next = sigma - diff / vega
        if sigma_next < 1e-8:
            sigma_next = 1e-8

        sigma = sigma_next

    raise ValueError("Newton-Raphson did not converge.")

#################################
## 1.3) IMPLIED VOLATILITY
# Parameters are with respect to a volatility cube

def implied_volatility_bachelier(R, T, K, A, N, quoted_price):
    imp_vol = np.zeros((quoted_price.shape[0], quoted_price.shape[1]))
    B = 'bachelier'
    check = np.zeros((quoted_price.shape[0], quoted_price.shape[1]))
    durata = np.zeros((quoted_price.shape[0], quoted_price.shape[1]))
    count = np.zeros((quoted_price.shape[0], quoted_price.shape[1]))
    
    for i in range(len(quoted_price)):
        
        for j in range(quoted_price.shape[1]):
            
            if j <= 5:
                w = -1
                
            else:
                w=1 
                
            if quoted_price[i,j] == 0:
                imp_vol[i,j] = 0 
            
            elif R[i] == K[i,j]:
                imp_vol[i,j] = mt.sqrt(2*mt.pi/T[i]) * quoted_price[i,j]/(A[i]*N)
            
            else:
                start = time.time()
                value = bisection_scheme(R[i], T[i], K[i,j], A[i], N, w, B, quoted_price[i,j])
                durata[i,j] = time.time() - start
                imp_vol[i,j] = value[0]
                check[i,j] = value[1]
                count[i,j] = value[2]
    
    return imp_vol, check, durata, count

def implied_volatility_black_numerical(R, T, K, A, N, quoted_price, shift):
    imp_vol = np.zeros((quoted_price.shape[0], quoted_price.shape[1]))
    B = 'black'
    check = np.zeros((quoted_price.shape[0], quoted_price.shape[1]))

    for i in range(len(quoted_price)):
        
        for j in range(quoted_price.shape[1]):
            
            if j <= 5:
                w = -1
                
            else:
                w=1 
                
            if quoted_price[i,j] == 0:
                imp_vol[i,j] = 0 
            
            else:
                
                value = bisection_scheme(R[i]+shift, T[i], K[i,j]+shift, A[i], N, w, B, quoted_price[i,j])
                imp_vol[i,j] = value[0]
                check[i,j] = value[1]
                
    return imp_vol, check


#################################
## 1.4) CALCOLO VOLATILITIES

imp_vol_bach = implied_volatility_bachelier(R, T, K, A, N, quoted_price) 
np.save("imp_vol_bach.npy", imp_vol_bach[0])

shift = 0.05
imp_vol_black = implied_volatility_black_numerical(R, T, K, A, N, quoted_price, shift)
np.save("imp_vol_black.npy", imp_vol_black[0])

shift = 0.06
imp_vol_black_6 = implied_volatility_black_numerical(R, T, K, A, N, quoted_price, shift)

shift = 0.07
imp_vol_black_7 = implied_volatility_black_numerical(R, T, K, A, N, quoted_price, shift)


#################################
## 1.5) VALIDATION PRECISION AND PERFORMANCE
# Parameters are with respect to a smile section

def smile(sigma, K, B):
    
    mask = sigma > 0 # To eliminate zeros from the plot
    
    if B == 'black':
        title = 'Volatility Smile Black'
        
    else:
        title = 'Volatility Smile Bachelier'
        
    plt.figure()
    plt.plot(K[mask], sigma[mask], 'o-')
    plt.xlabel('Moneyness')
    plt.ylabel('Volatility')
    plt.title(title)
    plt.grid(True)
    plt.show()

n = 35 # smile section to plot for smiles

smile(imp_vol_bach[0][n], moneyness, 'bachelier')
smile(imp_vol_black[0][n], moneyness, 'black')

#Check how many bisection scheme doesn't get convergence till 100 iterations for Black volatilities
n_black_conv = len(quoted_price)*len(quoted_price[0])-np.count_nonzero(quoted_price == 0)- np.sum(imp_vol_black[1])
n_black_6_conv = len(quoted_price)*len(quoted_price[0])-np.count_nonzero(quoted_price == 0)- np.sum(imp_vol_black_6[1])
n_black_7_conv = len(quoted_price)*len(quoted_price[0])-np.count_nonzero(quoted_price == 0)- np.sum(imp_vol_black_7[1])
n_bach_conv = len(quoted_price)*len(quoted_price[0])-np.count_nonzero(quoted_price == 0)- np.sum(imp_vol_bach[1])-quoted_price.shape[0]

table = [ ['Bachelier', n_bach_conv ], ['Black 5% shift',  n_black_conv], ['Black 6% shift',  n_black_6_conv], ['Black 7% shift',  n_black_7_conv] ]
headers = ['Method', 'Number of fail in estimation']

tabulate_ = tabulate(table, headers=headers, tablefmt='pretty')
print(tabulate_)

## Save in a txt
# with open(r'C:\Users\Jacopo Zavalloni\Desktop\ADVANCED RATE MODEL\TABLES\fail_table.txt', 'w') as f:
#     f.write(tabulate_)

# PLOT OF DIFFERENT SHIFTED BLACK VOLATILITIES

def plot_shifted_vol(primo, secondo, terzo, K):
    mask = primo > 0 #In order to eliminate zeros from the plot
    plt.figure()
    plt.plot(K[mask], primo[mask], 'o-', label = '5% shift')
    plt.plot(K[mask], secondo[mask], 'x--', label = '6% shift')
    plt.plot(K[mask], terzo[mask], '^-', label = '7% shift')

    plt.xlabel('Moneyness')
    plt.ylabel('Volatility')
    plt.title('Shifted Volatilities comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_shifted_vol(imp_vol_black[0][n], imp_vol_black_6[0][n], imp_vol_black_7[0][n], moneyness)


# CHECK ESTIMATIONS

def plot_estimated_vs_quoted(e, market, K, L, B, method = False):
    mask = market > 0 #In order to eliminate zeros from the plot
    plt.figure()
    line1, = plt.plot(K[mask], market[mask], 'o-', label = 'Market')
    line2, = plt.plot(K[mask], e[mask], 'x--', label = 'estimated')
    plt.xlabel('Moneyness')
    
    if L == 'volatility':
        y = 'Volatility'
        title = 'Volatility '
    else:
        y = 'Price'
        title = 'Price '
    
    if B =='black':
        title = title + 'Black vs market'
    
    else:
        title = title + 'Bachelier vs market'
    
    if method:
        title = 'Volatility smile Bisection vs Newton-Raphson'
        line1.set_label('Bisection Scheme')
        line2.set_label('Newton Raphson')

    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

price_bach = PRICE(imp_vol_bach[0], R, K, T, A, N, 0, 'bachelier')
price_black_ = PRICE(imp_vol_black[0], R, K, T, A, N, 0.05, 'black')


plot_estimated_vs_quoted(price_bach[n], quoted_price[n], moneyness, 'Price', 'bachelier')
plot_estimated_vs_quoted(price_black_[n], quoted_price[n], moneyness,'Price', 'black')

# CHECK BISECTION VS NEWTON-RAPHSON

implied_var = np.zeros((quoted_price.shape[0],quoted_price.shape[1]))
durata = np.zeros((quoted_price.shape[0],quoted_price.shape[1]))
count = np.zeros((quoted_price.shape[0],quoted_price.shape[1]))
check = np.zeros((quoted_price.shape[0],quoted_price.shape[1]))

for i in range(K.shape[0]):
    for j in range(K.shape[1]):
        if j <= 5:
            w=-1
        else:
            w =1
            
        if quoted_price[i,j] < 1e-4 :
            implied_var[i,j] = 0
            
        elif K[i,j] == R[i]:
            implied_var[i,j] = mt.sqrt(2*mt.pi/T[i]) * quoted_price[i,j]/(A[i]*N)
        
        else:
            start = time.time()
            value = newton_raphson(R[i], K[i,j], T[i], quoted_price[i,j], A[i], N, w)
            implied_var[i,j] = value[0]
            count[i,j] = value[1]
            durata[i,j] = time.time() - start
            check[i,j] = value[2]

plot_estimated_vs_quoted(imp_vol_bach[0][n], implied_var[n], moneyness, 'volatility', 'Bachelier', method = True)

n_bach_conv_newton = len(quoted_price)*len(quoted_price[0])-np.count_nonzero(quoted_price == 0)- np.sum(check)-quoted_price.shape[0]


table = [ ['Bisection Scheme', n_bach_conv, np.mean(imp_vol_bach[2]), np.sum(imp_vol_bach[3])], ['Newton Raphson', n_bach_conv_newton, np.mean(durata), np.sum(count)]]
headers = ['Method', 'Number of fail in convergence', 'Time to convergence', 'Number of Iterations']

tabulate_ = tabulate(table, headers=headers, tablefmt='pretty')
print(tabulate_)

with open(r'C:\Users\Jacopo Zavalloni\Desktop\ADVANCED RATE MODEL\TABLES\algorithmic_table.txt', 'w') as f:
    f.write(tabulate_)


#####################################
## 1.6) INTERNAL RATE OF RETURN SWAP

IRR_bachelier = PRICE(imp_vol_bach[0], R, K, T, A_IRR, N, 0, 'bachelier')
IRR_black = PRICE(imp_vol_black[0], R, K, T, A_IRR, N, 0.05, 'black')
IRR_black_6 = PRICE(imp_vol_black_6[0], R, K, T, A_IRR, N, 0.06, 'black')
IRR_black_7 = PRICE(imp_vol_black_7[0], R, K, T, A_IRR, N, 0.07, 'black')

# Check how many IRR swaption has not been priced
c_bach = np.count_nonzero(IRR_bachelier == 0) - np.count_nonzero(imp_vol_black[0] == 0) 
c_black = np.count_nonzero(IRR_black == 0) - np.count_nonzero(imp_vol_black[0] == 0)
c_black_6 = np.count_nonzero(IRR_black_6 == 0) - np.count_nonzero(imp_vol_black[0] == 0)
c_black_7 = np.count_nonzero(IRR_black_7 == 0) - np.count_nonzero(imp_vol_black[0] == 0)

table = [ ['Bachelier',  c_bach], ['Black 5% shift',  c_black], ['Black 6% shift',  c_black_6], ['Black 7% shift',  c_black_7]]
headers = ['Method', 'Not priced IRR']

tabulate_ = tabulate(table, headers=headers, tablefmt='pretty')
print(tabulate_)

# with open(r'C:\Users\Jacopo Zavalloni\Desktop\ADVANCED RATE MODEL\TABLES\no_priced__table.txt', 'w') as f:
#     f.write(tabulate_)

mask = quoted_price[n] > 0
plt.figure()
plt.plot(moneyness[mask], IRR_bachelier[n][mask], 'o', label = 'Bachelier', color='red')
plt.plot(moneyness[mask], IRR_black[n][mask], 'x', label = 'Black 5% shift', color='yellow')
plt.plot(moneyness[mask], IRR_black_6[n][mask], '^', label = 'Black 6% shift', color = 'blue')
plt.plot(moneyness[mask], IRR_black_7[n][mask], '*', label = 'Black 7% shift', color = 'green' )
plt.xlabel('Moneyness')
plt.ylabel('Price')
plt.title('IRR pricing comparison')
plt.legend()
plt.grid(True)
plt.show()

####################################
## 1.7) OUTPUT

vol_5 = pd.DataFrame(imp_vol_black[0])
vol_6 = pd.DataFrame(imp_vol_black_6[0])
vol_7 = pd.DataFrame(imp_vol_black_7[0])
IRR_bach = pd.DataFrame(IRR_bachelier)
IRR_black_1 = pd.DataFrame(IRR_black)
IRR_black_2 = pd.DataFrame(IRR_black_6)
IRR_black_3 = pd.DataFrame(IRR_black_7)
vol_bach = pd.DataFrame(imp_vol_bach[0])

vol = pd.concat([vol_5, vol_6, vol_7], axis=1) # Glue together the implied forward volatilities evaluated with three different shift
vol.index = INDEX
vol.columns = moneyness.tolist() * 3
IRR = pd.concat([IRR_bach, IRR_black_1, IRR_black_2, IRR_black_3], axis=1)
IRR.index = INDEX
IRR.columns = moneyness.tolist() * 4
vol_bach.index = INDEX
vol_bach.columns = moneyness

file_path = r'C:\Users\Jacopo Zavalloni\Desktop\ADVANCED RATE MODEL\12AIRMM-MarketData31Oct2019bis.xlsx'


# with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
#     #vol_bach.to_excel(writer, sheet_name='Volatility_bachelier_cube')
#     vol.to_excel(writer, sheet_name='Volatilities_black_cube')
#     IRR.to_excel(writer, sheet_name='IRR_cubes')












