import pandas as pd
import numpy as np

np.set_printoptions(suppress=True, precision=5) # Set precision for number visualization in output

#################################
## IMPORTAZIONE DATI
df = pd.read_excel(r'C:\Users\Jacopo Zavalloni\Desktop\ADVANCED RATE MODEL\12AIRMM-MarketData31Oct2019bis.xlsx', sheet_name='IMPORT', header=0)

INDEX = np.array(df['TIME'].copy()) # Array of maturity-tenor pairs
T = np.array(df['MATURITY'].copy()) # Array of maturities
A = np.array(df['ANNUITY'].copy()) # Array with annuities for each IRS (each section smile is written for a IRR)
R = np.array(df['IRS_R'].copy())/100 # Array of IRS rate
quoted_price = df.iloc[:, 3:16].copy() # Quoted price cube
A_IRR = np.array(df['A_IRR'].copy()) # Array of Annuity for Internal Rate of Return
N = 10000 #Nominal

moneyness = np.array(quoted_price.columns.astype(float)) #Take column as array of quoted_price

# Strike cube costruction
K = np.zeros((quoted_price.shape[0],quoted_price.shape[1])) 

for i in range(K.shape[1]):
    K[:,i] = R + moneyness[i]

quoted_price = np.array(quoted_price.fillna(0))

np.save("INDEX.npy", INDEX)
np.save("R.npy", R)
np.save("K.npy", K)
np.save("quoted_price.npy", quoted_price)
np.save("A.npy", A)
np.save("T.npy", T)
np.save("moneyness.npy", moneyness)
np.save("A_IRR.npy", A_IRR)
