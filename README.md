# Interest Rate Driven Financial Instruments Project

This repository contains the Python code developed for a graduate-level exam project on the use and modelling of interest rate driven financial instruments.

The project focuses on the calibration and implementation of models used to study implied volatility and SABR dynamics, with comparisons between the Bachelier and Black pricing frameworks.

## Repository Contents

- `ASSIGNMENT.pdf`  
  Contains the general assignment and the statement of the project.

- `AIRM-8.pdf`  
  Contains the complete worked solution, with all explanations and derivations.

- `12AIRM-MarketData31Oct2019bis.xlsx`  
  Contains the market data used in the project. The data are located in the `IMPORT` sheet.  
  These data were provided by the professor exclusively for educational and non-financial purposes, and were uploaded only to allow the full code to be checked and reproduced.

- `IMPORT DATI.py`  
  Python script used to import the Excel data and prepare the dataset for the remaining scripts.

- `IMPLIED FORWARD VOLATILITY.py`  
  Python script for the implied forward volatility part of the project.

- `SABR.py`  
  Python script implementing the SABR model using the Bachelier formula.

- `SABR WITH BLACK.py`  
  Python script implementing the SABR model using the Black formula.

## How the Code is Organized

The Python files are divided into sections so that they can be used either as full scripts or as a source of individual functions.

If your goal is only to use or implement the models, you can copy the relevant sections of code containing the function definitions:

- For the SABR model, it is sufficient to take the code up to section **2.5**.
- For the implied volatility part, it is sufficient to take the code up to section **1.3**.

## How to Reproduce the Full Project

To reproduce the complete exercise, follow these steps:

1. Open `IMPORT DATI.py`.
2. Update the path to the Excel file `12AIRM-MarketData31Oct2019bis.xlsx` according to your local environment.
3. Run `IMPORT DATI.py` first.
4. Then run `IMPLIED FORWARD VOLATILITY.py`.
5. Finally, run either:
   - `SABR WITH BLACK.py`, or
   - `SABR.py`

## Notes

- The scripts are intended for educational purposes.
- The repository includes both the implementation and the documentation needed to understand the full workflow.
- The project was developed as part of a master's degree exam on interest rate driven financial instruments.
