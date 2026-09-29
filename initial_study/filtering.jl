
#=
Filtering 

In this script we will conduct some simple 
filtering on the Intel Lab Data using Gaussian Random Fields. 
The goal is to predict the temperature in the whole lab 
based on the temperature readings from the sensors.

We will use our processed version of the data, 
which has been averaged over 1 minute intervals. 
The data is stored in a feather file for efficient reading.
=#

using Arrow, DataFrames

DATA_PATH = "intel_data_processed_1000.feather"

