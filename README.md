# REMD-Scheduler
This script computes the optimum temperatures for an REMD simulation. 

These computations assume that the distribution of energies at a particular temperature are Gaussian distributed. The probability of exchange between two temperature simulations is determined by computing the thermally averaged probability of finding each of the two temperature systems at their respective total energies.

More information for determining the acceptance ratio can be found from the works of
Garcia *et al*, 2006 (https://doi.org/10.1016/S1574-1400(06)02005-6) and
Patriksson and van der Spoel, 2008 (http://dx.doi.org/10.1039/b716554d).

# Requirements
*Python 3.9 or above

*Scipy 1.7.9

# Usage
 Simply type ```python3 scheduler.py```
 
 ```
 usage: scheduler.py [-h] [-mode {1,2,3}] [-data DATA] [-T_min T_MIN] [-T_max T_MAX] [-P_acc P_ACC] [-n_rep N_REP]
                    [-plot_exchange] [-plot_distros] [-plot_evt]

optional arguments:
  -h, --help      show this help message and exit
  -mode {1,2,3}   Type of calculation to run.
                  
                  -mode 1 makes the number of replicas the solvable quantity.
                  This mode takes in a maximum temperature with -T_max and a 
                  probability of swapping with -P_acc to assign the replicas
                  between -T_min and -T_max.
                  
                  -mode 2 makes the maximum temperature the solvable quantity.
                  This modes takes in an acceptance probability with -P_acc 
                  and a number of replicas with -n_rep to assign a temperature 
                  range that start at -T_min.
                  
                  -mode 3 Makes the acceptance probability the solvable 
                  quantity. This mode takes in -T_max and -n_rep to determine 
                  the replicas that give the maximum acceptance probability
                  that fit between -T_min and -T_max.
                  
  -data DATA      Input file.
                  If using -mode 1-3, -plot_distros, or -plot_evt, this has
                  must have three columns (temperature, mean energy, RMSD energy). 
                  If using -plot_exchange, then this is a GROMACS .xvg file.
                  
  -T_min T_MIN    Lowest temperature to start replicas at (default=300 K)
                  
  -T_max T_MAX    Highest temperature to use in Kelvin
                  
  -P_acc P_ACC    Acceptance probability between each replica (between 0 and 1)
                  
  -n_rep N_REP    Integer number of replicas to use
                  
  -plot_exchange  polt the exchange rate from a GROMACS .xvg file.
                  No computations are performed, only plotting.
                  
  -plot_distros   polt the energy distributions from the -data file.
                  No computations are performed, only plotting.
                  
  -plot_evt       polt the energy vs. temperature polynomial fits.
                  No computations are performed, only plotting.
                  
```
