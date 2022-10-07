# REMD-Scheduler
This script computes the optimum temperatures for an REMD simulation. 

These computations assume that the distribution of energies at a particular temperature are Gaussian distributed. The probability of exchange between two temperature simulations $P_\mathrm{acc}$ is determined by computing the thermally averaged probability of finding an energy difference $E_{12} = E_2 - E_1$ between two uncorrelated temperature systems
$$P_\mathrm{acc} = \langle P(E_{12}|\mu_{12},\sigma_{12}) \rangle $$
where $P(E_{12}|\mu_{12},\sigma_{12})$ is assumed to be a gaussian function centered at $\mu_{12} = \mu_2 - \mu_1$ with width $\sigma_{12} = \sqrt{ \sigma_{1}^2 + \sigma_{2}^2}$. The means and widths of the individual probability distributions, as functions of temperature, are determined by least-squares fitting against initial calibration simulation data at various temperatures. For example, one might run 10 short simulations at equal temperature intervals, say 300K, 325K, 350K..., and then compute the mean and RMSD total energies of each simulation. The means and widths as continueous functions of $T$ are then approximated as piecewise polynomials by the program

$$
\mu_i(T) = 
  \begin{cases} 
      \sum_i a_i T^i & T_\mathrm{0} \le T \le  T_\mathrm{N}\\
      m_\mathrm{0}T + b_\mathrm{0} & T < T_0 \\
      m_\mathrm{N}T + b_\mathrm{N} & T > T_N \\
   \end{cases}
$$

$T_0$ and $T_N$ are the minimum and maximum temperatures used in the calibration data, respectively. $m_0$ and $m_N$ are the chosen such that the slopes at $T_0$ and $T_N$ are equal. The default is to use a quadratic polynomial between $T_0$ and $T_N$, but this can be adjusted with the `-deg` option. The same function is fit for $\sigma_i(T)$ as well, with different parameters of course.

More information for determining the acceptance ratio can be found from the works of
Garcia *et al*, 2006 (https://doi.org/10.1016/S1574-1400(06)02005-6) and
Patriksson and van der Spoel, 2008 (http://dx.doi.org/10.1039/b716554d).

# Requirements
The following was used to create and test the scheduler:
* Python 3.9
* Scipy 1.7.9
* Numpy 1.21.4
* Matplotlib 3.3.1 (optional, but required for plotting functionality)

# Usage
 Simply type ```python3 scheduler.py```
 
 ```
 usage: scheduler.py [-h] [-mode {1,2,3}] [-data DATA] [-T_min T_MIN] [-T_max T_MAX] [-P_acc P_ACC] [-n_rep N_REP] [-deg DEG]
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
                  
  -T_min T_MIN    Lowest temperature to start replicas at (default = 300)
                  
  -T_max T_MAX    Highest temperature to use in Kelvin
                  
  -P_acc P_ACC    Acceptance probability between each replica (between 0 and 1)
                  
  -n_rep N_REP    Integer number of replicas to use
                  
  -deg DEG        Polynomial degree to fit Energy vs Temperature to (default = 2)
                  
  -plot_exchange  polt the exchange rate from a GROMACS .xvg file.
                  No computations are performed, only plotting.
                  
  -plot_distros   polt the energy distributions from the -data file.
                  No computations are performed, only plotting.
                  
  -plot_evt       polt the energy vs. temperature polynomial fits.
                  No computations are performed, only plotting.
```
