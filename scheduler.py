import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy import optimize
from scipy.special import erf, erfc
import argparse
#from openmm.openmm import NonbondedForce

#   Boltzmann's Constant in kJ/mol/K
KB = 8.314462618E-3

class poly_function():

    def __init__(self, T, eng, deg=3):
        '''
        Fits a polynomial of degree 'deg' with deg > 0 to energy data

        Parameters
        ----------
        T: array of temperatures
        eng: array of energies to fit to
        deg: degree of polynomial that is greater than 0
        
        Returns
        -------
        Callable object as a function of temperature
        '''
        self._degree = deg
        self._poly_coeff = np.flip(np.polyfit(T, eng, self._degree))

        self._T_data = np.copy(T)
        self._eng_data = np.copy(eng)
        self.T_max = np.max(T)
        self.T_min = np.min(T)

    def __call__(self, T):
        powers = np.array([np.array(T)**n for n in range(self._degree + 1)]).T
        return powers @ self._poly_coeff

class REMDSolver():
    def __init__(self, temperatures, energies, stddevs) -> None:
        self.energy_func = poly_function(temperatures, energies, 3)
        self.sigma_func = poly_function(temperatures, stddevs, 2)

    def R_acc(self, T2, T1):
        '''
            Computes the REMD acceptance rate using the algorithm
            from Patriksson and van der Spoel, 2008
            https://doi.org/10.1039/B716554D

            Parameters
            ----------
            T2: Temperature 2 of system
            T1: Temperature 1 of system with T1 < T2
            energy_func: callable function of temperature that returns the average energy
            sigma_func: callable function of temperature that returns the RMSD energy
            
            Returns
            -------
            Acceptance rate between 0 and 1
            '''
        ''' assumes T2 > T1 and that E2 > E1'''
        E2 = self.energy_func(T2)
        E1 = self.energy_func(T1)
        s2 = self.sigma_func(T2)
        s1 = self.sigma_func(T1)
        C = 1/(T2*KB) - 1/(T1*KB)           # eq. 5 from paper
        sigma_12 = np.sqrt(s1*s1 + s2*s2)   # defined in text between eq. 7 and eq. 8
        mu_12 = E2 - E1                     # defined in text between eq. 7 and eq. 8

        #    the following computes eq. 9 from the paper
        term1 = 0.5*(1 + erf(-mu_12/sigma_12))  
        exp_term = np.exp(C*mu_12 + 0.5 * C**2 * sigma_12**2)
        term2 = 0.5*(1 + erf((mu_12 + C*sigma_12**2)/(sigma_12*np.sqrt(2))))

        return term1 + exp_term*term2

    def calc_replicas(self, T_min, P_acc, T_max=None, n_replicas=None) -> list:
        '''
            Computes the set of temperatures with an expected
            acceptance rate of swapping between each replica.
            The replicas start at T_min and neighbor each other
            with a fixed probability P_Accof swapping between 
            each neighbor.

            Parameters
            ----------
            T_min:      Minimum temperature to start replicas at
            P_acc:      Acceptance probability between 0 and 1
            T_max:      Maximum temperature to compute replicas at
            n_replicas: Number of replicas to use

            Returns
            -------
            array of temperatures
        '''

        #   set both bounds if neither are specified to use max_temperature
        if T_max is None and n_replicas is None:
            T_max = self.energy_func.T_max
            n_replicas = 100000

        #   n_replicas were specified, so max temperature is unbounded
        elif T_max is None:
            T_max = 1000000

         #   max temp was specified, so n_replicas is unbounded
        elif n_replicas is None:
            n_replicas = 1000000

        replicas = [T_min]
        root_func = lambda T2, T1: self.R_acc(T2, T1) - P_acc
        while(True):
            T1 = replicas[-1]
            T_guess = replicas[-1] + 1.0
            args_in = (T1)

            T_new = optimize.fsolve(root_func, T_guess, args_in)[0]
            replicas.append(T_new)

            if T_new >= T_max:
                break
            if len(replicas) == n_replicas:
                break

        return replicas

    def calc_replicas_optimized(self, T_min, T_max, n_replicas):
        '''
            Computes the optimum acceptance probability between
            a fixed number of replicas  in a fixed temperature range.

            Parameters
            ----------
            T_min:      Minimum temperature to start replicas at
            T_max:      Maximum temperature of replicas
            n_replicas: Number of temperatures to use

            Returns
            -------
            P_acc:      Swapping probability between two neighboring replicas and
            replicas:   np.array of temperatures for each replica
        '''

        def length_func(r_acc):
            if r_acc < 0:
                return 1.7976931348623157e+308
            replicas = self.calc_replicas(T_min, r_acc, n_replicas=n_replicas)
            return (replicas[-1] - T_max)

        #    make sure we have enough replicas first
        max_length = length_func(0.0001)
        if max_length < 0:
            message =  " Too little replicas specified.\n"
            message += " Could not find an acceptance rate less than 0.01%"
            print_error(message)

        # tmp = np.arange(0.01, 0.1, 0.01)
        # vals = [length_func(x) for x in tmp]
        # plt.plot(tmp, vals, marker='.')
        # plt.show()
        # exit()

        optrimum_acc = optimize.fsolve(length_func, 0.01)[0]
        replicas = self.calc_replicas(T_min, optrimum_acc, n_replicas=n_replicas)
        return replicas, optrimum_acc

    def print_results(self, replicas, Racc):
        print('\n')
        print(' Exchange probability: {:.4f}'.format(Racc))
        print(' Number of replicas computed: {:5d}'.format(len(replicas)))
        print(' ----------------------------------')
        for n, rep in enumerate(replicas):
            print('     Replica {:5d}:  {:8.3f}'.format(n+1, rep))
        print(' ----------------------------------')

    def _plot_poly_fits(self):
        temps = np.linspace(self.energy_func.T_min, self.energy_func.T_max, 100)
        temps = self.energy_func._T_data
        plt.plot(temps, self.sigma_func(temps), marker='.')
        plt.show()

    def plot_gaussians(self, outfile='distros.png'):
        '''
            Plot a series of normalized gaussian functions

            Parameters
            ----------
            energies: array of centers for the Gaussian distributions
            sigmas: array of std. dev. for the Gaussians
        '''

        temps = self.energy_func._T_data
        order = np.argsort(temps)
        temps = temps[order]
        means = self.energy_func._eng_data[order]
        sigmas = self.sigma_func._eng_data[order]

        fig, ax = plt.subplots()
        x_min = np.min(means)
        x_min = x_min - np.abs(x_min)*0.
        x_max = np.max(means)
        x_max = x_max + np.abs(x_max)*0.
        x_vals = np.linspace(x_min, x_max, 10000)

        normal = lambda x, m,s:  (1/np.sqrt(2*np.pi*s*s))*np.exp(-(x - m)**2 / (2*s*s))

        for n, mean in enumerate(means):
            #print("TEMP: ", n+1, temps[n])
            P_acc = 0.0
            if n > 0:
                P_acc += self.R_acc(temps[n], temps[n-1])
            if n < len(temps) - 1:
                P_acc += self.R_acc(temps[n + 1], temps[n])
            print("     Replica {:3d}:  {:8.2f}  {:10.4e}  {:10.4e} {:8.3f} % Estimated Swap Prob.".format(n+1, temps[n], mean, sigmas[n], P_acc*100))
            ax.plot(x_vals, normal(x_vals, mean, sigmas[n]))
        #fig.savefig(outfile)
        #plt.show()
        plt.close()

def print_title():
    print(" ---------------------------------------------")
    print("          REMD Temperature Assigner           ")
    print(" ---------------------------------------------")
    print("")

def print_error(message, close_program=True):
    print("\n\n  xxxxxxxxxxxx   ERROR   xxxxxxxxxxxx")
    print(message)
    if close_program:
        exit()

def plot_exchange_rate(xvg_file):
    data = np.loadtxt(xvg_file, dtype=float)
    reps = data[:, 1:].astype(int)

    print(" Found {:d} replicas".format(len(reps)))
    print(" Computing exchange probabilities...")
    n_used = 0
    sums = np.zeros_like(reps[0])
    for n, step in enumerate(reps):
        if n == 0: continue
        if n == 2500: break
        diff = step - reps[n -1]
        sums[diff != 0] += 1
        n_used += 1
    print(" ...Done!\n")

    means = sums/n_used
    print(' ----------------------------------')
    for n, mean in enumerate(means):
        print('     Replica {:5d}:  {:8.3f} %'.format(n+1, mean*100))
    print(' ----------------------------------')


    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)) + 1, means*100)
    ax.set_xlabel("Replica No.")
    ax.set_ylabel("Probability (%)")
    plt.show()




if __name__ == "__main__":
    print_title()
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', help='Input data file', required=True, type=str)
    parser.add_argument('-T_min', help='Lowest temperature to start replicas at', type=float, default=300)
    parser.add_argument('-mode', help='Type of calculation to run\n stuff asdfasdf asdf asdfasdf asdf \\nasdfa sdfasd fasdfasdfasd asdfasd fasdf asdfasdf asdfa sdfas', type=int, default=1, choices=[1,2,3])
    parser.add_argument('-T_max', help='Highest temperature to use', type=float)
    parser.add_argument('-P_acc', help='Acceptance probability between each replica ', type=float)
    parser.add_argument('-n_rep', help='Number of replicas to use', type=int)
    parser.add_argument('-plot_exchange', help='polt the exchange rate from a GROMACS .xvg file', action='store_true')
    parser.add_argument('-plot_distros', help='polt the energy distributions from the energy data file', action='store_true')
    args = parser.parse_args()

    #   plot calls do not require any more checking of parameters
    #   and exit upon completion.
    if args.plot_exchange:
        plot_exchange_rate(args.data)
        exit()
    if args.plot_distros:
        data = np.loadtxt(args.data).T
        solver = REMDSolver(data[0], data[1], data[2])
        # solver._plot_poly_fits()
        # exit()
        solver.plot_gaussians()
        exit()

    #   no plots were called, run analysis modes
    if args.mode == 1:
        if args.T_max is None or args.P_acc is None:
            raise print_error(' For Mode 1, both -T_max and -P_acc arguments are required')
    if args.mode == 2:
        if args.P_acc is None or args.n_rep is None:
            raise print_error(' For Mode 2, both -P_acc and -n_rep arguments are required')
    if args.mode == 3:
        if args.T_max is None or args.n_rep is None:
            raise print_error(' For Mode 2, both -T_max and -n_rep arguments are required')
    if args.mode in [1, 3]:
        if args.T_min >= args.T_max:
            raise print_error(" -T_max must be greater than -T_min (default = 300 K)")

    #   import energy data with columns (temperature, mean_energy, stdev_energy)
    data = np.loadtxt(args.data).T
    solver = REMDSolver(data[0], data[1], data[2])

    if args.mode == 1:
        print(" Running in Mode 1: Fixed temperature range and P_acc")
        print(" Requested temperature range: {:.2f} K - {:.2f} K ".format(args.T_min, args.T_max))
        print(" Requested swapping probability: {:.4f}".format(args.P_acc))

        replicas = solver.calc_replicas(args.T_min, args.P_acc, T_max=args.T_max)
        solver.print_results(replicas, args.P_acc)

    elif args.mode == 2:
        print(" Running in Mode 2: Fixed number of replicas and P_acc")
        print(" Minimum temperature: {:.2f}".format(args.T_min))
        print(" Number of replicas: {:d}".format(args.n_rep))
        print(" Requested swapping probability: {:.4f}".format(args.P_acc))

        replicas = solver.calc_replicas(args.T_min, args.P_acc, n_replicas=args.n_rep)
        solver.print_results(replicas, args.P_acc)

    elif args.mode == 3:
        print(" Running in Mode 3: Fixed number of replicas and temperature range")
        print(" Requested temperature range: {:.2f} K - {:.2f} K ".format(args.T_min, args.T_max))
        print(" Number of replicas: {:d}".format(args.n_rep))

        replicas, optimum_r_acc = solver.calc_replicas_optimized(args.T_min, args.T_max, args.n_rep)
        solver.print_results(replicas, optimum_r_acc)

