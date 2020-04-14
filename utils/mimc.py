from abc import ABC, abstractmethod
import numpy as np
import numpy
import torch
import torch.nn as nn
import math
from scipy.stats import linregress
import sys
import time
from itertools import product
from sklearn.linear_model import LinearRegression

class WeakConvergenceFailure(Exception):
    pass


class MIMC(ABC):
    """Base class for MIMC

    """

    def __init__(self, Lmin, Lmax, N0):
        """Multi-level Monte Carlo estimation
        Parameters:
            Lmin : int
                minimum level of refinement >=2
            Lmax : int
                maximum level of refinement >= Lmin
            N0 : int
                initial number of samples > 0

        """
        self.Lmin = Lmin
        self.Lmax = Lmax
        self.N0 = N0

        self.M = None # refinement factor

        self.alpha = LinearRegression()
        self.beta = LinearRegression()
        self.gamma = LinearRegression()
    
        self.avg_Pf_Pc = None
        self.avg_Pf = None
        self.var_Pf_Pc = None
        self.var_Pf = None

    def _mimc(self,eps):
        """Multi-level Monte Carlo estimation achieving 
        MSE = Bias^2 + Var < eps^2, and minimising total cost for fixed variance <= 0.5 * eps^2
        
        
        Note
        ----
        Since MSE = Bias^2 + Var<eps^2,
        if at Lmax there isn't weak convergence (that is, Bias > 1/sqrt(2) * eps
        then the function returns an error

        Parameters
        ----------
        eps : float
            desired accuracy (rms error) > 0

        Returns
        -------
        P : float
            MLMC estimator
        Nl : np.ndarray of size (L,L)
            Number of samples per multi-index level
            
        """
        
        # sanity checks
        if self.Lmax < self.Lmin:
            raise ValueError("Need Lmax >= Lmin")
        if any([eps<=0, self.alpha<0, self.beta<0]):
            raise ValueError("Need N0>0, eps>0, gamma>0, alpha_0>0, beta_0>0")
        

        theta = 0.5
        L = self.Lmin
        Nl = np.zeros([L+1, L+1]) # this will store number of MC samples per level
        suml = np.zeros([2,L+1,L+1]) # first matrix:   second matrix:
        dNl = self.N0 * np.ones_like(Nl) # this will store the number of remaining samples per level to generate to achieve target variance
        Cl = np.zeros_like(Nl)

        while sum(dNl)>0:
            # update sample sums
            for l1,l2 in product(range(L+1), range(L+1)):
                if dNl[l1,l2]>0:
                    sums_level_l = self.mlmc_fn(l1,l2, int(dNl[l])) # \sum (Y_l-Y_{l-1)}
                    Nl[l1,l2] += dNl[l1,l2]
                    suml[0,l1,l2] += sums_level_l[0]
                    suml[1,l1,l2] += sums_level_l[1]
        
            # compute the absolute average and variance **at each level**, necessary to calculate additional samples
            ml = np.abs(suml[0,:,:]/Nl)
            Vl = np.maximum(0, suml[1,:,:]/Nl - ml**2)
            
            
            # set optimal number of additional samples (dNl) in order to minimise total cost for a fixed variance
            for l1,l2 in product(range(L+1, range(L+1))):
                Cl[l1,l2] = 2**(self.gamma.predict(np.array([[l1,l2]])))
            
            Ns = np.ceil(np.sqrt(Vl/Cl) * sum(np.sqrt(Vl*Cl)) / ((1-theta)*eps**2)) # check http://people.maths.ox.ac.uk/~gilesm/files/acta15.pdf page 4
            dNl = np.maximum(0, Ns-Nl)

            # if (almost) converged, i.e. if there are very 
            # few samples to add, 
            # estimate remaining error and decide whether a new level is required
            converged = (dNl < 0.01 * Nl)
            if all(converged):
                if self.get_weak_error(ml) > np.sqrt(1/2) * eps:
                    if L == self.Lmax:
                        raise WeakConvergenceFailure("Failed to achieve weak convergence")
                    else:
                        L = L+1
                        Vl = np.zeros([L+1,L+1])
                        for l1, l2 in product(range(L+1),range(L+1)):
                            Vl[l1,l2] = 2**(self.beta.predict(np.array([[l1,l2]])))
                        Nl = np.pad(Nl, ((0,1),(0,1)), constant_values = 0.0)
                        suml = np.pad(suml, ((0,0),(0,1),(0,1)), constant_values=0.0)
                        
                        # we decide how many samples need to be added in the new level
                        for l1,l2 in product(range(L+1, range(L+1))):
                            Cl[l1,l2] = 2**(self.gamma.predict(np.array([[l1,l2]])))
                        
                        Ns = np.ceil(np.sqrt(Vl/Cl) * sum(np.sqrt(Vl*Cl)) / ((1-theta)*eps**2)) # check http://people.maths.ox.ac.uk/~gilesm/files/acta15.pdf page 4
                        dNl = np.maximum(0, Ns-Nl)
                else:
                    pass
        
        # finally, evaluate the multi-level estimator
        P = sum(suml[0,:]/Nl)
        return P, Nl


    
    def estimate_alpha_beta_gamma(self, L, N, logfile):
        """Returns alpha, beta, gamma
        """
        
        format_string = "{:<5}{:<5}{:<15}{:<15}{:<15}{:<15}"
        self.write(logfile, "**********************************************************\n")
        self.write(logfile, "*** Convergence tests, kurtosis, telescoping sum check ***\n")
        self.write(logfile, "**********************************************************\n\n")
        #write(logfile, "\n level    ave(Pf-Pc)    ave(Pf)   var(Pf-Pc)    var(Pf)    kurtosis     check") 
        self.write(logfile, format_string.format("level1","level2","avg(Pf-Pc)","avg(Pf)","var(Pf-Pc)","var(Pf)"))
        self.write(logfile, "\n----------------------------------------------------------------------------\n")

        avg_Pf_Pc = np.zeros([L+1,L+1])
        avg_Pf = np.zeros([L+1,L+1])
        var_Pf_Pc = np.zerps([L+1,L+1])
        var_Pf = np.zeros([L+1,L+1])
        kur1 = []
        #chk1 = np.zeros([L+1,L+1])
        cost = np.zeros([L+1,L+1])

        for l1,l2 in product(range(L+1), range(L+1)):
            init_time = time.time()
            sums_level_l = self.mlmc_fn(l, N)
            end_time = time.time()
            #cost.append(end_time - init_time)
            cost[l1,l2]= end_time - init_time
            sums_level_l = sums_level_l/N
            avg_Pf_Pc[l1,l2] = sums_level_l[0]
            avg_Pf[l1,l2] = sums_level_l[4]
            var_Pf_Pc[l1,l2] = (sums_level_l[1]-sums_level_l[0]**2)
            var_Pf[l1,l2] = (sums_level_l[5]-sums_level_l[4]**2)
                

            #if (l1,l2)==(0,0):
            #    check = 0
            #else:
            #    check = abs(avg_Pf_Pc[l] + avg_Pf[l] - avg_Pf[l-1])
            #    check = check / (3*(math.sqrt(var_Pf_Pc[l]) + math.sqrt(var_Pf[l-1]) +\
            #             math.sqrt(var_Pf[l]) )/math.sqrt(N) )
            #chk1.append(check)
            format_string = "{:<5}{:,5}{:<15.4e}{:<15.4e}{:<15.4e}{:<15.4e}\n"
            self.write(logfile, format_string.format(l1,l2,avg_Pf_Pc[l1,l2], avg_Pf[l1,l2], var_Pf_Pc[l1,l2], var_Pf[l1,l2]))

        L1 = int(np.ceil(0.4*L))
        L2 = L+1
        
        # we approximate alpha, beta, gamma, see Theorem 2
        # in http://people.maths.ox.ac.uk/~gilesm/files/acta15.pdf
        x = list(product(range(L1,L2), product(range(L1,L2))))
        xx = np.array(x)
        
        #alpha
        y = np.array([avg_Pf_Pc[idx] for idx in x]).reshape(-1,1)
        self.alpha.fit(xx,np.log2(np.abs(y)))

        #beta
        y = np.array([var_Pf_Pc[idx] for idx in x]).reshape(-1,1)
        self.beta.fit(xx,np.log2(np.abs(y)))
        
        # gamma
        y = np.array([cost[idx] for idx in x]).reshape(-1,1)
        self.gamma.fit(xx, np.log2(cost))
        
        
        self.write(logfile, "\n******************************************************\n")
        self.write(logfile, "*** Linear regression estimates of MLMC parameters ***\n")
        self.write(logfile, "******************************************************\n")
        self.write(logfile, "alpha intercept = {:.4f}\n".format(alpha.intercept_))
        self.write(logfile "alpha weights = {:.4f},{:.4f}\n\n").format(alpha.coef_[0], alpha.coef_[1])

        self.write(logfile, "beta intercept = {:.4f}\n".format(beta.intercept_))
        self.write(logfile "beta weights = {:.4f},{:.4f}\n\n").format(beta.coef_[0], beta.coef_[1])
        
        self.write(logfile, "gamma intercept = {:.4f}\n".format(gamma.intercept_))
        self.write(logfile "gamma weights = {:.4f},{:.4f}\n\n").format(gamma.coef_[0], gamma.coef_[1])
        
        
        # estimates of averages and variances of different levels
        self.avg_Pf_Pc = avg_Pf_Pc
        self.avg_Pf = avg_Pf
        self.var_Pf_Pc = var_Pf_Pc
        self.var_Pf = var_Pf

        return 0


    def get_complexities(self,Eps, logfile):
        """Function that returns cost of MLMC and cost of standard Monte Carlo
        for a given epsilon

        Parameters
        ----------
        Eps : np.ndarray
            array of epsilons
        
        Returns
        -------
        Nl_list : List(np.ndarray)
            list of Nl for each epsilon
        mlmc_cost : np.ndarray
            MLMC cost for each eps in Eps
        std_cst : np.ndarray
            Monte Carlo cost for each eps in Eps

        """
        self.write(logfile, "\n");
        self.write(logfile, "***************************** \n");
        self.write(logfile, "*** MLMC complexity tests *** \n");
        self.write(logfile, "***************************** \n\n");
        format_string = "{:<10}{:<15}{:<15}{:<15}{}\n"
        self.write(logfile, format_string.format("eps","mlmc_cost","std_cost","savings", "N_l"))
        self.write(logfile, "-------------------------------------------------------------------- \n");

        mlmc_cost = np.zeros_like(Eps)
        std_cost = np.zeros_like(Eps)
        Nl_list = []
        for idx, eps in enumerate(Eps):
            P, Nl = self._mlmc(eps)
            Nl_list.append(Nl)
            mlmc_cost[idx] = self.get_cost_MLMC(eps, Nl)
            std_cost[idx] = self.get_cost_std_MC(eps, Nl)
            self.write(logfile, "{:<10.4f}{:<15.3e}{:<15.3e}{:<15.2f}".format(eps,mlmc_cost[idx],std_cost[idx],std_cost[idx]/mlmc_cost[idx]))
            self.write(logfile, " ".join(["%9d" % n for n in Nl]))
            self.write(logfile, "\n")

        return Nl_list, mlmc_cost, std_cost


    def plot(self, Eps, Nl_list, mlmc_cost, std_cost, filename):
        L = len(self.avg_Pf)
        l = np.arange(L)
        fig, ax = plt.subplots(nrows=2,ncols=2,sharex=False, sharey=False)
        ax[0,0].plot(l, np.log2(self.var_Pf), '*-', label='$P_l$')
        ax[0,0].plot(l[1:], np.log2(self.var_Pf_Pc[1:]), '*--', label='$P_l-P_{l-1}$')
        ax[0,0].set_xlabel('level $l$')
        ax[0,0].set_ylabel('$\mathrm{log}_2(\mathrm{variance})$')
        ax[0,0].legend(loc='lower left', fontsize='x-small')

        ax[0,1].plot(l, np.log2(self.avg_Pf), '*-', label='$P_l$')
        ax[0,1].plot(l[1:], np.log2(np.abs(self.avg_Pf_Pc[1:])), '*--', label='$P_l-P_{l-1}$')
        ax[0,1].set_xlabel('level $l$')
        ax[0,1].set_ylabel('$\mathrm{log}_2(|\mathrm{mean}|)$')
        ax[0,1].legend(loc='lower left', fontsize='x-small')
        
        for eps, Nl in zip(Eps, Nl_list):
            ax[1,0].plot(Nl, label=str(eps))
        
        ax[1,0].legend()
        ax[1,0].set_yscale('log')
        
        Eps = np.array(Eps)
        ax[1,1].plot(Eps, Eps**2 * mlmc_cost, '*--', label='MLMC')
        ax[1,1].plot(Eps, Eps**2 * mtd_cost, '*--', label='std MC')
        ax[1,1].set_yscale('log')
        ax[1,1].set_xscale('log')

        fig.savefig("filename")


    
    
    def get_alpha(self):
        return self.alpha
    
    def get_beta(self):
        return self.beta

    def get_gamma(self):
        return self.gamma
        

    @abstractmethod
    def get_cost_MLMC(self, eps, Nl):
        """Get the cost of MLMC
        """
        ...
    
    @abstractmethod
    def get_cost_std_MC(self, eps, Nl):
        """get cost of standard Monte Carlo

        """
        ...
    
    @abstractmethod
    def get_weak_error(self, ml):
        """Test weak convergence

        """
        ...
    
    @abstractmethod
    def mlmc_fn(self, L):
        """MLMC user low-level routine: if we want to approximate E[F(X)] 
        then we aim to create the Multi-Level estimator
        m := 1/N_0 Y_0 + 1/N_1 (Y_1) + ... + 1/N_L Y_L
        where each Y_l = \sum F(X_l^i) - F(X_{l-1}^i)
        This function will calculate Y_l and higher order moments

        Parameters
        ----------
        L : int
            level

        Returns
        -------
        sums_level_l : np.ndarray of size (6)
            sums_level_l[0] := \sum F(X_L^i) - F(X_{L-1}^i)
            sums_level_l[1] := \sum (F(X_L^i) - F(X_{L-1}^i))^2
            sums_level_l[2] := \sum (F(X_L^i) - F(X_{L-1}^i))^3
            sums_level_l[3] := \sum (F(X_L^i) - F(X_{L-1}^i))^4
            sums_level_l[4] := \sum F(X_L^i)
            sums_level_l[5] := \sum (F(X_L^i)^2) 
            
        """
        ...

    @abstractmethod
    def Func(self, X):
        """Function for which we want to approximate E[F(X)] 
        with X r.v. from which we sample using MLMC
        """
        ...

    @staticmethod
    def write(logfile,msg):
        """
        Write to both sys.stdout and to a logfile
        """
        with open(logfile, "a") as f:
            f.write(msg)
        sys.stdout.write(msg)





