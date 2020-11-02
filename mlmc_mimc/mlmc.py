from abc import ABC, abstractmethod
import numpy as np
import numpy
import torch
import torch.nn as nn
import math
from scipy.stats import linregress
import sys
import time
import matplotlib.pyplot as plt
import pickle

class WeakConvergenceFailure(Exception):
    pass


class MLMC(ABC):
    """Base class for MLMC

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

        self.alpha = None
        self.beta = None
        self.gamma = None
    
        self.avg_Pf_Pc = None
        self.avg_Pf = None
        self.var_Pf_Pc = None
        self.var_Pf = None
        self.N_samples_convergence = None

        self.target = None

    def _mlmc(self,eps):
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
        Nl : np.ndarray
            Number of samples per level, ie number of samples of (P_l - P_{l-1})
            
        """
        
        # sanity checks
        if self.Lmax < self.Lmin:
            raise ValueError("Need Lmax >= Lmin")
        if any([eps<=0, self.alpha<0, self.beta<0]):
            raise ValueError("Need N0>0, eps>0, gamma>0, alpha_0>0, beta_0>0")
        
        # initialisation
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        theta = 0.5
        L = self.Lmin
        Nl = np.zeros(L+1) # this will store number of MC samples per level
        suml = np.zeros([2,L+1]) # first row:   second row:
        dNl = self.N0 * np.ones(L+1) # this will store the number of remaining samples per level to generate to achieve target variance
        
        while sum(dNl)>0:
            # update sample sums
            for l in range(0,L+1):
                if dNl[l]>0:
                    sums_level_l = self.mlmc_fn(l, int(dNl[l])) # \sum (Y_l-Y_{l-1)}
                    Nl[l] += dNl[l]
                    suml[0,l] += sums_level_l[0]
                    suml[1,l] += sums_level_l[1]
        
            # compute the absolute average and variance **at each level**, necessary to calculate additional samples
            ml = np.abs(suml[0,:]/Nl)
            Vl = np.maximum(0, suml[1,:]/Nl - ml**2)
            if self.N_samples_convergence:
                for idx, _ in enumerate(Nl):
                    if Nl[idx]<self.N_samples_convergence and idx>1:
                        Vl[idx] = Vl[idx-1]/2.0**beta
                    elif Nl[idx]<self.N_samples_convergence and idx==1:
                        Vl[idx] = self.var_Pf_Pc[1]
                    elif Nl[idx]<self.N_samples_convergence and idx==0:
                        Vl[idx] = self.var_Pf[0]
                    else:
                        pass
            # set optimal number of additional samples (dNl) in order to minimise total cost for a fixed variance
            Cl = np.arange(L+1)
            for idx, nl in enumerate(Cl):
                Cl[idx] = self.get_cost(idx)
            
            Ns = np.ceil(np.sqrt(Vl/Cl) * sum(np.sqrt(Vl*Cl)) / ((1-theta)*eps**2)) # check http://people.maths.ox.ac.uk/~gilesm/files/acta15.pdf page 4
            dNl = np.maximum(0, Ns-Nl)

            # if (almost) converged, i.e. if there are very 
            # few samples to add, 
            # estimate remaining error and decide whether a new level is required
            converged = (dNl < 0.01 * Nl)
            if all(converged):
                
                #P = sum(suml[0,:]/Nl)
                #if self.get_weak_error_from_target(P) > np.sqrt(1/2) * eps:
                if L==0 and self.target is not None:
                    weak_error = abs(sum(suml[0,:]/Nl)-self.target)
                else:
                    weak_error = self.get_weak_error(ml)
                    
                #if self.get_weak_error(ml) > np.sqrt(1/2) * eps:
                if weak_error > np.sqrt(1/2) * eps:
                    if L == self.Lmax:
                        raise WeakConvergenceFailure("Failed to achieve weak convergence")
                    else:
                        L = L+1
                        if L>1:
                            Vl = np.append(Vl, Vl[-1] / 2.0**beta) # MLMC theorem, variance is O(2^{-beta})
                        else:
                            Vl = np.append(Vl, self.var_Pf_Pc[1])
                        Nl = np.append(Nl, 0.0)
                        suml = np.column_stack([suml, [0,0]])
                        
                        # we decide how many samples need to be added in the new level
                        Cl = np.arange(L+1)
                        for idx, nl in enumerate(Cl):
                            Cl[idx] = self.get_cost(idx)

                        Ns = np.ceil(np.sqrt(Vl/Cl) * sum(np.sqrt(Vl*Cl)) / ((1-theta)*eps**2)) # check http://people.maths.ox.ac.uk/~gilesm/files/acta15.pdf page 4
                        dNl = np.maximum(0, Ns-Nl)
                else:
                    pass
        
        # finally, evaluate the multi-level estimator
        P = sum(suml[0,:]/Nl)
        Ns = np.ceil(np.sqrt(Vl/Cl) * sum(np.sqrt(Vl*Cl)) / ((1-theta)*eps**2)) # check http://people.maths.ox.ac.uk/~gilesm/files/acta15.pdf page 4
        return P, Nl


    @abstractmethod
    def get_cost(self, l):
        """
        """
        ...
    
    def estimate_alpha_beta_gamma(self, L, N, logfile):
        """Returns alpha, beta, gamma
        """
        
        format_string = "{:<10}{:<15}{:<15}{:<15}{:<15}{:<15}"
        self.write(logfile, "**********************************************************\n")
        self.write(logfile, "*** Convergence tests, kurtosis, telescoping sum check ***\n")
        self.write(logfile, "**********************************************************\n\n")
        #write(logfile, "\n level    ave(Pf-Pc)    ave(Pf)   var(Pf-Pc)    var(Pf)    kurtosis     check") 
        self.write(logfile, format_string.format("level","avg(Pf-Pc)","avg(Pf)","var(Pf-Pc)","var(Pf)","check"))
        self.write(logfile, "\n----------------------------------------------------------------------------\n")

        avg_Pf_Pc = []
        avg_Pf = []
        var_Pf_Pc = []
        var_Pf = []
        kur1 = []
        chk1 = []
        cost = []

        for l in range(0, L+1):
            init_time = time.time()
            sums_level_l = self.mlmc_fn(l, N)
            end_time = time.time()
            #cost.append(end_time - init_time)
            cost.append(self.get_cost(l))
            sums_level_l = sums_level_l/N
            avg_Pf_Pc.append(sums_level_l[0])
            avg_Pf.append(sums_level_l[4])
            var_Pf_Pc.append(sums_level_l[1]-sums_level_l[0]**2)
            var_Pf.append(sums_level_l[5]-sums_level_l[4]**2)
                

            if l==0:
                check = 0
            else:
                check = abs(avg_Pf_Pc[l] + avg_Pf[l] - avg_Pf[l-1])
                check = check / (3*(math.sqrt(var_Pf_Pc[l]) + math.sqrt(var_Pf[l-1]) +\
                         math.sqrt(var_Pf[l]) )/math.sqrt(N) )
            chk1.append(check)
            format_string = "{:<10}{:<15.4e}{:<15.4e}{:<15.4e}{:<15.4e}{:<15.4e}\n"
            self.write(logfile, format_string.format(l,avg_Pf_Pc[-1], avg_Pf[-1], var_Pf_Pc[-1], var_Pf[-1], check))

        L1 = int(np.ceil(0.4*L))
        L2 = L+1
        
        x = np.arange(L1+1, L2+1)
        
        # we approximate alpha, beta, gamma, see Theorem 1
        # in http://people.maths.ox.ac.uk/~gilesm/files/acta15.pdf
        alpha, _, _, _, _ = linregress(x, np.log2(np.abs(avg_Pf_Pc[L1:L2]))) # see Theorem 1 in http://people.maths.ox.ac.uk/~gilesm/files/acta15.pdf
        alpha = -alpha

        beta, _,_, _, _  = linregress(x, np.log2(np.abs(var_Pf_Pc[L1:L2])))
        beta = -beta 

        #gamma = np.log2(self.M) #
        gamma = np.log2(cost[-1]/cost[-2])

        self.write(logfile, "\n******************************************************\n")
        self.write(logfile, "*** Linear regression estimates of MLMC parameters ***\n")
        self.write(logfile, "******************************************************\n")
        self.write(logfile, "alpha = {:.4f}\n".format(alpha))
        self.write(logfile, "beta = {:.4f}\n".format(beta))
        self.write(logfile, "gamma = {:.4f}\n".format(gamma))
        
        # estimates of alpha, beta, gamma
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        
        # estimates of averages and variances of different levels
        self.avg_Pf_Pc = avg_Pf_Pc
        self.avg_Pf = avg_Pf
        self.var_Pf_Pc = var_Pf_Pc
        self.var_Pf = var_Pf


        return alpha, beta, gamma


    
    def estimate_V_N_C_unbiased_estimor(self, eps, levels):

        # initialisation
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        theta = 0.5
        L = self.Lmin + 1
        Nl = np.zeros(L+1) # this will store number of MC samples per level
        suml = np.zeros([2,L+1]) # first row:   second row:
        dNl = 100 * np.ones(L+1) # this will store the number of remaining samples per level to generate to achieve target variance
        
        for l in range(0,L+1):
            if dNl[l]>0:
                sums_level_l = self.mlmc_fn(l, int(dNl[l])) # \sum (Y_l-Y_{l-1)}
                Nl[l] += dNl[l]
                suml[0,l] += sums_level_l[0]
                suml[1,l] += sums_level_l[1]
    
        # compute the absolute average and variance **at each level**, necessary to calculate additional samples
        ml = np.abs(suml[0,:]/Nl)
        Vl = np.maximum(0, suml[1,:]/Nl - ml**2)
        
        
        Cl = np.zeros(levels+1)
        for idx, _ in enumerate(Cl):
            Cl[idx] = self.get_cost(idx)
        for idx in range(len(Vl), levels+1):
            Vl = np.append(Vl, Vl[-1]/2**beta)
    
        Ns = np.ceil(np.sqrt(Vl/Cl) * sum(np.sqrt(Vl*Cl)) / ((1-theta)*eps**2)) # check http://people.maths.ox.ac.uk/~gilesm/files/acta15.pdf page 4
        cost_MLMC = self.get_cost_MLMC(eps, Ns)
        return Vl, Ns, cost_MLMC
        
    
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
        
        styles = ['o--', 'x--', 'd--', '*--', 's--', 't--']
        for idx, (eps, Nl) in enumerate(zip(Eps, Nl_list)):
            ax[1,0].plot(Nl, styles[idx], label=str(eps))
        
        ax[1,0].legend(loc="upper right")
        ax[1,0].set_yscale('log')
        ax[1,0].set_xlabel("level $l$")
        ax[1,0].set_ylabel("$N_l$")
        
        Eps = np.array(Eps)
        ax[1,1].plot(Eps, Eps**2 * mlmc_cost, '*--', label='MLMC')
        ax[1,1].plot(Eps, Eps**2 * std_cost, '*--', label='std MC')
        ax[1,1].set_yscale('log')
        ax[1,1].set_xscale('log')
        ax[1,1].set_xlabel("accuracy $\epsilon$")
        ax[1,1].set_ylabel("$\epsilon^2$ cost")
        ax[1,1].legend(loc="upper right")
        fig.tight_layout()


        fig.savefig(filename)


    def save_convergence_test(self, filename):
        output = {'avg_Pf_Pc':self.avg_Pf_Pc,
                'avg_Pf':self.avg_Pf,
                'var_Pf':self.var_Pf,
                'var_Pf_Pc':self.var_Pf_Pc,
                'alpha':self.alpha,
                'beta':self.beta,
                'gamma':self.gamma}

        with open(filename,"wb") as f:
            pickle.dump(output, f)
    
    
    def save(self, Eps, Nl_list, mlmc_cost, std_cost, filename):
        output = {'Eps':Eps,
                'Nl_list':Nl_list,
                'mlmc_cost':mlmc_cost,
                'std_cost':std_cost,
                'avg_Pf_Pc':self.avg_Pf_Pc,
                'avg_Pf':self.avg_Pf,
                'var_Pf':self.var_Pf,
                'var_Pf_Pc':self.var_Pf_Pc,
                'alpha':self.alpha,
                'beta':self.beta,
                'gamma':self.gamma}

        with open(filename,"wb") as f:
            pickle.dump(output, f)

    
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
    def get_weak_error_from_target(self, P):
        """Test weak convergence

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





