import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
'''
File to store the classes used in the project for the streamlit app
Maybe use a super class since all the classes are related to the same starting point
'''

class EfficientFrontier:
    '''
    This class implements the Markowitz Efficient Frontier.
    '''
    def __init__(self, prices, risk_free_rate=None, short=False):
        '''
        Initialize the EfficientFrontier object.
        Compute the expected returns, volatilities, correlation matrix and covariance matrix.
        '''
        self.prices = prices  # DataFrame with historical prices
        self.risk_free_rate = risk_free_rate  # Risk-free rate
        self.returns = prices.pct_change().dropna()  # Compute returns
        self.returns = self.returns[self.returns.sum(axis=1) != 0]  # Remove days with zero returns
        self.short = short  # Short-selling allowed or not

        # Compute returns and covariances for the risky assets
        self.mu = self.returns.mean().values * 252  # Annualized expected returns
        self.vol = self.returns.std().values * np.sqrt(252)  # Annualized volatilities
        self.correl_matrix = self.returns.corr().values  # Correlation matrix
        self.covmat = self.vol.reshape(1, -1) * self.correl_matrix * self.vol.reshape(-1, 1)  # Covariance matrix
        self.n = self.mu.shape[0]
        self.x0 = np.ones(self.n) / self.n

        # If a risk-free rate is provided, modify the mu, vol, and covmat
        if risk_free_rate is not None:
            self.mu_mod = np.append(self.mu, self.risk_free_rate)  # Append risk-free rate to the expected returns
            self.vol_mod = np.append(self.vol, 0)
            self.covmat_mod = np.zeros((self.n+1, self.n+1))
            self.covmat_mod[:self.n, :self.n] = self.covmat
            self.x0_mod = np.ones(self.n+1) / (self.n+1)
            self.SR = (self.mu - self.risk_free_rate) / self.vol
        else:
            self.mu_mod = None
            self.vol_mod = None
            self.covmat_mod = None
            self.x0_mod = None
            self.SR = self.mu / self.vol


    def QP(self, x, sigma, mu, gamma):
        '''
        Standard QP problem function.
        Maximize utility by minimizing variance for a given return target (gamma).
        '''
        return 0.5 * x.T @ sigma @ x - gamma * x.T @ mu

    def efficient_frontier(self, n, x0, covmat, mu, gam):
        '''
        Compute the efficient frontier for a given gamma.
        We need to give the constraints and bounds to the optimizer.
        '''
        constraints = LinearConstraint(np.ones(n), lb=1, ub=1)  # Adjust constraints
        if self.short:
            bounds = [(None, None) for _ in range(n)]  # Short-selling allowed
        else:
            bounds = [(0, 1) for _ in range(n)]  # No short-selling allowed

        res = minimize(self.QP, x0, args=(covmat, mu, gam),
                    bounds=bounds, constraints=constraints)

        optimized_weights = res.x
        mu_optimized = optimized_weights @ mu
        vol_optimized = np.sqrt(optimized_weights @ covmat @ optimized_weights)

        return mu_optimized, vol_optimized, optimized_weights
    
    def get_efficient_frontier(self):
        '''
        Plot the efficient frontier.
        '''
        gammas = np.linspace(-5, 5, 500)  # Range of gamma values for the efficient frontier
        # Compute efficient frontier for risky assets
        frontier1 = [self.efficient_frontier(self.n, self.x0, self.covmat, self.mu, g) for g in gammas]
        # If risk-free rate is provided, compute and plot the extended frontier
        if self.risk_free_rate is not None:
            maxvalue = 1
            it = 100
            gammas2 = np.linspace(0, maxvalue, it)
            frontier2 = [self.efficient_frontier(self.n + 1, self.x0_mod, self.covmat_mod, self.mu_mod, g) for g in gammas2]

            # Compute the tangency portfolio by combining two arbitrary portfolios
            x1_pos = round(0.25/maxvalue * (it-1), 0) # Position of the portfolio with gamma = 0.25
            x2_pos = round(0.5/maxvalue * (it-1), 0) # Position of the portfolio with gamma = 0.5
            x1 = frontier2[int(x1_pos)][2] # Weights of the portfolio with gamma = 0.25
            x2 = frontier2[int(x2_pos)][2] # Weights of the portfolio with gamma = 0.5

            alpha_star = x1[-1] / (x1[-1] - x2[-1])
            tangency_pf = (1-alpha_star) * x1 + alpha_star * x2

            tangency_mu = tangency_pf @ self.mu_mod
            tangency_vol = np.sqrt(tangency_pf @ self.covmat_mod @ tangency_pf)

        if self.risk_free_rate is not None:
            return frontier1, frontier2, tangency_pf, tangency_mu, tangency_vol
        else:
            return frontier1
    def get_portfolio_based_on_gamma(self,gamma):
        '''
        Compute the minimum variance portfolio.
        '''
        port = self.efficient_frontier(self.n, self.x0, self.covmat, self.mu, gamma)


        return port

class Portfolio: 
    # Equal Risk Contribution Portfolio maybe use only one class and change the name to portfolio 
    def __init__(self, prices, risk_free_rate=None, short=False):
        ''' Initialize the Equal Risk Contribution Portfolio object.'''
        self.prices = prices  # DataFrame with historical prices
        self.risk_free_rate = risk_free_rate  # Risk-free rate
        self.returns = prices.pct_change().dropna()  # Compute returns
        self.returns = self.returns[self.returns.sum(axis=1) != 0]  # Remove days with zero returns
        self.short = short  # Short-selling allowed or not

        # Compute returns and covariances for the risky assets
        self.mu = self.returns.mean().values * 252  # Annualized expected returns
        self.vol = self.returns.std().values * np.sqrt(252)  # Annualized volatilities
        self.correl_matrix = self.returns.corr().values  # Correlation matrix
        self.covmat = self.vol.reshape(1, -1) * self.correl_matrix * self.vol.reshape(-1, 1)  # Covariance matrix
        self.n = self.mu.shape[0]
        self.x0 = np.ones(self.n) / self.n

        # If a risk-free rate is provided, modify the mu, vol, and covmat
        if risk_free_rate is not None:
            self.mu_mod = np.append(self.mu, self.risk_free_rate)  # Append risk-free rate to the expected returns
            self.vol_mod = np.append(self.vol, 0)
            self.covmat_mod = np.zeros((self.n+1, self.n+1))
            self.covmat_mod[:self.n, :self.n] = self.covmat
            self.x0_mod = np.ones(self.n+1) / (self.n+1)
            self.SR = (self.mu - self.risk_free_rate) / self.vol
        else:
            self.mu_mod = None
            self.vol_mod = None
            self.covmat_mod = None
            self.x0_mod = None
            self.SR = self.mu / self.vol
    
    def EW(self):
        '''Compute the equally weighted portfolio.'''
        return np.ones(self.n) / self.n
    
    def ARC(self, x, sigma):
        '''Compute the risk contribution of each asset.'''
        return np.dot(sigma, x) * x / np.sqrt(np.dot(x, np.dot(sigma, x)))

    def RRC(self, x, sigma):
        '''Compute the average risk contribution of the portfolio.'''
        return np.dot(sigma, x) * x / np.dot(x, np.dot(sigma, x))

    def MRC(self, x, sigma):
        '''Compute the marginal risk contribution of each asset.'''
        return sigma @ x / np.sqrt(np.dot(x, np.dot(sigma, x)))

    def QP(self, x, sigma):
        '''Standard QP problem function.'''
        return 0.5 * x.T @ sigma @ x 

    def ERC(self):
        '''Compute the equal risk contribution portfolio.'''
        n = self.prices.shape[1]
        logy = lambda y: np.sum(np.log(y))
        constraints = [LinearConstraint(np.eye(self.x0.shape[0]), lb = 0, ub = 1), NonlinearConstraint(logy, lb = -n * np.log(n) - 2, ub=np.inf)]
        res = minimize(self.QP, self.x0, args=(self.covmat), constraints=constraints) 
        optimized_weights_erc = res.x / np.sum(res.x)

        return optimized_weights_erc

    def DR(self,x,vol,sigma):
        '''Compute the diversification ratio of the portfolio.'''
        return -np.log(x.T@vol / np.sqrt(x @ sigma @ x))
    def MDP(self):
        '''Compute the maximum diversification portfolio.'''
        constraints = [LinearConstraint(np.ones(self.x0.shape[0]), lb = 1, ub = 1)] #no long only see lecture6 for the reasons
        res = minimize(self.DR, self.x0, args=(self.vol,self.covmat), constraints=constraints)
        optimized_weights_mdp = res.x


        return optimized_weights_mdp
    
class BlackLitterman:
    def __init__(self, prices, risk_free_rate, short=False):
        ''' Initialize the Black-Litterman object.'''
        self.prices = prices  # DataFrame with historical prices
        self.risk_free_rate = risk_free_rate  # Risk-free rate
        self.returns = prices.pct_change().dropna()  # Compute returns
        self.returns = self.returns[self.returns.sum(axis=1) != 0]  # Remove days with zero returns
        self.short = short  # Short-selling allowed or not

        # Compute returns and covariances for the risky assets
        self.mu = self.returns.mean().values * 252  # Annualized expected returns
        self.vol = self.returns.std().values * np.sqrt(252)  # Annualized volatilities
        self.correl_matrix = self.returns.corr().values  # Correlation matrix
        self.covmat = self.vol.reshape(1, -1) * self.correl_matrix * self.vol.reshape(-1, 1)  # Covariance matrix
        self.n = self.mu.shape[0]
        self.x0 = np.ones(self.n) / self.n

        # If a risk-free rate is provided, modify the mu, vol, and covmat
        
        self.mu_mod = np.append(self.mu, self.risk_free_rate)  # Append risk-free rate to the expected returns
        self.vol_mod = np.append(self.vol, 0)
        self.covmat_mod = np.zeros((self.n+1, self.n+1))
        self.covmat_mod[:self.n, :self.n] = self.covmat
        self.x0_mod = np.ones(self.n+1) / (self.n+1)
        self.SR = (((self.mu - self.risk_free_rate) / self.vol)@ np.ones(self.n)) / self.n


        self.implied_mu = self.risk_free_rate + self.SR * (self.covmat_mod @ self.x0_mod) / np.sqrt(self.x0_mod @ self.covmat_mod @ self.x0_mod)
        self.P = None
        self.Q = None
        self.Omega = None
        self.tau = 0.05
        self.implied_phi = self.SR/self.vol_mod
    def QP(x, sigma, mu, gamma ):
    
        v = 0.5 * x.T @ sigma @ x - gamma * x.T @ mu
    
        return v

    def add_views(self, P, Q, Omega):
        '''Add views to the Black-Litterman model.'''

        if self.P is None:
            self.P = P
            self.Q = Q
            self.Omega = Omega
        else:
            self.P = np.vstack((self.P, P))
            self.Q = np.append(self.Q, Q)
            self.Omega = sp.linalg.block_diag(self.Omega, Omega)
    def gamma_matrix(self,tau):
        return tau * self.covmat_mod

    def target_tau(self, x_start, target_vol):
        '''Compute the target tau for the Black-Litterman model.'''
        constraints = [LinearConstraint(np.ones(self.x0_mod.shape), ub = 1), 
               LinearConstraint(-np.ones(self.x0_mod.shape), ub = -1),
              LinearConstraint(np.eye(self.x0_mod.shape[0]), lb = 0)]
        mu_bar = self.implied_mu + (self.gamma_matrix(self.x0) @ self.P.T) @ np.linalg.inv(self.P @ self.gamma_matrix(self.x0) @ self.P.T + self.omega) @ (self.Q - self.P @ self.implied_mu)
        gam = 1/self.implied_phi
        res = minimize(self.QP, self.x0, args = (self.covmat_mod, mu_bar, gam) , options={'disp': False}, constraints = constraints)
        optimized_weights = res.x
        return np.sqrt((optimized_weights-self.x0_mod) @ self.covmat_mod @ (optimized_weights-self.x0_mod)) - target_vol

    def optimal_tau(self):
        opti_tau = fsolve(self.target_tau, x_start = 0.05, args = 0.02)[0]
        return opti_tau

    def BL(self,tau):
        '''Compute the Black-Litterman portfolio.'''
        if self.P is None:
            raise ValueError('No views added to the model.')
        
        constraints = [LinearConstraint(np.ones(self.x0_mod.shape), ub = 1), 
               LinearConstraint(-np.ones(self.x0_mod.shape), ub = -1),
              LinearConstraint(np.eye(self.x0_mod.shape[0]), lb = 0)]

        mu_bar = self.implied_mu + (self.gamma_matrix(tau) @ self.P.T) @ np.linalg.inv(self.P @ self.gamma_matrix(tau) @ self.P.T + self.Omega) @ (self.Q - self.P @ self.implied_mu)
        gam = 1/self.implied_phi
        res = minimize(self.QP, self.x0_mod, args = (self.covmat_mod, mu_bar, gam) , options={'disp': False}, constraints = constraints)
        optimized_weights_bl = res.x
    
        return optimized_weights_bl

def get_performance(prices, weights):
    '''Show the performance of the portfolio with a graph across time.'''
    # Compute the returns of the portfolio
    returns = prices.pct_change().dropna()
    returns = returns[returns.sum(axis=1) != 0]
    returns = returns @ weights
    returns = returns + 1
    returns = returns.cumprod()
    return returns
   

