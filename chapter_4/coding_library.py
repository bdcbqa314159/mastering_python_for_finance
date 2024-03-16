#pylint: disable-all
import math
import numpy as np
import scipy.linalg as linalg

class StockOption():
    def __init__(self, S0, K, r, T, N, params):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.N = max(1,N)
        self.STs = None

        self.pu = params.get("pu",0)
        self.pd = params.get("pd",0)
        self.div = params.get("div",0)
        self.sigma = params.get("sigma",0)
        self.is_call = params.get("is_call", True)
        self.is_european = params.get("is_eu", True)

        self.dt = T/float(N)
        self.df = math.exp(-(r-self.div)*self.dt)

class BinomialEuropeanOption(StockOption):
    def _setup_parameters_(self):
        self.M = self.N+1
        self.u = 1+self.pu
        self.d = 1-self.pd
        self.qu = (math.exp((self.r-self.div)*self.dt)-self.d)/(self.u-self.d)
        self.qd = 1-self.qu
        return
    
    def _initialize_stock_price_tree_(self):
        self.STs = np.zeros(self.M)
        
        for i in range(self.M):
            self.STs[i] = self.S0*(self.u**(self.N-i))*(self.d**i)

        return
    
    def _initialize_payoffs_tree_(self):
        payoffs = np.maximum(0, (self.STs-self.K) if self.is_call else (self.K-self.STs))
        return payoffs
    
    def _travere_tree_(self, payoffs):
        for i in range(self.N):
            payoffs = (payoffs[:-1]*self.qu + payoffs[1:]*self.qd)*self.df
        return payoffs
    
    def __begin_tree_traversal__(self):
        payoffs = self._initialize_payoffs_tree_()
        return self._travere_tree_(payoffs)
    
    def price(self):
        self._setup_parameters_()
        self._initialize_stock_price_tree_()
        payoffs = self.__begin_tree_traversal__()

        return payoffs[0]
    
class BinomialTreeOption(StockOption):
    def _setup_parameters_(self):
        self.u = 1+self.pu
        self.d = 1-self.pd
        self.qu = (math.exp((self.r-self.div)*self.dt)-self.d)/(self.u-self.d)
        self.qd = 1-self.qu
        return
    
    def _initialize_stock_price_tree_(self):
        self.STs = [np.array([self.S0])]
        for i in range(self.N):
            prev_branches = self.STs[-1]
            st = np.concatenate((prev_branches*self.u, [prev_branches[-1]*self.d]))
            self.STs.append(st)
        return
       
    def _initialize_payoffs_tree_(self):
        return np.maximum(0, (self.STs[self.N]-self.K) if self.is_call else (self.K-self.STs[self.N]))
    
    def __check_early_exercise__(self, payoffs, node):
        early_ex_payoff = (self.STs[node]-self.K) if self.is_call else (self.K-self.STs[node])
        return np.maximum(payoffs, early_ex_payoff)
    
    def _traverse_tree_(self, payoffs):
        for i in reversed(range(self.N)):
            payoffs = (payoffs[:-1]*self.qu+payoffs[1:]*self.qd)*self.df
            if not self.is_european:
                payoffs = self.__check_early_exercise__(payoffs,i)

        return payoffs
    
    def __begin_tree_traversal__(self):
        payoffs = self._initialize_payoffs_tree_()
        return self._traverse_tree_(payoffs)
    
    def price(self):
        self._setup_parameters_()
        self._initialize_stock_price_tree_()
        payoffs = self.__begin_tree_traversal__()
        return payoffs[0]
    
class BinomialCRROption(BinomialTreeOption):
    def _setup_parameters_(self):
        self.u = math.exp(self.sigma * math.sqrt(self.dt))
        self.d = 1./self.u
        self.qu = (math.exp((self.r-self.div)*self.dt)-self.d)/(self.u-self.d)
        self.qd = 1-self.qu
        return

class BinomialLROption(BinomialTreeOption):
    def _setup_parameters_(self):
        odd_N = self.N if (self.N%2 == 1) else (self.N+1)
        d1 = (math.log(self.S0/self.K)+((self.r-self.div)+(self.sigma**2)*0.5)*self.T)/(self.sigma*math.sqrt(self.T))
        d2 = (math.log(self.S0/self.K)+((self.r-self.div)-(self.sigma**2)*0.5)*self.T)/(self.sigma*math.sqrt(self.T))

        pp2_inversion = lambda z,n:.5+math.copysign(1,z)*math.sqrt(.25- 0.25*math.exp(-((z/(n+1./3 +.1/(n+1)))**2.)*(n+1./6)))
        pbar = pp2_inversion(d1, odd_N)

        self.p = pp2_inversion(d2, odd_N)
        self.u = 1/self.df * pbar/self.p
        self.d = (1/self.df - self.p*self.u)/(1-self.p)
        self.qu = self.p
        self.qd = 1-self.p

class BinomialLRWithGreeks(BinomialLROption):
    def __new_stock_price_tree__(self):
        self.STs = [np.array([self.S0*self.u/self.d, self.S0, self.S0*self.d/self.u])]

        for i in range(self.N):
            prev_branches = self.STs[-1]
            st = np.concatenate((prev_branches*self.u, [prev_branches[-1]*self.d]))
            self.STs.append(st)
        return 
    
    def price(self):
        self._setup_parameters_()
        self.__new_stock_price_tree__()
        payoffs = self.__begin_tree_traversal__()

        option_value = payoffs[1]
        payoff_up = payoffs[0]
        payoff_down = payoffs[-1]

        S_up = self.STs[0][0]
        S_down = self.STs[0][-1]

        dS_up = S_up - self.S0
        dS_down = self.S0 - S_down

        dS = S_up-S_down
        dV = payoff_up-payoff_down

        delta = dV/dS

        gamma = ((payoff_up-option_value)/dS_up - (option_value-payoff_down)/dS_down)/((self.S0 + S_up)*0.5 - (self.S0+S_down)*0.5)
        return option_value, delta, gamma
    
class TrinomialTreeOption(BinomialTreeOption):
    def _setup_parameters_(self):
        self.u = math.exp(self.sigma*math.sqrt(2.*self.dt))
        self.d = 1./self.u
        self.m = 1

        sqrt_semi_step = math.sqrt(0.5*self.dt)
        term_1 = math.exp((self.r - self.div)*0.5*self.dt)
        term_2 = math.exp(self.sigma*sqrt_semi_step)
        term_3 = math.exp(-self.sigma*sqrt_semi_step)

        self.qu = ((term_1 - term_3)/(term_2-term_3))**2
        self.qd = ((-term_1 + term_2)/(term_2-term_3))**2
        self.qm = 1-self.qu-self.qd
        return 
    
    def _initialize_stock_price_tree_(self):
        self.STs = [np.array([self.S0])]
        for i in range(self.N):
            prev_nodes = self.STs[-1]
            self.ST = np.concatenate((prev_nodes*self.u, [prev_nodes[-1]*self.m, prev_nodes[-1]*self.d]))
            self.STs.append(self.ST)
        return
    
    def _traverse_tree_(self, payoffs):
        for i in reversed(range(self.N)):
            payoffs = (payoffs[:-2]*self.qu + payoffs[1:-1]*self.qm+payoffs[2:]*self.qd)*self.df
            if not self.is_european:
                payoffs = self.__check_early_exercise__(payoffs, i)
        return payoffs
    
class BinomialCRRLattice(BinomialCRROption):
    def _setup_parameters_(self):
        super()._setup_parameters_()
        self.M = 2*self.N+1
        return
    
    def _initialize_stock_price_tree_(self):
        self.STs = np.zeros(self.M)
        self.STs[0] = self.S0*self.u**self.N

        for i in range(self.M)[1:]:
            self.STs[i] = self.STs[i-1]*self.d

    def _initialize_payoffs_tree_(self):
        odd_nodes = self.STs[::2]
        return np.maximum(0, (odd_nodes-self.K) if self.is_call else (self.K-odd_nodes))
    
    def __check_early_exercise__(self, payoffs, node):
        self.STs = self.STs[1:-1]
        odd_STs = self.STs[::2]
        early_ex_payoffs = (odd_STs-self.K) if self.is_call else (self.K-odd_STs)
        payoffs = np.maximum(payoffs, early_ex_payoffs)
        return payoffs
    
class TrinomialLattice(TrinomialTreeOption):
    def _setup_parameters_(self):
        super()._setup_parameters_()
        self.M = 2*self.N+1
        return 
    
    def _initialize_stock_price_tree_(self):
        self.STs = np.zeros(self.M)
        self.STs[0] = self.S0*self.u**self.N

        for i in range(self.M)[1:]:
            self.STs[i] = self.STs[i-1]*self.d
    
    def _initialize_payoffs_tree_(self):
        return np.maximum(0, (self.STs-self.K) if self.is_call else (self.K-self.STs))
    
    def __check_early_exercise__(self, payoffs, node):
        self.STs = self.STs[1:-1]
        early_ex_payoffs = (self.STs-self.K) if self.is_call else (self.K-self.STs)
        payoffs = np.maximum(payoffs, early_ex_payoffs)
        return payoffs
    
class FiniteDifferences():
    def __init__(self, S0, K, r, T, sigma, Smax, M, N, is_call = True):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.Smax = Smax
        self.M, self.N = int(M), int(N)
        self.is_call = is_call

        self.dS = Smax/float(self.M)
        self.dt = T/float(self.N)
        self.i_values = np.arange(self.M)
        self.j_values = np.arange(self.N)
        self.grid = np.zeros(shape = (self.M+1, self.N+1))
        self.boundary_conds = np.linspace(0, Smax, self.M+1)

    def _setup_boudary_conditions_(self):
        pass

    def _setup_coefficients_(self):
        pass

    def _traverse_grid_(self):
        pass

    def _interpolate_(self):
        return np.interp(self.S0, self.boundary_conds, self.grid[:, 0])
    
    def price(self):
        self._setup_boudary_conditions_()
        self._setup_coefficients_()
        self._traverse_grid_()
        return self._interpolate_()
    
class FDExplicitEu(FiniteDifferences):
    def _setup_boudary_conditions_(self):
        if self.is_call:
            self.grid[:,-1] = np.maximum(self.boundary_conds - self.K, 0)
            self.grid[-1,:-1] = (self.Smax - self.K)*np.exp(-self.r*self.dt*(self.N - self.j_values))
        else:

            self.grid[:, -1] = np.maximum(self.K - self.boundary_conds, 0)
            self.grid[0, :-1] = (self.K -self.Smax)*np.exp(-self.r*self.dt*(self.N-self.j_values))
        return
    
    def _setup_coefficients_(self):
        self.a = 0.5*self.dt*((self.sigma**2)*(self.i_values**2)-self.r*self.i_values)
        self.b = 1-self.dt*((self.sigma**2)*(self.i_values**2)+self.r)
        self.c = 0.5*self.dt*((self.sigma**2)*(self.i_values**2)+self.r*self.i_values)
        return
    
    def _traverse_grid_(self):
        for j in reversed(self.j_values):
            for i in range(self.M)[2:]:
                self.grid[i,j] = self.a[i]*self.grid[i-1,j+1]+self.b[i]*self.grid[i,j+1]+self.c[i]*self.grid[i+1,j+1]
        return
    
class FDImplicitEu(FDExplicitEu):
    def _setup_coefficients_(self):
        self.a = 0.5*(self.r*self.dt*self.i_values - (self.sigma**2)*self.dt*(self.i_values**2))
        self.b = 1+(self.sigma**2)*self.dt*(self.i_values**2)+self.r*self.dt
        self.c = -0.5*(self.r*self.dt*self.i_values + (self.sigma**2)*self.dt*(self.i_values**2))

        self.coeffs = np.diag(self.a[2:self.M], -1)+np.diag(self.b[1:self.M])+np.diag(self.c[1:self.M-1], 1)
        return
    
    def _traverse_grid_(self):
        P,L,U = linalg.lu(self.coeffs)
        aux = np.zeros(self.M-1)

        for j in reversed(range(self.N)):
            aux[0] = np.dot(-self.a[1], self.grid[0,j])
            x1 = linalg.solve(L, self.grid[1:self.M, j+1]+aux)
            x2 = linalg.solve(U,x1)
            self.grid[1:self.M,j] = x2
        return

class FDCnEu(FDExplicitEu):

    def _setup_coefficients_(self):
        self.alpha = 0.25*self.dt*((self.sigma**2)*(self.i_values**2)-self.r*self.i_values)
        self.beta = -self.dt*0.5*((self.sigma**2)*(self.i_values**2)+self.r)
        self.gamma = 0.25*self.dt*((self.sigma**2)*(self.i_values**2)+self.r*self.i_values)

        self.M1 = -np.diag(self.alpha[2:self.M], -1)+ np.diag(1-self.beta[1:self.M])-np.diag(self.gamma[1:self.M-1],1)
        self.M2 = np.diag(self.alpha[2:self.M], -1)+np.diag(1+self.beta[1:self.M])+np.diag(self.gamma[1:self.M-1], 1)
        return
    
    def _traverse_grid_(self):
        _ ,L,U = linalg.lu(self.M1)
        for j in reversed(range(self.N)):
            x1 = linalg.solve(L, np.dot(self.M2, self.grid[1:self.M, j+1]))
            x2 = linalg.solve(U,x1)

            self.grid[1:self.M,j] = x2
        return

class FDCnDo(FDCnEu):

    def __init__(self, S0, K, r, T, sigma, Sbarrier, Smax, M, N, is_call=True):
        super().__init__(S0, K, r, T, sigma, Smax, M, N, is_call)
        self.dS = (Smax-Sbarrier)/float(self.M)
        self.boundary_conds = np.linspace(Sbarrier, Smax, self.M+1)
        self.i_values = self.boundary_conds/self.dS
        return 


if __name__ == "__main__":
    pass
