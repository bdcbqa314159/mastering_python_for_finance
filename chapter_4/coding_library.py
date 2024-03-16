#pylint: disable-all
import math
import numpy as np

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

if __name__ == "__main__":
    pass
