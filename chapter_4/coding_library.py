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
    def __setup_parameters__(self):
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
        self.__setup_parameters__()
        self._initialize_stock_price_tree_()
        payoffs = self.__begin_tree_traversal__()

        return payoffs[0]


if __name__ == "__main__":
    pass

