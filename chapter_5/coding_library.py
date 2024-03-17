#pylint: disable-all
import math
import numpy as np
import scipy.optimize as optimize
import sys
import matplotlib.pyplot as plt

def zero_coupon_bond(par, y, t):
    return par/(1+y)**t

class BootstrapYieldCurve():
    def __init__(self):
        self.zero_rates = dict()
        self.instruments = dict()

    def add_instrument(self, par, T, coup, price, compounding_freq = 2):
        self.instruments[T] = (par, coup, price, compounding_freq)

    def get_zero_rates(self):
        self.__bootstrap_zero_coupons__()
        self.__get_bond_spot_rates__()
        return [self.zero_rates[T] for T in self.get_maturities()]
    
    def get_maturities(self):
        return sorted(self.instruments.keys())
    
    def __bootstrap_zero_coupons__(self):
        for T in self.instruments:
            par, coup, price, freq = self.instruments[T]
            if coup == 0:
                self.zero_rates[T] = self.zero_coupon_spot_rate(par, price, T)
    
    def __get_bond_spot_rates__(self):
        for T in self.get_maturities():
            instrument = self.instruments[T]
            par, coup, price, freq = instrument

            if coup != 0:
                self.zero_rates[T] = self.__calculate_bond_spot_rate__(T, instrument)

    def __calculate_bond_spot_rate__(self, T, instrument):
        try:
            par, coup, price, freq = instrument

            periods = T*freq
            value = price
            per_coupon = coup/freq

            for i in range(int(periods)-1):
                t = (i+1)/float(freq)

                spot_rate = self.zero_rates[t]
                discounted_coupon = per_coupon*math.exp(-spot_rate*t)
                value -= discounted_coupon

            last_period = int(periods)/float(freq)
            spot_rate = -math.log(value/(par+per_coupon))/last_period
            return spot_rate
        
        except:
            print(f"Error: spot rate not found for T={t}")

    def zero_coupon_spot_rate(self, par, price, T):
        spot_rate = math.log(par/price)/T
        return spot_rate
    
class ForwardRates():
    def __init__(self):
        self.forward_rates = []
        self.spot_rates = dict()

    def add_spot_rate(self, T, spot_rate):
        self.spot_rates[T] = spot_rate
    
    def __calculate_forwar_rate__(self, T1, T2):
        R1 = self.spot_rates[T1]
        R2 = self.spot_rates[T2]

        forward_rate = (R2*T2 - R1*T1)/(T2-T1)
        return forward_rate
    
    def get_forward_rates(self):
        periods = sorted(self.spot_rates.keys())
        for T2,T1 in zip(periods, periods[1:]):
            forward_rate = self.__calculate_forwar_rate__(T1,T2)
            self.forward_rates.append(forward_rate)
        return self.forward_rates
    
def bond_ytm(price, par, T, coup, freq = 2, guess = 0.05):
    freq = float(freq)
    periods = T*freq

    coupon = (coup/100.)*(par/freq)
    dt = [(i+1)/freq for i in range(int(periods))]

    ytm_func = lambda y: sum([coupon/(1+y/freq)**(freq*t) for t in dt])+ par/(1+y/freq)**(freq*T) - price

    return optimize.newton(ytm_func, guess)

def bond_price(par, T, ytm, coup, freq = 2):
    freq = float(freq)
    periods = T*freq

    coupon = (coup/100.)*(par/freq)
    dt = [(i+1)/freq for i in range(int(periods))]

    price = sum([coupon/(1+ytm/freq)**(freq*t) for t in dt])+ par/(1+ytm/freq)**(freq*T)

    return price


def bond_mod_duration(price, par, T, coup, freq, dy = 0.01):
    ytm = bond_ytm(price, par, T, coup, freq)
    ytm_minus = ytm-dy
    ytm_plus = ytm+dy

    price_minus = bond_price(par, T, ytm_minus, coup, freq)
    price_plus = bond_price(par, T, ytm_plus, coup, freq)

    mduration = (price_minus-price_plus)/(2*price*dy)
    return mduration

def bond_convexity(price, par, T, coup, freq, dy = 0.01):
    ytm = bond_ytm(price, par, T, coup, freq)
    ytm_minus = ytm-dy
    ytm_plus = ytm+dy

    price_minus = bond_price(par, T, ytm_minus, coup, freq)
    price_plus = bond_price(par, T, ytm_plus, coup, freq)

    mconvexity = (price_minus+price_plus-2*price)/(price*dy**2)
    return mconvexity


def vasicek(r0, K, theta, sigma, T = 1., N = 10, seed = 777):
    np.random.seed(seed)
    dt = T/float(N)
    rates = [r0]

    for i in range(N):
        dr = K*(theta-rates[-1])*dt + sigma*np.random.normal()
        rates.append(rates[-1]+dr)
    return range(N+1), rates

def cir(r0, K, theta,sigma, T = 1., N=10, seed=777):
    np.random.seed(seed)
    dt = T/float(N)
    rates = [r0]
    for i in range(N):
        dr = K*(theta-rates[-1])*dt + sigma*math.sqrt(rates[-1])*np.random.normal()
        rates.append(rates[-1] + dr)
    return range(N+1), rates

def rendleman_bartter(r0, theta, sigma, T = 1., N=10, seed = 777):
    np.random.seed(777)
    dt = T/float(N)
    rates = np.array((N+1)*[0.])
    rates[0] = r0
    for i in range(1, N+1):
        dr = theta*rates[i-1]*dt+ sigma*rates[i-1]*np.random.normal()
        rates[i] = rates[i-1]+dr
    return range(N+1), rates

def brennan_schwartz(r0, K, theta, sigma, T = 1., N=10, seed = 777):
    np.random.seed(777)
    dt = T/float(N)
    rates = np.array((N+1)*[0.])
    rates[0] = r0
    for i in range(1, N+1):
        dr = K*(theta-rates[i-1])*dt+ sigma*rates[i-1]*np.random.normal()
        rates[i] = rates[i-1]+dr
    return range(N+1), rates




if __name__ == "__main__":
    pass
