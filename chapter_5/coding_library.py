#pylint: disable-all
import math
import numpy as np
import scipy.linalg as linalg
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


if __name__ == "__main__":
    pass
