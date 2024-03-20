#pylint: disable-all
import math
import numpy as np
import scipy.optimize as optimize
import scipy.stats as st
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


def exact_zcb(theta, kappa, sigma, tau, r0 = 0.):
    B = (1 - np.exp(-kappa*tau))/kappa
    A = np.exp((theta-(sigma**2)/(2*(kappa**2)))*(B-tau)-(sigma**2)/(4*kappa)*(B**2))
    return A*np.exp(-r0*B)

def exercise_value(K,R,t):
    return K*np.exp(-R*t)

class VasicekCZCB():
    def __init__(self):
        self.norminv = st.distributions.norm.ppf
        self.norm = st.distributions.norm.cdf

    def vasicek_czcb_values(self, r0, R, ratio, T, sigma, kappa, theta, M, prob = 1e-6,max_policy_iter = 10, grid_struct_const = 0.25, rs = None):
        r_min, dr, N, dtau = self.vasicek_params(r0, M, sigma, kappa, theta, T, prob, grid_struct_const, rs)
        r = np.r_[0:N]*dr+r_min
        v_mplus1 = np.ones(N)
        for i in range(1, M+1):
            K = self.exercise_call_price(R, ratio, i*dtau)
            eex = np.ones(N)*K
            subdiagonal, diagonal, superdiagonal = self.vasicek_diagonals(sigma, kappa, theta, r_min, dr, N, dtau)
            v_mplus1, iterations = self.iterate(subdiagonal, diagonal, superdiagonal, v_mplus1, eex, max_policy_iter)

        return r, v_mplus1
    
    def vasicek_limits(self, r0, sigma, kappa, theta, T, prob = 1e-6):
        er = theta+(r0-theta)*math.exp(-kappa*T)
        variance = (sigma**2)*T if kappa == 0 else (sigma**2)/(2*kappa)*(1-math.exp(-2*kappa*T))
        stdev = math.sqrt(variance)
        r_min = self.norminv(prob, er, stdev)
        r_max = self.norminv(1-prob, er, stdev)

        return r_min, r_max
    
    def vasicek_params(self, r0, M, sigma, kappa, theta, T, prob, grid_struct_const = 0.25, rs = None):
        r_min, r_max = rs[0], rs[-1] if not rs is None else self.vasicek_limits(r0, sigma, kappa, theta, T, prob)
        dt = T/float(M)

        N = self.calculate_N(grid_struct_const, dt, sigma, r_max, r_min)
        dr = (r_max-r_min)/(N-1)
        return r_min, dr, N, dt
    
    def calculate_N(self, max_structure_const, dt, sigma, r_max, r_min):
        N = 0
        while True:
            N+=1

            grid_structure_interval = dt*(sigma**2)/(((r_max-r_min)/float(N))**2)
            if grid_structure_interval>max_structure_const:
                break

        return N

    def vasicek_diagonals(self, sigma, kappa, theta, r_min, dr, N, dtau):
        rn = np.r_[0:N]*dr +r_min
        subdiagonals = kappa*(theta-rn)*dtau/(2*dr) - 0.5*(sigma**2)*dtau/(dr**2)
        diagonals = 1+rn*dtau+sigma**2*dtau/(dr**2)
        superdiagonals = -kappa*(theta-rn)*dtau/(2*dr)-0.5*(sigma**2)*dtau/(dr**2)

        if N>0:
            v_subd0 = subdiagonals[0]
            superdiagonals[0] = superdiagonals[0]-subdiagonals[0]
            diagonals[0]+=2*v_subd0
            subdiagonals[0] = 0

        if N>1:
            v_superd_last = superdiagonals[-1]
            superdiagonals[-1] = superdiagonals[-1]-subdiagonals[-1]
            diagonals[-1] += 2*v_superd_last
            superdiagonals[-1] = 0

        return subdiagonals, diagonals, superdiagonals

    def check_exercise(self, V, eex):
        return V>eex

    def exercise_call_price(self, R, ratio, tau):
        K = ratio*np.exp(-R*tau)
        return K

    def vasicek_policy_diagonals(self, subdiagonal, diagonal, superdiagonal, v_old, v_new, eex):
        has_early_exercise = self.check_exercise(v_new, eex)
        subdiagonal[has_early_exercise] = 0
        superdiagonal[has_early_exercise] = 0
        policy = v_old/eex
        policy_values = policy[has_early_exercise]
        diagonal[has_early_exercise] = policy_values
        return subdiagonal, diagonal, superdiagonal

    def iterate(self, subdiagonal, diagonal, superdiagonal, v_old, eex, max_policy_iter = 10):
        v_mplus1 = v_old
        v_m = v_old
        change = np.zeros(len(v_old)) 
        prev_changes = np.zeros(len(v_old))

        iterations = 0

        while iterations <= max_policy_iter:
            iterations+=1

            v_mplus1 = self.tridiagonal_solve(subdiagonal, diagonal, superdiagonal, v_old)

            subdiagonal, diagonal, superdiagonal = self.vasicek_policy_diagonals(subdiagonal, diagonal, superdiagonal, v_old, v_mplus1, eex)
            is_eex = self.check_exercise(v_mplus1, eex)
            change[is_eex] = 1
            if iterations>1:
                change[v_mplus1 != v_m] = 1

            is_no_more_eex = False if True in is_eex else True
            if is_no_more_eex:
                break
            v_mplus1[is_eex] = eex[is_eex]
            changes = (change==prev_changes)

            is_no_further_changes = all((x==1) for x in changes)
            if is_no_further_changes:
                break
            prev_changes = change

            v_m = v_mplus1

        return v_mplus1, (iterations-1)
    
    def tridiagonal_solve(self, a, b, c, d):
        nf = len(a)
        ac, bc, cc, dc = map(np.array, (a,b,c,d))

        for it in range(1, nf):
            mc = ac[it]/bc[it-1]
            bc[it] = bc[it]-mc*cc[it-1]
            dc[it] = dc[it]-mc*dc[it-1]

        xc = ac
        xc[-1] = dc[-1]/bc[-1]

        for il in range(nf-2, -1, -1):
            xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

        del bc,cc,dc

        return xc
    


if __name__ == "__main__":
    pass
