# Chapter 6 - Interactive Financial Analytics with Python and VSTOXX

The methods and the data websites are a bit too old to be adapted - these kind of techniques can be learn in datacamp.
We can try to code the code from the second edition book though, but not now.

## Summary

In this chapter, we looked at volatility derivatives and their uses by investors
to diversify and hedge their risk in equity and credit portfolios. Since long-term
investors in equity funds are exposed to downside risk, volatility can be used as a
hedge for the tail risk and in replacement for the put options. In the United States,
the CBOE Volatility Index (VIX) measures the short-term volatility implied by S&P
500 stock index option prices. In Europe, the VSTOXX market index is based on the
market prices of a basket of OESX and measures the implied market volatility over
the next 30 days on the EURO STOXX 50 Index. Many people around the world
use the VIX as a popular measurement tool for the stock market volatility over the
next 30-day period. To help us better understand how the VSTOXX market index
is calculated, we looked at its components and at formulas used in determining
its value.
The STOXX and Eurex Exchange websites provide the historical daily data of the main
index and its sub-indexes. To help us determine the relationship between the EURO
STOXX 50 Index and VSTOXX, we downloaded this data with Python, merged them,
and performed a variety of financial analytics. We came to the conclusion that they
are negatively correlated. This relationship presents a viable way of avoiding frequent
rebalancing costs by trading strategies based on benchmarking. The statistical nature
of volatility allows volatility derivative traders to generate returns by utilizing mean-
reverting strategies, dispersion trading, and volatility spread trading, among others.
The VSTOXX consists of eight sub-indexes that represent the calculated volatility
index from the EURO STOXX 50 Index options expiring in 1, 2, 3, 6, 9, 12, 18, and
24 months. Since the VSTOXX index represents the volatility outlook for the next
30 days, we gathered the OESX call and put prices from the Eurex website and
calculated the sub-indexes for the 2 month forward expiry date.
Finally, we studied the component weighing formula of the VSTOXX sub-indexes
and used Python to calculate the VSTOXX main index value to give us an estimate
of the volatility outlook for the next 30 days.
In the next chapter, we will take a look at managing big data in finance with Python.

