# Chapter 3 - Nonlinearity in Finance

## Summary

In this chapter, we briefly discussed the persistence of nonlinearity in economics
and finance. We looked at some nonlinear models that are commonly used in finance
to explain certain aspects of data left unexplained by linear models: the Black-Scholes
implied volatility model, Markov switching model, threshold model, and smooth
transition models.
In Black-Scholes implied volatility modeling, we discussed the volatility smile
that was made up of implied volatilities derived via the Black-Scholes model from
the market prices of call or put options for a particular maturity. You may be
interested enough to seek the lowest implied volatility value possible, which can be
useful for inferring theoretical prices and comparing against the market prices for
potential opportunities. However, since the curve is nonlinear, linear algebra cannot
adequately solve for the optimal point. To do so, we will require the use of root-
finding methods.
Root-finding methods attempt to find the root of a function or its zero. We discussed
common root-finding methods: the bisection method, Newton's method, and secant
method. Using a combination of root-finding algorithms may help us to seek roots of
complex functions faster. One such example is Brent's method.
We explored functionalities in the scipy.optimize module that contains these
root-finding methods, albeit with constraints. One of these constraints requires
that the two boundary input values be evaluated with a pair of a negative value
and positive value for the solution to converge successfully. In implied volatility
modeling, this evaluation is almost impossible since volatilities do not have negative
values. Implementing our own root-finding methods might perhaps give us more
authority over how our application should perform.
Using general solvers is another way of finding roots. They may also converge to
our solution more quickly, but such a convergence is not guaranteed by the initial
given values.
Nonlinear modeling and optimization are inherently a complex task, and there is
no universal solution or a sure way to reach a conclusion. This chapter serves to
introduce nonlinearity studies for finance in general.
In the next chapter, we will take a look at numerical methods commonly used for
options pricing. By pairing a numerical procedure with a root-finding algorithm,
we will learn how to build an implied volatility model with the market prices of an
equity option.
