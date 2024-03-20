# Chapter 5 - Interest Rates and Derivatives

## Summary

In this chapter, we focused on interest rate and related derivative pricing with Python.
Most bonds, such as US Treasury bonds, pay a fixed amount of interest semi-annually,
while other bonds may pay quarterly, or annually. It is a characteristic of bonds that
their prices are closely related to current interest rate levels in an inversely related
manner. The normal or positive yield curve, where long-term interest rates are higher
than short-term interest rates, is said to be upward sloping. In certain economic
conditions, the yield curve can be inverted and is said to be downward sloping.
A zero-coupon bond is a bond that pays no coupons during its lifetime, except
on maturity when the principal or face value is repaid. We implemented a simple
zero-coupon bond calculator in Python.
The yield curve can be derived from the short-term zero or spot rates of securities,
such as zero-coupon bonds, T-bills, notes, and Eurodollar deposits using a
bootstrapping process. Using Python, we used a lot of bond information to plot a
yield curve, and derived forward rates, yield-to-maturity, and bond prices from the
yield curve.
Two important metrics to bond traders are duration and convexity. Duration is a
sensitivity measure of bond prices to yield changes. Convexity is the sensitivity
measure of the duration of a bond to yield changes. We implemented calculations
using the modified duration model and convexity calculator in Python.
Short rate models are frequently used in the evaluation of interest rate derivatives.
Interest rate modeling is a fairly complex topic since they are affected by a multitude
of factors, such as economic states, political decisions, government intervention,
and the laws of supply and demand. A number of interest rate models have been
proposed to account for various characteristics of interest rates. Some of the interest
rate models we have discussed are the Vasicek model, CIR model, and Rendleman
and Bartter model.
Bond issuers may embed options within a bond to allow them the right, but not
the obligation, to buy or sell the issued bond at a predetermined price during a
specified period of time. The price of a callable bond can be thought of as the price
difference of a bond without an option and the price of the embedded call option.
Using Python, we took a look at pricing a callable zero-coupon bond by applying
the Vasicek model to the implicit method of finite differences. This method is,
however, just one of the many methods that quantitative analysts use in bond
options modeling.
In the next chapter, we will explore analytics with Python and VSTOXX.
