# Chapter 8 - Algorithmic Trading

The methods and the data websites are a bit too old to be adapted.
We can try to code the code from the second edition book though, but not now.

## Summary

In this chapter, we were introduced to the evolution of trading from the pits to the
electronic trading platform, and learned how algorithmic trading came about. We
looked at some brokers offering API access to their trading service offering. To help
us get started on our journey in developing an algorithmic trading system, we used
the TWS of IB and the IbPy Python module.
In our first trading program, we successfully sent an order to our broker through the
TWS API using a demonstration account. Next, we developed a simple algorithmic
trading system. We started by requesting the market data and account updates
from the server. With the captured real-time information, we implemented a mean-
reverting algorithm to trade the markets. Since this trading system uses only one
indicator, more work would be required to build a robust, reliable, and profitable
trading system.
We also discussed currency trading with the OANDA REST API with the help of the
oandapy Python module. After setting up our account for API access, our first step to
explore the OANDA API is to fetch rates for a single currency pair and send a limit
order to the server. Using the fxTrade Practice platform, we can track our current
trades, orders, and positions. Next, we developed a trend-following algorithm to
trade the EUR/USD currency pair with the use of streaming rates and market orders.
One critical aspect of trading is to manage risk effectively. In the financial industry,
the VaR is the most common technique used to measure risk. Using Python, we took
a practical approach to calculate the daily VaR of a set of stock prices from Yahoo!
Finance.
Once we have built a working algorithmic trading system, we can explore the other
ways to measure the performance of our trading strategy. One of these areas is
backtesting. We will discuss this topic in the next chapter.