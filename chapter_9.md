# Chapter 9 - Backtesting

The mini project coded there is not in my top priority right now. The algorithms mentioned
below can be found in datacamp.
We can try to code the code from the second edition book though, but not now.

## Summary

A backtest is a simulation of a model-driven investment strategy's response to
historical data. The purpose of performing experiments with backtests is to make
discoveries about a process or system and to compute various factors related to
either risk or return. The factors are typically used together to find a combination
that is predictive of return.
While working on designing and developing a backtester, to achieve functionalities,
such as simulated market pricing, ordering environment, order matching engine,
order book management, as well as account and position updates, we can explore
the concept of an event-driven backtesting system.
In this chapter, we designed and implemented an event-driven backtesting system
using the TickData class, the MarketDataSource class, the Order class, the
Position class, the Strategy class, the MeanRevertingStrategy class, and the
Backtester class. We plotted our resulting profits and losses onto a graph to
help us visualize the performance of our trading strategy.
Backtesting involves a lot of research that merits a literature on its own. In this
chapter, we explored ten considerations for designing a backtest model. To
help improve our models on a continuous basis, a number of algorithms can be
employed in backtesting. We briefly discussed some of these: k-means clustering,
k-nearest neighbor, classification and regression tree, 2k factorial design, and
genetic algorithm.
In the next chapter, we will discuss Excel with Python, using the Component
Object Model (COM).