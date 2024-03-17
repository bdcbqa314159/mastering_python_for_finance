# Chapter 4 - Numerical Procedures

## Summary
In this chapter, we looked at a number of numerical procedures in derivative pricing
the most common being options. One such procedure is the use of trees, with
binomial trees being the simplest structure to model asset information, where one
node extends to two other nodes in each time step, representing an up state and a
down state respectively. In trinomial trees, each node extends to three other nodes in
each time step, representing an up state, a down state, and a state with no movement
respectively. As the tree traverses upwards, the underlying asset is computed and
represented at each node. The option then takes on the structure of this tree and,
starting from the terminal payoffs, the tree traverses backward and toward the
root, which converges to the current discounted option price. Besides binomial and
trinomial trees, trees can take on the form of the Cox-Ross-Rubinstein, Jarrow-Rudd,
Tian, or Leisen-Reimer parameters.
By adding another layer of nodes around our tree, we introduced additional
information from which we can derive the Greeks such as the delta and gamma
without incurring additional computational cost.
Lattices were introduced as a way of saving storage costs over binomial and
trinomial trees. In lattice pricing, nodes with new information are saved only
once and reused later on nodes that require no change in the information.
We also discussed the finite difference schemes in option pricing, consisting of
terminal and boundary conditions. From the terminal conditions, the grid traverses
backward in time using the explicit method, implicit method, and the Crank-
Nicolson method. Besides pricing European and American options, finite difference
pricing schemes can be used to price exotic options, where we looked at an example
of pricing a down-and-out barrier option.
By importing the bisection root-finding method learned in Chapter 3, Nonlinearity in
Finance and the binomial Leisen-Reimer tree model in this chapter, we used market
prices of an American option to create an implied volatility curve for further studies.
In the next chapter, we will take a look at working with interest rate instruments.
