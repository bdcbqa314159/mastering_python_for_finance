# Chapter 2 - The Importance of Linearity in Finance

## Summary
In this chapter, we took a brief look at the use of the CAPM model and APT model in
finance. In the CAPM model, we visited the efficient frontier with the capital market
line to determine the optimal portfolio and the market portfolio. Then, we solved
for the security market line using regression that helped us to determine whether
an asset is undervalued or overvalued. In the APT model, we explored how various
factors affect security returns other than using the mean-variance framework. We
performed a multivariate regression to help us determine the coefficients of these
factors that led to the valuation of our security price.

In portfolio allocation, portfolio managers are typically mandated by investors to
achieve a set of objectives while following certain constraints. We can model this
problem using linear programming. Using the PuLP Python package, we defined a
maximization or minimization objective function, and added inequality constraints
to our problems to solve for unknown variables. The three outcomes in linear
optimization can either be an unbounded solution, only one solution, or no solution
at all.

Another form of linear optimization is integer programming, where all the variables
are restricted to be integers instead of fractional values. A special case of an integer
variable is a binary variable, which can either be 0 or 1, and it is especially useful
to model decision making given a set of choices. We worked on a simple integer
programming model containing binary conditions and saw how easy it is to run into
a pitfall. Careful planning on the design of integer programming models is required
for them to be useful in decision making.

The portfolio allocation problem may also be represented as a system of linear
equations with equalities, which can be solved using matrices in the form of Ax=B.
To find the values of x, we solved for inv(A)*B using various types of decomposition
of the matrix A. The two types of matrix decomposition methods are the direct and
indirect methods.

The direct method performs matrix algebra in a fixed number of
iterations. They are namely the LU decomposition, Cholesky decomposition, and
QR decomposition methods. The indirect or iterative method iteratively computes
the next values of x until a certain tolerance of accuracy is reached. This method
is particularly useful for computing large matrices, but it also faces the risk of not
having the solution converge. The indirect methods we have used are the Jacobi
method and the Gauss-Seidel method.
In the next chapter, we will take a look at nonlinear models and methods of
solving them.