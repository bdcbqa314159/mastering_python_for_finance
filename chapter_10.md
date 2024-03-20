# Chapter 10 - Excel with Python

The code here is outdated and also can not be reproduced directly in macOS.
Besides this there is no need to use vba anymore since the popular xlwings library exists.

## Summary

In this chapter, we looked at the use of the Component Object Model (COM) to allow
the reuse of objects across different software and hardware environments to interface
with each other, without the knowledge of its internal implementation.
To build the server component of the COM interface, we used the pythoncom
module to create a Black-Scholes pricing COM server with the three magic variables:
_public_methods_, _reg_progid_, and _reg_clsid_. Using topics in Chapter 4,
Numerical Procedures, we created COM server components using the binomial tree
by the CRR model and trinomial lattice model. We learned how to register and
unregister these COM server components with the Windows registry.
In Microsoft Excel, we can input a number of parameters for a particular option and
numerically compute the theoretical option prices using the COM server components
we built. These functions are made available in the formula cells using Visual Basic.
We created the Black-Scholes model, binomial tree CRR model, and trinomial lattice
model COM client VBA functions. These functions accept the same input values
from the spreadsheet cells to perform numerical pricing on the COM server. We also
saw how to update the input parameters in the spreadsheet that dynamically update
the option prices on the fly.