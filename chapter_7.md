# Chapter 7 - Big Data with Python

The methods and the data websites are a bit too old to be adapted - these kind of techniques can be learn in datacamp.
We can try to code the code from the second edition book though, but not now.

## Summary

In this chapter, we were introduced to big data and its uses in finance. Big data tools
provide the scalability and reliability of analyzing big data in the area of risk and
credit analytics, handling data coming in from multiple sources. Apache Hadoop
is one popular tool for financial institutions and enterprises in meeting these big
data needs.
Apache Hadoop is an open source framework and written in Java. To help us get
started quickly with Hadoop, we downloaded a QuickStart VM from Cloudera that
comes with CentOS and Hadoop 2.0 running on VirtualBox. The main components
in Hadoop are the HDFS file store, YARN, and MapReduce. We learned about
Hadoop by writing a map and reduce program in Python to perform a word count
on an e-book. Moving on, we downloaded a dataset of the daily prices of a stock and
counted the number of percentage intraday price changes. The outputs were taken
for further analysis.
Before we begin to manage big data, we will need an avenue to store this data.
The nature of digital data is varied, and other means of storing data became the
motivation for non-SQL products. One non-relational database language is NoSQL.
Because of its simple design, it can also be said to be faster in certain circumstances.
One use of NoSQL for finance is the storage of incoming tick data.
We looked at obtaining MongoDB as our NoSQL database server, and the PyMongo
module as a way of using Python to interact with MongoDB. After performing
a simple test connection to the server with Python, we learned the concepts of
databases and collections that are used to store data with PyMongo. Then, a few
sample tick data were created and stored as a collection in the BSON format.
We investigated how to delete, count, sort, and filter this tick data. These simple
operations will enable us to begin storing data continuously and can be later
retrieved for further analysis.
In the next chapter, we will take a look at developing an algorithmic trading system
with Python.