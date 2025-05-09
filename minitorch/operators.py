"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List


#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    return x*y

def id(x: float) -> float:
    return x

def add(x :float, y :float):
    return x+y

def neg(x :float):
    return -x

def lt(x:float, y:float):
    return x < y

def eq(x: float, y :float):
    return x == y

def max(x :float, y :float):
    if x > y :
        return x
    return y

def is_close(x:float, y:float):
    if abs(x-y) < 1e-2:
        return True
    return False

def sigmoid(x: float):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))
    
# ReLU 激活函数
def relu(x: float):
    return max(0, x)



# 自然对数函数
def log(x:float):
    if x <= 0:
        raise ValueError("log function is only defined for positive numbers")
    return math.log(x)

# 自然对数的反向传播函数
def log_back(x:float, grad_output:float):
    if x <= 0:
        raise ValueError("log function is only defined for positive numbers")
    return grad_output / x

# 指数函数
def exp(x:float):
    return math.exp(x)

# 指数函数的反向传播函数
def exp_back(x:float, grad_output:float):
    return grad_output * math.exp(x)

# 倒数函数
def inv(x:float):
    if x == 0:
        raise ValueError("inv function is not defined for zero")
    return 1.0 / x

# 倒数函数的反向传播函数
def inv_back(x:float, grad_output:float):
    if x == 0:
        raise ValueError("inv function is not defined for zero")
    return -grad_output / (x ** 2)

# ReLU 的反向传播函数
def relu_back(x:float, grad_output: float):
    if x > 0:
        return grad_output
    else:
        return 0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(arr:  Iterable, fn: Callable):
    return [fn(i) for i in arr]

def zipWith(a: Iterable, b: Iterable):
    return [(x1, x2) for x1, x2 in zip(a, b)]

def reduce(fn: Callable, arr: Iterable, initialValue: float = 0):
    sum = initialValue
    for x in arr:
        sum = fn(sum, x)
    return sum

def negList(arr: Iterable):
    return map(arr, neg)

def addLists(arr1: Iterable, arr2: Iterable):
    return [x[0] + x[1] for x in zipWith(arr1, arr2)]

def sum(arr: Iterable):
    def sum(current: float, x: float):
        return current+x
    return reduce(sum, arr)


def prod(arr:Iterable):
    return reduce(lambda p, x: p*x, arr, initialValue=1)
