"""Directed binomial-log intervals using exact small counts / Robbins bounds.
Only 3 < pi < 4 is needed; no floating-point pi constant is assumed.
"""
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache
from math import comb
import arithmetic as ar


def add(*intervals):
    lo=hi=D(0)
    for left,right in intervals:
        lo=ar.down(lambda:lo+left);hi=ar.up(lambda:hi+right)
    return lo,hi


def mul(interval,value):
    value=F(value)
    lo,hi=interval if value>=0 else interval[::-1]
    return (ar.down(lambda:ar.down(lambda:lo*D(value.numerator))/D(value.denominator)),
            ar.up(lambda:ar.up(lambda:hi*D(value.numerator))/D(value.denominator)))


@lru_cache(None)
def log_e_interval():
    with localcontext() as ctx:
        ctx.prec=ar.PREC
        x=D(2).ln();lo,hi=x.next_minus(),x.next_plus()
    return ar.down(lambda:D(1)/hi),ar.up(lambda:D(1)/lo)


@lru_cache(None)
def log_binom(n,s):
    if not 0<=s<=n:raise ValueError('binomial index outside range')
    if s in (0,n):return D(0),D(0)
    if n<=4096:return ar.log2_interval(comb(n,s))
    v=n-s
    entropy=add(mul(ar.log2_interval(n),n),mul(ar.log2_interval(s),-s),mul(ar.log2_interval(v),-v))
    l6,_=ar.log2_interval(6);_,u8=ar.log2_interval(8)
    prefactor=mul(add(ar.log2_interval(n),mul(ar.log2_interval(s),-1),mul(ar.log2_interval(v),-1),(-u8,-l6)),F(1,2))
    low_c=F(1,12*n+1)-F(1,12*s)-F(1,12*v)
    high_c=F(1,12*n)-F(1,12*s+1)-F(1,12*v+1)
    correction=mul(log_e_interval(),low_c)[0],mul(log_e_interval(),high_c)[1]
    lo,hi=add(entropy,prefactor,correction)
    return max(D(0),lo),max(D(0),hi)
