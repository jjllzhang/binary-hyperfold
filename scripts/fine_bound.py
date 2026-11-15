#!/usr/bin/env python3
"""Certified fast UPPER enclosure of Binary HyperFold B_fine, NOT exact DP.

Keep the a=2..r grouping and the minimum shortening-count constraints.
a=2: exact coefficient identity binom(n-s,m)*r**m (logarithm enclosed).
a>=3: binomial coefficient upper bound with the correct truncated support.
Integer-ratio geometric tails tighten the sum around each regional maximum.
No polynomial expansion or linear-in-n summation. Large-field preconditions
are checked; callers must not silently substitute this formula outside them.
"""
import sys
from pathlib import Path
from decimal import Decimal as D
from fractions import Fraction as F
from functools import lru_cache
from math import comb

sys.path.insert(0,str(Path(__file__).resolve().parent/'source-helpers'))
import arithmetic as ar
from binomial_interval import log_binom


class UnsupportedFastBound(ValueError):
    pass


def require(value, reason):
    if not value: raise ValueError(reason)


def ceil_div(a,b):
    return -((-a)//b)


def supports_fast(n,r,q):
    # q**a>n for all a>=2; plateau monotonicity for all groups.
    return q*q>n and (q-1)**r >= n*(r*n)**r


@lru_cache(None)
def binom_interval(n,s):
    return log_binom(n,s)


def ratio_interval(n,k,t,b,a,q,s):
    low,high=binom_interval(n-b,s-b)
    ql,qh=ar.log2_interval(q)
    exponent=a*(t-k+b-s)
    el=ar.down(lambda:D(exponent)*(ql if exponent>=0 else qh))
    eh=ar.up(lambda:D(exponent)*(qh if exponent>=0 else ql))
    return ar.down(lambda:low+el),ar.up(lambda:high+eh)


def crossing(n,k,t,b,a,q,lo,hi):
    original_lo,original_hi=lo,hi
    while lo<=hi:
        mid=(lo+hi)//2
        lower,upper=ratio_interval(n,k,t,b,a,q,mid)
        if lower>=0:lo=mid+1
        elif upper<=0:hi=mid-1
        else:
            left,right=mid-1,mid+1
            while left>=original_lo and ratio_interval(n,k,t,b,a,q,left)[0]<0:left-=1
            while right<=original_hi and ratio_interval(n,k,t,b,a,q,right)[1]>0:right+=1
            require(right-left-1<=16,'count-envelope crossing interval unexpectedly wide')
            return left,right
    return hi,lo


def mode_ratio_pair(n,s,r,a,T,q):
    m=T-r*s;v=n-s
    if a==2:
        left=(q-1)**r
        right=(s+1)*q**a*r**r
        for j in range(r):left*=m-j
        for j in range(1,r):right*=v-m+j
    else:
        left=v*(q-1)**r
        right=(s+1)*q**a
        for j in range(r):
            left*=m-j
            right*=r*v-j
    return left,right


def mode_increasing(n,s,r,a,T,q):
    left,right=mode_ratio_pair(n,s,r,a,T,q)
    return left>right


def mode(n,r,a,T,q,lo,hi):
    while lo<hi:
        mid=(lo+hi)//2
        if mode_increasing(n,mid,r,a,T,q):lo=mid+1
        else:hi=mid
    return lo


def coef_log_upper(n,s,r,a,T):
    m=T-r*s;v=n-s
    if m<0 or m>(a-1)*v:return D('-Infinity')
    if a==2:
        _,rh=ar.log2_interval(r)
        return ar.up(lambda:log_binom(v,m)[1]+D(m)*rh)
    return log_binom(r*v,m)[1]


def term_upper(n,k,t,b,r,a,T,q,s,kind):
    if kind=='rank':
        rank=k-s;count=log_binom(n,s)[1]
    elif kind=='plateau':
        rank=k-b
        count=ar.up(lambda:log_binom(n,b)[1]-log_binom(s,b)[0])
    elif kind=='singleton':
        rank=t-s;count=log_binom(n,s)[1]
    else:raise ValueError(kind)
    if rank<=0:return D('-Infinity')
    coefficient=coef_log_upper(n,s,r,a,T)
    if coefficient==D('-Infinity'):return coefficient
    _,qh=ar.log2_interval(q);Ql,_=ar.log2_interval(q-1)
    m=T-r*s
    return ar.up(lambda:count+ar.up(lambda:D(a*rank)*qh)+coefficient-ar.down(lambda:D(m+1)*Ql))


def window_sum_upper(n,k,t,b,r,a,T,q,lo,hi,kind):
    if kind=='plateau':
        s=hi
        lower_ratio=F((q-1)**r,n*(r*n)**r)
        factor=F(hi-lo+1)
        if lower_ratio>1:factor=min(factor,lower_ratio/(lower_ratio-1))
    else:
        s=mode(n,r,a,T,q,lo,hi)
        factor=F(1)
        if lo<s:
            num,den=mode_ratio_pair(n,s-1,r,a,T,q)
            rho=F(den,num)
            factor+=min(F(s-lo),rho/(1-rho)) if rho<1 else F(s-lo)
        if s<hi:
            num,den=mode_ratio_pair(n,s,r,a,T,q)
            rho=F(num,den)
            factor+=min(F(hi-s),rho/(1-rho)) if rho<1 else F(hi-s)
    _,factor_hi=ar.fraction_log2_interval(factor)
    return ar.up(lambda:term_upper(n,k,t,b,r,a,T,q,s,kind)+factor_hi)


def group_upper(n,k,t,b,r,a,T,q):
    lo=max(0,ceil_div(T-(a-1)*n,r-a+1));hi=t-1
    if lo>hi:return D('-Infinity'),[]
    candidates=[];window_logs=[]
    rank_hi=min(hi,b,k-1)
    if lo<=rank_hi:
        s=mode(n,r,a,T,q,lo,rank_hi)
        candidates.append(('rank',s,term_upper(n,k,t,b,r,a,T,q,s,'rank')))
        window_logs.append(window_sum_upper(n,k,t,b,r,a,T,q,lo,rank_hi,'rank'))
    if b<k:
        l=max(lo,b+1)
        if l<=hi:
            last_p,first_s=crossing(n,k,t,b,a,q,l,hi)
            if l<=last_p:
                candidates.append(('plateau',last_p,term_upper(n,k,t,b,r,a,T,q,last_p,'plateau')))
                window_logs.append(window_sum_upper(n,k,t,b,r,a,T,q,l,last_p,'plateau'))
            if first_s<=hi:
                s=mode(n,r,a,T,q,first_s,hi)
                candidates.append(('singleton',s,term_upper(n,k,t,b,r,a,T,q,s,'singleton')))
                window_logs.append(window_sum_upper(n,k,t,b,r,a,T,q,first_s,hi,'singleton'))
            for s in range(max(l,last_p+1),min(hi,first_s-1)+1):
                value=min(term_upper(n,k,t,b,r,a,T,q,s,'plateau'),term_upper(n,k,t,b,r,a,T,q,s,'singleton'))
                candidates.append(('uncertain_min',s,value))
                window_logs.append(value)
    if not candidates:return D('-Infinity'),[]
    _,count_hi=ar.log2_interval(hi-lo+1)
    _,group_hi=ar.log2_interval(comb(r,a))
    counted=ar.up(lambda:max(z[2] for z in candidates)+count_hi+group_hi)
    _,window_count_hi=ar.log2_interval(len(window_logs))
    geometric=ar.up(lambda:max(window_logs)+window_count_hi+group_hi)
    return min(counted,geometric),candidates


def log_upper(n,k,t,b,r,T,q,include_coarse=True):
    require(1<=k<n and k<=t<=n and 0<=b<=k,'invalid code-state parameters')
    require(r*t<=T<=r*n,'invalid threshold interval')
    if T>=(r-1)*n+t:return D('-Infinity')
    if not supports_fast(n,r,q):
        raise UnsupportedFastBound('Need q^2>n and (q-1)^r>=n(rn)^r; no silent small-field fallback')
    logs=[group_upper(n,k,t,b,r,a,T,q)[0] for a in range(2,r+1)]
    logs=[z for z in logs if z!=D('-Infinity')]
    if not logs:return D('-Infinity')
    _,groups_hi=ar.log2_interval(len(logs))
    value=min(D(0),ar.up(lambda:max(logs)+groups_hi))
    # B_fine <= B: for any fixed constraint ell, summing the nonempty-subset
    # counts gives <= q^(r R_ell)-1; taking individual minima only decreases
    # the sum. Every P_(r,a) coefficient is <= the F_r coefficient.
    if not include_coarse:return value
    _,qh=ar.log2_interval(q);Ql,_=ar.log2_interval(q-1)
    coarse=ar.up(lambda:log_binom(r*n,T)[1]+ar.up(lambda:D(r*t)*qh)-ar.down(lambda:D(T+1)*Ql))
    return min(value,coarse)


def threshold(n,k,t,b,r,q,depth,family_bits):
    require(supports_fast(n,r,q),'unsupported fast-field conditions; use explicitly named alternative')
    _,dh=ar.log2_interval(depth)
    target=ar.down(lambda:-D(family_bits)-dh)
    lo,hi=r*t,(r-1)*n+t
    while lo<hi:
        mid=(lo+hi)//2
        if log_upper(n,k,t,b,r,mid,q)<=target:hi=mid
        else:lo=mid+1
    value=log_upper(n,k,t,b,r,lo,q)
    require(value<=target,'final passing endpoint not certified')
    return lo,value


def chain(mu,mu0,r,c,w0,code_bits):
    nu=r.bit_length()-1
    require(2**nu==r and (mu-mu0)%nu==0 and mu>mu0,'invalid geometry')
    depth=(mu-mu0)//nu;q=2**w0
    n=c*2**mu0;k=2**mu0;t=k;tau=n-k
    require(n<=q,'additive RS base domain too large')
    out=[]
    for level in range(1,depth+1):
        T,e1=threshold(n,k,t,n-tau,r,q,depth,code_bits+1)
        U,e2=threshold(n,n-k,tau,n-t,r,q,depth,code_bits+1)
        out.append({'level':level,'n_previous':n,'k_previous':k,'t_previous':t,'tau_previous':tau,
                    'n':r*n,'k':r*k,'t':T,'tau':U,'Delta_exact':str(F(r*n-T+1,r*n)),
                    'primal_log2_upper':str(e1),'dual_log2_upper':str(e2),
                    'family_layer_budget_exact':str(F(1,depth*2**(code_bits+1))),
                    'method':'fine_group_binomial_geometric_upper_with_coarse_min',
                    'exact_Bfine_minimum_claimed':False})
        n,k,t,tau=r*n,r*k,T,U
    return out
