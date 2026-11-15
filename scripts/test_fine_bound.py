import unittest
from fractions import Fraction as F
from math import comb
from functools import lru_cache
import fine_bound as fb


@lru_cache(None)
def polynomial_coefficients(v,r,a):
    values=[1]
    for _ in range(v):
        nxt=[0]*(len(values)+a-1)
        for i,x in enumerate(values):
            for j in range(a):nxt[i+j]+=x*comb(r,j)
        values=nxt
    return tuple(values)


def exact_fine(n,k,t,b,r,T,q):
    total=F(0)
    for a in range(2,r+1):
        for s in range(t):
            degree=T-r*s;co=polynomial_coefficients(n-s,r,a)
            if not 0<=degree<len(co):continue
            count=min(F(comb(n,l),comb(s,l))*(q**min(t-l,k-min(l,b))-1)**a for l in range(s+1))
            prob=min(F(1),F(co[degree],(q-1)**degree))
            total+=comb(r,a)*count*prob/F(q-1)
    return min(F(1),total)


def exact_basic(n,k,t,b,r,T,q):
    total=F(0)
    for s in range(t):
        degree=T-r*s;co=polynomial_coefficients(n-s,r,r)
        if not 0<=degree<len(co):continue
        count=min(F(comb(n,l),comb(s,l))*(q**(r*min(t-l,k-min(l,b)))-1) for l in range(s+1))
        total+=count*min(F(1),F(co[degree],(q-1)**degree))/F(q-1)
    return min(F(1),total)


def exact_upper_group_term(n,k,t,b,r,a,T,q,s):
    degree=T-r*s;v=n-s
    if not 0<=degree<=(a-1)*v:return F(0)
    if s<=b:
        rank=k-s
        count=F(comb(n,s))*q**(a*rank) if rank>0 else F(0)
    elif b==k:return F(0)
    else:count=min(F(comb(n,b),comb(s,b))*q**(a*(k-b)),F(comb(n,s))*q**(a*(t-s)))
    coefficient=comb(v,degree)*r**degree if a==2 else comb(r*v,degree)
    return count*coefficient/F((q-1)**(degree+1))


class Tests(unittest.TestCase):
    def test_fine_upper_and_basic_relation(self):
        cases=0;q=65536
        for n,k in [(4,2),(6,2),(8,4)]:
            for t in sorted({k,min(n,k+2),n}):
                for b in sorted({0,k//2,k}):
                    for r in [2,4,8]:
                        for T in range(r*t,(r-1)*n+t+1):
                            exact=exact_fine(n,k,t,b,r,T,q)
                            self.assertLessEqual(exact,exact_basic(n,k,t,b,r,T,q))
                            upper=fb.log_upper(n,k,t,b,r,T,q)
                            if exact:
                                lo,_=fb.ar.fraction_log2_interval(exact)
                                self.assertGreaterEqual(upper,lo)
                            else:self.assertLessEqual(upper,0)
                            cases+=1
        print('Exact B_fine/B/enclosure comparisons:',cases)

    def test_group_maxima_enclose_enumeration(self):
        cases=0;q=65536
        for n,k,t,b,r in [(8,4,6,2,8),(8,4,6,0,4),(8,4,4,4,8),(6,2,4,1,2)]:
            for T in range(r*t,(r-1)*n+t+1):
                for a in range(2,r+1):
                    exact=sum(exact_upper_group_term(n,k,t,b,r,a,T,q,s) for s in range(t))*comb(r,a)
                    upper,_=fb.group_upper(n,k,t,b,r,a,T,q)
                    if exact:
                        lo,_=fb.ar.fraction_log2_interval(exact)
                        self.assertGreaterEqual(upper,lo)
                    else:self.assertEqual(upper,fb.D('-Infinity'))
                    cases+=1
        print('Exact grouped-sum enclosure comparisons:',cases)

    def test_count_endpoint_minima(self):
        cases=0;q=65536
        for n,k,t,b in [(8,4,7,2),(8,4,4,4),(6,2,5,0)]:
            for a in range(2,9):
                for s in range(t):
                    exact=min(F(comb(n,l),comb(s,l))*q**(a*min(t-l,k-min(l,b)))
                              if min(t-l,k-min(l,b))>0 else F(0) for l in range(s+1))
                    if s<=b:
                        closed=F(comb(n,s))*q**(a*(k-s)) if k>s else F(0)
                    elif b==k:closed=F(0)
                    else:closed=min(F(comb(n,b),comb(s,b))*q**(a*(k-b)),F(comb(n,s))*q**(a*(t-s)))
                    self.assertEqual(exact,closed);cases+=1
        print('Count endpoint identities:',cases)

    def test_binomial_intervals(self):
        count=0
        for n in [16,4096,4097,8192,10000]:
            for s in sorted({0,1,n//4,n//2,n-1,n}):
                lo,hi=fb.log_binom(n,s)
                exact_lo,exact_hi=fb.ar.log2_interval(comb(n,s))
                self.assertLessEqual(lo,exact_hi)
                self.assertGreaterEqual(hi,exact_lo)
                count+=1
        print('Exact binomial interval comparisons:',count)

    def test_modes_and_ratios(self):
        count=0;q=65536
        for n in [4,6,8]:
            for r in [2,4,8]:
                for a in range(2,r+1):
                    for T in range(r,r*n):
                        lo=max(0,fb.ceil_div(T-(a-1)*n,r-a+1))
                        hi=min(n-1,T//r)
                        if lo>hi:continue
                        values=[]
                        for s in range(lo,hi+1):
                            degree=T-r*s;v=n-s
                            co=comb(v,degree)*r**degree if a==2 else comb(r*v,degree)
                            values.append(F(comb(n,s)*co*(q-1)**(r*s),q**(a*s)))
                        peak=fb.mode(n,r,a,T,q,lo,hi)
                        self.assertEqual(values[peak-lo],max(values))
                        for s in range(lo,hi):
                            numerator,denominator=fb.mode_ratio_pair(n,s,r,a,T,q)
                            self.assertEqual(F(numerator,denominator),values[s-lo+1]/values[s-lo])
                        count+=1
        print('Mode/ratio enumeration cases:',count)

    def test_preconditions_are_explicit(self):
        self.assertTrue(fb.supports_fast(2**30,8,2**128))
        self.assertFalse(fb.supports_fast(2**20,2,2**16))
        with self.assertRaises(fb.UnsupportedFastBound):
            fb.log_upper(2**20,2**19,2**19,2**19,2,2**20,2**16)


if __name__=='__main__':unittest.main()
