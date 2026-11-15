#!/usr/bin/env python3
"""Field-independent directed arithmetic helpers.
Selected unchanged functions from the prior parameter-audit arithmetic module;
no prime-field constants, QA formulas or prior policy budgets are imported.
"""
from decimal import Decimal as D, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as F
from functools import lru_cache
import math

PREC=80
ROOT_SCALE=10**40

def directed(fn, mode):
    with localcontext() as ctx:
        ctx.prec = PREC
        ctx.rounding = mode
        return +fn()


def up(fn):
    return directed(fn, ROUND_CEILING)


def down(fn):
    return directed(fn, ROUND_FLOOR)


@lru_cache(None)
def log2_interval(n: int):
    assert n >= 1
    if n & (n - 1) == 0:
        return D(n.bit_length() - 1), D(n.bit_length() - 1)
    with localcontext() as ctx:
        ctx.prec = PREC
        a, b = D(n).ln(), D(2).ln()
        al, ah = a.next_minus(), a.next_plus()
        bl, bh = b.next_minus(), b.next_plus()
    return down(lambda: al / bh), up(lambda: ah / bl)


def fraction_log2_interval(x: F):
    assert x > 0
    al, ah = log2_interval(x.numerator)
    bl, bh = log2_interval(x.denominator)
    return down(lambda: al - bh), up(lambda: ah - bl)


def integer_cuberoot(n: int):
    assert n >= 0
    if not n:
        return 0
    x = 1 << ((n.bit_length() + 2) // 3)
    while True:
        y = (2 * x + n // (x * x)) // 3
        if y >= x:
            break
        x = y
    while (x + 1)**3 <= n:
        x += 1
    while x**3 > n:
        x -= 1
    return x


def root_interval(x: F, degree: int):
    assert 0 < x < 1
    a = x.numerator * ROOT_SCALE**degree // x.denominator
    z = math.isqrt(a) if degree == 2 else integer_cuberoot(a)
    lower, upper = F(z, ROOT_SCALE), F(z + 1, ROOT_SCALE)
    assert lower**degree <= x <= upper**degree
    return lower, upper

