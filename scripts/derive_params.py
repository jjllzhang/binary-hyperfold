#!/usr/bin/env python3
"""Exact rational reproduction of specification section 9.2; no specification file is bundled."""
from fractions import Fraction as F

q = 1 << 128
r, nu, depth = 4, 2, 6
n0, k0 = 512, 256

# 1. 验证各层零坐标阈值与相对最小距离下界 Delta = 29/64
n, t = n0, k0
rows = [(0, n, t, n - t + 1)]
for i in range(1, depth + 1):
    n *= r
    assert n % 128 == 0
    prev_t = t
    t = r * t + n // 128
    assert r * (prev_t - 1) < t <= n
    rows.append((i, n, t, n - t + 1))

assert [row[2] for row in rows[1:]] == [1040, 4224, 17152, 69632, 282624, 1146880]
N = n
Delta = F(29, 64)
assert all(F(row[3], row[1]) >= Delta for row in rows)
child_length_sum = (N - n0) // (r - 1)
assert child_length_sum == 698880

setup_upper = F(12, q)  # epsilon_code < 12 / 2^128

# 2. 验证逼近 1.5-Johnson 界的 MCA 与总知识健全性界
eta = F(1, 64)
delta = F(17, 100)
a_val = 1 - Delta + eta
assert a_val == F(9, 16)
assert F(89, 108) ** 3 < a_val < F(83, 100) ** 3
assert F(89, 108) - F(3, 4) == F(2, 27)  # 证明 B^\star - sigma > 2/27 => K_eta < 27
assert 2 * delta < Delta                 # 证明 L = 1

field_num = 2 * nu * depth + 1 + F(nu, 1) / eta * (F(1, 4) * child_length_sum + depth * 27)
assert field_num == 22384921

total_ell400 = setup_upper + field_num / q + (1 - delta) ** 400
total_ell373 = setup_upper + field_num / q + (1 - delta) ** 373
assert total_ell400 < F(1, 1 << 103)
assert total_ell373 < F(1, 1 << 100)

print("All exact rational parameter checks passed successfully!")
