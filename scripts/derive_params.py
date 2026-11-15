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

# Optional rational verification of schedules printed by the Rust CLI as JSON.
# This does not replace the B_fine setup certificate for the primal/dual mode.
def certify_schedule(config):
    from decimal import Decimal, localcontext
    from fractions import Fraction
    p = config['params']
    assert p['distance_mode'] == 'Analytic'
    r, c, mu = p['r'], p['c'], p['mu']
    width = p['base_field_bits']
    nu = r.bit_length()-1
    mu0 = config['mu_0']
    assert mu0 >= 0 and (mu-mu0) % nu == 0
    depth = (mu-mu0)//nu
    n = [c*(1 << (mu0+nu*i)) for i in range(depth+1)]
    t = [1 << mu0]
    for ni in n[1:]:
        t.append(r*t[-1] + (ni+width-1)//width)
    assert config['thresholds'] == t and config['n'] == n
    assert all(2*(ti+1) < 1 << width for ti in t)
    exact_distance = min(Fraction(ni-ti+1, ni) for ni, ti in zip(n,t))
    distance = Fraction.from_float(config['distance'])
    assert distance <= exact_distance
    eta, delta = map(Fraction.from_float, (config['eta'], config['delta']))
    a = 1-distance+eta
    assert 0 < eta < distance and delta > 0 and (1-delta)**3 >= a
    with localcontext() as ctx:
        ctx.prec = 90
        da = Decimal(a.numerator)/Decimal(a.denominator)
        sigma = Fraction(da.sqrt())
        cube = da ** (Decimal(1)/Decimal(3))
        bstar = Fraction(cube)
    e = Fraction(1,10**80)
    slo, shi = sigma-e, sigma+e
    blo = bstar-e
    assert slo**2 < a < shi**2 and blo**3 < a and blo > shi
    Kupper = max(Fraction(2)/(blo-shi), 6)
    L = 1 if 2*delta < distance else int((distance-delta)/(distance-2*delta+delta**2))
    assert config['list_size'] >= L
    field = (Fraction(L*(L-1),2)*mu+(2*nu*depth+1)*L + Fraction(nu)/eta*((1-slo)*sum(n[:-1])+depth*Kupper))/ (1 << 128)
    code = Fraction(2*depth,1 << width)
    ell = config['queries']
    assert code+field+(1-delta)**ell < Fraction(1,1 << p['security_bits'])
    return {'mu':mu,'r':r,'c':c,'mu0':mu0,'queries':ell,'certificate':'analytic integer ceilings + exact rational radius/budget check'}

if __name__ == '__main__':
    import argparse, json
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-json', help='JSON schedule file produced by bench_pcs --schedule-json')
    args=parser.parse_args()
    if args.verify_json:
        schedules=json.load(open(args.verify_json))
        if isinstance(schedules,dict): schedules=[schedules]
        print(json.dumps([certify_schedule(row) for row in schedules],indent=2))
