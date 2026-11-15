//! Fast B_fine upper-envelope *search*, using exact integer mode ratios.
//! Floating logarithms are proposals, not directed certificates. Certify every
//! exported schedule with the included decimal-interval reference tool.
use crate::pcs::error::ConfigError;
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};
const EPS: f64 = 1e-4;
fn log_fact(n: usize, upper: bool) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let x = n as f64;
    let pi2 = if upper { 8f64 } else { 6f64 };
    (x + 0.5) * x.log2() - x * std::f64::consts::LOG2_E
        + 0.5 * pi2.log2()
        + std::f64::consts::LOG2_E / if upper { 12.0 * x } else { 12.0 * x + 1.0 }
}
fn binom(n: usize, k: usize) -> (f64, f64) {
    if k == 0 || k == n {
        return (0.0, 0.0);
    }
    if n <= 4096 {
        let k = k.min(n - k);
        let mut x = BigUint::one();
        for i in 0..k {
            x = x * BigUint::from(n - i) / BigUint::from(i + 1);
        }
        let bits = x.bits();
        let top = if bits > 53 {
            (&x >> (bits - 53) as usize).to_u64().unwrap() as f64
        } else {
            x.to_u64().unwrap() as f64
        };
        let log = top.log2() + (bits.saturating_sub(53)) as f64;
        return (log - EPS, log + EPS);
    }
    let lo = log_fact(n, false) - log_fact(k, true) - log_fact(n - k, true);
    let hi = log_fact(n, true) - log_fact(k, false) - log_fact(n - k, false);
    (lo.max(0.0) - EPS, hi.max(0.0) + EPS)
}
fn choose(n: usize, k: usize) -> usize {
    (0..k).fold(1, |a, i| a * (n - i) / (i + 1))
}
#[derive(Clone, Copy)]
enum Kind {
    Rank,
    Plateau,
    Singleton,
}
struct Bound {
    n: usize,
    k: usize,
    t: usize,
    b: usize,
    r: usize,
    target: usize,
    w: usize,
    q: BigUint,
}
impl Bound {
    fn ratio(&self, s: usize, a: usize) -> (BigUint, BigUint) {
        let m = self.target - self.r * s;
        let v = self.n - s;
        let qm = &self.q - BigUint::one();
        let mut left = qm.pow(self.r as u32);
        let mut right = BigUint::from(s + 1) * self.q.pow(a as u32);
        if a == 2 {
            right *= BigUint::from(self.r).pow(self.r as u32);
            for j in 0..self.r {
                left *= BigUint::from(m - j)
            }
            for j in 1..self.r {
                right *= BigUint::from(v - m + j)
            }
        } else {
            left *= BigUint::from(v);
            for j in 0..self.r {
                left *= BigUint::from(m - j);
                right *= BigUint::from(self.r * v - j)
            }
        }
        (left, right)
    }
    fn mode(&self, a: usize, mut lo: usize, mut hi: usize) -> usize {
        while lo < hi {
            let mid = (lo + hi) / 2;
            let (num, den) = self.ratio(mid, a);
            if num > den { lo = mid + 1 } else { hi = mid }
        }
        lo
    }
    fn q_lower(&self) -> f64 {
        let e = 2f64.powi(-(self.w as i32));
        self.w as f64 - e * std::f64::consts::LOG2_E / (1.0 - e)
    }
    fn term(&self, a: usize, s: usize, kind: Kind) -> f64 {
        let (rank, count) = match kind {
            Kind::Rank => (self.k - s, binom(self.n, s).1),
            Kind::Plateau => (
                self.k - self.b,
                binom(self.n, self.b).1 - binom(s, self.b).0,
            ),
            Kind::Singleton => (self.t - s, binom(self.n, s).1),
        };
        if rank == 0 {
            return f64::NEG_INFINITY;
        }
        let m = self.target - self.r * s;
        let v = self.n - s;
        if m > (a - 1) * v {
            return f64::NEG_INFINITY;
        }
        let coeff = if a == 2 {
            binom(v, m).1 + m as f64 * (self.r as f64).log2()
        } else {
            binom(self.r * v, m).1
        };
        count + (a * rank) as f64 * self.w as f64 + coeff - (m + 1) as f64 * self.q_lower() + EPS
    }
    fn ratio_f64(num: &BigUint, den: &BigUint) -> f64 {
        // Keep a shared exponent so ratios near one do not overflow at 128-bit fields.
        let shift = num.bits().max(den.bits()).saturating_sub(500) as usize;
        let n = (num >> shift).to_f64().unwrap();
        let d = (den >> shift).to_f64().unwrap();
        n / d
    }
    fn window(&self, a: usize, lo: usize, hi: usize, kind: Kind) -> f64 {
        let s = if matches!(kind, Kind::Plateau) {
            hi
        } else {
            self.mode(a, lo, hi)
        };
        let mut factor = 1.0;
        if matches!(kind, Kind::Plateau) {
            let qminus = &self.q - BigUint::one();
            let num = qminus.pow(self.r as u32);
            let den = BigUint::from(self.n) * BigUint::from(self.r * self.n).pow(self.r as u32);
            let rho = Self::ratio_f64(&den, &num);
            factor = (hi - lo + 1) as f64;
            if rho < 1.0 {
                factor = factor.min(1.0 / (1.0 - rho));
            }
        } else {
            if lo < s {
                let (n, d) = self.ratio(s - 1, a);
                let rho = Self::ratio_f64(&d, &n);
                factor += if rho < 1.0 {
                    ((s - lo) as f64).min(rho / (1.0 - rho))
                } else {
                    (s - lo) as f64
                };
            }
            if s < hi {
                let (n, d) = self.ratio(s, a);
                let rho = Self::ratio_f64(&n, &d);
                factor += if rho < 1.0 {
                    ((hi - s) as f64).min(rho / (1.0 - rho))
                } else {
                    (hi - s) as f64
                };
            }
        }
        self.term(a, s, kind) + factor.log2() + EPS
    }
    fn crossing(&self, a: usize, lo: usize, hi: usize) -> Result<(isize, usize), ConfigError> {
        let (mut l, mut h) = (lo as isize, hi as isize);
        let compare = |s: usize| {
            let (x, y) = binom(self.n - self.b, s - self.b);
            let v = a as f64
                * (self.t as f64 - self.k as f64 + self.b as f64 - s as f64)
                * self.w as f64;
            (x + v, y + v)
        };
        while l <= h {
            let m = ((l + h) / 2) as usize;
            let (low, high) = compare(m);
            if low >= 0.0 {
                l = m as isize + 1
            } else if high <= 0.0 {
                h = m as isize - 1
            } else {
                let mut left = m as isize - 1;
                let mut right = m + 1;
                while left >= lo as isize && compare(left as usize).0 < 0.0 {
                    left -= 1
                }
                while right <= hi && compare(right).1 > 0.0 {
                    right += 1
                }
                if right as isize - left > 18 {
                    return Err(ConfigError::Unsupported(
                        "uncertain shortening-count crossing",
                    ));
                }
                return Ok((left, right));
            }
        }
        Ok((h, l as usize))
    }
    fn group(&self, a: usize) -> Result<f64, ConfigError> {
        let signed = self.target as isize - ((a - 1) * self.n) as isize;
        let lo = if signed > 0 {
            (signed as usize).div_ceil(self.r - a + 1)
        } else {
            0
        };
        let hi = self.t - 1;
        if lo > hi {
            return Ok(f64::NEG_INFINITY);
        }
        let mut logs = vec![];
        let rank_hi = hi.min(self.b).min(self.k - 1);
        if lo <= rank_hi {
            logs.push(self.window(a, lo, rank_hi, Kind::Rank));
        }
        if self.b < self.k {
            let l = lo.max(self.b + 1);
            if l <= hi {
                let (last, first) = self.crossing(a, l, hi)?;
                if last >= l as isize {
                    logs.push(self.window(a, l, last as usize, Kind::Plateau));
                }
                if first <= hi {
                    logs.push(self.window(a, first, hi, Kind::Singleton));
                }
                for s in l.max((last + 1) as usize)..=hi.min(first.saturating_sub(1)) {
                    logs.push(
                        self.term(a, s, Kind::Plateau)
                            .min(self.term(a, s, Kind::Singleton)),
                    );
                }
            }
        }
        if logs.is_empty() {
            return Ok(f64::NEG_INFINITY);
        }
        Ok(logs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            + (logs.len() as f64).log2()
            + (choose(self.r, a) as f64).log2()
            + EPS)
    }
    fn evaluate(&self) -> Result<f64, ConfigError> {
        if self.target >= (self.r - 1) * self.n + self.t {
            return Ok(f64::NEG_INFINITY);
        }
        let mut logs = vec![];
        for a in 2..=self.r {
            let x = self.group(a)?;
            if x.is_finite() {
                logs.push(x);
            }
        }
        let fine = if logs.is_empty() {
            f64::NEG_INFINITY
        } else {
            logs.iter().copied().fold(f64::NEG_INFINITY, f64::max) + (logs.len() as f64).log2()
        };
        let coarse = binom(self.r * self.n, self.target).1
            + (self.r * self.t) as f64 * self.w as f64
            - (self.target + 1) as f64 * self.q_lower();
        Ok(fine.min(coarse).min(0.0) + EPS)
    }
}
fn threshold(
    n: usize,
    k: usize,
    t: usize,
    b: usize,
    r: usize,
    w: usize,
    d: usize,
    bits: usize,
) -> Result<usize, ConfigError> {
    let q = BigUint::one() << w;
    let qm = &q - BigUint::one();
    if &q * &q <= BigUint::from(n)
        || qm.pow(r as u32) < BigUint::from(n) * BigUint::from(r * n).pow(r as u32)
    {
        return Err(ConfigError::Unsupported("B_fine fast-field preconditions"));
    }
    let mut bound = Bound {
        n,
        k,
        t,
        b,
        r,
        target: 0,
        w,
        q,
    };
    let (mut lo, mut hi) = (r * t, (r - 1) * n + t);
    let target = -(bits as f64) - (d as f64).log2();
    while lo < hi {
        let mid = (lo + hi) / 2;
        bound.target = mid;
        if bound.evaluate()? <= target {
            hi = mid
        } else {
            lo = mid + 1
        }
    }
    bound.target = lo;
    if bound.evaluate()? > target {
        return Err(ConfigError::NoBudget);
    }
    Ok(lo)
}
pub fn chain(
    mu: usize,
    mu0: usize,
    r: usize,
    c: usize,
    w: usize,
    security_bits: usize,
) -> Result<(Vec<usize>, Vec<usize>), ConfigError> {
    if mu > 32 {
        return Err(ConfigError::Unsupported(
            "floating B_fine proposer currently limited to mu <= 32; certify externally",
        ));
    }
    let d = (mu - mu0) / (r.ilog2() as usize);
    let (mut n, mut k) = (c * (1 << mu0), 1 << mu0);
    let mut ts = vec![k];
    let mut us = vec![n - k];
    for _ in 0..d {
        let t = *ts.last().unwrap();
        let u = *us.last().unwrap();
        let next = threshold(n, k, t, n - u, r, w, d, security_bits + 2)?;
        let dual = threshold(n, n - k, u, n - t, r, w, d, security_bits + 2)?;
        ts.push(next);
        us.push(dual);
        n *= r;
        k *= r;
    }
    Ok((ts, us))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_thresholds() {
        let (t, u) = chain(20, 8, 4, 2, 128, 100).unwrap();
        assert_eq!(t, [256, 1040, 4181, 16810, 67599, 271885, 1093717]);
        assert_eq!(u, t);
        let (t, _) = chain(20, 8, 8, 2, 128, 100).unwrap();
        assert_eq!(t, [256, 2080, 16689, 133922, 1074800]);
        let (t, _) = chain(20, 8, 4, 2, 32, 100).unwrap();
        assert_eq!(t, [256, 1090, 4483, 18466, 76176, 314615, 1300489]);
    }
}
