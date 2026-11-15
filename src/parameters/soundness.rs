//! Floating point schedule search; exact-rational checking lives in scripts/derive_params.py.
use super::HyperFoldParams;
use crate::pcs::error::ConfigError;
pub fn field_error(
    mu: usize,
    nu: usize,
    d: usize,
    n: &[usize],
    distance: f64,
    eta: f64,
    delta: f64,
) -> Option<(usize, f64)> {
    let a = 1.0 - distance + eta;
    let b = a.cbrt();
    let sigma = a.sqrt();
    if !(eta > 0.0 && eta < distance && delta > 0.0 && delta <= 1.0 - b) {
        return None;
    }
    let list = if 2.0 * delta < distance {
        1
    } else {
        ((distance - delta) / (distance - 2.0 * delta + delta * delta)).floor() as usize
    };
    let l = list as f64;
    let k = (2.0 / (b - sigma)).max(6.0);
    let sum = n[..d].iter().map(|x| *x as f64).sum::<f64>();
    let err = (l * (l - 1.0) / 2.0 * mu as f64
        + (2 * nu * d + 1) as f64 * l
        + nu as f64 / eta * ((1.0 - sigma) * sum + d as f64 * k))
        / 2f64.powi(128);
    Some((list, err))
}
pub fn query_count(security: usize, code: f64, field: f64, delta: f64) -> Option<usize> {
    let budget = 2f64.powi(-(security as i32)) - code - field;
    if budget <= 0.0 {
        return None;
    }
    let mut ell = (budget.ln() / (-delta).ln_1p()).ceil() as usize;
    while code + field + (1.0 - delta).powf(ell as f64) >= 2f64.powi(-(security as i32)) {
        ell += 1;
    }
    Some(ell)
}
pub fn optimize(
    p: &HyperFoldParams,
    nu: usize,
    d: usize,
    n: &[usize],
    distance: f64,
    code: f64,
) -> Result<(f64, f64, usize, usize, f64), ConfigError> {
    let mut best = None;
    for i in 1..=1024 {
        let eta = distance * i as f64 / 1025.0;
        let delta = (1.0 - (1.0 - distance + eta).cbrt()) * (1.0 - 1e-12);
        let Some((list, field)) = field_error(p.mu, nu, d, n, distance, eta, delta) else {
            continue;
        };
        if let Some(q) = query_count(p.security_bits, code, field, delta) {
            if best
                .as_ref()
                .is_none_or(|x: &(f64, f64, usize, usize, f64)| q < x.3)
            {
                best = Some((eta, delta, list, q, field));
            }
        }
    }
    best.ok_or(ConfigError::NoBudget)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paper_example() {
        let n: Vec<_> = (0..=6).map(|i| 512usize * 4usize.pow(i)).collect();
        let (l, f) = field_error(20, 2, 6, &n, 29.0 / 64.0, 1.0 / 64.0, 0.17).unwrap();
        assert_eq!(l, 1);
        let code = 12.0 / 2f64.powi(128);
        assert_eq!(query_count(100, code, f, 0.17), Some(373));
        assert!(code + f + 0.83f64.powi(400) < 2f64.powi(-103));
    }
}
