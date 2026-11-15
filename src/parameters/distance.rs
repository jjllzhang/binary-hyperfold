use super::{DistanceMode,HyperFoldParams};
use crate::pcs::error::ConfigError;
/// Analytic certificate uses integer ceilings, not rounded-down zero thresholds.
pub fn derive(p:&HyperFoldParams,k:&[usize],n:&[usize])->Result<(Vec<usize>,f64,f64),ConfigError>{
 if matches!(p.distance_mode,DistanceMode::PrimalDual){return Err(ConfigError::Unsupported("PrimalDual evaluator not yet implemented"))}
 let mut t=vec![k[0]];for i in 1..n.len(){t.push(p.r*t[i-1]+n[i].div_ceil(p.base_field_bits));}
 let delta=n.iter().zip(&t).map(|(&n,&t)| (n as f64-t as f64+1.0)/n as f64).fold(1.0,f64::min);
 let q=2f64.powi(p.base_field_bits as i32);
 // Bernoulli: (1-1/q)^(T+1)>1/2 suffices for the 2/q bound.
 if t.iter().any(|&t|(t as f64+1.0)*2.0>=q)||delta<=0.0{return Err(ConfigError::NoBudget)}
 Ok((t,delta,2.0*(n.len()-1) as f64/q))
}
