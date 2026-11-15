pub mod distance;
pub mod soundness;
use crate::pcs::error::ConfigError;
#[derive(Clone,Copy,Debug,PartialEq,Eq,serde::Serialize,serde::Deserialize)]
pub enum MatrixKind { Specialized, General }
use serde::{Serialize,Deserialize};
#[derive(Clone,Copy,Debug,Serialize,Deserialize)]
pub enum DistanceMode {Analytic,PrimalDual}
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct HyperFoldParams {
 pub mu:usize,pub security_bits:usize,pub r:usize,pub c:usize,
 pub base_field_bits:usize,pub ext_field_bits:usize,pub mu_0:Option<usize>,
 pub distance_mode:DistanceMode,pub matrix_kind:MatrixKind,
}
impl Default for HyperFoldParams {fn default()->Self{Self{mu:20,security_bits:100,r:4,c:2,base_field_bits:128,ext_field_bits:128,mu_0:None,distance_mode:DistanceMode::Analytic,matrix_kind:MatrixKind::Specialized}}}
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct HyperFoldConfig {
 pub params:HyperFoldParams,pub nu:usize,pub mu_0:usize,pub d:usize,
 pub k:Vec<usize>,pub n:Vec<usize>,pub thresholds:Vec<usize>,
 pub distance:f64,pub epsilon_code:f64,pub eta:f64,pub delta:f64,pub list_size:usize,
 pub queries:usize,pub epsilon_field:f64,
}
impl HyperFoldConfig {
 pub fn derive(p:&HyperFoldParams)->Result<Self,ConfigError>{
  if ![2,4,8].contains(&p.r)||![2,4].contains(&p.c)||p.ext_field_bits!=128||![16,32,64,128].contains(&p.base_field_bits){return Err(ConfigError::Unsupported("field/arity/rate"))}
  let nu=p.r.ilog2() as usize;
  let mu_0=if let Some(m)=p.mu_0{m}else{(4..=p.mu).find(|&m|(p.mu-m)%nu==0).ok_or(ConfigError::Unsupported("mu0"))?};
  if mu_0>p.mu||(p.mu-mu_0)%nu!=0||p.mu>=usize::BITS as usize-2{return Err(ConfigError::Unsupported("dimension"))}
  if mu_0+p.c.ilog2() as usize>p.base_field_bits{return Err(ConfigError::Unsupported("base code evaluation domain exceeds base field"))}
  let d=(p.mu-mu_0)/nu;let k:Vec<_>=(0..=d).map(|i|1usize<<(mu_0+nu*i)).collect();let n:Vec<_>=k.iter().map(|x|p.c*x).collect();
  let (thresholds,distance,epsilon_code)=distance::derive(p,&k,&n)?;
  let (eta,delta,list_size,queries,epsilon_field)=soundness::optimize(p,nu,d,&n,distance,epsilon_code)?;
  Ok(Self{params:p.clone(),nu,mu_0,d,k,n,thresholds,distance,epsilon_code,eta,delta,list_size,queries,epsilon_field})
 }
}

#[cfg(test)]mod tests {use super::*;#[test]fn smallest_base_dimension(){for mu in [18,24]{let c=HyperFoldConfig::derive(&HyperFoldParams{mu,r:8,..Default::default()}).unwrap();assert_eq!(c.mu_0,6);assert_eq!(c.d,(mu-6)/3);assert_eq!(c.k[0],64);}}}

#[cfg(test)]mod final_rule {use super::*;#[test]fn lower_bound_four(){for (mu,r,want) in [(20,2,4),(20,4,4),(20,8,5),(18,8,6)]{let c=HyperFoldConfig::derive(&HyperFoldParams{mu,r,..Default::default()}).unwrap();assert_eq!(c.mu_0,want);}}}
