#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("unsupported parameter: {0}")]
    Unsupported(&'static str),
    #[error("no soundness budget for these parameters")]
    NoBudget,
}
#[derive(Debug, thiserror::Error)]
pub enum VerifierError {
    #[error("proof does not verify")]
    InvalidProof,
}
