use std::collections::HashMap;

use anyhow::{Context, Result};

use crate::{
    ebi_objects::stochastic_labelled_petri_net::StochasticLabelledPetriNet,
    ebi_traits::{
        ebi_trait_finite_stochastic_language::EbiTraitFiniteStochasticLanguage,
        ebi_trait_queriable_stochastic_language::EbiTraitQueriableStochasticLanguage,
    },
    math::fraction::Fraction,
};

pub trait StochasticMarkovianAbstraction {
    /// Compute the uEMSC conformance measure for two stochastic languages
    fn markovian_uemsc(
        &self,
        language2: Box<dyn EbiTraitQueriableStochasticLanguage>,
        k: usize,
    ) -> Result<Fraction>;
}

/// Represents a k-th order Markovian abstraction of a stochastic language
pub struct MarkovianAbstraction {
    /// The order of the abstraction (k)
    pub k: usize,
    /// The mapping from subtraces to normalized frequencies
    pub abstraction: HashMap<Vec<String>, Fraction>,
}


// Trait Implementation

// Implementation for finite stochastic languages (logs)
impl StochasticMarkovianAbstraction for dyn EbiTraitFiniteStochasticLanguage {
    
    fn markovian_uemsc(
        &self,
        language2: Box<dyn EbiTraitQueriableStochasticLanguage>,
        k: usize,
    ) -> Result<Fraction> {
        // Validate k
        if k < 2 {
            return Err(anyhow::anyhow!("k must be at least 2"));
        }
        
        // Discard the language2 parameter for now
        let _ = &language2;

        // Step 1: Compute abstraction for the log (self)
        let abstraction1 = compute_abstraction_for_log(self, k)
            .context("Computing abstraction for first language")?;
        
        // Step 2: Compute abstraction for language2
        // Might need to cast to known types like StochasticLabelledPetriNet
        // TODO: Implement this ; currently just returns a dummy abstraction
        let abstraction2 = {
            let mut abstraction = HashMap::new();
            let example_subtrace = vec!["b".to_string(), "c".to_string()];
            abstraction.insert(example_subtrace, Fraction::from((1, 1)));
            
            MarkovianAbstraction { k, abstraction }
        };
        
        // Step 3: Compute the distance between the abstractions
        compute_uemsc_distance(&abstraction1, &abstraction2)
            .context("Computing uEMSC distance between abstractions")
    }
}

/// Compute the k-th order Markovian abstraction for a log (finite stochastic language)
pub fn compute_abstraction_for_log(
    _log: &dyn EbiTraitFiniteStochasticLanguage, 
    k: usize
) -> Result<MarkovianAbstraction> {

    // TODO: Implement the full algorithm to compute Markovian abstraction for logs
    // Just dummy for now
    let mut abstraction = HashMap::new();
    let example_subtrace = vec!["a".to_string(), "b".to_string()];
    abstraction.insert(example_subtrace, Fraction::from((1, 1)));

    Ok(MarkovianAbstraction { k, abstraction })
}

/// Compute the k-th order Markovian abstraction for a stochastic labelled Petri net
pub fn compute_abstraction_for_petri_net(
    _petri_net: &StochasticLabelledPetriNet,
    k: usize
) -> Result<MarkovianAbstraction> {

    // TODO: Implement the algorithm
    // 1. Get the embedded SNFA from the SLPN
    // 2. Perform t-removal to get a tau-free SNFA
    // 3. Patch the SNFA with +/- markers
    // 4. Solve the linear system (I-Δ)^Tx = [1 0...0]ᵀ
    // 5. Compute f_l^k and normalize to get m_l^k
    // Just dummy for now
    let mut abstraction = HashMap::new();
    let example_subtrace = vec!["c".to_string(), "d".to_string()];
    abstraction.insert(example_subtrace, Fraction::from((1, 1)));
    
    Ok(MarkovianAbstraction { k, abstraction })
}

/// Compute a distance measure between two Markovian abstractions
pub fn compute_uemsc_distance(
    _abstraction1: &MarkovianAbstraction,
    _abstraction2: &MarkovianAbstraction
) -> Result<Fraction> {
    // TODO: Implement the m^k-uEMSC distance calculation
    
    Ok(Fraction::from((1, 2))) // 0.5 as a fraction
}

#[cfg(test)]
mod tests {
    // Tests will be added when implementation is complete
}