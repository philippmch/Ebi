use std::collections::HashMap;
use std::sync::Arc;

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
    /// Uses Arc<[String]> to reduce memory usage through shared ownership
    pub abstraction: HashMap<Arc<[String]>, Fraction>,
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
            // Convert to Arc<[String]> to match the expected type
            let arc_subtrace = Arc::<[String]>::from(example_subtrace);
            abstraction.insert(arc_subtrace, Fraction::from((1, 1)));
            
            MarkovianAbstraction { k, abstraction }
        };
        
        // Step 3: Compute the distance between the abstractions
        compute_uemsc_distance(&abstraction1, &abstraction2)
            .context("Computing uEMSC distance between abstractions")
    }
}


/// This implements the calculation of the k-th order Stochastic Markovian abstraction
/// for the log (finite stochastic language). First it computes the k-th order multiset
/// markovian abstraction by adding the special start '+' and end '-' markers and then
/// computing the k-trimmed subtraces. Afterwards, the k-th order stochastic markovian
/// abstraction gets computed by normalizing the multiset markovian abstraction.
pub fn compute_abstraction_for_log(
    log: &dyn EbiTraitFiniteStochasticLanguage, 
    k: usize
) -> Result<MarkovianAbstraction> {
    // Validate k
    if k < 2 {
        return Err(anyhow::anyhow!("k must be at least 2 for Markovian abstraction"));
    }
    
    // Initialize f_l^k which stores the expected number of occurrences of each subtrace
    let mut f_l_k: HashMap<Arc<[String]>, Fraction> = HashMap::new();
    
    // For each trace in the log with its probability
    for (trace, probability) in log.iter_trace_probability() {
        // Create a Vec<String> from the Vec<Activity>
        let string_trace: Vec<String> = trace.iter()
            .map(|activity| activity.to_string())
            .collect();
        
        // Compute M_σ^k for this trace (k-th order multiset Markovian abstraction)
        let m_sigma_k = compute_multiset_abstraction_for_trace(&string_trace, k);
        
        // Add contribution to f_l^k
        for (subtrace, occurrences) in m_sigma_k {
            let occurrences_as_fraction = Fraction::from((occurrences, 1));
            // Create an owned contribution using explicit reference operations
            let contribution = {  
                let p_ref: &Fraction = probability;
                let o_ref: &Fraction = &occurrences_as_fraction;
                p_ref * o_ref  // This returns an owned FractionEnum
            };
            
            // Update the expected occurrence count in f_l^k
            f_l_k.entry(subtrace)
                .and_modify(|current| {
                    *current += &contribution; // Add this trace's contribution to existing count
                })
                .or_insert(contribution); // Insert new count if subtrace not seen before
        }
    }
    
    // Calculate the total sum for normalization
    let mut total = Fraction::from((0, 1));
    for value in f_l_k.values() {
        total += value;
    }
    
    // Normalize f_l^k to get m_l^k
    let mut abstraction = HashMap::new();
    for (subtrace, count) in f_l_k {
        let count_ref: &Fraction = &count;
        let total_ref: &Fraction = &total;
        abstraction.insert(subtrace, count_ref / total_ref);
    }
    
    Ok(MarkovianAbstraction { k, abstraction })
}

/// Compute the multiset of k-trimmed subtraces for a given trace 
/// This implements M_σ^k = S_{+σ-}^k
fn compute_multiset_abstraction_for_trace(trace: &[String], k: usize) -> HashMap<Arc<[String]>, usize> {
    
    // Create a new trace with start '+' and end '-' markers
    let mut augmented_trace = Vec::with_capacity(trace.len() + 2);
    augmented_trace.push("+".to_string());
    augmented_trace.extend(trace.iter().cloned());
    augmented_trace.push("-".to_string());
    
    // Compute S_σ^k for the trace
    compute_multiset_k_trimmed_subtraces_iterative(&augmented_trace, k)
}

/// Compute the multiset of k-trimmed subtraces for a given trace
/// This implements S_σ^k, but uses an iterative approach rather than recursion
/// to avoid stack overflow for very long traces.
fn compute_multiset_k_trimmed_subtraces_iterative(trace: &[String], k: usize) -> HashMap<Arc<[String]>, usize> {
    let mut result = HashMap::new();
    
    if trace.len() <= k {
        // If trace length <= k, add the whole trace once - directly create Arc
        let arc = Arc::<[String]>::from(trace.to_vec());
        result.insert(arc, 1);
        return result;
    }
    
    // Compute k-length subtraces using sliding window
    for window in trace.windows(k) {
        let arc = Arc::<[String]>::from(window.to_vec());
        *result.entry(arc).or_insert(0) += 1; // Increment count for this subtrace
    }
    
    result
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
    // Convert to Arc<[String]> to match the expected type
    let arc_subtrace = Arc::<[String]>::from(example_subtrace);
    abstraction.insert(arc_subtrace, Fraction::from((1, 1)));
    
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
    use crate::{
        ebi_objects::finite_stochastic_language::FiniteStochasticLanguage,
        ebi_framework::activity_key::ActivityKey
    };
    use super::*;

    #[test]
    fn test_compute_abstraction_for_example_log() {
        // Example log L_1 = [⟨a,b⟩^{50}, ⟨a,a,b,c⟩^{20}, ⟨a,a,c,b⟩^{10}]
        let mut activity_key = ActivityKey::new();
        let mut traces = HashMap::new();
        
        // Create traces with probabilities 5/8, 1/4, 1/8
        traces.insert(
            activity_key.process_trace(&vec!["a".to_string(), "b".to_string()]),
            Fraction::from((50, 80)) // 5/8
        );
        traces.insert(
            activity_key.process_trace(&vec!["a".to_string(), "a".to_string(), "b".to_string(), "c".to_string()]),
            Fraction::from((20, 80)) // 1/4
        );
        traces.insert(
            activity_key.process_trace(&vec!["a".to_string(), "a".to_string(), "c".to_string(), "b".to_string()]),
            Fraction::from((10, 80)) // 1/8
        );
        
        // Compute k=2 abstraction
        let abstraction = compute_abstraction_for_log(
            &FiniteStochasticLanguage::new_raw(traces, activity_key), 2
        ).unwrap();
        
        println!("\nComputed abstraction for example log with k=2:");

        // Expected values for k=2 with activity mapping: ac0=a, ac1=b, ac2=c
        assert_eq!(abstraction.abstraction.len(), 8, "Should be exactly 8 entries");
        
        // For each entry in the abstraction, print it and check its value
        for (subtrace, probability) in abstraction.abstraction.iter() {
            let subtrace_str = subtrace.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(",");
            println!("{:<12} : {}", subtrace_str, probability);
            
            // Verify the value is correct
            match subtrace_str.as_str() {
                "+,ac0"    => assert_eq!(*probability, Fraction::from((4, 15))),
                "ac0,ac0"  => assert_eq!(*probability, Fraction::from((1, 10))),
                "ac0,ac1"  => assert_eq!(*probability, Fraction::from((7, 30))),
                "ac0,ac2"  => assert_eq!(*probability, Fraction::from((1, 30))),
                "ac1,-"    => assert_eq!(*probability, Fraction::from((1, 5))),
                "ac1,ac2"  => assert_eq!(*probability, Fraction::from((1, 15))),
                "ac2,-"    => assert_eq!(*probability, Fraction::from((1, 15))),
                "ac2,ac1"  => assert_eq!(*probability, Fraction::from((1, 30))),
                _ => panic!("Unexpected subtrace found: {}", subtrace_str)
            }
        }
    }
}