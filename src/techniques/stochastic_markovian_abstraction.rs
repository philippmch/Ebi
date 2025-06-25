use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::{
    ebi_objects::stochastic_labelled_petri_net::StochasticLabelledPetriNet,
    ebi_traits::{
        ebi_trait_finite_stochastic_language::EbiTraitFiniteStochasticLanguage,
        ebi_trait_queriable_stochastic_language::EbiTraitQueriableStochasticLanguage,
    },
    marking::Marking,
    math::fraction::Fraction,
    math::traits::Zero,
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
    k: usize,
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
                p_ref * o_ref // This returns an owned FractionEnum
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
        let arc = Arc::<[String]>::from(trace);
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

use crate::ebi_objects::stochastic_nondeterministic_finite_automaton::{
    StochasticNondeterministicFiniteAutomaton as Snfa,
    State as SnfaState,
    Transition as SnfaTransition,
};

// Embedded-SNFA construction
fn build_embedded_snfa(net: &StochasticLabelledPetriNet) -> Result<Snfa> {
    use std::collections::VecDeque;

    // Create a new SNFA with empty states list
    let mut snfa = Snfa::new();
    snfa.states.clear(); // Remove the initial state
    
    // Map reachable markings to state indices
    let mut marking2idx: HashMap<Marking, usize> = HashMap::new();
    let mut queue: VecDeque<Marking> = VecDeque::new();
    
    // Initialize with initial marking
    let initial_marking = net.get_initial_marking().clone();
    marking2idx.insert(initial_marking.clone(), 0);
    snfa.states.push(SnfaState { transitions: vec![], p_final: Fraction::from((0, 1)) });
    queue.push_back(initial_marking.clone());

    while let Some(marking) = queue.pop_front() {
        let src_idx = *marking2idx.get(&marking).unwrap();

        // Collect enabled transitions and their total weight
        let mut enabled: Vec<(usize, Fraction)> = vec![]; // (transition index, weight)
        let mut weight_sum = Fraction::from((0, 1));
        let n_transitions = net.weights.len();
        for t in 0..n_transitions {
            if is_enabled(&net, &marking, t) {
                let weight = net.weights[t].clone();
                enabled.push((t, weight.clone()));
                weight_sum += &weight;
            }
        }

        if enabled.is_empty() {
            // Deadlock marking -> p_final = 1
            snfa.states[src_idx].p_final = Fraction::from((1, 1));
            continue;
        }

        for (t, w) in enabled {
            // Compute probability weight
            let prob = &w / &weight_sum;
            
            // Fire transition
            let new_marking = fire_transition(&net, &marking, t)?;
            
            // Handle new marking
            let tgt_idx;
            if !marking2idx.contains_key(&new_marking) {
                // Create new state
                let idx = snfa.states.len();
                snfa.states.push(SnfaState { 
                    transitions: vec![],
                    p_final: Fraction::from((0, 1)), // Not final by default
                });
                marking2idx.insert(new_marking.clone(), idx);
                queue.push_back(new_marking.clone());
                tgt_idx = idx;
            } else {
                tgt_idx = *marking2idx.get(&new_marking).unwrap();
            };

            // Use only the transition label from the Petri net without state information
            let label = if let Some(a) = net.get_transition_label(t) {
                a.to_string() // Keep just the plain transition label
            } else {
                "".to_string() // Tau transition
            };

            // Add the transition to the source state
            snfa.states[src_idx].transitions.push(SnfaTransition {
                target: tgt_idx,
                label,
                probability: prob,
            });
        }
    }

    // Set the initial state to 0
    snfa.initial = 0;
    Ok(snfa)
}

// Helper function to check if a transition is enabled in a given marking
fn is_enabled(net: &StochasticLabelledPetriNet, marking: &Marking, t: usize) -> bool {
    for (pos, place) in net.transition2input_places[t].iter().enumerate() {
        let card = net.transition2input_places_cardinality[t][pos] as u64;
        if marking.get_place2token()[*place] < card {
            return false;
        }
    }
    true
}

fn fire_transition(net: &StochasticLabelledPetriNet, marking: &Marking, t: usize) -> Result<Marking> {
    let mut new_m = marking.clone();
    // Consume tokens
    for (i, place) in net.transition2input_places[t].iter().enumerate() {
        let card = net.transition2input_places_cardinality[t][i] as u64;
        new_m.decrease(*place, card)?;
    }
    // Produce tokens
    for (i, place) in net.transition2output_places[t].iter().enumerate() {
        let card = net.transition2output_places_cardinality[t][i] as u64;
        new_m.increase(*place, card)?;
    }
    Ok(new_m)
}

// Patching - assumes SNFA is already tau-free
fn patch_snfa(snfa: &Snfa) -> Snfa {
    // Create a new vector of states to avoid borrow conflicts
    let mut states = snfa.states.clone();

    let q_plus = states.len();
    let q_minus = states.len() + 1;

    // Redirect original finals to q₋ via '-'
    for i in 0..states.len() {
        let s = &mut states[i];
        if !s.p_final.is_zero() {
            // Copy the existing p_final value
            let final_prob = s.p_final.clone();
            // Reset p_final to zero
            s.p_final = Fraction::from((0, 1));
            // Add a "-" transition with the original final probability
            s.transitions.push(SnfaTransition { 
                target: q_minus, 
                label: "-".to_string(), 
                probability: final_prob
            });
        }
    }

    // q₊ with +-transition to original initial state
    states.push(SnfaState { 
        transitions: vec![SnfaTransition { 
            target: snfa.initial, 
            label: "+".to_string(), 
            probability: Fraction::from((1, 1)) 
        }], 
        p_final: Fraction::from((0, 1))
    });

    // q₋ absorbing final state
    states.push(SnfaState { 
        transitions: vec![], 
        p_final: Fraction::from((1, 1))
    });

    // Return the patched automaton
    Snfa {
        states,
        initial: q_plus,
    }
}

// Build transition matrix
fn build_delta(snfa: &Snfa) -> Vec<Vec<Fraction>> {
    let n = snfa.states.len();
    let mut delta = vec![vec![Fraction::from((0, 1)); n]; n];
    
    for (i, state) in snfa.states.iter().enumerate() {
        for t in &state.transitions {
            delta[i][t.target] += &t.probability;
        }
    }
    delta
}

// Solve (I − Δ)ᵀ x = e₊ using exact Gaussian elimination on Fractions (needs more efficient implementation, TODO)
fn solve_linear_system(a: &mut [Vec<Fraction>], b: &mut [Fraction]) -> Result<Vec<Fraction>> {
    let n = b.len();
    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut pivot = i;
        while pivot < n && a[pivot][i].is_zero() {
            pivot += 1;
        }
        if pivot == n {
            return Err(anyhow::anyhow!("Matrix is singular"));
        }
        if pivot != i {
            a.swap(i, pivot);
            b.swap(i, pivot);
        }
        // Normalize row i
        let inv = a[i][i].clone().recip();
        for j in i..n {
            a[i][j] *= &inv;
        }
        b[i] *= &inv;
        // Eliminate other rows
        for r in 0..n {
            if r == i { continue; }
            let factor = a[r][i].clone();
            if factor.is_zero() { continue; }
            for c in i..n {
                let product = &factor * &a[i][c];
                a[r][c] = &a[r][c] - &product;
            }
            b[r] -= &factor * &b[i];
        }
    }
    Ok(b.to_vec())
}

// Expected frequencies
fn compute_phi(snfa: &Snfa, k: usize) -> Vec<HashMap<Arc<[String]>, Fraction>> {
    let n = snfa.states.len();
    let mut phi: Vec<HashMap<Arc<[String]>, Fraction>> = vec![HashMap::new(); n];

    // DFS to compute for each state and subtrace
    fn dfs(
        start_idx: usize,
        cur_idx: usize,
        snfa: &Snfa,
        phi: &mut Vec<HashMap<Arc<[String]>, Fraction>>,
        path: &mut Vec<String>,
        k: usize,
        f_q: &Fraction                      // current path weight
    ) {
        // If the run has terminated with ‘-’ before reaching k,
        // store the whole trace and stop exploring.
        // The first symbol must be ‘+’; otherwise this is just a suffix inside a longer run and must be ignored.
        if matches!(path.last(), Some(s) if s == "-")
            && path.len() <= k
            && matches!(path.first(), Some(p) if p == "+")
        {
            let subtrace = Arc::from(&path[..]);
            let entry = phi[start_idx]
                .entry(subtrace)
                .or_insert_with(|| Fraction::from((0, 1)));
            *entry += f_q;                 // weight already includes ‘-’
            return;                        // nothing follows after ‘-’
        }

        // Only accumulate frequency when path reaches exactly length k
        // This keeps the k-length subtraces
        if path.len() == k {
            let subtrace = Arc::from(&path[..]);
            let entry = phi[start_idx]
                .entry(subtrace)
                .or_insert_with(|| Fraction::from((0, 1)));
            *entry += f_q;
            return;                        // stop DFS when path reaches k
        }

        // Recursively follow outgoing transitions until we reach length k
        for tr in &snfa.states[cur_idx].transitions {
            path.push(tr.label.clone());
            let next_f_q = &tr.probability * f_q; // multiply path weight
            dfs(start_idx, tr.target, snfa, phi, path, k, &next_f_q);
            path.pop();
        }
    }

    // Start DFS from each state with initial weight 1
    let one = Fraction::from((1, 1));
    for q in 0..n {
        dfs(q, q, snfa, &mut phi, &mut Vec::new(), k, &one);
    }

    phi
}


// Main Petri-net abstraction function
pub fn compute_abstraction_for_petri_net(
    petri_net: &StochasticLabelledPetriNet,
    k: usize,
) -> Result<MarkovianAbstraction> {
    if k < 2 {
        return Err(anyhow::anyhow!("k must be at least 2 for Markovian abstraction"));
    }

    // 1 Build embedded SNFA (tau transitions may be present but we assume none for now) TODO: handle tau transitions
    let snfa_raw = build_embedded_snfa(petri_net)?;

    // 2 Patch it
    let snfa = patch_snfa(&snfa_raw);

    // 3 Build matrix and solve for x
    let delta = build_delta(&snfa);
    let n = delta.len();
    let mut a: Vec<Vec<Fraction>> = vec![vec![Fraction::from((0, 1)); n]; n];
    // A = (I − Δ)ᵀ
    for i in 0..n {
        for j in 0..n {
            let identity = if i == j { Fraction::from((1, 1)) } else { Fraction::from((0, 1)) };
            let val = &identity - &delta[i][j];
            a[j][i] = val; // transpose while filling
        }
    }
    let mut b = vec![Fraction::from((0, 1)); n];
    b[snfa.initial] = Fraction::from((1, 1)); // e₊
    let mut a_ref: Vec<Vec<Fraction>> = a.into_iter().map(|row| row.into_iter().collect()).collect();
    let mut b_ref = b.clone();
    let x = solve_linear_system(&mut a_ref, &mut b_ref)?;

    // 4 Compute φ for each state
    let phi = compute_phi(&snfa, k);

    // 5 Compute f_l^k
    let mut f_l_k: HashMap<Arc<[String]>, Fraction> = HashMap::new();
    for (q, map) in phi.iter().enumerate() {
        for (gamma, phi_val) in map {
            let contribution = &x[q] * phi_val;
            f_l_k.entry(gamma.clone())
                .and_modify(|v| *v = &*v + &contribution)
                .or_insert(contribution.clone());
        }
    }

    // 6 Normalize
    let mut total = Fraction::from((0, 1));
    for v in f_l_k.values() {
        total += v;
    }
    let mut abstraction = HashMap::new();
    for (gamma, val) in f_l_k {
        abstraction.insert(gamma, &val / &total);
    }

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
    use std::fs;
    use crate::ebi_objects::finite_stochastic_language::FiniteStochasticLanguage;
    use crate::ebi_framework::activity_key::ActivityKey;
    use crate::ebi_objects::stochastic_labelled_petri_net::StochasticLabelledPetriNet;
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
    
    #[test]
    fn test_compute_abstraction_for_petri_net() {
        // This test verifies the computation of Markovian abstractions
        // for a simple SLPN model with the following structure:
        // - Initial token in place 0
        // - Transition 'a' moves token to place 1
        // - From place 1, transition 'b' (weight 3/4) or 'c' (weight 1/4) to deadlock
        let file_content = fs::read_to_string("testfiles/simple_markovian_abstraction.slpn").unwrap();
        let petri_net = file_content.parse::<StochasticLabelledPetriNet>().unwrap();
        
        // Check that k < 2 is rejected
        let result = compute_abstraction_for_petri_net(&petri_net, 1);
        assert!(result.is_err(), "Should reject k < 2");
        
        // Compute abstraction with k=2
        let abstraction = compute_abstraction_for_petri_net(&petri_net, 2).unwrap();
        

        // Check that probabilities sum to 1
        let mut total = Fraction::from((0, 1));
        for probability in abstraction.abstraction.values() {
            total += probability;
        }
        assert_eq!(total, Fraction::from((1, 1)), "Total probability should be 1");

        // Expected internal traces and probabilities
        let expected_traces = [
            ("+ ac0", "1/3"),
            ("ac0 ac1", "1/4"),
            ("ac1 -", "1/4"),
            ("ac0 ac2", "1/12"),
            ("ac2 -", "1/12")
        ];

        // Verify expected traces have the correct probabilities
        for (trace_str, expected_prob) in expected_traces.iter() {
            let key: Vec<String> = trace_str.split_whitespace().map(|s| s.to_string()).collect();
            let arc_key = Arc::from(key.as_slice());
            if let Some(prob) = abstraction.abstraction.get(&arc_key) {
                println!("Found ⟨{}⟩ with probability {}", trace_str, prob);
                assert_eq!(prob.to_string(), *expected_prob,
                    "Probability mismatch for trace ⟨{}⟩: expected {}, got {}",
                    trace_str, expected_prob, prob);
            } else {
                panic!("Expected trace ⟨{}⟩ not found in abstraction", trace_str);
            }
        }

        println!("Test passed: Abstraction matches expected internal trace probabilities!");
    }
}