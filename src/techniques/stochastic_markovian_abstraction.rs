use std::any::Any;
use rustc_hash::FxHashMap as HashMap;
use rayon::prelude::*;

use std::sync::Arc;

use anyhow::{Context, Result};

use num_traits::ToPrimitive;

use crate::{
    ebi_traits::{ebi_trait_semantics::Semantics, ebi_trait_stochastic_semantics::StochasticSemantics},
    ebi_framework::activity_key::TranslateActivityKey,
    ebi_objects::finite_stochastic_language::FiniteStochasticLanguage,
    ebi_objects::stochastic_labelled_petri_net::StochasticLabelledPetriNet,
    ebi_objects::labelled_petri_net::LPNMarking,
    ebi_objects::stochastic_nondeterministic_finite_automaton::{
        StochasticNondeterministicFiniteAutomaton as Snfa,
        State as SnfaState,
        Transition as SnfaTransition,
    },
    ebi_traits::{
        ebi_trait_finite_stochastic_language::EbiTraitFiniteStochasticLanguage,
        ebi_trait_queriable_stochastic_language::EbiTraitQueriableStochasticLanguage,
    },

    math::fraction::{Fraction, MaybeExact},
    math::traits::{Zero, Signed},
    techniques::bounded::Bounded,
    techniques::livelock_patch,
    techniques::sample::Sampler,
};

/// Supported distance metrics for Markovian abstraction comparison.
/// The list can be extended later – variants that are not yet implemented
/// will return an `anyhow::Error`.
#[derive(Clone, Copy, Debug)]
pub enum DistanceMetric {
    /// Asymmetric m^k-uEMSC distance.
    Uemsc,
    /// Symmetric total-variation distance 1/2 Sum|p−m|.
    TotalVariation,
    /// Square-root Jensen–Shannon distance.
    JensenShannon,
    /// Hellinger distance.
    Hellinger,
}

/// Default number of traces to sample when falling back to simulation for
/// unbounded Petri nets. (maybe optional parameter later)
const DEFAULT_SAMPLE_SIZE: usize = 10_000;

pub trait StochasticMarkovianAbstraction {
    /// Compare `self` with another stochastic language using the selected
    /// distance metric over their k-th order Markovian abstractions and
    /// return a conformance score in the range [0,1] where 1 means perfect
    /// match.
    ///
    /// Implemented metrics:
    /// * `DistanceMetric::Uemsc` – returns `1 – uEMSC distance`
    /// * `DistanceMetric::TotalVariation` – returns `1 – TV distance`
    /// * `DistanceMetric::JensenShannon` – returns `1 – √JSD`
    /// * `DistanceMetric::Hellinger` – returns `1 – Hellinger`
    ///
    /// Any other variant (if added in the future) will yield an error until
    /// its computation is implemented.
    fn markovian_conformance(
        &self,
        language2: Box<dyn EbiTraitQueriableStochasticLanguage>,
        k: usize,
        metric: DistanceMetric,
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
    fn markovian_conformance(
        &self,
        language2: Box<dyn EbiTraitQueriableStochasticLanguage>,
        k: usize,
        metric: DistanceMetric,
    ) -> Result<Fraction> {
        // Validate k
        if k < 1 {
            return Err(anyhow::anyhow!("k must be at least 1"));
        }

        // Step 1: Compute abstraction for the first language
        let abstraction1 = compute_abstraction_for_log(self, k)
            .context("Computing abstraction for first language (finite log)")?;

        // Step 2: Before computing the abstraction for the second language we must
        // make sure it uses the same activity labels for the same activities
        let mut shared_key = self.get_activity_key().clone();
        let mut language2 = language2;
        let abstraction2 = if let Some(pn) = (&mut *language2 as &mut dyn Any)
            .downcast_mut::<StochasticLabelledPetriNet>()
        {
            // If the Petri net is unbounded, fall back to random sampling.
            if !pn.bounded()? {
                log::warn!("Model is unbounded; falling back to random sampling. If a livelock is also present this may not terminate.");
                // Sample a finite stochastic language of DEFAULT_SAMPLE_SIZE traces
                let sampled: FiniteStochasticLanguage = pn
                    .sample(DEFAULT_SAMPLE_SIZE)
                    .context("Sampling unbounded Petri net")?;
                let mut boxed_sample: FiniteStochasticLanguage = sampled;
                boxed_sample.translate_using_activity_key(&mut shared_key);
                compute_abstraction_for_log(&boxed_sample, k)
                    .context("Computing abstraction for second language (sampled log)")?
            } else {
                pn.translate_using_activity_key(&mut shared_key);
                compute_abstraction_for_petri_net(pn, k)
                    .context("Computing abstraction for second language (Petri net)")?
            }
        } else if let Some(flog) = (&mut *language2 as &mut dyn Any)
            .downcast_mut::<FiniteStochasticLanguage>()
        {
            flog.translate_using_activity_key(&mut shared_key);
            compute_abstraction_for_log(flog, k)
                .context("Computing abstraction for second language (finite log)")?
        } else {
            return Err(anyhow::anyhow!(
                "markovian_conformance: unsupported type for second language"
            ));
        };

        // Step 3: Compute the conformance between the abstractions depending on the metric
        match metric {
            DistanceMetric::Uemsc => {
                let d = compute_uemsc_conformance(&abstraction1, &abstraction2)?;
                // already returns the conformance score not the distance
                Ok(d)
            }
            DistanceMetric::TotalVariation => {
                let tv = compute_total_variation_distance(&abstraction1, &abstraction2)?;
                let one = Fraction::from((1, 1));
                Ok(&one - &tv)
            }
                    DistanceMetric::JensenShannon => {
                let js = compute_jensen_shannon_distance(&abstraction1, &abstraction2)?;
                let one = Fraction::from((1, 1));
                Ok(&one - &js)
            }
            DistanceMetric::Hellinger => {
                let h = compute_hellinger_distance(&abstraction1, &abstraction2)?;
                let one = Fraction::from((1, 1));
                Ok(&one - &h)
            }

        }
    }
}

/// Compute the Jensen–Shannon distance between two abstractions.
fn compute_jensen_shannon_distance(
    abstraction1: &MarkovianAbstraction,
    abstraction2: &MarkovianAbstraction,
) -> Result<Fraction> {

    // Helper: convert an exact or approximate Fraction to f64 quickly.
    // For Exact fractions we compute n / d in double precision which is 
    // good enough for a Jensen–Shannon distance that is approximated to 1 e-15.
    #[inline]
    fn frac_to_f64(fr: &Fraction) -> f64 {
        if fr.is_zero() {
            return 0.0;
        }

        // Try the fast path for approximate fractions first
        if let Ok(v) = fr.extract_approx() {
            return v;
        }

        // Exact fraction -> fall back to numerator / denominator conversion
        if let Ok(exact) = fr.extract_exact() {
            if let (Some(n), Some(d)) = (exact.numer(), exact.denom()) {
                // Convert the first 53 bits of each part (f64 mantissa size).
                let n_f = n.to_f64().unwrap_or(0.0);
                let d_f = d.to_f64().unwrap_or(1.0);
                return n_f / d_f;
            }
        }
        0.0 // Should not happen, but keeps the compiler happy
    }

    #[inline]
    fn n_log_n(x: f64) -> f64 {
        if x <= 0.0 { 0.0 } else { x * x.log2() }
    }

    let mut h = 0.0;

    // Keys from abstraction1
    for (gamma, p1_frac) in &abstraction1.abstraction {
        let p1 = frac_to_f64(p1_frac);
        let p2 = abstraction2
            .abstraction
            .get(gamma)
            .map(|f| frac_to_f64(f))
            .unwrap_or(0.0);
        if p1 + p2 == 0.0 {
            continue;
        }
        let pq = p1 + p2;
        h += n_log_n(p1) + n_log_n(p2) - n_log_n(pq) + pq;
    }

    // Keys only in abstraction2
    for (gamma, p2_frac) in &abstraction2.abstraction {
        if abstraction1.abstraction.contains_key(gamma) {
            continue;
        }
        let p2 = frac_to_f64(p2_frac);
        if p2 == 0.0 {
            continue;
        }
        h += n_log_n(0.0) + n_log_n(p2) - n_log_n(p2) + p2; // pq == p2
    }

    h *= 0.5; // scale by 1/2

    // Identical abstractions or tiny difference
    if h <= 2.220_446_049_250_313e-16 {
        return Ok(Fraction::from((0, 1)));
    }

    let d = h.sqrt();
    const DEN: u64 = 1_000_000_000_000_000; // 1e15
    let num = (d * DEN as f64).round() as u64;
    Ok(Fraction::from((num, DEN)))
}



/// Compute the Hellinger distance between two abstractions.
fn compute_hellinger_distance(
    abstraction1: &MarkovianAbstraction,
    abstraction2: &MarkovianAbstraction,
) -> Result<Fraction> {
    let zero = Fraction::from((0, 1));
    let mut bc = Fraction::from((0, 1));

    for (gamma, p1) in &abstraction1.abstraction {
        let p2 = abstraction2.abstraction.get(gamma).unwrap_or(&zero);
        // sqrt(p1 * p2)
        let prod = p1 * p2;
        if !prod.is_zero() {
            bc += prod.sqrt_abs(15);
        }
    }
    for (gamma, _) in &abstraction2.abstraction {
        if abstraction1.abstraction.contains_key(gamma) {
            continue;
        }
        // p1 = 0 for these keys -> contribution to BC is zero -> nothing to add
    }
    let one = Fraction::from((1, 1));
    let h_sq = &one - &bc;
    Ok(h_sq.sqrt_abs(15))
}


/// Compute the symmetric total-variation distance between two abstractions.
fn compute_total_variation_distance(
    abstraction1: &MarkovianAbstraction,
    abstraction2: &MarkovianAbstraction,
) -> Result<Fraction> {
    if abstraction1.k != abstraction2.k {
        return Err(anyhow::anyhow!(
            "Cannot compare abstractions of different order: k1={}, k2={}",
            abstraction1.k, abstraction2.k
        ));
    }

    let mut diff_sum = Fraction::from((0, 1));
    let zero = Fraction::from((0, 1));

    // Iterate over union of keys; first pass abstraction1
    for (gamma, p1) in &abstraction1.abstraction {
        let p2 = abstraction2.abstraction.get(gamma).unwrap_or(&zero);
        let d = (p1 - p2).abs();
        diff_sum += d;
    }
    // Now add keys that are only in abstraction2
    for (gamma, p2) in &abstraction2.abstraction {
        if !abstraction1.abstraction.contains_key(gamma) {
            diff_sum += p2.clone();
        }
    }

    // Multiply by 1/2
    diff_sum /= 2usize;
    Ok(diff_sum)
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
    if k < 1 {
        return Err(anyhow::anyhow!("k must be at least 1 for Markovian abstraction"));
    }

    // Initialize f_l^k which stores the expected number of occurrences of each subtrace
    let mut f_l_k: HashMap<Arc<[String]>, Fraction> = HashMap::default();

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
    let mut abstraction = HashMap::default();
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
    let mut result = HashMap::default();

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

// Embedded-SNFA construction
fn build_embedded_snfa(net: &StochasticLabelledPetriNet) -> Result<Snfa> {
    use std::collections::VecDeque;

    // Create a new SNFA without the default initial state
    let mut snfa = Snfa::new();
    snfa.states.clear();

    // Reachability exploration queue
    let mut marking2idx: HashMap<LPNMarking, usize> = HashMap::default();
    let mut queue: VecDeque<LPNMarking> = VecDeque::new();

    // Insert the initial state of the Petri net
    let initial_marking = net
        .get_initial_state()
        .context("SLPN has no initial state")?;
    marking2idx.insert(initial_marking.clone(), 0);
    snfa.states.push(SnfaState {
        transitions: vec![],
        p_final: Fraction::from((0, 1)),
    });
    queue.push_back(initial_marking);

    while let Some(state) = queue.pop_front() {
        let src_idx = *marking2idx.get(&state).unwrap();

        // Collect enabled transitions and the total enabled weight in this state
        let enabled_transitions = net.get_enabled_transitions(&state);
        if enabled_transitions.is_empty() {
            // Deadlock state -> make it final with probability 1
            snfa.states[src_idx].p_final = Fraction::from((1, 1));
            continue;
        }

        let weight_sum = net.get_total_weight_of_enabled_transitions(&state)?;

        for &t in &enabled_transitions {
            let w = net.get_transition_weight(&state, t).clone();
            let prob = &w / &weight_sum;

            // Fire transition (creates a successor state)
            let mut next_state = state.clone();
            net.execute_transition(&mut next_state, t)?;

            // Map / enqueue successor
            let tgt_idx = *marking2idx.entry(next_state.clone()).or_insert_with(|| {
                let idx = snfa.states.len();
                snfa.states.push(SnfaState {
                    transitions: vec![],
                    p_final: Fraction::from((0, 1)),
                });
                queue.push_back(next_state.clone());
                idx
            });

            // Transition label (empty string for tau transition)
            let label = if let Some(a) = net.get_transition_label(t) {
                a.to_string()
            } else {
                "".to_string()
            };

            snfa.states[src_idx].transitions.push(SnfaTransition {
                target: tgt_idx,
                label,
                probability: prob,
            });
        }
    }

    // The first discovered marking is the initial state of the SNFA
    snfa.initial = 0;
    Ok(snfa)
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

/// Sparse exact Gaussian elimination where each row is a **sorted** Vec<(usize, Fraction)>.
/// We convert the incoming HashMap representation once and then run a cache friendly merge based
/// elimination (A := A - factor * pivot).
fn solve_sparse_linear_system_optimized(a_hash: &mut [HashMap<usize, Fraction>], mut b: Vec<Fraction>) -> Result<Vec<Fraction>> {
    // Convert the incoming HashMap rows into sorted vec rows
    fn to_vec_rows(a: &mut [HashMap<usize, Fraction>]) -> Vec<Vec<(usize, Fraction)>> {
        a.iter_mut()
            .map(|row| {
                let mut v: Vec<(usize, Fraction)> = row.drain().collect();
                v.sort_by_key(|(c, _)| *c);
                v
            })
            .collect()
    }

    // Binary search helper
    fn find_col(row: &[(usize, Fraction)], col: usize) -> Option<usize> {
        row.binary_search_by_key(&col, |(c, _)| *c).ok()
    }

    /// target = target - factor * pivot   (skips column i)
    fn saxpy_row(target: &mut Vec<(usize, Fraction)>, i: usize, pivot: &[(usize, Fraction)], factor: &Fraction) {
        let mut out = Vec::with_capacity(target.len() + pivot.len());
        let mut t = 0;
        let mut p = 0;
        while t < target.len() || p < pivot.len() {
            match (target.get(t), pivot.get(p)) {
                (Some(&(c_t, ref v_t)), Some(&(c_p, ref v_p))) if c_t == c_p => {
                    if c_t != i {
                        let mut new_val = v_t.clone();
                        new_val -= &(factor * v_p);
                        if !new_val.is_zero() {
                            out.push((c_t, new_val));
                        }
                    }
                    t += 1;
                    p += 1;
                }
                (Some(&(c_t, ref v_t)), Some(&(c_p, _))) if c_t < c_p => {
                    if c_t != i {
                        out.push((c_t, v_t.clone()));
                    }
                    t += 1;
                }
                (Some(&(c_t, ref v_t)), None) => {
                    if c_t != i {
                        out.push((c_t, v_t.clone()));
                    }
                    t += 1;
                }
                (None, Some(&(c_p, ref v_p))) | (Some(&(_, _)), Some(&(c_p, ref v_p))) => {
                    // c_p < c_t  OR target exhausted
                    if c_p != i {
                        let new_val = -(factor * v_p);
                        if !new_val.is_zero() {
                            out.push((c_p, new_val));
                        }
                    }
                    p += 1;
                }
                (None, None) => unreachable!(),
            }
        }
        out.shrink_to_fit();
        *target = out;
    }
    
    // Convert matrix once
    let mut a: Vec<Vec<(usize, Fraction)>> = to_vec_rows(a_hash);
    let n = b.len();

    for i in 0..n {
        // 1. Pivot search (find first row r >= i with non zero col i)
        let pivot = (i..n)
            .find(|&r| {
                find_col(&a[r], i)
                    .map_or(false, |idx| !a[r][idx].1.is_zero())
            })
            .ok_or_else(|| anyhow::anyhow!("Matrix is singular"))?;

        if pivot != i {
            a.swap(i, pivot);
            b.swap(i, pivot);
        }

        // 2. Normalize pivot row so diagonal becomes 1
        let diag_idx = find_col(&a[i], i).expect("pivot exists");
        let inv = a[i][diag_idx].1.clone().recip();
        for &mut (_, ref mut v) in &mut a[i] {
            *v *= &inv;
        }
        b[i] *= &inv;

        // 3. Split mutable slice around pivot row
        let (left, rest) = a.split_at_mut(i);
        let (pivot_row, below) = rest.split_first_mut().expect("pivot row");
        let pivot_ref: &[(usize, Fraction)] = &pivot_row[..];
        let pivot_b = b[i].clone();

        // Thread local updates for RHS
        let mut updates: Vec<(usize, Fraction)> = Vec::with_capacity(left.len() + below.len());

        // rows above
        updates.extend(
            left.par_iter_mut()
                .enumerate()
                .filter_map(|(r, row)| {
                    find_col(row, i).map(|idx| {
                        let factor = row[idx].1.clone();
                        row[idx].1 = Fraction::zero(); // lazy zeroing avoids shift
                        if factor.is_zero() {
                            None
                        } else {
                            saxpy_row(row, i, pivot_ref, &factor);
                            Some((r, factor))
                        }
                    }).flatten()
                })
                .collect::<Vec<_>>()
        );

        // rows below
        updates.extend(
            below.par_iter_mut()
                .enumerate()
                .filter_map(|(off, row)| {
                    let r = i + 1 + off;
                    find_col(row, i).map(|idx| {
                        let factor = row[idx].1.clone();
                        row[idx].1 = Fraction::zero();
                        if factor.is_zero() {
                            None
                        } else {
                            saxpy_row(row, i, pivot_ref, &factor);
                            Some((r, factor))
                        }
                    }).flatten()
                })
                .collect::<Vec<_>>()
        );

        // Apply RHS updates sequentially
        for (r, factor) in updates {
            b[r] -= &factor * &pivot_b;
        }
    }

    Ok(b)
}

/// Naive sparse Gaussian elimination for Fraction matrices represented as Vec<HashMap<usize, Fraction>>
fn solve_sparse_linear_system(a: &mut [HashMap<usize, Fraction>], mut b: Vec<Fraction>) -> Result<Vec<Fraction>> {
    let n = b.len();
    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut pivot = i;
        while pivot < n && a[pivot].get(&i).map_or(true, |v| v.is_zero()) {
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
        let inv = a[i].get(&i).unwrap().clone().recip();
        // scale row i
        let keys: Vec<usize> = a[i].keys().cloned().collect();
        for j in keys {
            if let Some(val) = a[i].get_mut(&j) {
                *val *= &inv;
            }
        }
        b[i] *= &inv;
        let pivot_b = b[i].clone();
        // Eliminate other rows
        for r in 0..n {
            if r == i { continue; }
            if let Some(factor_val) = a[r].get(&i).cloned() {
                if !factor_val.is_zero() {
                    // subtract factor * row_i from row_r
                    let keys: Vec<(usize, Fraction)> = a[i].iter().map(|(k,v)| (*k, v.clone())).collect();
                    for (c, val_i) in keys {
                        let product = &factor_val * &val_i;
                        let entry = a[r].entry(c).or_insert_with(Fraction::zero);
                        *entry = &*entry - &product;
                        if entry.is_zero() {
                            a[r].remove(&c);
                        }
                    }
                    b[r] -= &factor_val * &pivot_b;
                }
            }
        }
    }
    Ok(b.to_vec())
}

fn compute_phi(snfa: &Snfa, k: usize) -> Vec<HashMap<Arc<[String]>, Fraction>> {
    let n = snfa.states.len();
    let mut phi: Vec<HashMap<Arc<[String]>, Fraction>> = vec![HashMap::default(); n];

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
    if k < 1 {
        return Err(anyhow::anyhow!("k must be at least 1 for Markovian abstraction"));
    }

    // 0.5 Patch bounded livelocks by adding timeout escapes (δ = 1e-4) (maybe make optional parameter later)
    let patched_net = livelock_patch::patch_livelocks(petri_net, Fraction::from((1, 10000)))?;

    // 1 Build embedded SNFA
    let mut snfa_raw = build_embedded_snfa(&patched_net)?;

    // 1.1 Remove tau transitions
    snfa_raw.remove_tau_transitions();

    // 2 Patch it
    let snfa = patch_snfa(&snfa_raw);

    // 3 Build matrix and solve for x
    let n = snfa.states.len();
    // Build sparse A = (I − Delta)^T
    let mut a_sparse: Vec<HashMap<usize, Fraction>> = vec![HashMap::default(); n];
    let delta = build_delta(&snfa);
    for i in 0..n {
        for j in 0..n {
            let identity = if i == j { Fraction::from((1, 1)) } else { Fraction::from((0, 1)) };
            let val = &identity - &delta[i][j];
            if !val.is_zero() {
                a_sparse[j].insert(i, val); // transpose while filling
            }
        }
    }
    let mut b = vec![Fraction::from((0, 1)); n];
    b[snfa.initial] = Fraction::from((1, 1)); // e₊
    let mut a_ref = a_sparse;
    let x = if n < 100{
        // small matrices -> simpler hash map solver avoids conversion overhead
        solve_sparse_linear_system(&mut a_ref, b)?
    } else {
        solve_sparse_linear_system_optimized(&mut a_ref, b)?
    };

    // 4 Compute Phi for each state
    let phi = compute_phi(&snfa, k);

    // 5 Compute f_l^k
    let mut f_l_k: HashMap<Arc<[String]>, Fraction> = HashMap::default();
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
    let mut abstraction = HashMap::default();
    for (gamma, val) in f_l_k {
        abstraction.insert(gamma, &val / &total);
    }

    Ok(MarkovianAbstraction { k, abstraction })
}

/// Compute the m^k-uEMSC conformance between two Markovian abstractions
pub fn compute_uemsc_conformance(
    abstraction1: &MarkovianAbstraction,
    abstraction2: &MarkovianAbstraction,
) -> Result<Fraction> {
    // Sanity-check: abstractions must be of the same order k
    if abstraction1.k != abstraction2.k {
        return Err(anyhow::anyhow!(
            "Cannot compare abstractions of different order: k1={}, k2={}",
            abstraction1.k, abstraction2.k
        ));
    }

    // Sum for Σ_γ max(m1(γ) − m2(γ), 0)
    let mut positive_diff = Fraction::from((0, 1));
    let zero = Fraction::from((0, 1));

    for (gamma, p1) in &abstraction1.abstraction {
        let p2 = abstraction2
            .abstraction
            .get(gamma)
            .unwrap_or(&zero);

        if p1 > p2 {
            let diff = &*p1 - &*p2;
            positive_diff += diff;
        }
    }

    // Conformance = 1 - positive_diff
    let one = Fraction::from((1, 1));
    let conformance = &one - &positive_diff;

    Ok(conformance)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use crate::ebi_objects::finite_stochastic_language::FiniteStochasticLanguage;
    use crate::ebi_objects::event_log::EventLog;
    use crate::ebi_objects::stochastic_labelled_petri_net::StochasticLabelledPetriNet;
    use crate::ebi_traits::ebi_trait_finite_stochastic_language::EbiTraitFiniteStochasticLanguage;
    use crate::ebi_framework::activity_key::{HasActivityKey, TranslateActivityKey};
    use super::*;

    #[test]
    fn test_compute_abstraction_for_example_log() {
        let file_content = fs::read_to_string("testfiles/simple_log_markovian_abstraction.xes").unwrap();
        let event_log = file_content.parse::<EventLog>().unwrap();
        let finite_lang: FiniteStochasticLanguage = Into::into(event_log);
        
        // Compute abstraction with k=2 for example log [⟨a,b⟩^{5}, ⟨a,a,b,c⟩^{2}, ⟨a,a,c,b⟩^{1}]
        let abstraction = compute_abstraction_for_log(&finite_lang, 2).unwrap();
        
        println!("\nComputed abstraction for example log with k=2:");

        assert_eq!(abstraction.abstraction.len(), 8, "Should be exactly 8 entries");
        
        // map for checking
        let mut check: std::collections::HashMap<String, Fraction> = std::collections::HashMap::default();
        for (subtrace, prob) in abstraction.abstraction.iter() {
            let key = subtrace.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(",");
            println!("{:<12} : {}", key, prob);
            check.insert(key, prob.clone());
        }

        assert_eq!(check["+,ac0"], Fraction::from((4, 15)));
        assert_eq!(check["ac0,ac0"], Fraction::from((1, 10)));

        let pair1 = [Fraction::from((7, 30)), Fraction::from((1, 30))];
        let pair2 = [Fraction::from((1, 5)), Fraction::from((1, 15))];
        let pair3 = [Fraction::from((1, 15)), Fraction::from((1, 30))];

        // Helper to check that a key has one of two expected values and that the other value is on the other key
        fn assert_pair(check: &std::collections::HashMap<String, Fraction>, k1: &str, k2: &str, exp: [Fraction; 2]) {
            let v1 = check.get(k1).expect("missing key");
            let v2 = check.get(k2).expect("missing key");
            assert!( (v1 == &exp[0] && v2 == &exp[1]) || (v1 == &exp[1] && v2 == &exp[0]),
                "Pair {{ {}, {} }} has unexpected values {{ {}, {} }}", k1, k2, v1, v2);
        }

        assert_pair(&check, "ac0,ac1", "ac0,ac2", pair1);
        assert_pair(&check, "ac1,-",   "ac2,-",   pair2);
        assert_pair(&check, "ac1,ac2", "ac2,ac1", pair3);
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
        
        // Check that k < 1 is rejected (k = 0)
        let result = compute_abstraction_for_petri_net(&petri_net, 0);
        assert!(result.is_err(), "Should reject k < 1");
        
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

    #[test]
    fn test_markovian_conformance_uemsc_log_vs_petri_net() {
        // Load the example log and convert to finite stochastic language
        let file_content = fs::read_to_string("testfiles/simple_log_markovian_abstraction.xes").unwrap();
        let event_log = file_content.parse::<EventLog>().unwrap();
        let mut finite_lang: FiniteStochasticLanguage = Into::into(event_log);

        // Load the Petri net model
        let file_content = fs::read_to_string("testfiles/simple_markovian_abstraction.slpn").unwrap();
        let mut petri_net = file_content.parse::<StochasticLabelledPetriNet>().unwrap();

        // Ensure both share a common ActivityKey to avoid nondeterministic mappings
        petri_net.translate_using_activity_key(finite_lang.get_activity_key_mut());

        // Compute the m^2-uEMSC distance (k = 2)
        let distance = (&finite_lang as &dyn EbiTraitFiniteStochasticLanguage)
            .markovian_conformance(Box::new(petri_net), 2, DistanceMetric::Uemsc)
            .unwrap();

        println!("Computed m^2-uEMSC distance: {}", distance);
        assert_eq!(distance, Fraction::from((4, 5))); // Expect 4/5
    }
}