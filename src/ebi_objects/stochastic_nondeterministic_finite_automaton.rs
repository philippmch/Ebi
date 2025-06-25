use crate::math::fraction::Fraction;
/// A transition in a stochastic non-deterministic finite automaton
#[derive(Clone, Debug)]
pub struct Transition {
    pub target: usize,
    pub label: String,
    pub probability: Fraction,
}

/// A state in a stochastic non-deterministic finite automaton
#[derive(Clone, Debug)]
pub struct State {
    pub transitions: Vec<Transition>,
    /// Probability of terminating in this state. This could be implied by missing
    /// probability mass on outgoing edges, but we keep it explicit because we
    /// often redirect the full final probability to a sink state.
    pub p_final: Fraction,
}

/// The semantics are: starting in the `initial` state we either terminate with
/// probability `p_final` or follow one of the outgoing transitions chosen
/// according to their probabilities and move to the target state.
#[derive(Clone, Debug)]
pub struct StochasticNondeterministicFiniteAutomaton {
    pub states: Vec<State>,
    pub initial: usize,
}

impl StochasticNondeterministicFiniteAutomaton {
    /// Create an empty SNFA with a single initial / final state
    pub fn new() -> Self {
        Self {
            states: vec![State {
                transitions: vec![],
                p_final: Fraction::from((1, 1)),
            }],
            initial: 0,
        }
    }

    /// Ensures that a state with index `idx` exists, extending the vector if necessary.
    fn ensure_state(&mut self, idx: usize) {
        while self.states.len() <= idx {
            self.states.push(State { transitions: vec![], p_final: Fraction::from((0, 1)) });
        }
    }

    /// Adds a transition. The caller must make sure that outgoing probabilities of each state sum up to <= 1.
    pub fn add_transition(&mut self, source: usize, label: String, target: usize, probability: Fraction) {
        self.ensure_state(source);
        self.ensure_state(target);
        self.states[source].transitions.push(Transition { target, label, probability });
    }

    /// Sets the final-probability of a state.
    pub fn set_final_probability(&mut self, state: usize, p_final: Fraction) {
        self.ensure_state(state);
        self.states[state].p_final = p_final;
    }

    /// Returns the number of states.
    pub fn len(&self) -> usize { self.states.len() }
    pub fn is_empty(&self) -> bool { self.states.is_empty() }
}