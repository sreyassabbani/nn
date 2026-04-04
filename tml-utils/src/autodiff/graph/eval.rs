//! Forward- and reverse-mode graph evaluation.

use crate::Float;

use super::{EvalTape, ExprGraph, Node, NodeId, ReverseTape};

impl ExprGraph {
    /// Pure forward evaluation that allocates its own tape.
    pub fn eval_fwd(&self, inputs: &[Float]) -> Vec<(Float, Vec<Float>)> {
        let mut tape = self.fwd_tape();
        self.eval_fwd_with_tape(inputs, &mut tape)
    }

    /// Forward evaluation that reuses the provided tape.
    pub fn eval_fwd_with_tape(
        &self,
        inputs: &[Float],
        tape: &mut EvalTape,
    ) -> Vec<(Float, Vec<Float>)> {
        assert_eq!(
            inputs.len(),
            self.inputs.len(),
            "expected {} inputs, got {}",
            self.inputs.len(),
            inputs.len()
        );

        tape.reset(self.nodes.len(), self.inputs.len(), self.max_arity);

        for (input_idx, node_id) in self.inputs.iter().enumerate() {
            let node_idx = node_id.index;
            tape.primals[node_idx] = inputs[input_idx];
            let tangent_idx = tape.tangent_index(node_idx, input_idx);
            tape.tangents[tangent_idx] = 1.0;
        }

        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                Node::AfterOperation(op, inputs) => {
                    let arity = inputs.len();
                    let input_primals = &mut tape.scratch_primals[..arity];
                    for (slot, &id) in input_primals.iter_mut().zip(inputs.iter()) {
                        *slot = tape.primals[id.index];
                    }

                    tape.primals[i] = op.apply(input_primals);

                    let partials = &mut tape.scratch_partials[..arity];
                    for (j, partial) in partials.iter_mut().enumerate() {
                        *partial = op.compute_derivative(input_primals, j);
                    }

                    let input_count = tape.input_count;
                    let tangents = &mut tape.tangents;
                    for input_dim in 0..input_count {
                        let mut total = 0.0;
                        for (j, &input_id) in inputs.iter().enumerate() {
                            let idx = input_id.index * input_count + input_dim;
                            total += tangents[idx] * partials[j];
                        }
                        let out_idx = i * input_count + input_dim;
                        tangents[out_idx] = total;
                    }
                }
                Node::Const(value) => {
                    tape.primals[i] = *value;
                }
                _ => {}
            }
        }

        for (i, node) in self.nodes.iter().enumerate() {
            if let Node::Output(input_id) = node {
                tape.primals[i] = tape.primals[input_id.index];
                let src_start = tape.tangent_index(input_id.index, 0);
                let dst_start = tape.tangent_index(i, 0);
                let len = tape.input_count;
                tape.tangents
                    .copy_within(src_start..(src_start + len), dst_start);
            }
        }

        self.outputs
            .iter()
            .map(|id| {
                let idx = id.index;
                let start = tape.tangent_index(idx, 0);
                let end = start + tape.input_count;
                (tape.primals[idx], tape.tangents[start..end].to_vec())
            })
            .collect()
    }

    /// Forward evaluation specialized to a single output.
    pub fn eval_fwd_one(&self, inputs: &[Float]) -> (Float, Vec<Float>) {
        let mut tape = self.fwd_tape();
        self.eval_fwd_one_with_tape(inputs, &mut tape)
    }

    /// Single-output forward evaluation with a reusable tape.
    pub fn eval_fwd_one_with_tape(
        &self,
        inputs: &[Float],
        tape: &mut EvalTape,
    ) -> (Float, Vec<Float>) {
        let mut outputs = self.eval_fwd_with_tape(inputs, tape);
        assert!(
            outputs.len() == 1,
            "expected a single output, got {}",
            outputs.len()
        );
        outputs.remove(0)
    }

    /// Forward evaluation with gradients returned as `(name, grad)` pairs.
    pub fn eval_fwd_named(&self, inputs: &[Float]) -> Vec<(Float, Vec<(String, Float)>)> {
        let mut tape = self.fwd_tape();
        self.eval_fwd_named_with_tape(inputs, &mut tape)
    }

    /// Named forward evaluation with a reusable tape.
    pub fn eval_fwd_named_with_tape(
        &self,
        inputs: &[Float],
        tape: &mut EvalTape,
    ) -> Vec<(Float, Vec<(String, Float)>)> {
        let outputs = self.eval_fwd_with_tape(inputs, tape);
        outputs
            .into_iter()
            .map(|(value, grads)| {
                let named = self
                    .input_names
                    .iter()
                    .cloned()
                    .zip(grads)
                    .collect::<Vec<_>>();
                (value, named)
            })
            .collect()
    }

    /// Reverse-mode evaluation that allocates its own tape.
    pub fn eval(&self, inputs: &[Float]) -> Vec<(Float, Vec<Float>)> {
        let mut tape = self.reverse_tape();
        self.eval_with_tape(inputs, &mut tape)
    }

    /// Reverse-mode evaluation that reuses the provided tape.
    pub fn eval_with_tape(
        &self,
        inputs: &[Float],
        tape: &mut ReverseTape,
    ) -> Vec<(Float, Vec<Float>)> {
        self.eval_for_with_tape(inputs, &self.outputs, tape)
    }

    /// Reverse-mode evaluation for a selected set of outputs.
    pub fn eval_for(&self, inputs: &[Float], outputs: &[NodeId]) -> Vec<(Float, Vec<Float>)> {
        let mut tape = self.reverse_tape();
        self.eval_for_with_tape(inputs, outputs, &mut tape)
    }

    /// Reverse-mode evaluation for selected outputs with a reusable tape.
    pub fn eval_for_with_tape(
        &self,
        inputs: &[Float],
        outputs: &[NodeId],
        tape: &mut ReverseTape,
    ) -> Vec<(Float, Vec<Float>)> {
        assert_eq!(
            inputs.len(),
            self.inputs.len(),
            "expected {} inputs, got {}",
            self.inputs.len(),
            inputs.len()
        );
        for &output in outputs {
            self.assert_valid_node(output, "requested output");
        }

        tape.reset(self.nodes.len(), self.max_arity);

        for (input_idx, node_id) in self.inputs.iter().enumerate() {
            tape.primals[node_id.index] = inputs[input_idx];
        }

        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                Node::AfterOperation(op, inputs) => {
                    let arity = inputs.len();
                    let input_primals = &mut tape.scratch_primals[..arity];
                    for (slot, &id) in input_primals.iter_mut().zip(inputs.iter()) {
                        *slot = tape.primals[id.index];
                    }
                    tape.primals[i] = op.apply(input_primals);
                }
                Node::Output(input_id) => {
                    tape.primals[i] = tape.primals[input_id.index];
                }
                Node::Const(value) => {
                    tape.primals[i] = *value;
                }
                Node::Input(_) => {}
            }
        }

        let mut results = Vec::with_capacity(outputs.len());

        for output_id in outputs {
            tape.adjoints.fill(0.0);
            tape.adjoints[output_id.index] = 1.0;

            for (i, node) in self.nodes.iter().enumerate().rev() {
                match node {
                    Node::Output(input_id) => {
                        tape.adjoints[input_id.index] += tape.adjoints[i];
                    }
                    Node::AfterOperation(op, inputs) => {
                        let arity = inputs.len();
                        let input_primals = &mut tape.scratch_primals[..arity];
                        for (slot, &id) in input_primals.iter_mut().zip(inputs.iter()) {
                            *slot = tape.primals[id.index];
                        }

                        let partials = &mut tape.scratch_partials[..arity];
                        for (j, partial) in partials.iter_mut().enumerate() {
                            *partial = op.compute_derivative(input_primals, j);
                        }

                        let adj = tape.adjoints[i];
                        if adj != 0.0 {
                            for (j, &input_id) in inputs.iter().enumerate() {
                                tape.adjoints[input_id.index] += adj * partials[j];
                            }
                        }
                    }
                    Node::Const(_) | Node::Input(_) => {}
                }
            }

            let grads = self
                .inputs
                .iter()
                .map(|id| tape.adjoints[id.index])
                .collect::<Vec<_>>();
            results.push((tape.primals[output_id.index], grads));
        }

        results
    }

    /// Single-output reverse-mode evaluation.
    pub fn eval_one(&self, inputs: &[Float]) -> (Float, Vec<Float>) {
        let mut tape = self.reverse_tape();
        self.eval_one_with_tape(inputs, &mut tape)
    }

    /// Single-output reverse-mode evaluation with a reusable tape.
    pub fn eval_one_with_tape(
        &self,
        inputs: &[Float],
        tape: &mut ReverseTape,
    ) -> (Float, Vec<Float>) {
        let mut outputs = self.eval_with_tape(inputs, tape);
        assert!(
            outputs.len() == 1,
            "expected a single output, got {}",
            outputs.len()
        );
        outputs.remove(0)
    }

    /// Reverse-mode evaluation with named gradients.
    pub fn eval_named(&self, inputs: &[Float]) -> Vec<(Float, Vec<(String, Float)>)> {
        let mut tape = self.reverse_tape();
        self.eval_named_with_tape(inputs, &mut tape)
    }

    /// Named reverse-mode evaluation with a reusable tape.
    pub fn eval_named_with_tape(
        &self,
        inputs: &[Float],
        tape: &mut ReverseTape,
    ) -> Vec<(Float, Vec<(String, Float)>)> {
        let outputs = self.eval_with_tape(inputs, tape);
        outputs
            .into_iter()
            .map(|(value, grads)| {
                let named = self
                    .input_names
                    .iter()
                    .cloned()
                    .zip(grads)
                    .collect::<Vec<_>>();
                (value, named)
            })
            .collect()
    }

    /// Reverse-mode evaluation for selected outputs with named gradients.
    pub fn eval_named_for(
        &self,
        inputs: &[Float],
        outputs: &[NodeId],
    ) -> Vec<(Float, Vec<(String, Float)>)> {
        let mut tape = self.reverse_tape();
        self.eval_named_for_with_tape(inputs, outputs, &mut tape)
    }

    /// Selected-output named reverse-mode evaluation with a reusable tape.
    pub fn eval_named_for_with_tape(
        &self,
        inputs: &[Float],
        outputs: &[NodeId],
        tape: &mut ReverseTape,
    ) -> Vec<(Float, Vec<(String, Float)>)> {
        let outputs = self.eval_for_with_tape(inputs, outputs, tape);
        outputs
            .into_iter()
            .map(|(value, grads)| {
                let named = self
                    .input_names
                    .iter()
                    .cloned()
                    .zip(grads)
                    .collect::<Vec<_>>();
                (value, named)
            })
            .collect()
    }
}
