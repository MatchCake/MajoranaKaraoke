# Tutorials

The notebooks of this folder are executed in continuous integration and rendered in the
[documentation](https://MatchCake.github.io/MajoranaKaraoke/).

- [Translating a Circuit into Matchgates](translating_a_circuit.ipynb): how `mk.qubit`, the `translate_to_matchgates`
  transform and the `MatchgateTranslator` translate a circuit operation by operation, which circuits are translated
  exactly, how far the cover of a generic circuit is from the original, how to write your own translation rules, and
  a 40-qubit circuit that a statevector simulator could not hold.
- [Quantum Kernel SVC on Iris: Unconstrained vs Matchgate](iris_kernel_svc.ipynb): one fidelity kernel evaluated with
  the same embedding circuit on `default.qubit` and on `mk.qubit`, used in support vector classifiers on the Iris
  dataset next to a classical RBF baseline, and the cost of both devices as the number of qubits grows.
