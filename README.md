# MajoranaKaraoke

[![Star on GitHub](https://img.shields.io/github/stars/MatchCake/MajoranaKaraoke.svg?style=social)](https://github.com/MatchCake/MajoranaKaraoke/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/MatchCake/MajoranaKaraoke?style=social)](https://github.com/MatchCake/MajoranaKaraoke/network/members)
[![Python 3.11 to 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/MatchCake/MajoranaKaraoke/blob/dev/LICENSE)

![Tests Workflow](https://github.com/MatchCake/MajoranaKaraoke/actions/workflows/tests.yml/badge.svg)
![Doc Workflow](https://github.com/MatchCake/MajoranaKaraoke/actions/workflows/docs.yml/badge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/MatchCake/MajoranaKaraoke/branch/main/graph/badge.svg)](https://codecov.io/github/MatchCake/MajoranaKaraoke)


# Description

MajoranaKaraoke is a minimal [PennyLane](https://pennylane.ai/) plugin that runs any unitary circuit on the `nif.qubit`
device of [MatchCake](https://github.com/MatchCake/MatchCake). Every operation of the circuit is translated into
matchgates, and the translated circuit is simulated by MatchCake in a time polynomial in the number of qubits for
expectation values and for the probability of a given outcome. The device is registered in PennyLane as `mk.qubit`.

A matchgate is a two-qubit gate acting on neighbouring wires whose matrix has the form

```math
M(A, B) = \begin{pmatrix} A_{00} & 0 & 0 & A_{01} \\ 0 & B_{00} & B_{01} & 0 \\ 0 & B_{10} & B_{11} & 0 \\ A_{10} & 0 & 0 & A_{11} \end{pmatrix}, \qquad \det A = \det B .
```

Through the Jordan–Wigner transformation, a circuit of matchgates describes non-interacting (free) fermions, which is
why it can be simulated efficiently. Matchgate circuits are not universal, so a general circuit cannot be rewritten as
one. Like a karaoke singer, MajoranaKaraoke performs a matchgate cover of the original circuit instead.

**The translation is an approximation, and it is exact only on a gate-level subset of free-fermion circuits**:
circuits made of two-qubit gates on neighbouring wires that are matchgates for every value of their parameters (such as
`IsingXX`, `IsingYY`, `IsingXY` or `SingleExcitation`), diagonal single-qubit gates, and the operations and state
preparations that `nif.qubit` executes. A gate that is a matchgate only at special values, such as `CRZ(0)`, and other
free-fermion gates, such as `PauliRot(theta, "XZX")`, are decomposed and translated approximately. On any other
circuit, the cover is a different circuit: the results of `mk.qubit` differ in general from
those of `default.qubit`, and nothing bounds how much. They are always the exact results of the translated circuit,
which the `translate_to_matchgates` transform lets you inspect.

## Translation rules

The `MatchgateTranslator` sends each operation through the first rule that applies to it:

1. **Adjoints.** An `Adjoint` operation, which `qml.adjoint` creates by default, is translated into the inverse of the
   translation of its base, so a circuit followed by its adjoint still returns to its initial state. The translation
   works gate by gate and is not multiplicative (the cover of `CNOT` does not square to the identity), so an inverse
   written gate by gate is not translated into an inverse.
2. **Native operations.** The operations that `nif.qubit` applies directly are kept as they are: `MatchgateOperation`,
   `SingleParticleTransitionMatrixOperation` (SPTM) and their subclasses, such as `fSWAP` or `SptmCompRxRx`, any
   operation that provides its SPTM, such as `FermionicSuperposition`, and the state preparations that `nif.qubit`
   executes, such as `BasisState`. MatchCake templates such as `SptmAngleEmbedding` are decomposed into these
   operations, as on `nif.qubit`. A `BasisState` on part of the wires is extended with zeros to all the wires.
   `StatePrep`, which `nif.qubit` lists but cannot execute, is decomposed.
3. **Rules by name.** `GlobalPhase` and `Identity` are dropped, and `CNOT` is written in its Mølmer–Sørensen form

   ```math
   \mathrm{CNOT}_{c,t} = e^{-i\pi/4} R_Y^{(c)}(-\pi/2) R_X^{(c)}(-\pi/2) R_X^{(t)}(-\pi/2) \mathrm{XX}_{c,t}(\pi/2) R_Y^{(c)}(\pi/2),
   ```

   whose entangling part $`\mathrm{XX}(\pi/2) = e^{-i \frac{\pi}{4} X \otimes X}`$ is a matchgate. Your own rules are
   added to these and take precedence over them.
4. **Two-qubit matchgates.** A two-qubit gate with scalar parameters that is a matchgate at generic values of them,
   such as `IsingXX`, `IsingYY` or `IsingXY`, becomes a `MatchgateOperation`. Such a gate that is a matchgate only at
   special values, such as `CRZ(0)`, is decomposed at every value, so the cover does not jump there. A gate with a
   matrix-valued parameter, such as `QubitUnitary`, is kept whenever its current matrix is a matchgate. When its wires are not neighbours, fermionic swaps (`fSWAP`) bring them together and
   back. This applies the fermionic version of the gate, with its Jordan–Wigner string, and not the qubit gate.
5. **Single-qubit gates.** A gate $`G`$ on the wire $`w`$ becomes the matchgate
   $`M(G, G) = \mathrm{CNOT}_{w,w+1} (G \otimes I) \mathrm{CNOT}_{w,w+1}`$ on the wires $`(w, w+1)`$. On the last wire,
   it becomes $`M(G, XGX) = \mathrm{CNOT}_{w,w-1} (I \otimes G) \mathrm{CNOT}_{w,w-1}`$ on the wires $`(w-1, w)`$.
   Both are exact when $`G`$ is diagonal.
6. **Everything else** is decomposed with PennyLane and each operation of the decomposition is translated in turn. An
   operation that has a matrix but no decomposition is translated as a `QubitUnitary` of its matrix, with a warning:
   PennyLane decomposes that matrix numerically, so the cover is not continuous in the parameters of the operation. An
   operation that none of these rules translates, such as a mid-circuit measurement, raises a `ValueError`.

Because matchgates act on neighbouring wires, the wires of the device must be consecutive increasing integers.

Differentiate with `diff_method="backprop"` and the `torch` interface, or with `diff_method="finite-diff"`. The
parameter-shift rule uses the shift rules of the original gates. They may not apply to the cover of a gate translated
through a decomposition, whose dependence on the parameter can contain other frequencies, so parameter-shift gradients
are wrong for gates such as `ControlledPhaseShift`.


# Requirements

| Requirement   | Supported                  |
|---------------|----------------------------|
| **Python**    | 3.11, 3.12, 3.13, 3.14     |
| **Platforms** | Linux, Windows             |
| **PennyLane** | 0.45 or newer              |
| **MatchCake** | 1.0.0 or newer             |

Every one of these Python versions is tested on Linux (x86-64 and aarch64) and on Windows in
[continuous integration](https://github.com/MatchCake/MajoranaKaraoke/actions/workflows/tests.yml).


# Installation

MajoranaKaraoke is not published on PyPI yet. Until it is, install it from GitHub into a virtual environment
(`git` must be available on your `PATH`):

| Method  | Commands                                                                                    |
|---------|---------------------------------------------------------------------------------------------|
| **pip** | `pip install git+https://github.com/MatchCake/MajoranaKaraoke`                              |
| **uv**  | `uv add "majorana-karaoke @ git+https://github.com/MatchCake/MajoranaKaraoke"` in a uv project, or `uv pip install git+https://github.com/MatchCake/MajoranaKaraoke` |

These install the `main` branch. Append `@dev` to the URL to install the development branch instead, for example
`pip install "git+https://github.com/MatchCake/MajoranaKaraoke@dev"`.

MajoranaKaraoke depends on MatchCake, which depends on PyTorch. On Linux, the default PyTorch wheel is the CUDA build,
which downloads several gigabytes of NVIDIA packages. For a CPU only environment, install PyTorch from the PyTorch CPU
index first, as explained in the
[installation guide of MatchCake](https://github.com/MatchCake/MatchCake#pytorch-build-and-installation-size):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/MatchCake/MajoranaKaraoke
```


# Quick Usage Preview

## Run any circuit on `mk.qubit`

On a free-fermion circuit, `mk.qubit` gives the same results as `default.qubit`:

```python
import numpy as np
import pennylane as qml


def circuit(theta):
    qml.BasisState(np.array([1, 0, 1, 0]), wires=range(4))
    qml.IsingXX(theta, wires=[0, 1])
    qml.IsingYY(theta, wires=[2, 3])
    qml.RZ(theta, wires=1)
    qml.IsingXY(theta, wires=[1, 2])
    return qml.probs(wires=range(4))


karaoke = qml.QNode(circuit, qml.device("mk.qubit", wires=4))
original = qml.QNode(circuit, qml.device("default.qubit", wires=4))
print(np.allclose(karaoke(0.3), original(0.3)))  # True
```

## See what `mk.qubit` actually simulates

`SWAP` is not a matchgate, so its cover is only an approximation. Given the wires of the device,
`translate_to_matchgates` applies the same translation as `mk.qubit`, so running the translated circuit on
`default.qubit` reproduces the results of `mk.qubit`, except for MatchCake's single-particle transition matrix
operations, whose `matrix()` is not a qubit unitary. Without `wire_order`, the transform uses the integers from the
smallest to the largest wire of the circuit, which dresses a gate on the last of them differently when the device has
more wires:

```python
from majorana_karaoke import translate_to_matchgates


def swap_circuit():
    qml.BasisState(np.array([1, 0]), wires=[0, 1])
    qml.SWAP(wires=[0, 1])
    return qml.probs(wires=[0, 1])


original = qml.QNode(swap_circuit, qml.device("default.qubit", wires=2))
karaoke = qml.QNode(swap_circuit, qml.device("mk.qubit", wires=2))
cover = translate_to_matchgates(original, wire_order=[0, 1])

print(original())  # [0. 1. 0. 0.], |10> becomes |01>
print(qml.math.toarray(karaoke()).round(3))  # [0. 0. 1. 0.], the cover leaves |10> in place
print(qml.math.toarray(cover()).round(3))  # [0. 0. 1. 0.], the same as mk.qubit
```

MatchCake builds its matchgates with PyTorch, so `mk.qubit` returns `torch` tensors; `qml.math.toarray` converts them to
NumPy arrays.

## Write your own translation rules

A rule maps the name of an operation to a function that returns the operations replacing it. For instance, the
fermionic swap $`\mathrm{fSWAP} = \mathrm{SWAP} \cdot \mathrm{CZ}`$ is a matchgate that acts as `SWAP` on every state
without a $`|11\rangle`$ component:

```python
from matchcake.operations import fSWAP

from majorana_karaoke import MatchgateTranslator

translator = MatchgateTranslator(rules={"SWAP": lambda op: [fSWAP(wires=op.wires)]})
karaoke = qml.QNode(swap_circuit, qml.device("mk.qubit", wires=2, translator=translator))
print(qml.math.toarray(karaoke()).round(3))  # [0. 1. 0. 0.]
```


# Tutorials

- [Translating a Circuit into Matchgates](https://github.com/MatchCake/MajoranaKaraoke/blob/main/tutorials/translating_a_circuit.ipynb)
- [Quantum Kernel SVC on Iris: Unconstrained vs Matchgate](https://github.com/MatchCake/MajoranaKaraoke/blob/main/tutorials/iris_kernel_svc.ipynb)


# Contributing

To contribute to the development of MajoranaKaraoke, please refer to the
[contributing guidelines](https://github.com/MatchCake/MajoranaKaraoke/blob/dev/CONTRIBUTING.md).


# Important Links

- Documentation at [https://MatchCake.github.io/MajoranaKaraoke/](https://MatchCake.github.io/MajoranaKaraoke/).
- GitHub at [https://github.com/MatchCake/MajoranaKaraoke/](https://github.com/MatchCake/MajoranaKaraoke/).
- Found a bug or have a feature request?
  [Click here to create a new issue.](https://github.com/MatchCake/MajoranaKaraoke/issues/new/choose)


# License

[Apache License 2.0](https://github.com/MatchCake/MajoranaKaraoke/blob/dev/LICENSE)


# Citation

If you use MajoranaKaraoke, please cite the software, whose metadata is in
[CITATION.cff](https://github.com/MatchCake/MajoranaKaraoke/blob/dev/CITATION.cff):

```
@software{majoranakaraoke_Gince2026,
  title={MajoranaKaraoke},
  author={Gince, Jérémie},
  year={2026},
  url={https://github.com/MatchCake/MajoranaKaraoke},
}
```

Please also cite MatchCake, which performs every simulation of MajoranaKaraoke. It is archived on
[Zenodo](https://doi.org/10.5281/zenodo.22917735) (this DOI resolves to the latest version), and described in its
[Journal of Open Source Software paper](https://joss.theoj.org/papers/91f77a47cbd519daac9794a1d2144361):

```
@software{matchcake_Gince2026,
  title={MatchCake: A Python Simulator for Non-Interacting Fermionic Quantum Circuits with Machine Learning Applications},
  author={Gince, Jérémie},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.22917735},
  url={https://doi.org/10.5281/zenodo.22917735},
}
```
