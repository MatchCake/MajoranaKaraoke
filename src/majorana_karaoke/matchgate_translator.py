import warnings
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pennylane as qml
from matchcake import NonInteractingFermionicDevice
from matchcake.operations import (
    MatchgateOperation,
    SingleParticleTransitionMatrixOperation,
    fSWAP,
)
from pennylane.operation import Operator
from pennylane.ops.op_math import Adjoint
from pennylane.tape import QuantumScript
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

TranslationRule = Callable[[Operator], Union[Operator, Sequence[Operator]]]


class MatchgateTranslator:
    r"""
    Translate any PennyLane gate into operations that MatchCake's ``nif.qubit`` device simulates.

    Mid-circuit measurements are not supported, as on ``nif.qubit``.

    Each operation goes through the first step that applies to it:

    1. An ``Adjoint`` operation, which ``qml.adjoint`` creates by default, becomes the reversed adjoints of the
       translation of its base, so it is translated into the inverse of the translation of its base. The translation
       works gate by gate and is not multiplicative: the translation of ``CNOT`` does not square to the identity, so
       an inverse written gate by gate is not translated into an inverse.
    2. The operations that ``nif.qubit`` applies directly are kept: ``MatchgateOperation``,
       ``SingleParticleTransitionMatrixOperation`` and their subclasses, any operation that provides its single-particle
       transition matrix (through ``to_sptm_operation`` or ``single_particle_transition_matrix``), such as
       ``FermionicSuperposition``, and the state preparations that ``nif.qubit`` executes. MatchCake templates such as
       ``SptmAngleEmbedding`` are decomposed into these operations, as on ``nif.qubit``. A ``BasisState`` on part of the
       wires is extended with zeros to all the wires. ``StatePrep``, which ``nif.qubit`` lists but cannot
       execute, is decomposed.
    3. An operation whose name is a key of ``rules`` is replaced by the operations its rule returns. By
       default, ``GlobalPhase`` and ``Identity`` are dropped and ``CNOT`` is written in its Mølmer–Sørensen form

       .. math::
           \mathrm{CNOT}_{c,t} = e^{-i\pi/4} R_Y^{(c)}(-\pi/2)\, R_X^{(c)}(-\pi/2)\, R_X^{(t)}(-\pi/2)\,
           \mathrm{XX}_{c,t}(\pi/2)\, R_Y^{(c)}(\pi/2),

       whose entangling part :math:`\mathrm{XX}(\pi/2) = e^{-i\frac{\pi}{4} X_c X_t}` is a matchgate.
    4. A two-qubit gate whose parameters are scalars and that is a matchgate at generic values of them, such as
       ``IsingXX``, is rebuilt as a :class:`~matchcake.operations.MatchgateOperation`. Such a gate that is a matchgate
       only at special values, such as ``CRZ(0)``, is decomposed at every value, so its translation does not jump
       there. A gate with a matrix-valued parameter, such as ``QubitUnitary``, is kept whenever its current matrix is a
       matchgate, to ``MATCHGATE_ATOL``, and decomposed otherwise. When the wires of the gate
       are not neighbours, fermionic swaps carry its higher wire next to its lower wire before the matchgate and back
       after it. The routed gate is the fermionic (Jordan–Wigner) version of the gate, with a :math:`Z` string on the
       wires in between, and not the qubit gate.
    5. A single-qubit gate :math:`G` on the wire :math:`w` is dressed into the matchgate

       .. math::
           M(G, G) = \mathrm{CNOT}_{w,w+1}\, (G \otimes I)\, \mathrm{CNOT}_{w,w+1}

       acting on :math:`(w, w + 1)`, which maps :math:`X_w \to X_w X_{w+1}`, :math:`Y_w \to Y_w X_{w+1}` and
       :math:`Z_w \to Z_w`. On the last wire, it becomes :math:`M(G, XGX)` acting on :math:`(w - 1, w)`, which is
       :math:`I \otimes G` conjugated by :math:`\mathrm{CNOT}_{w,w-1}`. Both are exact when :math:`G` is diagonal,
       so a circuit made of the gates kept by step 4 on neighbouring wires and of diagonal single-qubit gates is
       translated exactly.
    6. Any other operation is decomposed and every operation of its decomposition is translated.
    7. An operation that has a matrix but no decomposition is translated as a ``QubitUnitary`` of its matrix, with a
       warning: PennyLane decomposes that matrix numerically, so the translation is not continuous in the parameters
       of the operation, and it does not support trainable or broadcast parameters.

    The operations returned by a rule or a decomposition are translated in turn, so a rule may return any PennyLane
    operation.

    :param rules: Extra translation rules indexed by operation name. A rule returns an operation or a sequence of
        operations. They take precedence over the default rules.
    :type rules: Optional[Mapping[str, TranslationRule]]
    :param max_depth: Maximum number of nested translation steps before a translation is considered divergent.
    :type max_depth: int

    :ivar rules: The translation rules indexed by operation name, the default ones included.
    """

    DEFAULT_MAX_DEPTH = 64
    MATCHGATE_ATOL = 1e-8
    # nif.qubit lists StatePrep as supported, but none of its probability strategies executes it.
    NATIVE_OPS = frozenset(NonInteractingFermionicDevice._supported_ops - {"StatePrep"})

    @staticmethod
    def cnot_to_molmer_sorensen(op: Operator) -> List[Operator]:
        r"""
        Write a ``CNOT`` as single-qubit rotations around the matchgate ``IsingXX``.

        :param op: The ``CNOT`` operation.
        :type op: Operator
        :return: Operations whose product is :math:`e^{i\pi/4}` times ``op``.
        :rtype: List[Operator]
        """
        control, target = op.wires
        return [
            qml.RY(np.pi / 2, wires=control),
            qml.IsingXX(np.pi / 2, wires=[control, target]),
            qml.RX(-np.pi / 2, wires=control),
            qml.RX(-np.pi / 2, wires=target),
            qml.RY(-np.pi / 2, wires=control),
        ]

    @staticmethod
    def drop(op: Operator) -> List[Operator]:
        """
        Remove an operation that has no observable effect, such as a global phase.

        :param op: The operation to remove.
        :type op: Operator
        :return: An empty list.
        :rtype: List[Operator]
        """
        return []

    @staticmethod
    def is_native(op: Operator) -> bool:
        """
        Tell whether ``nif.qubit`` applies an operation directly, with the same rule as ``nif.qubit``.

        :param op: The operation to check.
        :type op: Operator
        :return: True for matchgates, single-particle transition matrix (SPTM) operations, operations that provide
            their SPTM, and the state preparations that ``nif.qubit`` executes.
        :rtype: bool
        """
        if isinstance(op, (MatchgateOperation, SingleParticleTransitionMatrixOperation)):
            return True
        if hasattr(op, "to_sptm_operation") or hasattr(op, "single_particle_transition_matrix"):
            return True
        return op.name in MatchgateTranslator.NATIVE_OPS

    @staticmethod
    def _adjoint(op: Operator) -> Operator:
        # The adjoint of an SPTM is its transpose, built with the base class method because some MatchCake subclasses,
        # such as SptmFermionicSuperposition, override adjoint() with themselves although they are not involutions.
        if isinstance(op, SingleParticleTransitionMatrixOperation):
            return SingleParticleTransitionMatrixOperation.adjoint(op)
        adjoint_op = qml.adjoint(op, lazy=False)
        if isinstance(adjoint_op, Adjoint) and hasattr(op, "to_sptm_operation"):
            # Operations such as FermionicSuperposition provide their SPTM but no adjoint of their own.
            return SingleParticleTransitionMatrixOperation.adjoint(op.to_sptm_operation())
        return adjoint_op

    @staticmethod
    def _check_wire_order(wire_order: WiresLike) -> Wires:
        wire_order = Wires(wire_order)
        labels = wire_order.tolist()
        is_consecutive = all(isinstance(label, (int, np.integer)) for label in labels) and all(
            right - left == 1 for left, right in zip(labels[:-1], labels[1:])
        )
        if not is_consecutive:
            raise ValueError(f"The wires must be consecutive increasing integers to host matchgates. Got {wire_order}.")
        return wire_order

    @staticmethod
    def _conjugate_by_x(gate: TensorLike) -> TensorLike:
        # X G X reverses the rows and the columns of G: (..., 2, 2) -> (..., 2, 2)
        first_row = qml.math.stack([gate[..., 1, 1], gate[..., 1, 0]], axis=-1)
        second_row = qml.math.stack([gate[..., 0, 1], gate[..., 0, 0]], axis=-1)
        return qml.math.stack([first_row, second_row], axis=-2)

    @staticmethod
    def _default_wire_order(tape: QuantumScript) -> Wires:
        labels = tape.wires.tolist()
        if len(labels) == 0 or not all(isinstance(label, (int, np.integer)) for label in labels):
            return tape.wires
        return Wires(range(min(labels), max(labels) + 1))

    @staticmethod
    def _extend_basis_state(op: qml.BasisState, wire_order: Wires) -> qml.BasisState:
        if op.wires == wire_order:
            return op
        extended_bits = np.zeros(len(wire_order), dtype=int)
        extended_bits[[wire_order.index(wire) for wire in op.wires]] = qml.math.to_numpy(op.parameters[0])
        return qml.BasisState(extended_bits, wires=wire_order)

    @classmethod
    def _is_matchgate_matrix(cls, matrix: TensorLike) -> bool:
        matrix = np.asarray(qml.math.to_numpy(matrix))  # (..., 4, 4)
        rows, columns = zip(*MatchgateOperation.MATCHGATE_ZERO_POSITIONS)
        outer_det = matrix[..., 0, 0] * matrix[..., 3, 3] - matrix[..., 0, 3] * matrix[..., 3, 0]
        inner_det = matrix[..., 1, 1] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 1]
        has_structure = np.allclose(matrix[..., list(rows), list(columns)], 0.0, rtol=0.0, atol=cls.MATCHGATE_ATOL)
        return bool(has_structure and np.allclose(outer_det, inner_det, rtol=0.0, atol=cls.MATCHGATE_ATOL))

    @classmethod
    def _is_matchgate_family(cls, op: Operator, wires: Sequence) -> bool:
        # Scalar parameters are replaced by generic values, so the decision depends on the gate and not on whether
        # its current parameters sit at a special point such as CRZ(0) = I.
        batched_scalar_ndim = 0 if op.batch_size is None else 1
        if len(op.data) == 0 or any(qml.math.ndim(param) > batched_scalar_ndim for param in op.data):
            return True
        generic_params = [np.sqrt(2.0) * (index + 1) for index in range(len(op.data))]
        try:
            generic_op = qml.ops.functions.bind_new_parameters(op, generic_params)
            generic_matrix = qml.matrix(generic_op, wire_order=wires)
        except (IndexError, TypeError, ValueError):
            # PennyLane cannot rebuild some operations from their data alone, such as the evolution of a Hamiltonian.
            return True
        return cls._is_matchgate_matrix(generic_matrix)

    def __init__(
        self,
        rules: Optional[Mapping[str, TranslationRule]] = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ):
        self.rules: Dict[str, TranslationRule] = {
            "CNOT": self.cnot_to_molmer_sorensen,
            "GlobalPhase": self.drop,
            "Identity": self.drop,
            **(rules or {}),
        }
        self.max_depth = max_depth

    def translate(self, op: Operator, wire_order: WiresLike) -> List[Operator]:
        """
        Translate an operation into operations that ``nif.qubit`` simulates.

        :param op: The operation to translate.
        :type op: Operator
        :param wire_order: The wires of the device, in order. They must be consecutive increasing integers because
            a matchgate acts on neighbouring wires.
        :type wire_order: WiresLike
        :return: The translated operations, in the order they are applied.
        :rtype: List[Operator]
        :raises ValueError: If ``wire_order`` is not made of consecutive increasing integers, if ``op`` acts on a
            wire that is not in ``wire_order``, or if ``op`` cannot be translated.
        """
        wire_order = self._check_wire_order(wire_order)
        missing_wires = Wires(op.wires) - wire_order
        if len(missing_wires) > 0:
            raise ValueError(f"The operation {op} acts on the wires {missing_wires}, which are not in {wire_order}.")
        with qml.QueuingManager.stop_recording():
            return self._translate(op, wire_order, depth=0)

    def translate_tape(self, tape: QuantumScript, wire_order: Optional[WiresLike] = None) -> QuantumScript:
        """
        Translate every operation of a tape and keep its measurements as they are.

        :param tape: The tape to translate.
        :type tape: QuantumScript
        :param wire_order: The wires of the device, in order. Defaults to every integer from the smallest to the
            largest wire of the tape.
        :type wire_order: Optional[WiresLike]
        :return: A copy of ``tape`` whose operations are translated.
        :rtype: QuantumScript
        """
        if wire_order is None:
            wire_order = self._default_wire_order(tape)
        operations = [new_op for op in tape.operations for new_op in self.translate(op, wire_order)]
        return tape.copy(operations=operations)

    def _dress_single_qubit_gate(self, op: Operator, wire_order: Wires) -> MatchgateOperation:
        if len(wire_order) < 2:
            raise ValueError(f"Translating {op} requires a neighbouring wire, but the only wires are {wire_order}.")
        gate = op.matrix()  # (..., 2, 2)
        index = wire_order.index(op.wires[0])
        if index + 1 < len(wire_order):
            return MatchgateOperation.from_sub_matrices(gate, gate, wires=wire_order[index : index + 2])
        return MatchgateOperation.from_sub_matrices(gate, self._conjugate_by_x(gate), wires=wire_order[index - 1 :])

    def _route_matchgate(self, op: Operator, wire_order: Wires) -> Optional[List[Operator]]:
        first_index, second_index = sorted(wire_order.index(wire) for wire in op.wires)
        sorted_wires = [wire_order[first_index], wire_order[second_index]]
        matrix = qml.matrix(op, wire_order=sorted_wires)  # (..., 4, 4)
        if not (self._is_matchgate_family(op, sorted_wires) and self._is_matchgate_matrix(matrix)):
            return None
        matchgate = MatchgateOperation(matrix, wires=wire_order[first_index : first_index + 2])
        route = [fSWAP(wires=wire_order[index - 1 : index + 1]) for index in range(second_index, first_index + 1, -1)]
        return route + [matchgate] + route[::-1]

    def _translate(self, op: Operator, wire_order: Wires, depth: int) -> List[Operator]:
        if depth > self.max_depth:
            raise ValueError(f"The translation of {op} did not end after {self.max_depth} nested steps.")
        if isinstance(op, Adjoint):
            base_ops = self._translate(op.base, wire_order, depth + 1)
            return [self._adjoint(base_op) for base_op in reversed(base_ops)]
        if self.is_native(op):
            return [self._extend_basis_state(op, wire_order) if isinstance(op, qml.BasisState) else op]
        if op.name in self.rules:
            return self._translate_all(self.rules[op.name](op), wire_order, depth)
        if len(op.wires) == 2 and op.has_matrix:
            matchgate_ops = self._route_matchgate(op, wire_order)
            if matchgate_ops is not None:
                return matchgate_ops
        if len(op.wires) == 1 and op.has_matrix:
            return [self._dress_single_qubit_gate(op, wire_order)]
        if op.has_decomposition:
            return self._translate_all(op.decomposition(), wire_order, depth)
        if op.has_matrix and not isinstance(op, qml.QubitUnitary):
            warnings.warn(
                f"{op} has no decomposition, so it is translated through a numerical decomposition of its matrix, "
                "whose translation is not continuous in the parameters of the operation.",
                UserWarning,
            )
            return self._translate_all(qml.QubitUnitary(op.matrix(), wires=op.wires), wire_order, depth)
        raise ValueError(f"The operation {op} has no matrix, no decomposition and no translation rule.")

    def _translate_all(self, ops: Union[Operator, Sequence[Operator]], wire_order: Wires, depth: int) -> List[Operator]:
        # Rules and some decompositions, such as the one of Pow, return a single operation.
        ops = [ops] if isinstance(ops, Operator) else ops
        return [new_op for op in ops for new_op in self._translate(op, wire_order, depth + 1)]
