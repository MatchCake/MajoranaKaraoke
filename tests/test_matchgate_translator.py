from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pennylane as qml
import pytest
import torch
from matchcake.operations import (
    FermionicSuperposition,
    MatchgateOperation,
    SingleParticleTransitionMatrixOperation,
    SptmCompRxRx,
    SptmFermionicSuperposition,
    fSWAP,
)
from pennylane.operation import Operator
from pennylane.tape import QuantumScript
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike
from scipy.stats import unitary_group

from majorana_karaoke import MatchgateTranslator

from .configs import (
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)

DIAGONAL_GATES: Dict[str, tuple] = {
    "RZ": (0.37,),
    "PhaseShift": (1.21,),
    "S": (),
    "T": (),
    "PauliZ": (),
}

GENERATED_MATCHGATES: Dict[str, Callable[[WiresLike], Operator]] = {
    "ising_xx": lambda wires: qml.IsingXX(0.7, wires=wires),
    "ising_yy": lambda wires: qml.IsingYY(0.4, wires=wires),
    "ising_xy": lambda wires: qml.IsingXY(0.9, wires=wires),
    "single_excitation": lambda wires: qml.SingleExcitation(1.3, wires=wires),
    "fermionic_swap": lambda wires: qml.FermionicSWAP(0.6, wires=wires),
}

MATCHGATES: Dict[str, Callable[[WiresLike], Operator]] = {
    **GENERATED_MATCHGATES,
    "random_matchgate": lambda wires: qml.QubitUnitary(
        MatchgateOperation.random(wires=[0, 1], seed=TEST_SEED).matrix(), wires=wires
    ),
}

NON_MATCHGATE_TWO_QUBIT_GATES: Dict[str, Callable[[WiresLike], Operator]] = {
    "cz": lambda wires: qml.CZ(wires=wires),
    "swap": lambda wires: qml.SWAP(wires=wires),
    "cry": lambda wires: qml.CRY(0.4, wires=wires),
    "ising_zz": lambda wires: qml.IsingZZ(0.3, wires=wires),
    "controlled_phase_shift": lambda wires: qml.ControlledPhaseShift(0.5, wires=wires),
}

IMAGINARY_OFF_STRUCTURE_GATES: Dict[str, Callable[[WiresLike], Operator]] = {
    "crx": lambda wires: qml.CRX(1.2, wires=wires),
    "pauli_rot_xz": lambda wires: qml.PauliRot(1.2, "XZ", wires=wires),
    "rx_tensor_identity": lambda wires: qml.QubitUnitary(
        np.kron(qml.matrix(qml.RX(1.2, wires=0)), np.eye(2)), wires=wires
    ),
    "cy": lambda wires: qml.CY(wires=wires),
    "pauli_rot_zx": lambda wires: qml.PauliRot(0.8, "ZX", wires=wires),
    "controlled_pauli_y": lambda wires: qml.ctrl(qml.PauliY(wires[1]), control=wires[0]),
}

SPECIAL_POINT_GATES: Dict[str, Callable[[TensorLike], Operator]] = {
    "crz": lambda angle: qml.CRZ(angle, wires=[0, 1]),
    "controlled_phase_shift": lambda angle: qml.ControlledPhaseShift(angle, wires=[0, 1]),
    "cry": lambda angle: qml.CRY(angle, wires=[0, 1]),
    "ising_zz": lambda angle: qml.IsingZZ(angle, wires=[0, 1]),
}

MATRIX_WITHOUT_DECOMPOSITION_OPS: Dict[str, Callable[[], Operator]] = {
    "square_root_of_swap": lambda: qml.pow(qml.SWAP(wires=[0, 1]), 0.5),
    "free_fermion_evolution_on_three_sites": lambda: qml.evolve(
        qml.Hamiltonian(
            [0.5, 0.5, 0.3, 0.3],
            [qml.X(0) @ qml.X(1), qml.Y(0) @ qml.Y(1), qml.X(1) @ qml.X(2), qml.Y(1) @ qml.Y(2)],
        ),
        0.7,
    ),
}

NATIVE_OPS: Dict[str, Callable[[], Operator]] = {
    "matchgate": lambda: MatchgateOperation.random(wires=[0, 1], seed=TEST_SEED),
    "fswap": lambda: fSWAP(wires=[1, 2]),
    "sptm": lambda: SptmCompRxRx(np.array([0.1, 0.2]), wires=[0, 1]),
    "random_sptm": lambda: SingleParticleTransitionMatrixOperation.random(wires=[0, 1, 2], seed=TEST_SEED),
    "provides_its_sptm": lambda: FermionicSuperposition(wires=[1, 2]),
    "basis_state": lambda: qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2]),
}

NON_NATIVE_OPS: Dict[str, Callable[[], Operator]] = {
    "rz": lambda: qml.RZ(0.3, wires=0),
    "ising_xx": lambda: qml.IsingXX(0.3, wires=[0, 1]),
    "cnot": lambda: qml.CNOT(wires=[0, 1]),
    "global_phase": lambda: qml.GlobalPhase(0.2),
    "state_prep": lambda: qml.StatePrep(np.array([1.0, 0.0, 0.0, 0.0]), wires=[0, 1]),
}

DROPPED_OPS: Dict[str, Callable[[], Operator]] = {
    "global_phase_without_wires": lambda: qml.GlobalPhase(0.3),
    "global_phase_on_a_wire": lambda: qml.GlobalPhase(0.3, wires=[1]),
    "identity": lambda: qml.Identity(wires=0),
    "identity_on_two_wires": lambda: qml.Identity(wires=[0, 1]),
}

ADJOINT_CASES: Dict[str, Callable[[], Operator]] = {
    "rx": lambda: qml.RX(0.3, wires=0),
    "hadamard_on_last_wire": lambda: qml.Hadamard(wires=2),
    "cnot": lambda: qml.CNOT(wires=[2, 0]),
    "ising_xy_reversed": lambda: qml.IsingXY(0.7, wires=[2, 1]),
    "ising_xx_routed": lambda: qml.IsingXX(0.5, wires=[0, 2]),
    "toffoli": lambda: qml.Toffoli(wires=[0, 1, 2]),
    "qft": lambda: qml.QFT(wires=[0, 1, 2]),
    "strongly_entangling": lambda: qml.StronglyEntanglingLayers(
        np.random.default_rng(TEST_SEED).uniform(size=(1, 3, 3)), wires=[0, 1, 2]
    ),
    "random_two_qubit_unitary": lambda: qml.QubitUnitary(unitary_group.rvs(4, random_state=TEST_SEED), wires=[0, 2]),
    "fswap": lambda: fSWAP(wires=[1, 2]),
    "matchgate": lambda: MatchgateOperation.random(wires=[0, 1], seed=TEST_SEED),
    "adjoint": lambda: qml.adjoint(qml.IsingXX(0.5, wires=[0, 2])),
}

BROADCAST_GATES: Dict[str, Callable[[TensorLike], Operator]] = {
    "rz_on_first_wire": lambda angles: qml.RZ(angles, wires=0),
    "phase_shift_on_last_wire": lambda angles: qml.PhaseShift(angles, wires=2),
    "rx_on_last_wire": lambda angles: qml.RX(angles, wires=2),
    "ising_xy_reversed": lambda angles: qml.IsingXY(angles, wires=[1, 0]),
    "ising_xx_routed": lambda angles: qml.IsingXX(angles, wires=[2, 0]),
    "cry_decomposed": lambda angles: qml.CRY(angles, wires=[0, 1]),
}


class TestMatchgateTranslator:
    @staticmethod
    def _matrix(ops: Sequence[Operator], wire_order: WiresLike) -> np.ndarray:
        return qml.math.to_numpy(qml.matrix(QuantumScript(list(ops)), wire_order=wire_order))

    @staticmethod
    def _cnot(control: int, target: int, wire_order: WiresLike) -> np.ndarray:
        return qml.matrix(qml.CNOT(wires=[control, target]), wire_order=wire_order)

    @staticmethod
    def _off_structure_entries(op: Operator) -> np.ndarray:
        matrix = qml.math.to_numpy(op.matrix())  # (..., 4, 4)
        return np.stack([matrix[..., row, column] for row, column in MatchgateOperation.MATCHGATE_ZERO_POSITIONS])

    @staticmethod
    def _random_unitary(seed: int) -> np.ndarray:
        return unitary_group.rvs(2, random_state=seed)

    @staticmethod
    def _with_jordan_wigner_string(op: Operator, wire_order: WiresLike) -> np.ndarray:
        # Every Pauli word acting on both wires of op gets a Z on each wire strictly between them.
        first, last = sorted(op.wires.tolist())
        string = {wire: "Z" for wire in range(first + 1, last)}
        sentence = qml.pauli.pauli_sentence(op.generator())
        fermionic_sentence = qml.pauli.PauliSentence(
            {
                (qml.pauli.PauliWord({**dict(word), **string}) if {first, last} <= set(word) else word): coefficient
                for word, coefficient in sentence.items()
            }
        )
        # PennyLane writes a parametrized gate as exp(i * theta * generator).
        generated = qml.exp(fermionic_sentence.operation(), 1j * op.parameters[0])
        return qml.matrix(generated, wire_order=wire_order)

    @classmethod
    def setup_class(cls) -> None:
        set_seed(TEST_SEED)

    def test_default_rules_and_max_depth(self) -> None:
        translator = MatchgateTranslator()
        assert set(translator.rules) == {"CNOT", "GlobalPhase", "Identity"}
        assert translator.rules["CNOT"] is MatchgateTranslator.cnot_to_molmer_sorensen
        assert translator.rules["GlobalPhase"] is MatchgateTranslator.drop
        assert translator.rules["Identity"] is MatchgateTranslator.drop
        assert translator.max_depth == MatchgateTranslator.DEFAULT_MAX_DEPTH

    def test_custom_rule_overrides_default_rule(self) -> None:
        def cnot_as_ising_xy(op: Operator) -> List[Operator]:
            return [qml.IsingXY(np.pi / 2, wires=op.wires)]

        translator = MatchgateTranslator(rules={"CNOT": cnot_as_ising_xy})
        translated = translator.translate(qml.CNOT(wires=[0, 1]), wire_order=[0, 1])

        assert translator.rules["CNOT"] is cnot_as_ising_xy
        assert set(translator.rules) == {"CNOT", "GlobalPhase", "Identity"}
        assert len(translated) == 1
        np.testing.assert_allclose(
            self._matrix(translated, [0, 1]),
            qml.matrix(qml.IsingXY(np.pi / 2, wires=[0, 1])),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_custom_rule_outputs_are_translated_in_turn(self) -> None:
        translator = MatchgateTranslator(
            rules={"Hadamard": lambda op: [qml.RZ(np.pi, wires=op.wires), qml.GlobalPhase(0.2), qml.CNOT([0, 1])]}
        )
        translated = translator.translate(qml.Hadamard(wires=1), wire_order=[0, 1])
        expected = [qml.RZ(np.pi, wires=1), *MatchgateTranslator.cnot_to_molmer_sorensen(qml.CNOT([0, 1]))]

        assert "CNOT" in translator.rules
        assert all(isinstance(op, MatchgateOperation) for op in translated)
        np.testing.assert_allclose(
            self._matrix(translated, [0, 1]),
            self._matrix([new_op for op in expected for new_op in translator.translate(op, [0, 1])], [0, 1]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("name", list(DROPPED_OPS))
    def test_global_phase_and_identity_are_dropped(self, name: str) -> None:
        op = DROPPED_OPS[name]()
        assert MatchgateTranslator.drop(op) == []
        assert MatchgateTranslator().translate(op, wire_order=[0, 1]) == []

    @pytest.mark.parametrize("name", list(NATIVE_OPS))
    def test_native_ops_are_kept_as_they_are(self, name: str) -> None:
        op = NATIVE_OPS[name]()
        translated = MatchgateTranslator().translate(op, wire_order=[0, 1, 2])
        assert MatchgateTranslator.is_native(op)
        assert len(translated) == 1
        assert translated[0] is op

    @pytest.mark.parametrize("name", list(NON_NATIVE_OPS))
    def test_non_native_ops_are_not_native(self, name: str) -> None:
        assert not MatchgateTranslator.is_native(NON_NATIVE_OPS[name]())

    @pytest.mark.parametrize("seed", range(3))
    @pytest.mark.parametrize("n_wires, wire", [(2, 0), (3, 0), (3, 1), (4, 2)])
    def test_single_qubit_gate_is_dressed_by_cnots(self, n_wires: int, wire: int, seed: int) -> None:
        gate = self._random_unitary(seed)
        pair = [wire, wire + 1]
        translated = MatchgateTranslator().translate(qml.QubitUnitary(gate, wires=wire), wire_order=range(n_wires))
        cnot = self._cnot(wire, wire + 1, pair)

        assert len(translated) == 1
        assert isinstance(translated[0], MatchgateOperation)
        assert translated[0].wires == Wires(pair)
        np.testing.assert_allclose(
            self._matrix(translated, pair),
            cnot @ np.kron(gate, np.eye(2)) @ cnot,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            self._matrix(translated, pair),
            qml.math.to_numpy(MatchgateOperation.from_sub_matrices(gate, gate, wires=pair).matrix()),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("seed", range(3))
    @pytest.mark.parametrize("n_wires", [2, 3, 4])
    def test_single_qubit_gate_on_last_wire_is_dressed_by_reversed_cnots(self, n_wires: int, seed: int) -> None:
        gate = self._random_unitary(seed)
        wire = n_wires - 1
        pair = [wire - 1, wire]
        translated = MatchgateTranslator().translate(qml.QubitUnitary(gate, wires=wire), wire_order=range(n_wires))
        reversed_cnot = self._cnot(wire, wire - 1, pair)

        assert len(translated) == 1
        assert isinstance(translated[0], MatchgateOperation)
        assert translated[0].wires == Wires(pair)
        np.testing.assert_allclose(
            self._matrix(translated, pair),
            reversed_cnot @ np.kron(np.eye(2), gate) @ reversed_cnot,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            self._matrix(translated, pair),
            qml.math.to_numpy(
                MatchgateOperation.from_sub_matrices(gate, PAULI_X @ gate @ PAULI_X, wires=pair).matrix()
            ),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("wire", [0, 1, 2])
    @pytest.mark.parametrize("name", list(DIAGONAL_GATES))
    def test_diagonal_gate_translation_is_exact(self, name: str, wire: int) -> None:
        op = getattr(qml, name)(*DIAGONAL_GATES[name], wires=wire)
        translated = MatchgateTranslator().translate(op, wire_order=[0, 1, 2])
        np.testing.assert_allclose(
            self._matrix(translated, [0, 1, 2]),
            qml.matrix(op, wire_order=[0, 1, 2]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("wires", [[0, 1], [1, 0], [0, 2], [2, 1]])
    def test_cnot_to_molmer_sorensen_is_cnot_up_to_global_phase(self, wires: List[int]) -> None:
        cnot = qml.CNOT(wires=wires)
        ops = MatchgateTranslator.cnot_to_molmer_sorensen(cnot)
        two_qubit_names = [op.name for op in ops if len(op.wires) == 2]

        assert two_qubit_names == ["IsingXX"]
        np.testing.assert_allclose(
            self._matrix(ops, [0, 1, 2]),
            np.exp(1j * np.pi / 4) * qml.matrix(cnot, wire_order=[0, 1, 2]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("wires", [[0, 1], [2, 1], [0, 2]])
    def test_cnot_is_translated_through_its_rule(self, wires: List[int]) -> None:
        translator = MatchgateTranslator()
        cnot = qml.CNOT(wires=wires)
        translated = translator.translate(cnot, wire_order=[0, 1, 2])
        expected = [
            new_op for op in translator.cnot_to_molmer_sorensen(cnot) for new_op in translator.translate(op, [0, 1, 2])
        ]

        assert all(isinstance(op, MatchgateOperation) for op in translated)
        np.testing.assert_allclose(
            self._matrix(translated, [0, 1, 2]),
            self._matrix(expected, [0, 1, 2]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("wires", [[0, 1], [1, 0], [1, 2], [2, 1]])
    @pytest.mark.parametrize("name", list(MATCHGATES))
    def test_adjacent_matchgate_is_translated_exactly(self, name: str, wires: List[int]) -> None:
        op = MATCHGATES[name](wires)
        translated = MatchgateTranslator().translate(op, wire_order=[0, 1, 2])

        assert len(translated) == 1
        assert isinstance(translated[0], MatchgateOperation)
        assert translated[0].wires == Wires(sorted(wires))
        np.testing.assert_allclose(
            self._matrix(translated, [0, 1, 2]),
            qml.matrix(op, wire_order=[0, 1, 2]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("wires", [[0, 2], [2, 0], [0, 3], [3, 1], [1, 3]])
    def test_non_adjacent_matchgate_is_routed_with_fswaps(self, wires: List[int]) -> None:
        first, second = sorted(wires)
        n_swaps = second - first - 1
        translated = MatchgateTranslator().translate(qml.IsingXX(0.7, wires=wires), wire_order=[0, 1, 2, 3])
        route, matchgate, route_back = translated[:n_swaps], translated[n_swaps], translated[n_swaps + 1 :]

        assert len(translated) == 2 * n_swaps + 1
        assert all(isinstance(op, fSWAP) for op in route + route_back)
        assert [op.wires for op in route] == [Wires([index - 1, index]) for index in range(second, first + 1, -1)]
        assert [op.wires for op in route_back] == [op.wires for op in reversed(route)]
        assert not isinstance(matchgate, fSWAP)
        assert matchgate.wires == Wires([first, first + 1])

    @pytest.mark.parametrize("wires", [[1, 2], [0, 2], [2, 0], [0, 3], [3, 1]])
    @pytest.mark.parametrize("name", list(GENERATED_MATCHGATES))
    def test_routed_matchgate_carries_a_jordan_wigner_string(self, name: str, wires: List[int]) -> None:
        op = GENERATED_MATCHGATES[name](wires)
        translated = MatchgateTranslator().translate(op, wire_order=[0, 1, 2, 3])
        np.testing.assert_allclose(
            self._matrix(translated, [0, 1, 2, 3]),
            self._with_jordan_wigner_string(op, [0, 1, 2, 3]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("wires", [[0, 1], [2, 0]])
    @pytest.mark.parametrize("name", list(NON_MATCHGATE_TWO_QUBIT_GATES))
    def test_non_matchgate_two_qubit_gate_is_decomposed(self, name: str, wires: List[int]) -> None:
        translator = MatchgateTranslator()
        op = NON_MATCHGATE_TWO_QUBIT_GATES[name](wires)
        translated = translator.translate(op, wire_order=[0, 1, 2])
        expected = [new_op for sub_op in op.decomposition() for new_op in translator.translate(sub_op, [0, 1, 2])]

        assert all(isinstance(new_op, MatchgateOperation) for new_op in translated)
        assert len(translated) == len(expected)
        np.testing.assert_allclose(
            self._matrix(translated, [0, 1, 2]),
            self._matrix(expected, [0, 1, 2]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("name", list(IMAGINARY_OFF_STRUCTURE_GATES))
    def test_gate_with_imaginary_off_structure_entries_is_not_kept_as_matchgate(self, name: str) -> None:
        translated = MatchgateTranslator().translate(IMAGINARY_OFF_STRUCTURE_GATES[name]([0, 1]), wire_order=[0, 1, 2])
        for new_op in translated:
            np.testing.assert_allclose(
                self._off_structure_entries(new_op),
                0.0,
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    @pytest.mark.parametrize("angle", [0.0, 1e-9, 1e-5, 2 * np.pi])
    @pytest.mark.parametrize("name", list(SPECIAL_POINT_GATES))
    def test_matchgate_detection_does_not_depend_on_parameter_values(self, name: str, angle: float) -> None:
        translator = MatchgateTranslator()
        op = SPECIAL_POINT_GATES[name](angle)
        translated = translator.translate(op, wire_order=[0, 1, 2])
        expected = [new_op for sub_op in op.decomposition() for new_op in translator.translate(sub_op, [0, 1, 2])]

        assert len(translated) == len(expected) > 1
        np.testing.assert_allclose(
            self._matrix(translated, [0, 1, 2]),
            self._matrix(expected, [0, 1, 2]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("name", list(MATRIX_WITHOUT_DECOMPOSITION_OPS))
    def test_op_with_matrix_but_no_decomposition_is_translated_as_qubit_unitary(self, name: str) -> None:
        translator = MatchgateTranslator()
        op = MATRIX_WITHOUT_DECOMPOSITION_OPS[name]()
        with pytest.warns(UserWarning, match="numerical decomposition of its matrix"):
            translated = translator.translate(op, wire_order=[0, 1, 2])
        expected = translator.translate(qml.QubitUnitary(op.matrix(), wires=op.wires), wire_order=[0, 1, 2])

        assert op.has_matrix and not op.has_decomposition
        assert all(MatchgateTranslator.is_native(new_op) for new_op in translated)
        np.testing.assert_allclose(
            self._matrix(translated, [0, 1, 2]),
            self._matrix(expected, [0, 1, 2]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("name", ["crz", "controlled_phase_shift", "cry"])
    def test_batch_mates_do_not_change_the_translation(self, name: str) -> None:
        translator = MatchgateTranslator()
        unbatched = self._matrix(translator.translate(SPECIAL_POINT_GATES[name](0.0), [0, 1, 2]), [0, 1, 2])
        for angles in (np.array([0.0, 0.0]), np.array([0.0, 0.5])):
            batched = self._matrix(translator.translate(SPECIAL_POINT_GATES[name](angles), [0, 1, 2]), [0, 1, 2])
            np.testing.assert_allclose(batched[0], unbatched, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_partly_batched_parameters_use_the_generic_family_check(self) -> None:
        translator = MatchgateTranslator()
        unbatched = self._matrix(translator.translate(qml.CRot(0.0, 0.0, 0.0, wires=[0, 1]), [0, 1, 2]), [0, 1, 2])
        batched_op = qml.CRot(0.0, np.array([0.0, 0.0]), 0.0, wires=[0, 1])
        batched = self._matrix(translator.translate(batched_op, [0, 1, 2]), [0, 1, 2])
        np.testing.assert_allclose(batched[0], unbatched, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_gate_that_is_a_matchgate_only_at_equal_parameters_is_decomposed(self) -> None:
        op = qml.prod(qml.IsingZZ(0.3, wires=[0, 1]), qml.adjoint(qml.IsingZZ(0.5, wires=[0, 1])))
        assert len(MatchgateTranslator().translate(op, wire_order=[0, 1, 2])) > 1

    def test_unitary_close_to_a_matchgate_is_decomposed(self) -> None:
        matrix = qml.matrix(qml.prod(qml.IsingXX(0.7, wires=[0, 1]), qml.IsingZZ(1e-6, wires=[0, 1])))
        assert len(MatchgateTranslator().translate(qml.QubitUnitary(matrix, wires=[0, 1]), wire_order=[0, 1])) > 1

    @pytest.mark.parametrize("coefficient", [0.5, -1.3])
    def test_evolution_of_a_hamiltonian_with_coefficients_is_kept_as_matchgate(self, coefficient: float) -> None:
        op = qml.evolve(coefficient * qml.X(0) @ qml.X(1), 0.3)
        translated = MatchgateTranslator().translate(op, wire_order=[0, 1, 2])

        assert len(translated) == 1
        assert isinstance(translated[0], MatchgateOperation)
        np.testing.assert_allclose(
            self._matrix(translated, [0, 1, 2]),
            qml.matrix(op, wire_order=[0, 1, 2]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_trotter_product_with_coefficients_is_translated(self) -> None:
        hamiltonian = qml.Hamiltonian([0.5, 0.3], [qml.X(0) @ qml.X(1), qml.Y(1) @ qml.Y(2)])
        op = qml.TrotterProduct(hamiltonian, 0.3, n=2)
        translated = MatchgateTranslator().translate(op, wire_order=[0, 1, 2])
        assert all(isinstance(new_op, MatchgateOperation) for new_op in translated)
        np.testing.assert_allclose(
            self._matrix(translated, [0, 1, 2]),
            qml.matrix(op, wire_order=[0, 1, 2]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_decomposition_returning_a_single_operation_is_translated(self) -> None:
        translator = MatchgateTranslator()
        op = qml.pow(qml.evolve(qml.X(0) @ qml.Z(1), 0.3), 2)
        translated = translator.translate(op, wire_order=[0, 1, 2])
        assert isinstance(op.decomposition(), Operator)
        assert all(MatchgateTranslator.is_native(new_op) for new_op in translated)
        np.testing.assert_allclose(
            self._matrix(translated, [0, 1, 2]),
            self._matrix(translator.translate(op.decomposition(), [0, 1, 2]), [0, 1, 2]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "bits, wires, expected_bits",
        [
            ([1, 1], [1, 2], [0, 1, 1, 0]),
            ([1, 0], [3, 0], [0, 0, 0, 1]),
            ([1, 0, 1, 1], [3, 2, 1, 0], [1, 1, 0, 1]),
        ],
    )
    def test_partial_basis_state_is_extended_to_all_wires(
        self, bits: List[int], wires: List[int], expected_bits: List[int]
    ) -> None:
        for state_cls in (qml.BasisState, qml.BasisEmbedding):
            translated = MatchgateTranslator().translate(state_cls(np.array(bits), wires=wires), wire_order=range(4))

            assert len(translated) == 1
            assert isinstance(translated[0], qml.BasisState)
            assert translated[0].wires == Wires(range(4))
            np.testing.assert_array_equal(qml.math.to_numpy(translated[0].parameters[0]), expected_bits)

    def test_state_prep_is_decomposed(self) -> None:
        op = NON_NATIVE_OPS["state_prep"]()
        translated = MatchgateTranslator().translate(op, wire_order=[0, 1, 2])
        assert len(translated) > 0
        assert all(isinstance(new_op, MatchgateOperation) for new_op in translated)

    def test_rule_may_return_a_single_operation(self) -> None:
        translator = MatchgateTranslator(rules={"SWAP": lambda op: fSWAP(wires=op.wires)})
        translated = translator.translate(qml.SWAP(wires=[1, 2]), wire_order=[0, 1, 2])
        assert len(translated) == 1
        assert isinstance(translated[0], fSWAP)

    @pytest.mark.parametrize("name", list(ADJOINT_CASES))
    def test_adjoint_is_translated_into_the_inverse(self, name: str) -> None:
        translator = MatchgateTranslator()
        op = ADJOINT_CASES[name]()
        translated = translator.translate(op, wire_order=[0, 1, 2])
        translated_adjoint = translator.translate(qml.adjoint(op), wire_order=[0, 1, 2])
        matrix = self._matrix(translated, [0, 1, 2])
        adjoint_matrix = self._matrix(translated_adjoint, [0, 1, 2])

        assert len(translated_adjoint) == len(translated)
        assert all(MatchgateTranslator.is_native(new_op) for new_op in translated_adjoint)
        np.testing.assert_allclose(
            adjoint_matrix,
            matrix.conj().T,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            adjoint_matrix @ matrix,
            np.eye(8),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "op_factory",
        [
            lambda: SptmCompRxRx(np.array([0.1, 0.2]), wires=[0, 1]),
            lambda: SptmFermionicSuperposition(wires=[1, 2]),
            lambda: SptmCompRxRx(np.array([[0.1, 0.2], [0.7, 1.3]]), wires=[0, 1]),
            lambda: FermionicSuperposition(wires=[1, 2]),
        ],
        ids=["sptm", "sptm_with_wrong_upstream_adjoint", "batched_sptm", "provides_its_sptm"],
    )
    def test_adjoint_of_sptm_operation_is_the_transposed_sptm(self, op_factory: Callable[[], Operator]) -> None:
        op = op_factory()
        translated = MatchgateTranslator().translate(qml.adjoint(op), wire_order=[0, 1, 2])
        sptm = op if isinstance(op, SingleParticleTransitionMatrixOperation) else op.to_sptm_operation()

        assert len(translated) == 1
        assert isinstance(translated[0], SingleParticleTransitionMatrixOperation)
        np.testing.assert_allclose(
            qml.math.to_numpy(translated[0].matrix()),
            np.swapaxes(qml.math.to_numpy(sptm.matrix()), -1, -2),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_translate_does_not_queue_operations(self) -> None:
        op = qml.IsingXX(0.3, wires=[0, 2])
        with qml.queuing.AnnotatedQueue() as queue:
            translated = MatchgateTranslator().translate(op, wire_order=[0, 1, 2])
        assert len(translated) == 3
        assert len(queue.queue) == 0

    @pytest.mark.parametrize("wire_order", [[0, 2], [1, 0], ["a", "b"], [0, 1, 3], [0.0, 1.0]])
    def test_invalid_wire_order_raises(self, wire_order: List) -> None:
        with pytest.raises(ValueError, match="consecutive increasing integers"):
            MatchgateTranslator().translate(qml.RZ(0.1, wires=wire_order[0]), wire_order=wire_order)

    @pytest.mark.parametrize("wire_order", [np.arange(3), Wires([0, 1, 2]), range(3), [5, 6, 7]])
    def test_valid_wire_order_is_accepted(self, wire_order: WiresLike) -> None:
        first_wire = list(wire_order)[0]
        translated = MatchgateTranslator().translate(qml.RZ(0.1, wires=first_wire), wire_order=wire_order)
        assert translated[0].wires == Wires([first_wire, first_wire + 1])

    @pytest.mark.parametrize(
        "op_factory",
        [lambda: qml.RZ(0.1, wires=3), lambda: qml.IsingXX(0.1, wires=[0, 5]), lambda: qml.RX(0.1, wires="a")],
        ids=["outside_range", "one_of_two_wires", "string_label"],
    )
    def test_wires_missing_from_wire_order_raise(self, op_factory: Callable[[], Operator]) -> None:
        with pytest.raises(ValueError, match="which are not in"):
            MatchgateTranslator().translate(op_factory(), wire_order=[0, 1, 2])

    @pytest.mark.parametrize("wire_order", [[0], [4]])
    def test_single_qubit_gate_without_neighbour_raises(self, wire_order: List[int]) -> None:
        with pytest.raises(ValueError, match="requires a neighbouring wire"):
            MatchgateTranslator().translate(qml.RZ(0.1, wires=wire_order[0]), wire_order=wire_order)

    @pytest.mark.parametrize(
        "op_factory, n_ops",
        [(lambda: qml.BasisState(np.array([1]), wires=[0]), 1), (lambda: qml.Identity(wires=0), 0)],
        ids=["basis_state", "identity"],
    )
    def test_single_wire_order_accepts_ops_without_dressing(
        self,
        op_factory: Callable[[], Operator],
        n_ops: int,
    ) -> None:
        assert len(MatchgateTranslator().translate(op_factory(), wire_order=[0])) == n_ops

    def test_op_without_matrix_or_decomposition_raises(self) -> None:
        with pytest.raises(ValueError, match="no matrix, no decomposition and no translation rule"):
            MatchgateTranslator().translate(qml.ops.MidMeasure(wires=0), wire_order=[0, 1])

    @pytest.mark.parametrize("max_depth", [0, 1, 5, MatchgateTranslator.DEFAULT_MAX_DEPTH])
    def test_divergent_rule_raises(self, max_depth: int) -> None:
        translator = MatchgateTranslator(rules={"PauliX": lambda op: [qml.PauliX(op.wires)]}, max_depth=max_depth)
        with pytest.raises(ValueError, match=f"did not end after {max_depth} nested steps"):
            translator.translate(qml.PauliX(wires=0), wire_order=[0, 1])

    @pytest.mark.parametrize("max_depth", [0, 1, 2])
    def test_divergent_adjoint_rule_raises(self, max_depth: int) -> None:
        translator = MatchgateTranslator(rules={"PauliX": lambda op: [qml.adjoint(op)]}, max_depth=max_depth)
        with pytest.raises(ValueError, match=f"did not end after {max_depth} nested steps"):
            translator.translate(qml.PauliX(wires=0), wire_order=[0, 1])

    @pytest.mark.parametrize(
        "op_factory",
        [lambda: qml.RZ(0.1, wires=1), lambda: qml.IsingXX(0.1, wires=[2, 0]), lambda: fSWAP(wires=[0, 1])],
        ids=["dressed", "routed", "native"],
    )
    def test_zero_max_depth_allows_direct_translations(self, op_factory: Callable[[], Operator]) -> None:
        translated = MatchgateTranslator(max_depth=0).translate(op_factory(), wire_order=[0, 1, 2])
        assert all(MatchgateTranslator.is_native(op) for op in translated)

    def test_zero_max_depth_rejects_decompositions(self) -> None:
        with pytest.raises(ValueError, match="did not end after 0 nested steps"):
            MatchgateTranslator(max_depth=0).translate(qml.Toffoli(wires=[0, 1, 2]), wire_order=[0, 1, 2])

    def test_translate_tape_translates_operations_and_keeps_measurements(self) -> None:
        translator = MatchgateTranslator()
        operations = [qml.RZ(0.3, wires=0), qml.IsingXX(0.5, wires=[0, 2]), qml.CNOT(wires=[2, 1])]
        measurements = [qml.probs(wires=[0, 1, 2]), qml.expval(qml.PauliZ(1))]
        tape = QuantumScript(operations, measurements, shots=25)
        translated_tape = translator.translate_tape(tape)
        expected = [new_op for op in operations for new_op in translator.translate(op, [0, 1, 2])]

        assert translated_tape is not tape
        assert tape.operations == operations
        assert translated_tape.shots == tape.shots
        assert all(qml.equal(new, old) for new, old in zip(translated_tape.measurements, measurements))
        assert len(translated_tape.measurements) == len(measurements)
        assert all(isinstance(op, MatchgateOperation) for op in translated_tape.operations)
        np.testing.assert_allclose(
            self._matrix(translated_tape.operations, [0, 1, 2]),
            self._matrix(expected, [0, 1, 2]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "wire_order, expected_wires",
        [
            (None, [[2, 3], [1, 2]]),
            (range(5), [[3, 4], [1, 2]]),
            ([1, 2, 3, 4], [[3, 4], [1, 2]]),
        ],
    )
    def test_translate_tape_wire_order(
        self,
        wire_order: Optional[WiresLike],
        expected_wires: List[List[int]],
    ) -> None:
        tape = QuantumScript([qml.RZ(0.3, wires=3), qml.RZ(0.5, wires=1)], [qml.probs(wires=[1, 3])])
        translated_tape = MatchgateTranslator().translate_tape(tape, wire_order=wire_order)
        assert [op.wires for op in translated_tape.operations] == [Wires(wires) for wires in expected_wires]

    @pytest.mark.parametrize(
        "measurements",
        [[], [qml.state()], [qml.probs(wires=[0, 1])], [qml.probs(wires=["a", "b"])]],
        ids=["no_measurement", "state", "integer_wires", "string_wires"],
    )
    def test_translate_tape_without_operations(self, measurements: List) -> None:
        tape = QuantumScript([], measurements)
        translated_tape = MatchgateTranslator().translate_tape(tape)
        assert translated_tape.operations == []
        assert len(translated_tape.measurements) == len(measurements)

    def test_translate_tape_with_string_wires_raises(self) -> None:
        tape = QuantumScript([qml.RZ(0.1, wires="a"), qml.IsingXX(0.2, wires=["a", "b"])], [qml.probs()])
        with pytest.raises(ValueError, match="consecutive increasing integers"):
            MatchgateTranslator().translate_tape(tape)

    @pytest.mark.parametrize("name", list(BROADCAST_GATES))
    def test_broadcast_translation_matches_each_element(self, name: str) -> None:
        translator = MatchgateTranslator()
        angles = np.array([0.1, 0.8, 2.3])
        translated = translator.translate(BROADCAST_GATES[name](angles), wire_order=[0, 1, 2])
        broadcast_matrix = self._matrix(translated, [0, 1, 2])  # (3, 8, 8)

        assert broadcast_matrix.shape == (len(angles), 8, 8)
        for index, angle in enumerate(angles):
            single = translator.translate(BROADCAST_GATES[name](angle), wire_order=[0, 1, 2])
            np.testing.assert_allclose(
                broadcast_matrix[index],
                self._matrix(single, [0, 1, 2]),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=RTOL_MATRIX_COMPARISON,
            )

    @pytest.mark.parametrize("name", list(BROADCAST_GATES))
    def test_translation_is_differentiable_with_torch(self, name: str) -> None:
        translator = MatchgateTranslator()

        def translated_matrix(angle: torch.Tensor) -> torch.Tensor:
            translated = translator.translate(BROADCAST_GATES[name](angle), wire_order=[0, 1, 2])
            return torch.view_as_real(qml.matrix(QuantumScript(translated), wire_order=[0, 1, 2]))

        angle = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            translated_matrix,
            (angle,),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
