from typing import Callable, Dict, List, Sequence

import numpy as np
import pennylane as qml
import pytest
from pennylane.operation import Operator
from pennylane.tape import QuantumScript
from pennylane.wires import Wires, WiresLike

from majorana_karaoke import MatchgateTranslator, translate_to_matchgates

from .configs import (
    ATOL_MATRIX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
    set_seed,
)

EXACT_CIRCUITS: Dict[str, Callable[[], List[Operator]]] = {
    "ising_and_diagonal_gates": lambda: [
        qml.IsingXX(0.7, wires=[0, 1]),
        qml.RZ(0.3, wires=2),
        qml.IsingXY(0.4, wires=[2, 1]),
        qml.T(wires=0),
        qml.IsingYY(1.1, wires=[1, 2]),
        qml.PhaseShift(0.9, wires=2),
    ],
    "basis_state_and_excitations": lambda: [
        qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2]),
        qml.SingleExcitation(0.8, wires=[1, 0]),
        qml.S(wires=1),
        qml.FermionicSWAP(0.5, wires=[1, 2]),
    ],
}

CIRCUITS: Dict[str, Callable[[], List[Operator]]] = {
    **EXACT_CIRCUITS,
    "hadamard_and_cnot": lambda: [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 2]), qml.RX(0.4, wires=1)],
    "toffoli": lambda: [qml.PauliX(wires=0), qml.Hadamard(wires=1), qml.Toffoli(wires=[0, 1, 2])],
    "routed_matchgate": lambda: [
        qml.BasisState(np.array([0, 1, 0]), wires=[0, 1, 2]),
        qml.Hadamard(wires=0),
        qml.IsingXX(0.9, wires=[2, 0]),
    ],
}


class TestTranslateToMatchgates:
    @staticmethod
    def _matrix(ops: Sequence[Operator], wire_order: WiresLike) -> np.ndarray:
        return qml.math.to_numpy(qml.matrix(QuantumScript(list(ops)), wire_order=wire_order))

    @staticmethod
    def _probs_qnode(device: qml.devices.Device, ops_factory: Callable[[], List[Operator]]) -> qml.QNode:
        # The operations are built outside of the QNode so that they are queued exactly once.
        ops = ops_factory()

        @qml.qnode(device)
        def circuit():
            for op in ops:
                qml.apply(op)
            return qml.probs(wires=[0, 1, 2])

        return circuit

    @classmethod
    def setup_class(cls) -> None:
        set_seed(TEST_SEED)

    def test_returns_one_tape_and_a_postprocessing_returning_its_result(self) -> None:
        tape = QuantumScript([qml.RZ(0.3, wires=0), qml.IsingXX(0.2, wires=[0, 1])], [qml.probs()])
        tapes, postprocessing = translate_to_matchgates(tape)
        assert len(tapes) == 1
        assert tapes[0] is not tape
        assert postprocessing(("result",)) == "result"

    @pytest.mark.parametrize("name", list(CIRCUITS))
    def test_default_translation_matches_translate_tape(self, name: str) -> None:
        tape = QuantumScript(CIRCUITS[name](), [qml.probs()])
        (translated_tape,), _ = translate_to_matchgates(tape)
        expected_tape = MatchgateTranslator().translate_tape(tape)

        assert all(MatchgateTranslator.is_native(op) for op in translated_tape.operations)
        assert [op.name for op in translated_tape.operations] == [op.name for op in expected_tape.operations]
        assert [op.wires for op in translated_tape.operations] == [op.wires for op in expected_tape.operations]
        np.testing.assert_allclose(
            self._matrix([op for op in translated_tape.operations if op.has_matrix], [0, 1, 2]),
            self._matrix([op for op in expected_tape.operations if op.has_matrix], [0, 1, 2]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_measurements_and_shots_are_kept(self) -> None:
        measurements = [qml.probs(wires=[0, 1]), qml.expval(qml.PauliZ(1)), qml.var(qml.PauliZ(0))]
        tape = QuantumScript([qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])], measurements, shots=30)
        (translated_tape,), _ = translate_to_matchgates(tape)

        assert translated_tape.shots == tape.shots
        assert len(translated_tape.measurements) == len(measurements)
        assert all(qml.equal(new, old) for new, old in zip(translated_tape.measurements, measurements))

    def test_translator_argument_is_used(self) -> None:
        translator = MatchgateTranslator(rules={"Hadamard": MatchgateTranslator.drop})
        tape = QuantumScript([qml.Hadamard(wires=0), qml.RZ(0.3, wires=1)], [qml.probs()])
        (translated_tape,), _ = translate_to_matchgates(tape, translator=translator)
        (default_tape,), _ = translate_to_matchgates(tape, translator=None)

        assert len(translated_tape.operations) == 1
        assert translated_tape.operations[0].wires == Wires([0, 1])
        assert len(default_tape.operations) == 2

    @pytest.mark.parametrize(
        "wire_order, expected_wires",
        [([0, 1], [0, 1]), ([0, 1, 2], [1, 2]), ([1, 2], [1, 2]), (range(4), [1, 2])],
    )
    def test_wire_order_argument_is_used(self, wire_order: WiresLike, expected_wires: List[int]) -> None:
        tape = QuantumScript([qml.RZ(0.3, wires=1)], [qml.probs(wires=[1])])
        (translated_tape,), _ = translate_to_matchgates(tape, wire_order=wire_order)
        assert [op.wires for op in translated_tape.operations] == [Wires(expected_wires)]

    def test_single_wire_tape_without_wire_order_raises(self) -> None:
        tape = QuantumScript([qml.RZ(0.3, wires=1)], [qml.probs(wires=[1])])
        with pytest.raises(ValueError, match="requires a neighbouring wire"):
            translate_to_matchgates(tape)

    def test_broadcast_tape_keeps_its_batch_size(self) -> None:
        angles = np.array([0.1, 0.5, 1.7])
        tape = QuantumScript([qml.RZ(angles, wires=2), qml.IsingXX(angles, wires=[0, 2])], [qml.probs()])
        (translated_tape,), _ = translate_to_matchgates(tape)
        assert translated_tape.batch_size == len(angles)

    @pytest.mark.parametrize("name", list(EXACT_CIRCUITS))
    def test_transformed_qnode_on_default_qubit_is_exact_on_free_fermion_circuits(self, name: str) -> None:
        device = qml.device("default.qubit", wires=3)
        circuit = self._probs_qnode(device, EXACT_CIRCUITS[name])
        transformed_circuit = translate_to_matchgates(circuit, wire_order=[0, 1, 2])
        np.testing.assert_allclose(
            qml.math.to_numpy(transformed_circuit()),
            qml.math.to_numpy(circuit()),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("name", list(CIRCUITS))
    def test_transformed_qnode_on_default_qubit_matches_mk_qubit(self, name: str) -> None:
        transformed_circuit = translate_to_matchgates(
            self._probs_qnode(qml.device("default.qubit", wires=3), CIRCUITS[name]),
            wire_order=[0, 1, 2],
        )
        mk_circuit = self._probs_qnode(qml.device("mk.qubit", wires=3), CIRCUITS[name])
        np.testing.assert_allclose(
            qml.math.to_numpy(mk_circuit()),
            qml.math.to_numpy(transformed_circuit()),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
