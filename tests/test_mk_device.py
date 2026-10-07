from importlib import metadata
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pennylane as qml
import pytest
import torch
from matchcake import NonInteractingFermionicDevice
from matchcake.operations import (
    CompHH,
    CompRxRx,
    FermionicSuperposition,
    SingleParticleTransitionMatrixOperation,
    SptmAngleEmbedding,
    SptmCompHH,
    SptmCompRxRx,
    SptmFermionicSuperposition,
)
from pennylane.devices import ExecutionConfig
from pennylane.operation import Operator
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

from majorana_karaoke import MajoranaKaraokeDevice, MatchgateTranslator, translate_to_matchgates

from .configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
    set_seed,
)

ENTRY_POINT_GROUP = "pennylane.plugins"
N_WIRES = 4
FSWAP_MATRIX = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]], dtype=complex)
STRONGLY_ENTANGLING_WEIGHTS = np.random.default_rng(TEST_SEED).uniform(0, 2 * np.pi, size=(2, N_WIRES, 3))

EXACT_CIRCUITS: Dict[str, Callable[[], List[Operator]]] = {
    "partial_basis_state": lambda: [
        qml.BasisState(np.array([1, 1]), wires=[2, 1]),
        qml.IsingXY(0.5, wires=[0, 1]),
        qml.RZ(0.4, wires=3),
        qml.SingleExcitation(0.8, wires=[2, 3]),
    ],
    "ising_and_rz": lambda: [
        qml.IsingXX(0.7, wires=[0, 1]),
        qml.RZ(0.3, wires=1),
        qml.IsingYY(1.1, wires=[1, 2]),
        qml.IsingXX(0.5, wires=[2, 3]),
        qml.RZ(0.8, wires=3),
        qml.IsingXY(0.6, wires=[1, 2]),
    ],
    "reversed_ising_xy_and_phase_shift": lambda: [
        qml.BasisState(np.array([1, 0, 0, 1]), wires=range(N_WIRES)),
        qml.IsingXY(0.9, wires=[1, 0]),
        qml.PhaseShift(0.4, wires=0),
        qml.IsingXY(1.3, wires=[3, 2]),
        qml.PhaseShift(1.1, wires=3),
        qml.IsingXY(0.6, wires=[2, 1]),
        qml.IsingXX(0.4, wires=[1, 0]),
    ],
    "clifford_diagonal_gates": lambda: [
        qml.IsingXX(np.pi / 3, wires=[0, 1]),
        qml.S(wires=0),
        qml.IsingYY(0.8, wires=[2, 1]),
        qml.T(wires=1),
        qml.PauliZ(wires=2),
        qml.IsingXX(1.0, wires=[2, 3]),
        qml.S(wires=3),
        qml.IsingXY(0.5, wires=[1, 2]),
        qml.T(wires=3),
    ],
    "excitations": lambda: [
        qml.BasisState(np.array([1, 1, 0, 0]), wires=range(N_WIRES)),
        qml.SingleExcitation(0.7, wires=[1, 2]),
        qml.FermionicSWAP(0.4, wires=[3, 2]),
        qml.SingleExcitation(1.2, wires=[0, 1]),
        qml.RZ(0.5, wires=2),
        qml.T(wires=3),
        qml.SingleExcitation(0.3, wires=[3, 2]),
    ],
}

GENERIC_CIRCUITS: Dict[str, Callable[[], List[Operator]]] = {
    "hadamard_and_cnot": lambda: [
        qml.Hadamard(wires=0),
        qml.CNOT(wires=[0, 1]),
        qml.RX(0.3, wires=2),
        qml.CNOT(wires=[2, 3]),
        qml.CNOT(wires=[3, 0]),
    ],
    "toffoli": lambda: [
        qml.Hadamard(wires=0),
        qml.Hadamard(wires=1),
        qml.Toffoli(wires=[0, 1, 3]),
        qml.RY(0.4, wires=2),
    ],
    "swap_and_cz": lambda: [
        qml.RX(0.3, wires=0),
        qml.RY(1.1, wires=3),
        qml.SWAP(wires=[0, 2]),
        qml.CZ(wires=[1, 3]),
        qml.Hadamard(wires=3),
    ],
    "qft": lambda: [qml.BasisState(np.array([1, 0, 1, 1]), wires=range(N_WIRES)), qml.QFT(wires=range(N_WIRES))],
    "strongly_entangling": lambda: [qml.StronglyEntanglingLayers(STRONGLY_ENTANGLING_WEIGHTS, wires=range(N_WIRES))],
    "angle_and_iqp_embedding": lambda: [
        qml.AngleEmbedding(np.array([0.1, 0.7, 1.3, 2.1]), wires=range(N_WIRES), rotation="Y"),
        qml.IQPEmbedding(np.array([0.4, 1.2, 0.8, 0.2]), wires=range(N_WIRES)),
    ],
    "basic_entangler": lambda: [
        qml.BasicEntanglerLayers(STRONGLY_ENTANGLING_WEIGHTS[:, :, 0], wires=range(N_WIRES)),
    ],
    "routed_non_adjacent_matchgates": lambda: [
        qml.Hadamard(wires=0),
        qml.IsingXX(0.7, wires=[0, 3]),
        qml.IsingYY(0.4, wires=[3, 1]),
        qml.PauliX(wires=2),
    ],
    "adjoint_template": lambda: [
        qml.adjoint(qml.StronglyEntanglingLayers(STRONGLY_ENTANGLING_WEIGHTS, wires=range(N_WIRES))),
    ],
}

IMAGINARY_OFF_STRUCTURE_CIRCUITS: Dict[str, Callable[[], List[Operator]]] = {
    "crx": lambda: [qml.BasisState(np.array([1, 0, 0, 0]), wires=range(N_WIRES)), qml.CRX(1.2, wires=[0, 1])],
    "cy": lambda: [qml.BasisState(np.array([1, 0, 0, 0]), wires=range(N_WIRES)), qml.CY(wires=[0, 1])],
    "pauli_rot_xz": lambda: [qml.Hadamard(wires=1), qml.PauliRot(0.9, "XZ", wires=[1, 2])],
}

KERNEL_ANSATZE: Dict[str, Callable[[TensorLike], List[Operator]]] = {
    "angle_and_strongly_entangling": lambda x: [
        qml.AngleEmbedding(x, wires=range(N_WIRES)),
        qml.StronglyEntanglingLayers(STRONGLY_ENTANGLING_WEIGHTS, wires=range(N_WIRES)),
    ],
    "iqp": lambda x: [qml.IQPEmbedding(x, wires=range(N_WIRES), n_repeats=2)],
    "angle_and_cnot_ladder": lambda x: [
        qml.AngleEmbedding(x, wires=range(N_WIRES), rotation="X"),
        *[qml.CNOT(wires=[wire, wire + 1]) for wire in range(N_WIRES - 1)],
    ],
}


# Operations that nif.qubit applies through their single-particle transition matrix, with the qubit unitary they encode.
SPTM_OPERATIONS: Dict[str, Callable[[], tuple]] = {
    "sptm_comp_rxrx": lambda: (
        SptmCompRxRx(np.array([0.1, 0.4]), wires=[1, 2]),
        qml.matrix(CompRxRx(np.array([0.1, 0.4]), wires=[1, 2])),
    ),
    "sptm_comp_hh": lambda: (SptmCompHH(wires=[0, 1]), qml.matrix(CompHH(wires=[0, 1]))),
    "sptm_fermionic_superposition": lambda: (
        SptmFermionicSuperposition(wires=[2, 3]),
        qml.matrix(FermionicSuperposition(wires=[2, 3])),
    ),
    "fermionic_superposition": lambda: (
        FermionicSuperposition(wires=[1, 2]),
        qml.matrix(FermionicSuperposition(wires=[1, 2])),
    ),
}


class TestMajoranaKaraokeDevice:
    @staticmethod
    def _entry_points() -> Dict[str, str]:
        return {entry.name: entry.value for entry in metadata.entry_points(group=ENTRY_POINT_GROUP)}

    @staticmethod
    def _sptm_preparation() -> List[Operator]:
        return [
            qml.BasisState(np.array([1, 0, 1, 0]), wires=range(N_WIRES)),
            qml.IsingXY(0.4, wires=[0, 1]),
            qml.IsingXY(0.9, wires=[2, 3]),
        ]

    @staticmethod
    def _sptm_mixing() -> List[Operator]:
        # Matchgates applied after the operation under test, so that U and U^dagger give different probabilities.
        return [qml.IsingXX(0.7, wires=[0, 1]), qml.IsingXY(1.1, wires=[1, 2]), qml.IsingXY(0.5, wires=[2, 3])]

    @staticmethod
    def _as_unitaries(ops: Sequence[Operator]) -> List[Operator]:
        return [qml.QubitUnitary(op.matrix(), wires=op.wires) if op.has_matrix else op for op in ops]

    @staticmethod
    def _translate(device: MajoranaKaraokeDevice, ops: Sequence[Operator]) -> List[Operator]:
        return [new_op for op in ops for new_op in device.translator.translate(op, wire_order=device.wires)]

    @staticmethod
    def _probs(device: qml.devices.Device, ops: Sequence[Operator]) -> np.ndarray:
        @qml.qnode(device)
        def circuit():
            for op in ops:
                qml.apply(op)
            return qml.probs(wires=device.wires)

        return qml.math.to_numpy(circuit())

    @staticmethod
    def _z_expvals(device: qml.devices.Device, ops: Sequence[Operator]) -> np.ndarray:
        @qml.qnode(device)
        def circuit():
            for op in ops:
                qml.apply(op)
            return [qml.expval(qml.PauliZ(wire)) for wire in device.wires]

        return np.asarray([qml.math.to_numpy(value) for value in circuit()])

    @staticmethod
    def _kernel_ops(name: str, x: np.ndarray, x_prime: np.ndarray) -> List[Operator]:
        return [
            *KERNEL_ANSATZE[name](x),
            *[qml.adjoint(op) for op in reversed(KERNEL_ANSATZE[name](x_prime))],
        ]

    @staticmethod
    def _exact_ansatz(params: TensorLike) -> None:
        qml.BasisState(np.array([1, 0, 0, 1]), wires=range(N_WIRES))
        qml.IsingXX(params[0], wires=[0, 1])
        qml.RZ(params[1], wires=0)
        qml.IsingXY(params[2], wires=[1, 0])
        qml.IsingXX(params[0], wires=[2, 3])
        qml.PhaseShift(params[3], wires=3)
        qml.IsingYY(params[2], wires=[3, 2])
        qml.SingleExcitation(params[1], wires=[1, 2])

    @staticmethod
    def _generic_ansatz(params: TensorLike) -> None:
        qml.Hadamard(wires=0)
        qml.RX(params[0], wires=1)
        qml.CNOT(wires=[0, 2])
        qml.IsingXX(params[1], wires=[3, 0])
        qml.RY(params[2], wires=0)
        qml.CRY(params[1], wires=[1, 2])
        qml.CZ(wires=[1, 2])
        qml.RZ(params[0], wires=3)

    @classmethod
    def setup_class(cls) -> None:
        set_seed(TEST_SEED)

    def test_entry_point_is_declared(self) -> None:
        entry_points = self._entry_points()
        module_path, _, attribute = entry_points["mk.qubit"].partition(":")
        assert attribute == MajoranaKaraokeDevice.__name__
        assert module_path == MajoranaKaraokeDevice.__module__

    def test_device_resolves_by_name(self) -> None:
        device = qml.device("mk.qubit", wires=3)
        assert isinstance(device, MajoranaKaraokeDevice)
        assert isinstance(device, NonInteractingFermionicDevice)
        assert device.name == "mk.qubit"
        assert device.wires == Wires([0, 1, 2])

    def test_default_constructor(self) -> None:
        device = MajoranaKaraokeDevice()
        assert device.wires == Wires([0, 1])
        assert isinstance(device.translator, MatchgateTranslator)
        assert device.translator.rules.keys() == MatchgateTranslator().rules.keys()

    def test_each_device_owns_its_default_translator(self) -> None:
        assert MajoranaKaraokeDevice(wires=2).translator is not MajoranaKaraokeDevice(wires=2).translator

    @pytest.mark.parametrize(
        "wires, expected_wires",
        [(3, [0, 1, 2]), ([0, 1, 2], [0, 1, 2]), (Wires([1, 2, 3]), [1, 2, 3]), (range(4), [0, 1, 2, 3])],
    )
    def test_wires_argument(self, wires: WiresLike, expected_wires: List[int]) -> None:
        assert MajoranaKaraokeDevice(wires=wires).wires == Wires(expected_wires)

    def test_custom_translator_is_used(self) -> None:
        translator = MatchgateTranslator(rules={"Hadamard": MatchgateTranslator.drop})
        device = qml.device("mk.qubit", wires=2, translator=translator)
        probs = self._probs(device, [qml.Hadamard(wires=0), qml.RZ(0.4, wires=1)])

        assert device.translator is translator
        np.testing.assert_allclose(
            probs,
            [1.0, 0.0, 0.0, 0.0],
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_shots_are_forwarded(self) -> None:
        n_shots = 40
        device = qml.device("mk.qubit", wires=3, shots=n_shots)
        ops = [
            qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2]),
            qml.RZ(0.3, wires=0),
            qml.S(wires=2),
            qml.IsingXX(np.pi, wires=[0, 1]),
        ]

        @qml.qnode(device)
        def circuit():
            for op in ops:
                qml.apply(op)
            return qml.sample()

        samples = np.asarray(circuit())
        assert samples.shape == (n_shots, 3)
        np.testing.assert_array_equal(samples, np.tile([0, 1, 1], (n_shots, 1)))

    @pytest.mark.parametrize("execution_config", [None, ExecutionConfig()])
    def test_preprocess_transforms_start_with_the_translation(
        self,
        execution_config: Optional[ExecutionConfig],
    ) -> None:
        device = qml.device("mk.qubit", wires=3)
        program = device.preprocess_transforms(execution_config)
        parent_program = NonInteractingFermionicDevice.preprocess_transforms(device, execution_config)

        assert program[0] == translate_to_matchgates(translator=device.translator, wire_order=device.wires)
        assert program[0].kwargs["translator"] is device.translator
        assert len(program) == len(parent_program) + 1

    @pytest.mark.parametrize("name", list(EXACT_CIRCUITS))
    def test_free_fermion_circuit_matches_default_qubit_probabilities(self, name: str) -> None:
        ops = EXACT_CIRCUITS[name]()
        np.testing.assert_allclose(
            self._probs(qml.device("mk.qubit", wires=N_WIRES), ops),
            self._probs(qml.device("default.qubit", wires=N_WIRES), ops),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("name", list(EXACT_CIRCUITS))
    def test_free_fermion_circuit_matches_default_qubit_expectation_values(self, name: str) -> None:
        ops = EXACT_CIRCUITS[name]()
        np.testing.assert_allclose(
            self._z_expvals(qml.device("mk.qubit", wires=N_WIRES), ops),
            self._z_expvals(qml.device("default.qubit", wires=N_WIRES), ops),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("wires", [[1, 2, 3], [5, 6, 7, 8]])
    def test_free_fermion_circuit_on_wires_not_starting_at_zero(self, wires: List[int]) -> None:
        ops = [
            qml.BasisState(np.array([1] + [0] * (len(wires) - 1)), wires=wires),
            qml.IsingXY(0.9, wires=[wires[1], wires[0]]),
            qml.RZ(0.7, wires=wires[-1]),
            qml.IsingXX(0.4, wires=wires[-2:]),
            qml.PhaseShift(0.2, wires=wires[0]),
        ]
        np.testing.assert_allclose(
            self._probs(qml.device("mk.qubit", wires=wires), ops),
            self._probs(qml.device("default.qubit", wires=wires), ops),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("wires", [[0, 2], [2, 0], [0, 3], [3, 1]])
    @pytest.mark.parametrize("name", ["IsingXX", "IsingYY", "IsingXY"])
    def test_routed_matchgate_matches_fswap_circuit(self, name: str, wires: List[int]) -> None:
        preparation = [
            qml.BasisState(np.array([0, 1, 1, 0]), wires=range(N_WIRES)),
            qml.IsingXX(0.5, wires=[1, 2]),
            qml.IsingYY(0.8, wires=[2, 3]),
        ]
        op = getattr(qml, name)(0.9, wires=wires)
        first, second = sorted(wires)
        route = [qml.QubitUnitary(FSWAP_MATRIX, wires=[index - 1, index]) for index in range(second, first + 1, -1)]
        matchgate = qml.QubitUnitary(qml.matrix(op, wire_order=[first, second]), wires=[first, first + 1])

        np.testing.assert_allclose(
            self._probs(qml.device("mk.qubit", wires=N_WIRES), [*preparation, op]),
            self._probs(qml.device("default.qubit", wires=N_WIRES), [*preparation, *route, matchgate, *route[::-1]]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "name",
        [
            *GENERIC_CIRCUITS,
            *EXACT_CIRCUITS,
            *IMAGINARY_OFF_STRUCTURE_CIRCUITS,
        ],
    )
    def test_matches_default_qubit_on_translated_operations(self, name: str) -> None:
        ops = {**GENERIC_CIRCUITS, **EXACT_CIRCUITS, **IMAGINARY_OFF_STRUCTURE_CIRCUITS}[name]()
        device = qml.device("mk.qubit", wires=N_WIRES)
        translated = self._as_unitaries(self._translate(device, ops))

        np.testing.assert_allclose(
            self._probs(device, ops),
            self._probs(qml.device("default.qubit", wires=N_WIRES), translated),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
        np.testing.assert_allclose(
            self._z_expvals(device, ops),
            self._z_expvals(qml.device("default.qubit", wires=N_WIRES), translated),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("name", list(SPTM_OPERATIONS))
    def test_sptm_operation_matches_its_qubit_unitary(self, name: str) -> None:
        op, unitary = SPTM_OPERATIONS[name]()
        np.testing.assert_allclose(
            self._probs(qml.device("mk.qubit", wires=N_WIRES), [*self._sptm_preparation(), op, *self._sptm_mixing()]),
            self._probs(
                qml.device("default.qubit", wires=N_WIRES),
                [
                    *self._sptm_preparation(),
                    qml.QubitUnitary(qml.math.to_numpy(unitary), wires=op.wires),
                    *self._sptm_mixing(),
                ],
            ),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("name", list(SPTM_OPERATIONS))
    def test_adjoint_of_sptm_operation_matches_the_inverse_unitary(self, name: str) -> None:
        op, unitary = SPTM_OPERATIONS[name]()
        inverse = qml.math.to_numpy(unitary).conj().T
        np.testing.assert_allclose(
            self._probs(
                qml.device("mk.qubit", wires=N_WIRES),
                [*self._sptm_preparation(), qml.adjoint(op), *self._sptm_mixing()],
            ),
            self._probs(
                qml.device("default.qubit", wires=N_WIRES),
                [*self._sptm_preparation(), qml.QubitUnitary(inverse, wires=op.wires), *self._sptm_mixing()],
            ),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("name", list(SPTM_OPERATIONS))
    def test_adjoint_of_sptm_operation_undoes_it(self, name: str) -> None:
        op, _ = SPTM_OPERATIONS[name]()
        device = qml.device("mk.qubit", wires=N_WIRES)
        np.testing.assert_allclose(
            self._probs(device, [*self._sptm_preparation(), op, qml.adjoint(op)]),
            self._probs(device, self._sptm_preparation()),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_sptm_circuit_matches_nif_qubit(self) -> None:
        ops = [
            *self._sptm_preparation(),
            SptmAngleEmbedding(np.array([0.3, 0.7, 1.1, 0.2]), wires=range(N_WIRES)),
            FermionicSuperposition(wires=[1, 2]),
            SingleParticleTransitionMatrixOperation.random(wires=[0, 1, 2], seed=TEST_SEED),
            SptmCompHH(wires=[2, 3]),
        ]
        np.testing.assert_allclose(
            self._probs(qml.device("mk.qubit", wires=N_WIRES), ops),
            self._probs(qml.device("nif.qubit", wires=N_WIRES), ops),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("seed", range(2))
    @pytest.mark.parametrize("name", list(KERNEL_ANSATZE))
    def test_kernel_of_identical_inputs_is_one(self, name: str, seed: int) -> None:
        x = np.random.default_rng(seed).uniform(0, np.pi, size=N_WIRES)
        probs = self._probs(qml.device("mk.qubit", wires=N_WIRES), self._kernel_ops(name, x, x))
        np.testing.assert_allclose(probs[0], 1.0, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("name", list(KERNEL_ANSATZE))
    def test_kernel_of_different_inputs_matches_translated_operations(self, name: str) -> None:
        rng = np.random.default_rng(TEST_SEED)
        x, x_prime = rng.uniform(0, np.pi, size=(2, N_WIRES))
        ops = self._kernel_ops(name, x, x_prime)
        device = qml.device("mk.qubit", wires=N_WIRES)
        np.testing.assert_allclose(
            self._probs(device, ops),
            self._probs(qml.device("default.qubit", wires=N_WIRES), self._as_unitaries(self._translate(device, ops))),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_broadcast_free_fermion_circuit_matches_default_qubit(self) -> None:
        angles = np.array([0.1, 0.9, 2.3])
        ops = [
            qml.IsingXX(angles, wires=[0, 1]),
            qml.RZ(angles, wires=3),
            qml.IsingXY(0.4, wires=[2, 1]),
            qml.IsingYY(angles, wires=[2, 3]),
            qml.PhaseShift(angles, wires=0),
        ]
        probs = self._probs(qml.device("mk.qubit", wires=N_WIRES), ops)

        assert probs.shape == (len(angles), 2**N_WIRES)
        np.testing.assert_allclose(
            probs,
            self._probs(qml.device("default.qubit", wires=N_WIRES), ops),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_broadcast_matches_individual_executions(self) -> None:
        angles = np.array([0.1, 0.9, 2.3])
        device = qml.device("mk.qubit", wires=N_WIRES)

        def ops(angle: TensorLike) -> List[Operator]:
            return [
                qml.Hadamard(wires=0),
                qml.RX(angle, wires=1),
                qml.CNOT(wires=[1, 2]),
                qml.IsingXX(angle, wires=[0, 3]),
                qml.RY(angle, wires=3),
            ]

        np.testing.assert_allclose(
            self._probs(device, ops(angles)),
            np.stack([self._probs(device, ops(angle)) for angle in angles]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_torch_backprop_matches_default_qubit_on_free_fermion_circuit(self) -> None:
        def circuit(params: torch.Tensor) -> qml.measurements.ProbabilityMP:
            self._exact_ansatz(params)
            return qml.probs(wires=range(N_WIRES))

        params = torch.tensor([0.3, 0.8, 1.4, 0.6], dtype=torch.float64, requires_grad=True)
        mk_circuit = qml.QNode(
            circuit, qml.device("mk.qubit", wires=N_WIRES), interface="torch", diff_method="backprop"
        )
        reference = qml.QNode(circuit, qml.device("default.qubit", wires=N_WIRES), interface="torch")
        mk_jacobian = torch.autograd.functional.jacobian(mk_circuit, params)  # (2**N_WIRES, n_params)
        reference_jacobian = torch.autograd.functional.jacobian(reference, params)  # (2**N_WIRES, n_params)

        assert torch.all(reference_jacobian.abs().sum(dim=0) > ATOL_MATRIX_COMPARISON)
        torch.testing.assert_close(
            mk_jacobian, reference_jacobian, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_torch_backprop_matches_translated_operations(self) -> None:
        device = qml.device("mk.qubit", wires=N_WIRES)

        def circuit(params: torch.Tensor) -> qml.measurements.ProbabilityMP:
            self._generic_ansatz(params)
            return qml.probs(wires=device.wires)

        def reference_circuit(params: torch.Tensor) -> qml.measurements.ProbabilityMP:
            ops = qml.tape.make_qscript(self._generic_ansatz)(params).operations
            # The unitaries are built without recording so that qml.apply queues each of them exactly once.
            with qml.QueuingManager.stop_recording():
                translated = self._as_unitaries(self._translate(device, ops))
            for op in translated:
                qml.apply(op)
            return qml.probs(wires=device.wires)

        params = torch.tensor([0.3, 0.8, 1.4], dtype=torch.float64, requires_grad=True)
        mk_circuit = qml.QNode(circuit, device, interface="torch", diff_method="backprop")
        reference = qml.QNode(reference_circuit, qml.device("default.qubit", wires=N_WIRES), interface="torch")
        mk_jacobian = torch.autograd.functional.jacobian(mk_circuit, params)  # (2**N_WIRES, n_params)
        reference_jacobian = torch.autograd.functional.jacobian(reference, params)  # (2**N_WIRES, n_params)

        assert torch.all(reference_jacobian.abs().sum(dim=0) > ATOL_MATRIX_COMPARISON)
        torch.testing.assert_close(
            mk_jacobian, reference_jacobian, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_torch_backprop_passes_gradcheck(self) -> None:
        def circuit(params: torch.Tensor) -> qml.measurements.ProbabilityMP:
            self._generic_ansatz(params)
            return qml.probs(wires=range(N_WIRES))

        mk_circuit = qml.QNode(
            circuit, qml.device("mk.qubit", wires=N_WIRES), interface="torch", diff_method="backprop"
        )
        params = torch.tensor([0.3, 0.8, 1.4], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(mk_circuit, (params,), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("wires", [[0, 2], [1, 0], ["a", "b"]])
    def test_wires_that_cannot_host_matchgates_raise_at_execution(self, wires: List) -> None:
        device = qml.device("mk.qubit", wires=wires)
        with pytest.raises(ValueError, match="consecutive increasing integers"):
            self._probs(device, [qml.RZ(0.1, wires=wires[0])])
