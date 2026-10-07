from typing import List, Optional, Union

from matchcake import NonInteractingFermionicDevice
from pennylane.devices import ExecutionConfig
from pennylane.transforms.core import CompilePipeline
from pennylane.wires import Wires

from .matchgate_translator import MatchgateTranslator
from .transforms import translate_to_matchgates


class MajoranaKaraokeDevice(NonInteractingFermionicDevice):
    """
    PennyLane device that runs any unitary circuit on MatchCake's ``nif.qubit`` device after translating it into
    matchgates.

    Before every execution, the operations of the circuit are translated by a
    :class:`~majorana_karaoke.matchgate_translator.MatchgateTranslator`. The translated circuit, which is made of
    matchgates and of the state preparations that ``nif.qubit`` executes, is then simulated by
    :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice`, in a time polynomial in the number of qubits
    for expectation values and for the probability of a given outcome. Probabilities, expectation values, samples,
    parameter broadcasting and backpropagation are inherited from it. The measurements are not translated.

    Differentiate with ``diff_method="backprop"`` and the ``torch`` interface, or with ``diff_method="finite-diff"``.
    The parameter-shift rule uses the shift rules of the original gates. They may not apply to the translation of a
    gate translated through a decomposition, whose dependence on the parameter can contain other frequencies, so
    parameter-shift gradients are wrong for gates such as ``ControlledPhaseShift``.

    The device is registered as ``mk.qubit``:

    .. code-block:: python

        import pennylane as qml

        dev = qml.device("mk.qubit", wires=4)

    :param wires: The number of wires of the device, or its wires. They must be consecutive increasing integers
        because a matchgate acts on neighbouring wires.
    :type wires: Union[int, Wires, List[int]]
    :param translator: The translator applied to every circuit. Defaults to a new
        :class:`~majorana_karaoke.matchgate_translator.MatchgateTranslator`.
    :type translator: Optional[MatchgateTranslator]
    :param kwargs: Keyword arguments forwarded to
        :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice`, such as ``shots``.
    """

    name = "mk.qubit"

    def __init__(
        self,
        wires: Union[int, Wires, List[int]] = 2,
        *,
        translator: Optional[MatchgateTranslator] = None,
        **kwargs,
    ):
        super().__init__(wires=wires, **kwargs)
        self.translator = translator if translator is not None else MatchgateTranslator()

    def preprocess_transforms(self, execution_config: Optional[ExecutionConfig] = None) -> CompilePipeline:
        """
        Return the preprocessing pipeline of ``nif.qubit`` preceded by the translation into matchgates.

        :param execution_config: Execution configuration options.
        :type execution_config: Optional[ExecutionConfig]
        :return: The preprocessing compile pipeline.
        :rtype: CompilePipeline
        """
        program = super().preprocess_transforms(execution_config)
        program.insert(0, translate_to_matchgates(translator=self.translator, wire_order=self.wires))
        return program
