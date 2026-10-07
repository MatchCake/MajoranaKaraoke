from typing import Optional

import pennylane as qml
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn, ResultBatch
from pennylane.wires import WiresLike

from .matchgate_translator import MatchgateTranslator


@qml.transform
def translate_to_matchgates(
    tape: QuantumScript,
    translator: Optional[MatchgateTranslator] = None,
    wire_order: Optional[WiresLike] = None,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """
    Translate every operation of a circuit into operations that MatchCake's ``nif.qubit`` device simulates.

    The measurements are kept as they are. See
    :class:`~majorana_karaoke.matchgate_translator.MatchgateTranslator` for the translation rules.

    :param tape: The circuit to translate.
    :type tape: QuantumScript
    :param translator: The translator to use. Defaults to a new
        :class:`~majorana_karaoke.matchgate_translator.MatchgateTranslator`.
    :type translator: Optional[MatchgateTranslator]
    :param wire_order: The wires of the device, in order. Defaults to every integer from the smallest to the largest
        wire of the circuit.
    :type wire_order: Optional[WiresLike]
    :return: The translated circuit and a post-processing function returning its single result.
    :rtype: tuple[QuantumScriptBatch, PostprocessingFn]
    """
    translator = translator if translator is not None else MatchgateTranslator()

    def null_postprocessing(results: ResultBatch):
        return results[0]

    return (translator.translate_tape(tape, wire_order=wire_order),), null_postprocessing
