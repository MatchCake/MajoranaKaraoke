from importlib.metadata import version

__author__ = "Jérémie Gince"
__email__ = "gincejeremie@gmail.com"
__copyright__ = "Copyright 2026, Jérémie Gince"
__license__ = "Apache 2.0"
__url__ = "https://github.com/MatchCake/MajoranaKaraoke"
__package__ = "majorana_karaoke"
__version__: str = version(__package__)

from .matchgate_translator import MatchgateTranslator, TranslationRule
from .mk_device import MajoranaKaraokeDevice
from .transforms import translate_to_matchgates
