from __future__ import annotations

from .cgw import CGWWatermark
from .dawa import DAWAWatermark
from .dipmark import DiPMarkWatermark
from .gaussmark import GaussMarkWatermark
from .hcw import HCWWatermark
from .heavywater import HeavyWaterWatermark
from .hybrid import HybridWatermark
from .kgw import KGWWatermark
from .kuditipudi import KuditipudiWatermark
from .semantic import PMarkWatermark, SemStampWatermark, SimMarkWatermark
from .simplexwater import SimplexWaterWatermark
from .unigram import UnigramWatermark
from .vanilla import VanillaWatermark


METHOD_CLASSES = {
    "vanilla": VanillaWatermark,
    "kgw": KGWWatermark,
    "unigram": UnigramWatermark,
    "dipmark": DiPMarkWatermark,
    "hcw": HCWWatermark,
    "heavywater": HeavyWaterWatermark,
    "simplexwater": SimplexWaterWatermark,
    "kuditipudi": KuditipudiWatermark,
    "semstamp": SemStampWatermark,
    "pmark": PMarkWatermark,
    "simmark": SimMarkWatermark,
    "cgw": CGWWatermark,
    "gaussmark": GaussMarkWatermark,
    "dawa": DAWAWatermark,
    "hybrid": HybridWatermark,
}


def get_method_class(method: str):
    return METHOD_CLASSES[method]
