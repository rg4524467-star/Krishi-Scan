"""
Recommendation / action layer.

After every diagnosis the farmer ALWAYS sees:
  * plain-language treatment advice (multilingual, authored per disease),
  * a next-step reference (nearest agri-input dealer guidance + the toll-free
    government helpline).

Both travel inside classes.json, which the PWA caches offline - so the action
layer is fully available with no network.  `get_advice` never ends a diagnosis
on a bare disease label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..inference.classes import Classes, load_classes

HELPLINE_DEFAULT = "1800-180-1551"

# Offline-available floor: even if a disease entry is missing, the section
# renders with the helpline reference so no diagnosis ends without a next step.
_FALLBACK_ADVICE = {
    "en": {"advice": "No specific advice is stored for this result.",
           "next": "Ask a local agriculture officer or call the Kisan Call Centre {helpline}."},
    "hi": {"advice": "इस परिणाम के लिए विशिष्ट सलाह उपलब्ध नहीं है।",
           "next": "स्थानीय कृषि अधिकारी से पूछें या किसान कॉल सेंटर {helpline} पर कॉल करें।"},
    "ta": {"advice": "இந்த முடிவுக்கு குறிப்பிட்ட ஆலோசனை இல்லை.",
           "next": "உள்ளூர் வேளாண் அலுவலரிடம் கேளுங்கள் அல்லது விவசாயி கால் சென்டர் {helpline} ஐ அழைக்கவும்."},
}


@dataclass
class Advice:
    advice: str
    next_steps: List[str]
    disease_label: str
    treatments: List[Dict] = None   # [{type,name,dose,when,phi,note}, ...]

    def __post_init__(self):
        if self.treatments is None:
            self.treatments = []


class RecommendationService:
    def __init__(self, classes: Classes = None, helpline: str = HELPLINE_DEFAULT):
        self.classes = classes or load_classes()
        self.helpline = helpline

    def for_disease(self, code: str, lang: str) -> Advice:
        entry = self.classes.recommendation(code, lang)
        label = self.classes.label(code, lang)
        fallback = _FALLBACK_ADVICE.get(lang, _FALLBACK_ADVICE["en"])
        advice = entry.get("advice") or fallback["advice"]
        next_txt = entry.get("next") or fallback["next"].format(helpline=self.helpline)
        next_txt = next_txt.format(helpline=self.helpline)
        steps = [next_txt, _helpline_step(lang)]
        treatments = entry.get("treatment") or []
        return Advice(advice=advice, next_steps=steps, disease_label=label,
                      treatments=treatments)

    # -- availability ---------------------------------------------------
    def known_diseases(self) -> List[str]:
        return list(self.classes.order)


def _helpline_step(lang: str) -> str:
    table = {
        "hi": "सरकारी हेल्पलाइन: किसान कॉल सेंटर 1800-180-1551 (टोल-फ्री)",
        "en": "Government helpline: Kisan Call Centre 1800-180-1551 (toll-free)",
        "ta": "அரசு உதவி எண்: விவசாயி கால் சென்டர் 1800-180-1551 (இலவசம்)",
    }
    return table.get(lang, table["en"])


def get_advice(code: str, lang: str, service: RecommendationService = None) -> Advice:
    service = service or RecommendationService()
    return service.for_disease(code, lang)