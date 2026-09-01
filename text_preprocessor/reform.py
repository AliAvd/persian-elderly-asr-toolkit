import re

from . import persian_alphabet
from .arabic_alphabet import *

to_reform = [
    {
        "characters": [
            ALEF_WASLA,
            HAMZA_BELOW_ALEF,
            HAMZA_ABOVE_ALEF,
        ],
        "to_be": persian_alphabet.ALEF,
    },
    {
        "characters": [HAMZA_ABOVE_WAW],
        "to_be": persian_alphabet.VAV,
    },
    {
        "characters": [
            ALEF_MAKSURA,
            YEH,
        ],
        "to_be": persian_alphabet.YE,
    },
    {
        "characters": [KAF],
        "to_be": persian_alphabet.KAF,
    },
    {
        "characters": [
            LAM_ALEF,
            LAM_ALEF_HAMZA_ABOVE,
            LAM_ALEF_HAMZA_BELOW,
            LAM_ALEF_MADDA_ABOVE,
        ],
        "to_be": persian_alphabet.LAM + persian_alphabet.ALEF,
    },
    {
        "characters": [TEH_MARBUTA],
        "to_be": persian_alphabet.HE2,
    },
]

replacements = {}
for rule in to_reform:
    for character in rule["characters"]:
        replacements[character] = rule["to_be"]

for original_form, shaped_forms in SHAPED_FORMS.items():
    for form in shaped_forms:
        replacements[form] = replacements.get(original_form, original_form)

reform_re = re.compile(f"({'|'.join(map(re.escape, replacements.keys()))})")


def reform(text: str) -> str:
    return reform_re.sub(
        lambda mo: replacements[mo.string[mo.start() : mo.end()]], text
    )
