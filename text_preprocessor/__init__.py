import re

from persian_tools import digits

from . import arabic_alphabet
from .reform import reform


class TextPreprocessor:
    def __init__(self):
        # Persian letters
        persian_letters = list("ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
        # Some Arabic letters
        arabic_letters = [
            arabic_alphabet.HAMZA_ABOVE_YEH,
            arabic_alphabet.ALEF_MADDA,
        ]

        # Combine all chars
        self.chars = [" "] + persian_letters + arabic_letters

        general_punctuations = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        persian_punctuations = ["‌", "،", "؛", "؟", "٪", "٬", "«", "»"]
        self.space_replaces = general_punctuations + persian_punctuations

        self.replace_re = re.compile(
            f"[{''.join(map(re.escape, self.space_replaces))}]+"
        )
        self.remove_re = re.compile(f"[^{''.join(map(re.escape, self.chars))}]+")

    def standardize(self, text: str) -> str:
        text = text.lower()
        text = reform(text)
        text = self.replace_re.sub(" ", text)
        text = self.remove_re.sub("", text)
        text = digits.convert_to_en(text)
        text = re.sub(
            r"\d+",
            lambda m: f" {digits.convert_to_word(int(m.string[m.start() : m.end()]))} ",
            text,
        )
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text
