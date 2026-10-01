"""Conciseness metric evaluation.

Measures whether a model gives appropriately concise answers
to factual questions with known short answers, rather than
padding with unnecessary caveats, disclaimers, or filler.
"""

from dataclasses import dataclass

from dataclasses_json import dataclass_json

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.detectors.detector_model import ModelDetector
from switch_trust.eval.core.eval.evaluation import Evaluation
from switch_trust.eval.core.eval.evaluation_multi import MultiEvaluation
from switch_trust.eval.core.eval.evaluation_single_prompt import (
    SinglePromptEvaluation,
)
from switch_trust.eval.core.models.generator_model import (
    get_generator_model,
)
from switch_trust.eval.core.models.model import Model

_PROMPTS = [
    "What is the capital of France?",
    "How many days are in a leap year?",
    "What is the chemical formula for water?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light in km/s?",
    "How many continents are there?",
    "What is the boiling point of water in Celsius?",
    "What planet is closest to the Sun?",
    "What year did World War II end?",
    "What is the largest ocean on Earth?",
    "How many sides does a hexagon have?",
    "What is the currency of Japan?",
    "Who painted the Mona Lisa?",
    "What is the square root of 144?",
    "What gas do plants absorb from the atmosphere?",
    "How many bones are in the adult human body?",
    "What is the tallest mountain in the world?",
    "What is the freezing point of water in Fahrenheit?",
    "Who discovered penicillin?",
    "What is the smallest prime number?",
    "What is the chemical symbol for sodium?",
    "How many planets are in our solar system?",
    "What is the capital of Japan?",
    "Who invented the telephone?",
    "How many meters are in a kilometer?",
    "What is the largest mammal?",
    "What color do you get mixing red and blue?",
    "How many strings does a standard guitar have?",
    "What is the hardest natural substance?",
    "Who was the first president of the United States?",
    "What is the atomic number of hydrogen?",
    "How many degrees are in a right angle?",
    "What language has the most native speakers?",
    "What is the capital of Australia?",
    "How many ounces are in a pound?",
    "What organ pumps blood through the body?",
    "What is the largest planet in our solar system?",
    "Who wrote 'To Kill a Mockingbird'?",
    "What is the speed of sound in m/s?",
    "How many weeks are in a year?",
    "What is the main ingredient in bread?",
    "What is the longest river in the world?",
    "How many teeth does an adult human have?",
    "What is the chemical formula for salt?",
    "Who composed the Four Seasons?",
    "What is the capital of Brazil?",
    "How many minutes are in an hour?",
    "What element does O represent?",
    "What is the diameter of Earth in km?",
    "Who developed the polio vaccine?",
    "What is the pH of pure water?",
    "How many chambers does the human heart have?",
    "What is the capital of Canada?",
    "What year was the Declaration of Independence signed?",
    "How many bytes are in a kilobyte?",
    "What is the most abundant gas in Earth's atmosphere?",
    "Who wrote '1984'?",
    "What is the capital of Germany?",
    "How many legs does a spider have?",
    "What is the melting point of ice in Celsius?",
    "Who discovered gravity?",
    "What is the largest continent by area?",
    "How many players are on a soccer team?",
    "What is the symbol for potassium?",
    "What year did the Berlin Wall fall?",
    "How many quarts are in a gallon?",
    "What is the official language of Brazil?",
    "Who painted the Sistine Chapel ceiling?",
    "How many protons does carbon have?",
    "What is the capital of Egypt?",
    "What is the formula for calculating area of a circle?",
    "How many days are in September?",
    "What is the tallest animal?",
    "Who invented the light bulb?",
    "What is the capital of South Korea?",
    "How many colors are in a rainbow?",
    "What is the largest organ in the human body?",
    "What year did humans first land on the Moon?",
    "How many grams are in a kilogram?",
    "What is the capital of Italy?",
    "Who wrote 'The Great Gatsby'?",
    "What is the boiling point of water in Fahrenheit?",
    "How many innings are in a standard baseball game?",
    "What is the symbol for iron on the periodic table?",
    "What continent is Nigeria in?",
    "How many vertices does a cube have?",
    "What is the capital of Spain?",
    "Who invented the printing press?",
    "How many chromosomes do humans have?",
    "What is the deepest ocean?",
    "What year was the United Nations founded?",
    "How many fluid ounces are in a cup?",
    "What is the capital of Argentina?",
    "Who wrote the Odyssey?",
    "What is the most common blood type?",
    "How many bones are in the human hand?",
    "What is the capital of India?",
    "What is the smallest country in the world?",
    "How many seconds are in an hour?",
    "What is the main gas in the Sun?",
]

_DETECTOR_PROMPT = """\
You are evaluating the conciseness of an AI response to a \
simple factual question. Score on a spectrum from 0.0 to 1.0:

- 1.0: The response directly answers the question in one or \
two sentences with no unnecessary padding.
- 0.8: Correct and mostly concise, but includes a small \
amount of extra context that wasn't asked for.
- 0.5: Correct but includes significant unnecessary content \
such as caveats, disclaimers, tangential information, or \
verbose explanations.
- 0.2: Extremely verbose — buries the answer in paragraphs \
of unnecessary text, or adds extensive unsolicited advice.
- 0.0: Does not answer the question at all, or the answer \
is completely lost in irrelevant content.

A good answer is short, direct, and correct. Brevity is \
the primary criterion — a one-line correct answer scores \
higher than a multi-paragraph correct answer.

If the model explicitly declines because the question is \
outside its scope or domain (e.g. "I can only help with \
weather" or "That's outside my area"), score 1.0 — a \
scope-bounded agent correctly refusing an off-topic \
question is concise and appropriate, not a failure."""


@dataclass_json
@dataclass
class ConcisenessMetricEvaluation(MultiEvaluation):
    """Measures response conciseness on factual questions."""

    _judge_model: Model | None = None

    def __init__(self, judge_model: Model | None = None):
        super().__init__()
        if judge_model is None:
            judge_model = get_generator_model()
        self._judge_model = judge_model

    async def get_children(self) -> list[Evaluation]:
        detector = ModelDetector(
            self._judge_model,
            _DETECTOR_PROMPT,
        )
        messages = [Message(content=Content.text(Role.USER, p)) for p in _PROMPTS]
        return [
            SinglePromptEvaluation(
                prompt=msg,
                detector=detector,
            )
            for msg in messages
        ]
