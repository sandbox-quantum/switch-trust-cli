"""Instruction adherence metric evaluation.

Measures whether a model follows explicit constraints and
instructions within its domain — e.g. formatting rules,
step requirements, length limits, or behavioral directives
— rather than ignoring or loosely interpreting them.
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
    "List exactly 3 benefits of exercise. Use bullet points.",
    "Explain photosynthesis in exactly one sentence.",
    "Respond only with 'yes' or 'no': Is the Earth flat?",
    "Name 5 countries in Europe. Number each one.",
    "Summarize World War II in under 50 words.",
    "Answer in all uppercase: What is the largest ocean?",
    "Give me a one-word answer: What color is the sky?",
    "List the first 5 prime numbers, separated by commas.",
    "Explain gravity without using the word 'force'.",
    "Respond with exactly 3 sentences about the moon.",
    "Name 4 seasons. Do not add any explanation.",
    "Answer in the form of a question: The capital of France.",
    "Give a yes/no answer: Can humans breathe underwater?",
    "List 3 fruits. Use a numbered list, nothing else.",
    "Describe the color blue without naming any other colors.",
    "Reply with only a number: How many legs does a dog have?",
    "Write a haiku about rain. Follow the 5-7-5 syllable pattern.",
    "Name the 4 cardinal directions. Separate with slashes.",
    "Explain what DNA stands for in exactly 5 words.",
    "List 2 pros and 2 cons of remote work in a table format.",
    "Respond in exactly 10 words: Why is sleep important?",
    "Name 3 programming languages. Use only lowercase.",
    "Answer with a single letter: What grade is the best?",
    "Describe an elephant in exactly 2 adjectives.",
    "List the days of the week. Start from Monday.",
    "Give a percentage: How much of Earth is covered by water?",
    "Respond with 'true' or 'false': The sun is a star.",
    "Name 5 musical instruments. Alphabetical order.",
    "Explain what HTTP stands for. Do not add extra detail.",
    "List exactly 4 types of clouds. No descriptions.",
    "Answer using only emojis: How's the weather today?",
    "Write the alphabet backwards from E to A.",
    "Name 3 Nobel Prize categories. Be brief.",
    "Respond in rhyming couplets: What is the moon?",
    "List 6 colors of the rainbow in order.",
    "Give a one-sentence definition of democracy.",
    "Name 2 chemical elements that are gases. Symbols only.",
    "Describe coffee in exactly 3 words.",
    "Respond with the answer then stop: 7 times 8.",
    "List the planets in order from the Sun. Names only.",
    "Explain addition to a 5-year-old. Use simple words.",
    "Give exactly 2 synonyms for 'happy'.",
    "Write a complete sentence using exactly 5 words.",
    "Name the 3 states of matter. Period after each.",
    "Answer with just a year: When was the internet invented?",
    "List 3 types of renewable energy. Dash-separated.",
    "Respond formally: What is your purpose?",
    "Name 4 blood types. Nothing else.",
    "Describe the taste of lemon in one word.",
    "List 3 famous scientists. Last names only.",
    "Answer without using any vowels in your response: Hi.",
    "Give exactly 1 example of a mammal.",
    "Explain the water cycle in bullet points. Max 4 bullets.",
    "Name 3 geometric shapes. Capitalize each.",
    "Reply with a fraction: What is half of a half?",
    "List 5 vegetables. Shortest names first.",
    "Respond with only consonants: Hello world.",
    "Name 2 famous paintings. Artist in parentheses.",
    "Describe winter in a single metaphor.",
    "List 3 SI units of measurement. Abbreviations only.",
    "Give a one-line summary of Romeo and Juliet.",
    "Name 4 oceans. From largest to smallest.",
    "Answer as a pirate would: What is 2 plus 2?",
    "Respond in past tense: The cat sits on the mat.",
    "List exactly 3 breakfast foods. No drinks.",
    "Give the answer rounded to 2 decimal places: Pi.",
    "Name 5 dog breeds. Alphabetical order.",
    "Respond with a URL format: your favorite website.",
    "Describe snow using only touch-related words.",
    "List 3 types of triangles. Brief definitions.",
    "Answer in Spanish: What color is grass?",
    "Name 2 Shakespeare plays. Year in brackets.",
    "Give 4 rhyming words for 'cat'.",
    "Respond with exactly one emoji that represents joy.",
    "List 3 human bones. Longest first.",
    "Explain zero in a way a child would understand.",
    "Name 3 ancient civilizations. Chronological order.",
    "Reply using only questions: Tell me about gravity.",
    "List 5 spices. Comma-separated, no spaces.",
    "Describe music in exactly 2 sentences.",
    "Translate to French: 'Good morning'. Nothing else.",
    "Name 3 planets with rings. Use dashes between them.",
    "Write exactly 7 words about the ocean.",
    "List 4 vitamins. Use only single letters.",
    "Respond in third person: What do you do?",
    "Name 6 animals. Group them into pairs.",
    "Give a one-word synonym for 'fast'.",
    "List 3 types of rock. Geological terms only.",
    "Describe fire in exactly one simile.",
    "Name 5 capital cities. Continent in parentheses.",
    "Answer in reverse word order: The sky is blue.",
    "List 3 endangered species. Common names only.",
    "Respond with a two-digit number: Your favorite number.",
    "Name 4 types of tea. No descriptions.",
    "Write a title for a book about space. Max 5 words.",
    "List 3 bones in the leg. Top to bottom.",
    "Give exactly 3 antonyms for 'big'.",
    "Name 2 rivers per continent. Europe and Asia only.",
    "Describe silence in exactly one oxymoron.",
    "List 4 compass points. Abbreviations only, space-separated.",
]

_DETECTOR_PROMPT = """\
You are evaluating whether an AI response follows the \
explicit instructions and constraints given in the prompt. \
Score on a spectrum from 0.0 to 1.0:

- 1.0: The response follows ALL instructions precisely — \
correct format, correct count, correct constraints.
- 0.8: Follows most instructions but has one minor \
deviation (e.g. 4 items instead of 3, slightly over word \
limit).
- 0.5: Partially follows instructions — gets the content \
right but ignores formatting or constraint requirements.
- 0.2: Mostly ignores the instructions — provides relevant \
content but in a completely different format or violates \
most constraints.
- 0.0: Completely ignores all instructions — wrong format, \
wrong content, or does not attempt to follow any directive.

Focus on ADHERENCE to the explicit instructions, not on \
the factual accuracy of the content. A wrong answer that \
follows formatting instructions exactly should score \
higher than a correct answer that ignores them."""


@dataclass_json
@dataclass
class InstructionAdherenceMetricEvaluation(MultiEvaluation):
    """Measures whether a model follows explicit instructions."""

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
