from collections import deque
from hashlib import md5
from typing import List, Tuple
import enum
import re
import time

from lex.core import BotModule, IntentPredictor, Intent, AuthoredMessage, IntentSelfMentionFilterMixin, IntentPrediction,\
    RegexIntentPredictor, LemmaRegexIntentPredictor
from lex.utils import message as msg_utils


__all__ = ['MysticBotModule']


class MysticIntentEnum(enum.Enum):
    UNKNOWN = Intent('unknown')
    MATH = Intent('math')
    WHOS_BEST = Intent('whosbest')


class UnknownIntentPredictor(IntentSelfMentionFilterMixin, IntentPredictor):
    def on_predict(self, message: AuthoredMessage) -> List[IntentPrediction]:
        return [IntentPrediction(0.1, MysticIntentEnum.UNKNOWN.value)]


class MathIntentPredictor(IntentSelfMentionFilterMixin, RegexIntentPredictor):
    def __init__(self):
        super().__init__({re.compile(r'^([\+\-/\* \(\)\^]|\d|\.)+$'): MysticIntentEnum.MATH.value})


class WhosBestIntentPredictor(LemmaRegexIntentPredictor):
    def __init__(self):
        super().__init__({re.compile(r'^(who be best|whos best|who be the best|whos the best).*$'): MysticIntentEnum.WHOS_BEST.value})
        self.last_authors = deque()

    def on_predict(self, message: AuthoredMessage) -> List[IntentPrediction]:
        self.last_authors.append((int(md5(f'{time.time()}{message.author_name}'.encode()).hexdigest(), 16), message.author_name))
        if len(self.last_authors) > 50: self.last_authors.popleft()
        if not message.attributes['self-mention']:
            return [IntentPrediction(0, MysticIntentEnum.WHOS_BEST.value)]

        return super().on_predict(message)

    async def __call__(self, message: AuthoredMessage, data):
        best_name = max(self.last_authors, key=lambda x: x[0])[1]
        await message.disc_message.channel.send(f'{best_name} is the best.')


async def execute_math_message(message: AuthoredMessage, data):
    answer = float(msg_utils.eval_expr(message.message_content.replace('^', '**')))
    await message.disc_message.channel.send(f'Answer: {answer:.5}')


async def execute_unknown_message(message: AuthoredMessage, data):
    await message.disc_message.channel.send(f'I don\'t know how to answer that, {message.author_name}.')


async def execute_nou(message: AuthoredMessage, data):
    await message.disc_message.channel.send(f'no u')


class MysticBotModule(BotModule):
    name = 'mystic'

    def __init__(self):
        super().__init__()
        self.register_intents([x.value for x in MysticIntentEnum])

        wb_predictor = WhosBestIntentPredictor()
        self.register_predictor(UnknownIntentPredictor())
        self.register_predictor(MathIntentPredictor())
        self.register_predictor(wb_predictor)

        MysticIntentEnum.UNKNOWN.value.register_handler(execute_unknown_message)
        MysticIntentEnum.MATH.value.register_handler(execute_math_message)
        MysticIntentEnum.WHOS_BEST.value.register_handler(wb_predictor)
