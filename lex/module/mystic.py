from collections import deque
from hashlib import md5
from typing import List, Tuple
import re
import time

from lex.core import BotModule, IntentPredictor, Intent, AuthoredMessage, IntentSelfMentionFilterMixin
from lex.utils import message as msg_utils


__all__ = ['MysticBotModule']


class UnknownIntent(Intent):
    name = 'unknown'


class MathIntent(Intent):
    name = 'math'


class WhosBestIntent(Intent):
    name = 'whosbest'


UNKNOWN_INTENT = UnknownIntent()
MATH_INTENT = MathIntent()
WHOS_BEST_INTENT = WhosBestIntent()


class UnknownIntentPredictor(IntentSelfMentionFilterMixin, IntentPredictor):
    def on_predict(self, message: AuthoredMessage) -> List[Tuple[float, Intent]]:
        return [(0.1, UNKNOWN_INTENT)]


class MathIntentPredictor(IntentSelfMentionFilterMixin, IntentPredictor):
    pattern = re.compile(r'^([\+\-/\* \(\)\^]|\d|\.)+$')

    def on_predict(self, message: AuthoredMessage) -> List[Tuple[float, Intent]]:
        rel = self.pattern.match(message.message_content) is not None
        return [(int(rel), MATH_INTENT)]


class WhosBestIntentPredictor(IntentPredictor):
    def __init__(self):
        self.last_authors = deque()

    def on_predict(self, message: AuthoredMessage) -> List[Tuple[float, Intent]]:
        self.last_authors.append((int(md5(f'{time.time()}{message.author_name}'.encode()).hexdigest(), 16), message.author_name))
        if len(self.last_authors) > 50: self.last_authors.popleft()
        if not message.contains_self_mention:
            return [(0, WHOS_BEST_INTENT)]

        doc = msg_utils.nlp(message.message_content)
        lemmatized = ' '.join([word.lemma for sent in doc.sentences for word in sent.words])
        rel = int(lemmatized.startswith('who be best') or lemmatized.startswith('whos best') or lemmatized.startswith('who be the best'))
        return [(rel, WHOS_BEST_INTENT)]

    async def __call__(self, message: AuthoredMessage):
        best_name = max(self.last_authors, key=lambda x: x[0])[1]
        await message.disc_message.channel.send(f'{best_name} is the best.')


async def execute_math_message(message: AuthoredMessage):
    answer = float(msg_utils.eval_expr(message.message_content.replace('^', '**')))
    await message.disc_message.channel.send(f'Answer: {answer:.5}')


async def execute_unknown_message(message: AuthoredMessage):
    await message.disc_message.channel.send(f'I don\'t know how to answer that, {message.author_name}-chan! ;-;')


class MysticBotModule(BotModule):
    name = 'mystic'

    def __init__(self):
        super().__init__()
        self.register_intent(UNKNOWN_INTENT)
        self.register_intent(MATH_INTENT)
        self.register_intent(WHOS_BEST_INTENT)

        wb_predictor = WhosBestIntentPredictor()
        self.register_predictor(UnknownIntentPredictor())
        self.register_predictor(MathIntentPredictor())
        self.register_predictor(wb_predictor)

        UNKNOWN_INTENT.register_handler(execute_unknown_message)
        MATH_INTENT.register_handler(execute_math_message)
        WHOS_BEST_INTENT.register_handler(wb_predictor)
