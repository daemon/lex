from collections import deque
from hashlib import md5
from typing import List, Tuple
import enum
import re
import time

from pydantic import BaseSettings
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

from lex.core import BotModule, IntentPredictor, Intent, AuthoredMessage, IntentSelfMentionFilterMixin, IntentPrediction,\
    RegexIntentPredictor, LemmaRegexIntentPredictor, MentionedRegexIntentPredictor
from lex.utils import message as msg_utils


__all__ = ['MysticBotModule']


class MysticIntentEnum(enum.Enum):
    UNKNOWN = Intent('unknown')
    MATH = Intent('math')
    WHOS_BEST = Intent('whosbest')
    SAMPLE = Intent('sample')


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


class MysticSettings(BaseSettings):
    sample_format: str = '{target} '
    sample_model: str = 'gpt2-medium'
    sample_model_path: str = 'gpt2-medium.pt'
    sample_cooldown: int = 30


class MysticBotModule(BotModule):
    name = 'mystic'

    def __init__(self, settings: MysticSettings):
        super().__init__()
        self.register_intents([x.value for x in MysticIntentEnum])

        wb_predictor = WhosBestIntentPredictor()
        self.register_predictor(UnknownIntentPredictor())
        self.register_predictor(MathIntentPredictor())
        self.register_predictor(wb_predictor)
        self.register_predictor(MentionedRegexIntentPredictor({re.compile(r'^!whobest$'): MysticIntentEnum.WHOS_BEST.value}))
        self.register_predictor(MentionedRegexIntentPredictor({re.compile(r'^(sample|generate|speak) (.+?)$', re.IGNORECASE): MysticIntentEnum.SAMPLE.value}))

        MysticIntentEnum.UNKNOWN.value.register_handler(execute_unknown_message)
        MysticIntentEnum.MATH.value.register_handler(execute_math_message)
        MysticIntentEnum.WHOS_BEST.value.register_handler(wb_predictor)
        MysticIntentEnum.SAMPLE.value.register_handler(self.sample_message)
        self.settings = settings
        self.tokenizer = GPT2Tokenizer.from_pretrained(settings.sample_model)
        self.model = GPT2LMHeadModel.from_pretrained(settings.sample_model)
        self.model.load_state_dict(torch.load(settings.sample_model_path))
        self.model = self.model.cuda()
        self.model.eval()
        self.last_send_map = dict()
        self.last_notif_map = dict()

    async def sample_message(self, message: AuthoredMessage, data, max_count=30):
        if max_count == 30 and time.time() - self.last_send_map.get(message.author_name, 0) < self.settings.sample_cooldown:
            if time.time() - self.last_notif_map.get(message.author_name, 0) > self.settings.sample_cooldown:
                self.last_notif_map[message.author_name] = time.time()
                await message.disc_message.channel.send(f'Please wait {self.settings.sample_cooldown} seconds between sample requests.')
            return
        self.last_send_map[message.author_name] = time.time()
        username = data['groups'][1]
        ids = [self.tokenizer.encode(self.settings.sample_format.format(target=username))]
        ids = torch.tensor(ids).cuda()
        token_ids = self.model.generate(ids, do_sample=True, max_length=64, eos_token_id=self.tokenizer.encode(' |')[0])
        token_ids = token_ids[0]
        token_ids = token_ids[ids.size(1):]
        text = self.tokenizer.decode(token_ids)
        text = text.replace(' |', '').strip()
        if '<|endoftext|>' in text:
            idx = text.find('<|endoftext|>')
            text = text[:idx]
        if len(text) < 20 and max_count > 0:
            await self.sample_message(message, data, max_count=max_count - 1)
        else:
            splits = username.split(' ', 1)
            if len(splits) > 1:
                text = f'{splits[1]} {text}'
            username = splits[0]
            await message.disc_message.channel.send(f'<{username}> {text}')
