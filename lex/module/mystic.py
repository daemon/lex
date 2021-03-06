from collections import deque, defaultdict
from hashlib import md5
from typing import List, Dict
import enum
import re
import time

from discord import TextChannel
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
    REPLY = Intent('reply')


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
    sample_cooldown: int = 5
    cooldowns: Dict[str, int] = ''


class MysticBotModule(BotModule):
    name = 'mystic'

    def __init__(self, settings: MysticSettings):
        super().__init__()
        self.register_intents([x.value for x in MysticIntentEnum])

        wb_predictor = WhosBestIntentPredictor()
        self.register_predictor(UnknownIntentPredictor())
        self.register_predictor(MathIntentPredictor())
        self.register_predictor(wb_predictor)
        self.register_predictor(MentionedRegexIntentPredictor({re.compile(r'^!whobest$'): MysticIntentEnum.WHOS_BEST.value,
                                                               re.compile(r'^(sample|generate|speak)\s+(.+?)$', re.IGNORECASE): MysticIntentEnum.SAMPLE.value,
                                                               re.compile(r'^(reply|respond)\s+(.+?)\s+(.+?)$', re.IGNORECASE): MysticIntentEnum.REPLY.value}))

        MysticIntentEnum.UNKNOWN.value.register_handler(execute_unknown_message)
        MysticIntentEnum.MATH.value.register_handler(execute_math_message)
        MysticIntentEnum.WHOS_BEST.value.register_handler(wb_predictor)
        MysticIntentEnum.SAMPLE.value.register_handler(self.sample_message)
        MysticIntentEnum.REPLY.value.register_handler(self.sample_reply)
        self.settings = settings
        self.tokenizer = GPT2Tokenizer.from_pretrained(settings.sample_model)
        self.model = GPT2LMHeadModel.from_pretrained(settings.sample_model)
        self.model.load_state_dict(torch.load(settings.sample_model_path))
        self.model = self.model.cuda()
        self.model.eval()
        self.last_send_map = dict()
        self.last_notif_map = dict()

    async def check_delay(self, author_name: str, disc_channel: TextChannel):
        cooldown = self.settings.cooldowns.get(disc_channel.name, self.settings.sample_cooldown)
        if time.time() - self.last_send_map.get(author_name, 0) < cooldown:
            if time.time() - self.last_notif_map.get(author_name, 0) > cooldown:
                self.last_notif_map[author_name] = time.time()
                await disc_channel.send(f'Please wait {cooldown} seconds between sample requests.')
            return False
        self.last_send_map[author_name] = time.time()
        return True

    async def sample_reply(self, message: AuthoredMessage, data):
        if not await self.check_delay(message.author_name, message.disc_message.channel):
            return
        target_username = data['groups'][1]
        source_text = data['groups'][2]
        await message.disc_message.channel.send(msg_utils.sample_gpt2_mc_dialogue(self.model,
                                                                                  self.tokenizer,
                                                                                  target_username,
                                                                                  message.author_name,
                                                                                  source_text))

    async def sample_message(self, message: AuthoredMessage, data):
        if not await self.check_delay(message.author_name, message.disc_message.channel):
            return
        username = data['groups'][1]
        format_text = self.settings.sample_format.format(target=username)
        if ' ' in username:  # contains conditional text
            format_text = format_text.rstrip()
        splits = format_text.split(' ', 1)
        text = msg_utils.sample_gpt2(self.model, self.tokenizer, format_text)
        if len(splits) > 1:
            text = f' {splits[1]}{text}'
        username = splits[0]
        await message.disc_message.channel.send(f'<{username}>{text}')
