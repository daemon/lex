from dataclasses import dataclass, field
from typing import List, Dict, Any
import re

import discord as disc

from .intent import IntentPredictor, Intent, IntentRegistry, IntentPrediction
import lex.core


__all__ = ['MinecraftDiscordCore', 'AuthoredMessage', 'BotModule']


@dataclass
class AuthoredMessage:
    disc_message: disc.Message
    message_content: str
    author_name: str
    attributes: Dict[str, Any] = field(default_factory=dict)


class BotModule:
    name: str = ''

    def __init__(self):
        self.predictors = []  # type: List[IntentPredictor]

    def on_finalize(self):
        pass

    def register_intent(self, intent: Intent):
        intent.namespace = self.name
        IntentRegistry.instance().register(intent)

    def register_intents(self, intents: List[Intent]):
        for intent in intents:
            self.register_intent(intent)

    def register_predictor(self, predictor: IntentPredictor):
        self.predictors.append(predictor)


class MinecraftDiscordCore(disc.Client):
    minecraft_message_rgx = re.compile(r'^‹\*\*(.+?)\*\*› (.+?)$')

    def __init__(self, settings, modules: List[BotModule]):
        super().__init__()
        self.settings = settings  # type: lex.core.BotSettings
        self.modules = modules
        for module in self.modules:
            module.on_finalize()

    async def on_message(self, message: disc.Message):
        if message.author == self.user:
            return
        m = self.minecraft_message_rgx.match(message.clean_content)
        if m:
            author_name = m.group(1).replace('\\_', '_')
            message_content = m.group(2)
        else:
            author_name = message.author.display_name
            message_content = message.clean_content
        mtext = self.settings.mention_workaround
        mentioned = self.user in message.mentions or (mtext and mtext in message.clean_content) or \
            f'@{self.user.display_name.lower()}' in message.clean_content.lower()
        message_content = message_content.replace(f'@{self.user.display_name}', '')
        message_content = message_content.replace(f'@{self.user.display_name.lower()}', '')
        if mtext:
            message_content = message_content.replace(mtext, '')
        message_content = message_content.strip()
        amsg = AuthoredMessage(message, message_content, author_name)
        amsg.attributes['self-mention'] = mentioned  # TODO: move to separate processing class
        await self.on_authored_message(amsg)

    async def on_authored_message(self, message: AuthoredMessage):
        print(f'{message.author_name}> {message.message_content}')
        intents = [x for m in self.modules for p in m.predictors for x in p.predict(message)]
        max_pred = max(intents, key=lambda x: x.rel)  # type: IntentPrediction
        if max_pred.rel > 0:
            await max_pred.intent.handle(message, max_pred.data)
