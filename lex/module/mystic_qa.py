from collections import defaultdict
from typing import List
import re

from pydantic import BaseSettings
import pandas as pd

from lex.core import BotModule, IntentPredictor, Intent, AuthoredMessage, IntentSelfMentionFilterMixin, IntentPrediction,\
    RegexIntentPredictor, LemmaRegexIntentPredictor, MentionedRegexIntentPredictor
from lex.utils import message as msg_utils


class QaSettings(BaseSettings):
    entities_path = 'data/entities.tsv'
    responses_path = 'data/responses.tsv'
    rules_path = 'data/simple-qa-rules.tsv'


class Answerer:
    def __init__(self, response):
        self.response = response

    async def __call__(self, message: AuthoredMessage, data):
        await message.disc_message.channel.send(self.response.format(author=message.author_name, **data.get('groupdict', {})))


class MysticQaBotModule(BotModule):
    name = 'mystic-qa'

    def __init__(self, settings: QaSettings):
        super().__init__()
        self.settings = settings
        entities_df = pd.read_csv(settings.entities_path, sep='\t', quoting=3)
        rules_df = pd.read_csv(settings.rules_path, sep='\t', quoting=3)
        responses_df = pd.read_csv(settings.responses_path, sep='\t', quoting=3)

        entity_map = {x.name: x.entity for x in entities_df.itertuples()}
        rules_map = defaultdict(list)
        for x in rules_df.itertuples():
            rules_map[x.response].append(re.compile(x.rule.format(**entity_map), re.IGNORECASE))

        match_map = {}
        for x in responses_df.itertuples():
            intent = Intent(x.name)
            intent.register_handler(Answerer(x.response))
            self.register_intent(intent)
            match_map.update({k: intent for k in rules_map[x.name]})
        self.register_predictor(LemmaRegexIntentPredictor(match_map))
