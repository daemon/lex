from dataclasses import dataclass, field
from typing import List, Coroutine, Iterable, Dict, Any, Pattern
import abc
import re

import lex.utils.message as msg_utils


HandlerType = Coroutine


class IntentRegistry:
    def __init__(self):
        self.intent_map = {}

    def register(self, intent: 'Intent'):
        assert intent.fq_name not in self.intent_map, 'Intent with FQ name already exists.'
        self.intent_map[intent.fq_name] = intent

    def intents(self) -> Iterable['Intent']:
        return self.intent_map.values()

    def __getitem__(self, name: str) -> 'Intent':
        return self.intent_map[name]

    @staticmethod
    def instance() -> 'IntentRegistry':
        if not hasattr(IntentRegistry, '_instance'):
            IntentRegistry._instance = IntentRegistry()
        return IntentRegistry._instance


class Intent:
    namespace: str = ''

    def __init__(self, name: str = ''):
        self.handlers = []  # type: List[HandlerType]
        self.name = name

    def register_handler(self, handler: HandlerType):
        self.handlers.append(handler)

    async def handle(self, message, data):
        for handler in self.handlers:
            await handler(message, data)

    @property
    def fq_name(self):
        return f'{self.namespace}:{self.name}'


NULL_INTENT = Intent()


@dataclass
class IntentPrediction:
    rel: float
    intent: Intent
    data: Dict[str, Any] = field(default_factory=dict)


class IntentPredictor:
    def predict(self, message) -> List[IntentPrediction]:
        return self.on_predict(message)

    @abc.abstractmethod
    def on_predict(self, message) -> List[IntentPrediction]:
        pass


class IntentSelfMentionFilterMixin:
    def predict(self, message) -> List[IntentPrediction]:
        if not message.attributes.get('self-mention'):
            return [IntentPrediction(0, NULL_INTENT)]
        return super().predict(message)


class ConstantIntentPredictor(IntentPredictor):
    def __init__(self, intent: Intent, value: float):
        self.intent = intent
        self.value = value

    def on_predict(self, message) -> List[IntentPrediction]:
        return [IntentPrediction(self.value, self.intent)]


class ConstantSelfMentionPredictor(IntentSelfMentionFilterMixin, ConstantIntentPredictor):
    def __init__(self, *args):
        super().__init__(*args)


class LemmaRegexIntentPredictor(IntentPredictor):
    def __init__(self, regex_intent_map: Dict[re.Pattern, Intent]):
        self.regex_intent_map = regex_intent_map

    def on_predict(self, message) -> List[IntentPrediction]:
        doc = msg_utils.nlp(message.message_content)
        lemmatized = ' '.join([word.lemma for sent in doc.sentences for word in sent.words])
        preds = []
        for rgx, intent in self.regex_intent_map.items():
            m = rgx.match(lemmatized)
            pred = IntentPrediction(int(m is not None), intent)
            if m is not None:
                pred.data['groups'] = m.groups()
            preds.append(pred)
        return preds


class RegexIntentPredictor(IntentPredictor):
    def __init__(self, regex_intent_map: Dict[Pattern[str], Intent]):
        self.regex_intent_map = regex_intent_map

    def on_predict(self, message) -> List[IntentPrediction]:
        preds = []
        for rgx, intent in self.regex_intent_map.items():
            m = rgx.match(message.message_content)
            pred = IntentPrediction(int(m is not None), intent)
            if m is not None:
                pred.data['groups'] = m.groups()
            preds.append(pred)
        return preds


class MentionedRegexIntentPredictor(IntentSelfMentionFilterMixin, RegexIntentPredictor):
    pass
