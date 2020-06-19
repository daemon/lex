from dataclasses import dataclass
from typing import List, Tuple, Coroutine, Iterable
import abc


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
    name: str = None

    def __init__(self):
        self.handlers = []  # type: List[HandlerType]

    def register_handler(self, handler: HandlerType):
        self.handlers.append(handler)

    async def handle(self, message):
        for handler in self.handlers:
            await handler(message)

    @property
    def fq_name(self):
        return f'{self.namespace}:{self.name}'


NULL_INTENT = Intent()


class IntentPredictor:
    def predict(self, message) -> List[Tuple[float, Intent]]:
        return self.on_predict(message)

    @abc.abstractmethod
    def on_predict(self, message) -> List[Tuple[float, Intent]]:
        pass


class IntentSelfMentionFilterMixin:
    def predict(self, message) -> List[Tuple[float, Intent]]:
        if not message.contains_self_mention:
            return [(0, NULL_INTENT)]
        return super().predict(message)
