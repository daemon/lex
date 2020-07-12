import enum
import random
import time

from pydantic import BaseSettings
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

from lex.core import BotModule, Intent, ConstantSelfMentionPredictor, AuthoredMessage


class DialogueIntentEnum(enum.Enum):
    DIALOGUE_INTENT = Intent('dialogue')


class DialogueSettings(BaseSettings):
    dialogue_format: str = '{selfname} {text} |{target} '
    dialogue_target_space: bool = True
    dialogue_target: str = 'Jumin'
    dialogue_capitalize: bool = True
    dialogue_punctuate: bool = True
    dialogue_selfname: str = '{0}'
    dialogue_model: str = 'gpt2-medium'
    dialogue_min_length: int = 10
    dialogue_model_path: str = 'gpt2-medium-mm.pt'


class DialogueModule(BotModule):
    name = 'dialogue'

    def __init__(self, settings: DialogueSettings):
        super().__init__()
        self.register_intent(DialogueIntentEnum.DIALOGUE_INTENT.value)
        self.register_predictor(ConstantSelfMentionPredictor(DialogueIntentEnum.DIALOGUE_INTENT.value, 0.5))
        DialogueIntentEnum.DIALOGUE_INTENT.value.register_handler(self.handle_dialogue)
        self.tokenizer = GPT2Tokenizer.from_pretrained(settings.dialogue_model)
        self.model = GPT2LMHeadModel.from_pretrained(settings.dialogue_model)
        self.model.load_state_dict(torch.load(settings.dialogue_model_path))
        self.model = self.model.cuda()
        self.model.eval()
        self.settings = settings
        self.thread = []
        self.eos_id = self.tokenizer.encode(' |')[0]

    def make_dialogue(self, text, author=None):
        if self.settings.dialogue_capitalize:
            text = text.capitalize().strip()
        if self.settings.dialogue_punctuate:
            if text.strip()[-1] not in ('?', '!', '.'):
                text = f'{text.strip()}.'
        if author is None:
            author = self.settings.dialogue_selfname
        self.thread.append((author, text))
        self.thread = self.thread[-7:]
        target = self.settings.dialogue_target
        join = ' | ' if self.settings.dialogue_target_space else ' |'
        thread_messages = [' '.join(x) for x in self.thread]
        cond_text = f'{join.join(thread_messages)}{join}{target} '
        return cond_text

    async def handle_dialogue(self, message: AuthoredMessage, _):
        async def step():
            await message.disc_message.channel.trigger_typing()
            cond_ids = self.tokenizer.encode(cond_text)
            cond_ids = cond_ids[-64:]
            try:
                cond_ids = cond_ids[cond_ids.find(self.eos_id) + 1:]
                print('truncated')
            except:
                pass
            cond_ids = torch.tensor([cond_ids]).cuda()
            length = -1
            attempts = 0
            min_length = self.settings.dialogue_min_length
            if random.random() < 0.2:
                min_length = 20
            while length < min_length or length == 0:
                token_ids = self.model.generate(cond_ids, do_sample=True, max_length=128, eos_token_id=self.eos_id).tolist()
                token_ids = token_ids[0]
                token_ids = token_ids[cond_ids.size(1):]
                text = self.tokenizer.decode(token_ids)
                text = text.replace(' |', '').strip()
                if '<|endoftext|>' in text:
                    idx = text.find('<|endoftext|>')
                    text = text[:idx]
                length = len(text)
                attempts += 1
                if attempts >= 20 and length != 0:
                    break
            await message.disc_message.channel.send(text.replace('{0}', message.author_name))
            return text
        orig_author = message.author_name
        cond_text = self.make_dialogue(message.message_content, author=self.settings.dialogue_selfname)
        text = await step()
        counter = 0

        while random.random() < 0.6:
            message.author_name = self.settings.dialogue_target
            message.message_content = text.replace('{0}', message.author_name)
            cond_text = self.make_dialogue(message.message_content.replace(orig_author, '{0}'),
                                           message.author_name)
            text = await step()
            time.sleep(1)
            counter += 1
            if counter > 5:
                break
        print(cond_text)
        self.thread.append((self.settings.dialogue_target, text))
