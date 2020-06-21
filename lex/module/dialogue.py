import enum

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
    dialogue_selfname: str = '{0}'
    dialogue_model: str = 'gpt2-medium'
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

    def make_dialogue(self, text):
        if self.settings.dialogue_capitalize:
            text = text.capitalize()
        self.thread.append((self.settings.dialogue_selfname, text))
        self.thread = self.thread[-10:]
        target = self.settings.dialogue_target
        join = ' | ' if self.settings.dialogue_target_space else ' |'
        thread_messages = [' '.join(x) for x in self.thread]
        cond_text = f'{join.join(thread_messages)}{join}{target} '
        return cond_text

    async def handle_dialogue(self, message: AuthoredMessage, _):
        cond_text = self.make_dialogue(message.message_content.replace(message.author_name, '{0}'))
        print(cond_text)
        cond_ids = torch.tensor([self.tokenizer.encode(cond_text)])
        cond_ids = cond_ids.cuda()
        cond_ids = cond_ids[:, -128:]
        token_ids = self.model.generate(cond_ids, do_sample=True, max_length=512, eos_token_id=self.eos_id).tolist()
        token_ids = token_ids[0]
        token_ids = token_ids[cond_ids.size(1):]
        text = self.tokenizer.decode(token_ids)
        text = text.replace(' |', '').strip()
        if '<|endoftext|>' in text:
            idx = text.find('<|endoftext|>')
            text = text[:idx]
        await message.disc_message.channel.send(text.replace('{0}', message.author_name))
        self.thread.append((self.settings.dialogue_target, text))
