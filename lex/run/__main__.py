from lex.core import MinecraftDiscordCore, BotSettings
from lex.module import mystic, dialogue


def main():
    settings = BotSettings()
    if settings.preset_name == 'mysticmessenger':
        core = MinecraftDiscordCore(settings, [dialogue.DialogueModule(dialogue.DialogueSettings())])
    else:
        core = MinecraftDiscordCore(settings, [mystic.MysticBotModule(mystic.MysticSettings())])
    core.run(settings.api_token)


if __name__ == '__main__':
    main()
