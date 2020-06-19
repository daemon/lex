from lex.core import MinecraftDiscordCore, BotSettings
from lex.module import mystic


def main():
    settings = BotSettings()
    core = MinecraftDiscordCore(settings, [mystic.MysticBotModule()])
    core.run(settings.api_token)


if __name__ == '__main__':
    main()
