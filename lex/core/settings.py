from pydantic import BaseSettings


__all__ = ['BotSettings']


class BotSettings(BaseSettings):
    api_token: str
    mention_workaround: str = '<@​&723317393870422146>'
