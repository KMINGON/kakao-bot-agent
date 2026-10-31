from kakao_bot.runtime.competition import ContestContextMiddleware


def build_middleware(pack):
    return [ContestContextMiddleware()]
