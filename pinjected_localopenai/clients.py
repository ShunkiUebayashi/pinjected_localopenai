import os
from pathlib import Path

from openai import AsyncOpenAI
from pinjected import instance


@instance
def async_openai_client(openai_api_key, base_url) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=openai_api_key,
        base_url=base_url
    )


@instance
def openai_api_key() -> str:
    from loguru import logger
    logger.warning(f"using openai api key from environment variable or ~/.openai_api_key.txt")
    if (api_key := os.environ.get('OPENAI_API_KEY', None)) is None:
        api_key = Path(os.path.expanduser("~/.openai_api_key.txt")).read_text().strip()
    return api_key
