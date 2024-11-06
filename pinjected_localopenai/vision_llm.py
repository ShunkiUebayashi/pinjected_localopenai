import asyncio
import base64
import io
import json
import os  # 追加: 環境変数を取得するため
import re
from asyncio import Lock
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Callable, Optional

import openai.types.chat
import pandas as pd
import reactivex
from PIL.Image import Image
from injected_utils.injected_cache_utils import async_cached, sqlite_dict
from loguru import logger
from math import ceil
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError
from openai.types.chat import ChatCompletion
from pinjected import injected, Injected, instances, instance
from pydantic import BaseModel, Field


class ChatCompletionWithCost(BaseModel):
    src: ChatCompletion
    total_cost_usd: float
    prompt_cost_usd: float
    completion_cost_usd: float


@instance
def chat_completion_costs_subject():
    return reactivex.Subject()


def to_content(img: Image, detail: Literal["auto", "low", "high"] = 'auto'):
    # ImageをJPEGバイトに変換
    jpg_bytes = io.BytesIO()
    img.convert('RGB').save(jpg_bytes, format='jpeg', quality=95)
    b64_image = base64.b64encode(jpg_bytes.getvalue()).decode('utf-8')
    mb_of_b64 = len(b64_image) / 1024 / 1024
    logger.info(f"image size: {mb_of_b64:.2f} MB in base64.")
    return {
        "type": 'image_url',
        "image_url": dict(
            url=f"data:image/jpeg;base64,{b64_image}",
            detail=detail
        )
    }


@dataclass
class UsageEntry:
    timestamp: datetime
    tokens: int

    class Config:
        arbitrary_types_allowed = True


class RateLimitManager(BaseModel):
    max_tokens: int
    max_calls: int
    duration: timedelta
    lock: Lock = asyncio.Lock()
    call_history: list[UsageEntry] = []

    async def acquire(self, approx_tokens):
        if await self.ready(approx_tokens):
            pass
        else:
            while not await self.ready(approx_tokens):
                await asyncio.sleep(1)

    async def ready(self, token):
        async with self.lock:
            remaining = await self.remaining_tokens()
            is_ready = remaining >= token and len(self.call_history) < self.max_calls
            if is_ready:
                self.call_history.append(UsageEntry(pd.Timestamp.now(), token))
            return is_ready

    async def remaining_tokens(self):
        return self.max_tokens - await self._current_usage()

    async def remaining_calls(self):
        return self.max_calls - len(self.call_history)

    async def _current_usage(self):
        t = pd.Timestamp.now()
        self.call_history = [e for e in self.call_history if e.timestamp > t - self.duration]
        return sum(e.tokens for e in self.call_history)

    class Config:
        arbitrary_types_allowed = True


class RateLimitKey(BaseModel):
    api_key: str
    organization: str
    model_name: str
    request_type: str


class BatchQueueLimits(BaseModel):
    tpm: Optional[int] = Field(None, alias="TPM")
    rpm: Optional[int] = Field(None, alias="RPM")
    tpd: Optional[int] = Field(None, alias="TPD")
    images_per_minute: Optional[int] = None


class ModelLimits(BaseModel):
    modeltoken_limits: Optional[int] = None
    request_limits: Optional[int] = None
    other_limits: Optional[int] = None
    batch_queue_limits: Optional[BatchQueueLimits] = None


class Limits(BaseModel):
    llava_1_6_mistral: ModelLimits = Field(..., alias="llava-1.6-mistral")
    llava_1_6_vicuna: ModelLimits = Field(..., alias="llava-1.6-vicuna")


class ModelPricing(BaseModel):
    input_cost: float
    output_cost: float


class PricingModel(BaseModel):
    llava_1_6_mistral: ModelPricing = ModelPricing(input_cost=0.0050, output_cost=0.0150)
    llava_1_6_vicuna: ModelPricing = ModelPricing(input_cost=0.0060, output_cost=0.0180)


pricing_model = PricingModel()


@instance
def openai_rate_limit_managers(
        openai_api_key: str,
        openai_organization: str,
        openai_rate_limits: Limits
) -> dict[Any, RateLimitManager]:
    managers = dict()
    for model, limits in openai_rate_limits.dict().items():
        key = RateLimitKey(
            api_key=openai_api_key,
            organization=openai_organization,
            model_name=model,
            request_type="completion"
        )
        managers[key] = RateLimitManager(
            max_tokens=limits.modeltoken_limits or 1000000,  # デフォルト値を設定
            max_calls=limits.request_limits or 1000,          # デフォルト値を設定
            duration=pd.Timedelta("1 minute"),
        )
    return managers


@injected
async def a_repeat_for_rate_limit(logger, /, task):
    while True:
        try:
            return await task()
        except RateLimitError as e:
            logger.error(f"rate limit error: {e}")
            pat = "Please retry after (\d+) seconds."
            match = re.search(pat, e.message)
            if match:
                seconds = int(match.group(1))
                logger.info(f"sleeping for {seconds} seconds")
                await asyncio.sleep(seconds)
            else:
                logger.warning(f"failed to parse rate limit error message: {e.message}")
                await asyncio.sleep(10)
        except APITimeoutError as e:
            logger.warning(f"API timeout error: {e}")
            await asyncio.sleep(10)
        except APIConnectionError as ace:
            logger.warning(f"API connection error: {ace}")
            await asyncio.sleep(10)


def resize(width: int, height: int) -> tuple[int, int]:
    if width > 1024 or height > 1024:
        if width > height:
            height = int(height * 1024 / width)
            width = 1024
        else:
            width = int(width * 1024 / height)
            height = 1024
    return width, height


@injected
def openai_count_image_tokens(width: int, height: int) -> int:
    width, height = resize(width, height)
    h = ceil(height / 512)
    w = ceil(width / 512)
    total = 85 + 170 * h * w
    return total


@injected
async def a_chat_completion_to_cost(
        openai_model_pricing_table: dict[str, ModelPricing],
        /,
        completion: ChatCompletion
) -> ChatCompletionWithCost:
    pricing = openai_model_pricing_table.get(completion.model)
    if not pricing:
        raise ValueError(f"Pricing information for model {completion.model} not found.")
    usage = completion.usage
    return ChatCompletionWithCost(
        src=completion,
        total_cost_usd=pricing.input_cost * usage.prompt_tokens / 1000 + pricing.output_cost * usage.completion_tokens / 1000,
        prompt_cost_usd=pricing.input_cost * usage.prompt_tokens / 1000,
        completion_cost_usd=pricing.output_cost * usage.completion_tokens / 1000
    )


@instance
def openai_model_pricing_table() -> dict[str, ModelPricing]:
    keys = pricing_model.dict().keys()
    return {k.replace("_", "-"): getattr(pricing_model, k) for k in keys}


@injected
async def a_vision_llm__openai(
        async_openai_client: AsyncOpenAI,
        a_repeat_for_rate_limit,
        a_chat_completion_to_cost,
        chat_completion_costs_subject: reactivex.Subject,
        a_enable_cost_logging: Callable,
        /,
        text: str,
        images: Optional[list[Image]] = None,
        model: str = "llava-1.6-mistral",  # デフォルトモデルを変更
        max_tokens: int = 2048,
        response_format: Optional[dict] = None,
        detail: Literal["auto", "low", "high"] = 'auto'
) -> str:
    assert isinstance(async_openai_client, AsyncOpenAI)
    await a_enable_cost_logging()
    if images is None:
        images = []
    if response_format is None:
        response_format = {"type": "text"}

    for img in images:
        assert isinstance(img, Image), f"image is not Image, but {type(img)}"

    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        API = async_openai_client.chat.completions.create
        def get_result(completion):
            return json.loads(completion.choices[0].message.content)
    else:
        API = async_openai_client.chat.completions.create
        def get_result(completion):
            return completion.choices[0].message.content

    async def task():
        chat_completion = await API(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": 'text',
                            "text": text
                        },
                        *[to_content(img, detail=detail) for img in images]
                    ]
                }
            ],
            model=model,
            max_tokens=max_tokens,
            response_format=response_format
        )
        cost: ChatCompletionWithCost = await a_chat_completion_to_cost(chat_completion)
        chat_completion_costs_subject.on_next(cost)
        return chat_completion

    chat_completion = await a_repeat_for_rate_limit(task)
    return get_result(chat_completion)


@instance
async def cost_logging_state():
    return dict(
        enabled=False,
    )


@injected
async def a_enable_cost_logging(
        cost_logging_state: dict,
        chat_completion_costs_subject: reactivex.Subject,
        /
):
    if cost_logging_state["enabled"]:
        return
    cumulative_cost = 0

    def on_next(cost: ChatCompletionWithCost):
        nonlocal cumulative_cost
        cumulative_cost += cost.total_cost_usd
        logger.info(f"cost: {cost.total_cost_usd:.4f} USD, cumulative: {cumulative_cost:.4f} USD")

    chat_completion_costs_subject.subscribe(on_next)
    cost_logging_state["enabled"] = True


# 新しいOpenAIクライアントの設定: ローカルサーバー用
@instance
def async_openai_client() -> AsyncOpenAI:
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("LOCAL_OPENAI_API_KEY", "token-abc123")  # 環境変数名を変更
    return AsyncOpenAI(api_key=api_key, base_url=base_url)  # base_urlを設定


# モデル名を変更して関数を部分適用
a_vision_llm__mistral = Injected.partial(a_vision_llm__openai, model="llava-1.6-mistral")
a_vision_llm__vicuna = Injected.partial(a_vision_llm__openai, model="llava-1.6-vicuna")

# キャッシュの設定
a_cached_vision_llm__mistral = async_cached(
    sqlite_dict(str(Path("~/.cache/pinjected_openai/a_vision_llm__llava-1.6-mistral.sqlite").expanduser()))
)(a_vision_llm__mistral)
a_cached_vision_llm__vicuna = async_cached(
    sqlite_dict(str(Path("~/.cache/pinjected_openai/a_vision_llm__llava-1.6-vicuna.sqlite").expanduser()))
)(a_vision_llm__vicuna)


@injected
async def a_llm__openai(
        async_openai_client: AsyncOpenAI,
        a_repeat_for_rate_limit,
        a_enable_cost_logging,
        /,
        text: str,
        model_name: str,
        max_completion_tokens: int = 4096,
) -> str:
    assert isinstance(async_openai_client, AsyncOpenAI)
    await a_enable_cost_logging()

    async def task():
        chat_completion = await async_openai_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": 'text',
                            "text": text
                        },
                    ]
                }
            ],
            model=model_name,
            max_tokens=max_completion_tokens
        )
        return chat_completion

    chat_completion = await a_repeat_for_rate_limit(task)
    res = chat_completion.choices[0].message.content
    assert isinstance(res, str)
    logger.info(f"result:\n{res}")
    return res


@injected
async def a_llm__mistral(
        a_llm__openai,
        /,
        text: str,
        max_completion_tokens: int = 4096
) -> str:
    return await a_llm__openai(text, model_name="llava-1.6-mistral", max_completion_tokens=max_completion_tokens)


a_llm__mistral_cached = async_cached(
    sqlite_dict(str(Path("~/.cache/a_llm__mistral.sqlite").expanduser()))
)(a_llm__mistral)


@injected
async def a_llm__vicuna(
        a_llm__openai,
        /,
        text: str,
        max_completion_tokens: int = 4096
) -> str:
    return await a_llm__openai(text, model_name="llava-1.6-vicuna", max_completion_tokens=max_completion_tokens)


@injected
async def a_json_llm__openai(
        logger,
        async_openai_client: AsyncOpenAI,
        a_repeat_for_rate_limit,
        /,
        text: str,
        max_completion_tokens: int = 4096,
        model: str = "llava-1.6-mistral"  # デフォルトモデルを変更
) -> dict:
    assert isinstance(async_openai_client, AsyncOpenAI)

    async def task():
        chat_completion = await async_openai_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": 'text',
                            "text": text
                        },
                    ]
                }
            ],
            model=model,
            max_tokens=max_completion_tokens,
            response_format={"type": "json_object"}
        )
        return chat_completion

    chat_completion = await a_repeat_for_rate_limit(task)
    res = chat_completion.choices[0].message.content
    assert isinstance(res, str)
    logger.info(f"\n{res}")
    return json.loads(res)


@injected
async def a_json_llm__vicuna(
        a_json_llm__openai,
        /,
        text: str,
        max_completion_tokens: int = 4096
) -> dict:
    return await a_json_llm__openai(
        text=text,
        max_completion_tokens=max_completion_tokens,
        model="llava-1.6-vicuna"  # モデル名を変更
    )


# テストケース
test_vision_llm__mistral = a_vision_llm__mistral(
    text="What are inside this image?",
    images=Injected.list(),
)

test_vision_llm__vicuna = a_vision_llm__vicuna(
    text="Describe the contents of this image.",
    images=Injected.list(),
)

test_llm__mistral = a_llm__mistral(
    "Hello world"
)

test_llm__vicuna = a_llm__vicuna(
    "Hello world"
)

test_json_llm__mistral = a_json_llm__openai(
    "Hello world, respond to me in json",
    model="llava-1.6-mistral"
)

test_json_llm__vicuna = a_json_llm__vicuna(
    "Hello world, respond to me in json"
)

__meta_design__ = instances()
