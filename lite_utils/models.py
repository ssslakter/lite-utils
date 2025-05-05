import litellm
import os
import fastcore.all as fc
from litellm import AllowedFailsPolicy, RetryPolicy, Router
from litellm.caching.caching import Cache
litellm.cache = Cache(type="disk")


def to_model_list(name: str, ld: list[dict]):
    return [{"model_name": name, "litellm_params": d} for d in ld]


deepinfra_args = {
    "rpm": 999,
    "api_key": os.getenv("DEEPINFRA_API_KEY", '_'),
    "max_parallel_requests": 19
}

model_configs = {
    "llama-3.1-8b-instruct": [
        {"model": "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct", **deepinfra_args},
    ],
    "llama-3.1-70b-instruct": [
        {"model": "deepinfra/meta-llama/Meta-Llama-3.1-70B-Instruct", **deepinfra_args},
    ],
    "gpt-4o-mini": [
        {"model": "gpt-4o-mini",
         "api_key": os.getenv("OPENAI_API_KEY", '_'),
         "rpm": 500,
         "tpm": 200_000}
    ],
    "mixtral-8x7b-instruct-v0.1": [
        {"model": "deepinfra/mistralai/Mixtral-8x7B-Instruct-v0.1", **deepinfra_args}
    ],
    "qwen2.5-72b-instruct": [
        {"model": "deepinfra/Qwen/Qwen2.5-72B-Instruct", **deepinfra_args}
    ],
    "mixtral-8x7b-instruct-v0.1": [
        {"model": "deepinfra/mistralai/Mixtral-8x7B-Instruct-v0.1", **deepinfra_args}
    ]
}

default_params = dict(
    drop_params=True,
    caching=True)

retry_policy = RetryPolicy(
    BadRequestErrorRetries=0,
    AuthenticationErrorRetries=1,
    TimeoutErrorRetries=3,
    RateLimitErrorRetries=5,
    ContentPolicyViolationErrorRetries=2,
    InternalServerErrorRetries=3
)

allowed_fails_policy = AllowedFailsPolicy(
    RateLimitErrorAllowedFails=5,
)


@fc.delegates(Router)
def make_router(model_name: str=None, config_list: list[dict]=None,
                model_list: list=None,
                cache_responses=True,
                retry_policy=retry_policy,
                allowed_fails_policy=allowed_fails_policy,
                cooldown_time=10,
                retry_after=5,
                routing_strategy="least-busy",
                default_litellm_params=default_params,
                **kwargs):
    model_list = model_list or to_model_list(model_name, config_list)
    return Router(model_list,
                  cache_responses=cache_responses,
                  retry_policy=retry_policy,
                  default_litellm_params=default_litellm_params,
                  allowed_fails_policy=allowed_fails_policy,
                  cooldown_time=cooldown_time,
                  retry_after=retry_after,
                  routing_strategy=routing_strategy,
                  **kwargs)

@fc.delegates(make_router)
def make_model_routers(model_configs: dict = model_configs, **kwargs):
    model_routers = {k: make_router(k, v, **kwargs) for k, v in model_configs.items()}
    return model_routers

default_routers = make_model_routers(model_configs)