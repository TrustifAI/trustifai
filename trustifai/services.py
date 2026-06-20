# services.py
"""
Service layer for External APIs (LLMs, Embeddings, Rerankers).
"""

import os
from typing import Optional, List, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from litellm import (
    completion,
    embedding,
    rerank,
    acompletion,
    batch_completion,
    aembedding,
    completion_cost,
    responses,
    aresponses,
)
import litellm
from dotenv import load_dotenv
from trustifai.config import Config
import httpx
import requests
import logging

logger = logging.getLogger(__name__)

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except Exception as e:
    MLFLOW_AVAILABLE = False
    mlflow = None

litellm.drop_params = True

RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    OSError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    requests.Timeout,
    litellm.exceptions.Timeout,
    litellm.exceptions.APIConnectionError,
)


def empty_decorator(fn):
    return fn


TRACE_DECORATOR = (
    mlflow.trace
    if MLFLOW_AVAILABLE and getattr(mlflow, "trace", None)
    else empty_decorator
)


class ExternalService:
    def __init__(self, config: Config):
        self.config = config
        if self.config.env_file:
            load_dotenv(self.config.env_file)
            logger.info("Environment variables loaded.")
        self.configure_tracing()

    def configure_tracing(self):
        """Initialize tracing if enabled in config"""
        if not MLFLOW_AVAILABLE:
            return

        if self.config.tracing.params["enabled"]:
            mlflow.litellm.autolog(silent=True)
            mlflow.set_tracking_uri(self.config.tracing.params["tracking_uri"])

            experiment_name = self.config.tracing.params["experiment_name"]
            try:
                mlflow.set_experiment(experiment_name)
            except Exception:
                mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
        else:
            mlflow.litellm.autolog(disable=True, silent=True)

    @staticmethod
    @TRACE_DECORATOR
    def log_metrics_by_category(
        metrics_data: dict, trust_score: float, decision: str, offline_metric_keys: set
    ):
        """Log metrics to MLflow categorized by type"""
        if not MLFLOW_AVAILABLE:
            return

        # Categorize metrics
        offline_metrics = {}
        online_metrics = {}

        for key, value in metrics_data.items():
            metric_score = value["score"]
            if key == "source_diversity":
                metric_score = min(1.0, metric_score)

            if key in offline_metric_keys:
                key_name = f"offline/{key}"
                offline_metrics[key_name] = metric_score
            else:
                key_name = f"online/{key}"
                online_metrics[key_name] = metric_score

        # Log categorized metrics as attributes of the trace
        mlflow.set_tag("decision", decision)
        mlflow.set_tag("trust_score/final", trust_score)
        for k, v in offline_metrics.items():
            mlflow.set_tag(k, v)
        for k, v in online_metrics.items():
            mlflow.set_tag(k, v)

    @staticmethod
    def extract_document(document: Any) -> str:
        """Helper to extract text content from various document formats"""
        if document is None:
            return ""
        elif isinstance(document, list):
            if len(document) == 0:
                return ""
            elif len(document) == 1:
                return ExternalService.extract_document(document[0])
            else:
                return "\n".join(
                    [ExternalService.extract_document(doc) for doc in document]
                )
        elif isinstance(document, dict):
            # Try common keys
            for key in ["page_content", "text", "content", "output", "document", "data", "content_text", "body", "description", "page_text"]:
                if key in document:
                    return str(document[key])
        elif hasattr(document, "page_content"):
            return document.page_content
        elif hasattr(document, "text"):
            return document.text
        elif hasattr(document, "content"):
            return document.content
        elif hasattr(document, "output"):
            return document.output
        elif hasattr(document, "document"):
            return document.document
        elif hasattr(document, "data"):
            return document.data
        elif hasattr(document, "content_text"):
            return document.content_text
        elif hasattr(document, "body"):
            return document.body
        elif hasattr(document, "description"):
            return document.description
        elif hasattr(document, "page_text"):
            return document.page_text
        return str(document)

    def get_llm_params(self, **kwargs):
        """Update LLM parameters in config at runtime"""
        cfg = self.config.llm
        model = f"{cfg.type}/{cfg.params.get('model_name')}"
        base_url = cfg.params.get("base_url") or os.environ.get("LLM_BASE_URL")
        api_base = cfg.params.get("api_base") or os.environ.get("LLM_API_BASE")
        azure_ad_token = cfg.params.get("azure_ad_token") or os.environ.get(
            "AZURE_AD_TOKEN"
        )
        api_version = cfg.params.get("api_version")
        deployment_id = cfg.params.get("deployment_id")
        api_type = cfg.params.get("api_type", "chat_completion")
        final_kwargs = cfg.kwargs.copy()
        final_kwargs.update(kwargs)
        final_kwargs["azure_ad_token"] = azure_ad_token
        params = {
            "model": model,
            "base_url": base_url,
            "api_base": api_base,
            "api_version": api_version,
            "deployment_id": deployment_id,
            "api_type": api_type,
            "final_kwargs": final_kwargs,
        }
        return params

    def get_embedding_params(self, **kwargs):
        """Update embedding parameters in config at runtime"""
        cfg = self.config.embeddings
        model = f"{cfg.type}/{cfg.params.get('model_name')}"
        api_base = (
            cfg.params.get("api_base")
            or cfg.params.get("base_url")
            or os.environ.get("EMBEDDING_API_BASE")
            or os.environ.get("EMBEDDING_BASE_URL")
        )
        input_type = cfg.params.get("input_type", None)
        azure_ad_token = cfg.params.get("azure_ad_token") or os.environ.get(
            "AZURE_AD_TOKEN"
        )
        final_kwargs = cfg.kwargs.copy()
        final_kwargs.update(kwargs)
        final_kwargs["azure_ad_token"] = azure_ad_token
        params = {
            "model": model,
            "api_base": api_base,
            "input_type": input_type,
            "final_kwargs": final_kwargs,
        }
        return params

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=500),
        reraise=True,
    )
    def llm_call(
        self, system_prompt: str = None, prompt: str = None, **kwargs
    ) -> Optional[dict]:
        """Safely call LLM with retries using Config object"""
        system_prompt = system_prompt or "You are a helpful assistant."
        params = self.get_llm_params(**kwargs)

        try:
            if params["api_type"] == "responses":
                response = responses(
                    model=params["model"],
                    base_url=params["base_url"],
                    api_base=params["api_base"],
                    api_version=params["api_version"],
                    deployment_id=params["deployment_id"],
                    seed=42,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    **params["final_kwargs"],
                )
            elif params["api_type"] == "chat_completion":
                response = completion(
                    model=params["model"],
                    base_url=params["base_url"],
                    api_base=params["api_base"],
                    api_version=params["api_version"],
                    deployment_id=params["deployment_id"],
                    seed=42,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    **params["final_kwargs"],
                )
            else:
                raise ValueError(
                    f"Unsupported API type: '{params['api_type']}'. Must be 'chat_completion' or 'responses'."
                )

            try:
                cost = completion_cost(completion_response=response) or 0.0
            except Exception:
                cost = 0.0

            response_logprobs = None
            if params["api_type"] == "chat_completion":
                if (
                    hasattr(response.choices[0], "logprobs")
                    and response.choices[0].logprobs
                ):
                    response_logprobs = [
                        token.logprob for token in response.choices[0].logprobs.content
                    ]

                return {
                    "response": response.choices[0].message.content,
                    "logprobs": response_logprobs,
                    "cost": f"{cost:.6f}",
                }
            else:
                return {
                    "response": response.output_text,
                    "logprobs": None,
                    "cost": f"{cost:.6f}",
                }

        except RETRYABLE_EXCEPTIONS as e:
            logger.exception(f"Retryable error (will retry): {e}")
            raise

        except Exception as e:
            logger.exception("Error calling LLM:", e)
            return {"response": None, "logprobs": None, "cost": 0.0}

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=500),
        reraise=True,
    )
    def llm_call_batch(
        self, system_prompt: str = None, prompts: List[str] = None, **kwargs
    ) -> Optional[dict]:
        """Safely call LLM in batch with retries using Config object"""
        system_prompt = system_prompt or "You are a helpful assistant."
        params = self.get_llm_params(**kwargs)

        # 1. Structure messages as a list of lists for independent batch processing
        batched_messages = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]

        if params["api_type"] != "chat_completion":
            raise ValueError(
                f"Batch LLM calls currently only support 'chat_completion' API type. Got '{params['api_type']}'."
            )

        try:
            # 2. Call batch_completion with the list of message lists
            responses = batch_completion(
                model=params["model"],
                base_url=params["base_url"],
                api_base=params["api_base"],
                api_version=params["api_version"],
                deployment_id=params["deployment_id"],
                seed=42,
                messages=batched_messages,  # Pass the list of lists here
                **params["final_kwargs"],
            )

            # 3. Process the LIST of responses
            extracted_responses = []
            extracted_logprobs = []
            total_cost = 0.0

            for r in responses:
                # Extract content
                extracted_responses.append(r.choices[0].message.content)

                # Extract logprobs if available
                if hasattr(r.choices[0], "logprobs") and r.choices[0].logprobs:
                    extracted_logprobs.append(
                        [token.logprob for token in r.choices[0].logprobs.content]
                    )
                else:
                    extracted_logprobs.append(None)

                try:
                    total_cost += completion_cost(completion_response=r) or 0.0
                except Exception:
                    pass

            return {
                "response": extracted_responses,
                "logprobs": (
                    extracted_logprobs
                    if any(x is not None for x in extracted_logprobs)
                    else None
                ),
                "cost": total_cost,
            }

        except RETRYABLE_EXCEPTIONS as e:
            logger.exception(f"Retryable error (will retry): {e}")
            raise

        except Exception as e:
            logger.exception("Error calling LLM batch:", e)
            # Return a list of Nones matching the input length to avoid index errors downstream
            return {
                "response": [None] * len(prompts) if prompts else [],
                "logprobs": None,
                "cost": 0.0,
            }

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=500),
        reraise=True,
    )
    async def llm_call_async(
        self, system_prompt: str = None, prompt: str = None, **kwargs
    ) -> Optional[dict]:
        """Safely call LLM asynchronously with retries using Config object"""
        system_prompt = system_prompt or "You are a helpful assistant."

        params = self.get_llm_params(**kwargs)

        try:
            if params["api_type"] == "responses":
                response = await aresponses(
                    model=params["model"],
                    base_url=params["base_url"],
                    api_base=params["api_base"],
                    api_version=params["api_version"],
                    deployment_id=params["deployment_id"],
                    seed=42,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    **params["final_kwargs"],
                )
            elif params["api_type"] == "chat_completion":
                response = await acompletion(
                    model=params["model"],
                    base_url=params["base_url"],
                    api_base=params["api_base"],
                    api_version=params["api_version"],
                    deployment_id=params["deployment_id"],
                    seed=42,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    **params["final_kwargs"],
                )
            else:
                raise ValueError(
                    f"Unsupported API type: '{params['api_type']}'. Must be 'chat_completion' or 'responses'."
                )

            try:
                cost = completion_cost(completion_response=response) or 0.0
            except Exception:
                cost = 0.0

            response_logprobs = None
            if (
                hasattr(response.choices[0], "logprobs")
                and response.choices[0].logprobs
            ):
                response_logprobs = [
                    token.logprob for token in response.choices[0].logprobs.content
                ]

            return {
                "response": response.choices[0].message.content,
                "logprobs": response_logprobs,
                "cost": cost,
            }

        except RETRYABLE_EXCEPTIONS as e:
            logger.exception(f"Retryable error (will retry): {e}")
            raise

        except Exception as e:
            logger.exception("Error calling LLM asynchronously:", e)
            return {"response": None, "logprobs": None, "cost": 0.0}

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=500),
        reraise=True,
    )
    def embedding_call(self, text: str, **kwargs) -> Optional[dict]:
        """Safely call embedding model"""
        params = self.get_embedding_params(**kwargs)

        try:
            response = embedding(
                model=params["model"],
                input=[text],
                api_base=params["api_base"],
                input_type=params["input_type"],
                **params["final_kwargs"],
            )

            try:
                cost = completion_cost(completion_response=response) or 0.0
            except Exception:
                cost = 0.0
            return {"embedding": response.data[0]["embedding"], "cost": cost}

        except RETRYABLE_EXCEPTIONS as e:
            logger.exception(f"Retryable error (will retry): {e}")
            raise
        except Exception as e:
            logger.exception(f"Error calling embedding: {e}")
            return None

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=500),
        reraise=True,
    )
    async def embedding_call_async(self, text: str, **kwargs) -> Optional[dict]:
        """Safely call embedding model"""
        params = self.get_embedding_params(**kwargs)

        try:
            response = await aembedding(
                model=params["model"],
                input=[text],
                api_base=params["api_base"],
                input_type=params["input_type"],
                **params["final_kwargs"],
            )
            try:
                cost = completion_cost(completion_response=response) or 0.0
            except Exception:
                cost = 0.0
            return {"embedding": response.data[0]["embedding"], "cost": cost}
        except RETRYABLE_EXCEPTIONS as e:
            logger.exception(f"Retryable error (will retry): {e}")
            raise
        except Exception as e:
            logger.exception(f"Error calling embedding asynchronously: {e}")
            return None

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=500),
        reraise=True,
    )
    def embedding_call_batch(self, texts: List[str], **kwargs) -> Optional[dict]:
        """Safely call embedding model with a batch of texts"""
        if not texts:
            return []

        params = self.get_embedding_params(**kwargs)

        try:
            # litellm handles batching when input is a list
            response = embedding(
                model=params["model"],
                input=texts,
                api_base=params["api_base"],
                input_type=params["input_type"],
                **params["final_kwargs"],
            )
            try:
                cost = completion_cost(completion_response=response) or 0.0
            except Exception:
                cost = 0.0
            # Extract list of embeddings in order
            return {
                "embedding": [data["embedding"] for data in response.data],
                "cost": cost,
            }
        except RETRYABLE_EXCEPTIONS as e:
            logger.exception(f"Retryable error (will retry): {e}")
            raise
        except Exception as e:
            logger.exception(f"Error calling embedding batch: {e}")
            return []

    # @retry(
    #     retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential(multiplier=2, min=3, max=500),
    #     reraise=True,
    # )
    # def reranker_call(self, query: str, documents: List[str], **kwargs) -> List[str]:
    #     """Rerank documents based on similarity to query"""
    #     if not self.config.reranker or not self.config.reranker.type:
    #         logger.warning(
    #             "Warning: Reranker call attempted but no reranker configured."
    #         )
    #         return []

    #     cfg = self.config.reranker
    #     model = f"{cfg.type}/{cfg.params.get('model_name')}"
    #     top_n = cfg.params.get("top_n", len(documents))

    #     try:
    #         response = rerank(
    #             model=model,
    #             query=query,
    #             documents=documents,
    #             top_n=top_n,
    #         )
    #         return response.results
    #     except RETRYABLE_EXCEPTIONS as e:
    #         logger.exception(f"Retryable error (will retry): {e}")
    #         raise
    #     except Exception as e:
    #         logger.exception(f"Error calling reranker: {e}")
    #         return []
