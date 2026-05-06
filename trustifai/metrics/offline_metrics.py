# offline_metrics.py
"""Offline Metric calculators"""

import json
import re
import numpy as np
from typing import List
from nltk.tokenize import sent_tokenize

from trustifai.config import Config
from trustifai.structures import (
    MetricContext,  
    SpanSchema,
    MetricResult,
    SpanCheckResult,
    TrustLevel,
)
from trustifai.services import ExternalService
from trustifai.metrics.calculators import SourceIdentifier
from trustifai.metrics.base import BaseMetric
import asyncio
import logging
import threading
import nest_asyncio
nest_asyncio.apply()

logger = logging.getLogger(__name__)

class EvidenceCoverageMetric(BaseMetric):
    def __init__(self, service: ExternalService, config: Config):
        super().__init__(service, config)
        self.strategy = LLMBasedEvidenceStrategy(self.service, self.config)

    def calculate(self, context: MetricContext) -> MetricResult:
        if not context.answer or context.documents is None or len(context.documents) == 0:
            return MetricResult(score=0.0, label="Empty Answer", details={"sentences_checked": 0})
        return self.strategy.calculate(context)
        
    async def a_calculate(self, context: MetricContext) -> MetricResult:
        return self.calculate(context) # Currently blocking in batch, can be optimized


class SemanticDriftMetric(BaseMetric):
    def calculate(self, context: MetricContext) -> MetricResult:
        if context.documents is None or len(context.documents) == 0:
            return MetricResult(
                score=0.0, label="No Documents", details={"docs_checked": 0}
            )

        all_sentences = []
        for doc in context.documents:
            text = self.service.extract_document(doc)
            all_sentences.extend(sent_tokenize(text))

        if not all_sentences:
            return MetricResult(score=0.0, label="Empty Context", details={})

        answer_emb = np.atleast_2d(context.answer_embeddings)
        query_emb = np.atleast_2d(context.query_embeddings)

        # Embed all document sentences at once (batched)
        try:
            embedding_result = self.service.embedding_call_batch(all_sentences)
        except Exception as e:
            logger.exception(f"Error during embedding call batch: {e}")
            return MetricResult(score=0.0, label="Embedding Error", details={"error": str(e), "sentences_checked": len(all_sentences)}, execution_metadata={"total_cost_usd": 0.0})
        
        sentence_embeddings = embedding_result['embedding'] if embedding_result and 'embedding' in embedding_result else []
        cost = embedding_result["cost"] if embedding_result else 0.0

        best_score = 0.0
        best_sentence = ""

        for sentence, sent_emb in zip(all_sentences, sentence_embeddings):
            sent_emb = np.atleast_2d(sent_emb)

            sim = max(
                self.cosine_calc.calculate(answer_emb, sent_emb),
                self.cosine_calc.calculate(query_emb, sent_emb),
            )

            if sim > best_score:
                best_score = sim
                best_sentence = sentence

        label, explanation = self.threshold_evaluator.evaluate_drift(best_score)

        return MetricResult(
            score=best_score,
            label=label,
            details={
                "explanation": explanation,
                "total_documents": len(context.documents),
                "total_sentences_checked": len(all_sentences),
                "best_matching_sentence": best_sentence[:150] + " ... [truncated]" if len(best_sentence) > 150 else best_sentence,
            },
            execution_metadata={"total_cost_usd": cost}
        )


class EpistemicConsistencyMetric(BaseMetric):
    def calculate(self, context: MetricContext) -> MetricResult:
        """Runs the async generation in a completely safe isolated thread to prevent event-loop clashes"""
        if self.config.k_samples == 0:
            return self._create_stable_result()

        result_container = []
        
        def run_async_isolated():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                res, cost = loop.run_until_complete(self._generate_samples_async(context))
                result_container.append((res, cost))
            finally:
                loop.close()

        # Isolate completely from any running event loops (FastAPI, Celery, Notebooks)
        thread = threading.Thread(target=run_async_isolated)
        thread.start()
        thread.join()

        samples, cost = result_container[0] if result_container else ([], 0.0)
        if not samples:
            return self._create_unreliable_result(cost)
        similarities = self._calculate_similarities(samples, context)
        if not similarities:
            return self._create_unreliable_result(cost)

        score = float(np.mean(similarities))
        std = float(np.std(similarities)) if len(similarities) > 1 else 0.0
        ci_95 = 1.96 * (std / np.sqrt(self.config.k_samples)) 

        return self._format_result(score, samples, std, ci_95, cost)

    async def a_calculate(self, context: MetricContext) -> MetricResult:
        """Native async pipeline for non-blocking server applications"""
        if self.config.k_samples == 0:
            return self._create_stable_result()

        samples, cost = await self._generate_samples_async(context)
        if not samples:
            return self._create_unreliable_result(cost)
        similarities = self._calculate_similarities(samples, context)

        if not similarities:
            return self._create_unreliable_result(cost)

        score = float(np.mean(similarities))
        std = float(np.std(similarities)) if len(similarities) > 1 else 0.0
        ci_95 = 1.96 * (std / np.sqrt(self.config.k_samples)) 

        return self._format_result(score, samples, std, ci_95, cost)

    async def _generate_samples_async(self, context: MetricContext):
        temperature_options = [0.7, 0.8, 0.9, 1.0]
        temps = np.random.choice(temperature_options, self.config.k_samples)

        doc_texts = []
        if context.documents:
            for doc in context.documents:
                extracted = self.service.extract_document(doc)
                doc_texts.append(extracted if extracted else str(doc))

        prompt = context.query
        if doc_texts and len(doc_texts) > 0:
            prompt += "\n\n---DOCUMENTS:---\n\n" + "\n".join(doc_texts)

        tasks = [self.service.llm_call_async(prompt=prompt, temperature=float(temp)) for temp in temps]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_responses = [r["response"] for r in responses if isinstance(r, dict) and r.get("response")]
        cost = sum([r.get("cost", 0.0) for r in responses if isinstance(r, dict)])
        
        return valid_responses, cost

    def _calculate_similarities(self, samples: List[str], context: MetricContext) -> List[float]:
        main_emb = np.atleast_2d(np.array(context.answer_embeddings))
        try:
            sample_embeddings = self.service.embedding_call_batch(samples)['embedding']
        except Exception as e:
            logger.exception(f"Error during embedding call batch: {e}")
            return []

        similarities = []
        for sample_emb_list in sample_embeddings:
            sample_emb = np.atleast_2d(np.array(sample_emb_list))
            if sample_emb is not None and sample_emb.size > 0:
                sim = self.cosine_calc.calculate(main_emb, sample_emb)
                similarities.append(sim)
        return similarities

    def _format_result(self, score: float, samples: List[str], std: float, ci_95: float, cost: float) -> MetricResult:
        label, explanation = self.threshold_evaluator.evaluate_consistency(score)
        return MetricResult(
            score=score, label=label,
            details={"explanation": explanation, "generated_responses": samples, "std_dev": round(std, 2), "uncertainty": round(ci_95, 2)},
            execution_metadata={"total_cost_usd": cost}
        )

    def _create_stable_result(self) -> MetricResult:
        return MetricResult(score=1.0, label=TrustLevel.STABLE.value, details={"explanation": "Assumed stable."}, execution_metadata={"total_cost_usd": 0.0})
    def _create_unreliable_result(self, cost: float) -> MetricResult:
        return MetricResult(score=0.0, label=TrustLevel.UNRELIABLE.value, details={"explanation": "No valid samples."}, execution_metadata={"total_cost_usd": cost})


class SourceDiversityMetric(BaseMetric):
    def calculate(self, context: MetricContext) -> MetricResult:
        if context.documents is None or len(context.documents) == 0:
            return MetricResult(
                score=0.0, label="No Trust", details={"unique_sources": 0}, execution_metadata={"total_cost_usd": 0.0}
            )

        source_identifier = SourceIdentifier()
        source_ids = {
            source_identifier.resolve_source_id(doc, self.service)
            for doc in context.documents
        }

        count = len(source_ids)
        total_docs = len(context.documents)
        
        # Check if low diversity is justified (only 1 relevant doc)
        relevant_docs_count = self._count_relevant_documents(context)
        is_justified_single_source = (count == 1 and relevant_docs_count <= 1)
        
        normalized_score = self._calculate_normalized_score(
            count, total_docs, is_justified_single_source
        )
        label, explanation = self.threshold_evaluator.evaluate_diversity(normalized_score)
        
        # Override explanation if single source is justified
        if is_justified_single_source:
            explanation = "Single source justified: only one document contains relevant information"
            label = "Acceptable"

        return MetricResult(
            score=normalized_score,
            label=label,
            details={
                "explanation": explanation,
                "unique_sources": count,
                "total_documents": total_docs,
                "relevant_documents": relevant_docs_count,
                "justified_single_source": is_justified_single_source,
            },
            execution_metadata={"total_cost_usd": 0.0}
        )

    def _count_relevant_documents(self, context: MetricContext) -> int:
        """Count documents semantically relevant to the query."""
        if context.query is None or context.documents is None or len(context.documents) == 0:
            return 0

        try:
            query_emb = np.atleast_2d(context.query_embeddings)
        except Exception as e:
            logger.exception(f"Error creating query embedding: {e}")
            return 0

        relevance_threshold = 0.5  # Configurable via config if needed
        
        relevant_count = 0
        for doc_emb in context.document_embeddings:
            try:
                doc_emb = np.atleast_2d(doc_emb)
                similarity = self.cosine_calc.calculate(query_emb, doc_emb)
                if similarity >= relevance_threshold:
                    relevant_count += 1
            except Exception as e:
                logger.exception(f"Error calculating similarity for document embedding: {e}")
                continue

        return max(relevant_count, 1)  # At least 1 to avoid division by zero

    @staticmethod
    def _calculate_normalized_score(
        count: int, total: int, is_justified: bool = False
    ) -> float:
        if total == 0:
            return 0.0
        
        # If single source is justified, don't penalize
        if is_justified:
            return 0.8  # High score, but not perfect (room for improvement)
        
        diversity_ratio = count / total
        count_score = 1 - np.exp(-count / 2)
        return 0.6 * diversity_ratio + 0.4 * count_score


class LLMBasedEvidenceStrategy(BaseMetric):

    def calculate(self, context: MetricContext) -> MetricResult:
        if context.documents is None or len(context.documents) == 0:
            return MetricResult(
                score=0.0, label="No Documents", details={"sentences_checked": 0}, execution_metadata={"total_cost_usd": 0.0}
            )

        extracted_docs = [
            self.service.extract_document(doc) for doc in context.documents
        ]
        
        result = self._verify_with_llm(context.query, context.answer, extracted_docs)

        score = (
            result.supported_count / result.total_count
            if result.total_count > 0
            else 0.0
        )
        label, explanation = self.threshold_evaluator.evaluate_grounding(score)

        return MetricResult(
            score=score,
            label=label,
            details={
                "explanation": explanation if result.failed_count == 0 else "",
                "total_sentences": result.total_count,
                "supported_sentences": result.supported_count,
                "unsupported_sentences": result.unsupported_spans,
                "failed_checks": result.failed_count,
                "failed_reason": result.fail_reason,
            },
            execution_metadata={"total_cost_usd": result.cost}
        )

    def _verify_with_llm(
        self, query: str, full_answer: str, extracted_docs: List[str]
    ) -> SpanCheckResult:
        supported = 0
        failed_checks = 0
        fail_reason = None
        unsupported_spans = []
        total_count = 0

        prompt = self._build_prompt(query, full_answer, extracted_docs)

        try:
            # Single LLM call instead of a batch of calls
            response = self.service.llm_call(prompt=prompt, response_format=SpanSchema)
        except Exception as e:
            logger.exception(f"Error calling LLM: {e}")
            return SpanCheckResult(0, [], failed_checks + 1, f"LLM call failed: {e}", 0, 0.0)

        if not response or not response.get("response"):
            return SpanCheckResult(0, [], failed_checks + 1, "LLM call failed or returned empty", 0, response.get("cost", 0.0) if response else 0.0)

        response_content = response["response"]

        try:
            # Parse the JSON array of evaluated sentences
            match = re.search(r'\{.*\}', response_content, re.DOTALL)
            clean_content = match.group(0) if match else response_content
            result_json = json.loads(clean_content)
            spans_result = result_json.get("spans", [])
            
            total_count = len(spans_result)
            
            for span in spans_result:
                if span.get("supported", False):
                    supported += 1
                else:
                    unsupported_spans.append(span.get("sentence", None))

        except Exception as e:
            failed_checks += 1
            fail_reason = f"Parse error: {e}"

        return SpanCheckResult(
            supported_count=supported,
            unsupported_spans=unsupported_spans,
            failed_count=failed_checks,
            fail_reason=fail_reason,
            total_count=total_count,
            cost=response.get("cost", 0.0)
        )

    @staticmethod
    def _build_prompt(query: str, full_answer: str, docs: List[str]) -> str:
        return f"""Evaluate if the provided ANSWER is factually supported by the DOCUMENTS.
        Think thoroughly and reason about the query and evidence in the documents before answering.

        **QUERY:**
        {query}
        =============

        **DOCUMENTS:**
        {docs}
        =============
        
        **FULL ANSWER:**
        {full_answer}
        =============
        
        **INSTRUCTIONS:**
        1. Understand the QUERY to grasp the intent.
        2. Read the FULL ANSWER to understand the context.
        3. Break the ANSWER down sentence by sentence.
        4. For EACH sentence, determine if it is fully supported by the DOCUMENTS.
        5. Return ONLY a JSON object matching this schema: 
           {{"spans": [{{"sentence": "<exact_sentence_string>", "supported": true/false, "reason": "<brief_reason>"}}]}}
        """