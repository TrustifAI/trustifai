# online_metrics.py
"""
Online metrics calculated during LLM generation time.
"""

import numpy as np
from typing import List, Dict, Any
from trustifai.metrics.calculators import ThresholdEvaluator
from trustifai.metrics.base import BaseMetric

class ConfidenceMetric(BaseMetric):
    """
    Calculates the confidence of the LLM response using log probabilities 
    captured during the generation process.
    """
    
    @staticmethod
    def calculate(logprobs: List[float], evaluator: ThresholdEvaluator) -> Dict[str, Any]:
        """
        Computes the confidence score from a list of token log probabilities.
        
        Args:
            logprobs: List of log probability values for generated tokens.
            evaluator: ThresholdEvaluator instance to classify the score.

        Returns:
            Dict containing score, label, and detailed explanation.
        """
        if not logprobs:
             return {
                "score": 0.0,
                "label": "N/A",
                "details": {"explanation": "No logprobs available for confidence calculation."}
            }
        
        try:
            # We use mean logprob as a proxy for sequence probability normalized by length
            avg_logprob = np.mean(logprobs)

            # Variance can indicate how consistent the generation was
            variance = np.var(logprobs)
            
            # exp(avg_logprob) gives the geometric mean of probabilities [0, 1]
            logprob_score = float(np.exp(avg_logprob))

            # Penalty for high variance (indicating uncertain or inconsistent generation)
            penalty = np.exp(-variance)

            score = round(logprob_score * penalty, 2)
            
            label, explanation = evaluator.evaluate_confidence(score)
            
            return {
                "score": score,
                "label": label,
                "details": {
                    "explanation": explanation,
                    "avg_logprob": round(avg_logprob, 2),
                    "variance": round(variance, 2),
                    "token_count": len(logprobs),
                }
            }
        except Exception as e:
            return {
                "score": 0.0,
                "label": "Error",
                "details": {"explanation": f"Calculation error: {str(e)}"}
            }