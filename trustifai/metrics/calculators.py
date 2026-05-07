# calculators.py
"""Calculator classes for various operations"""

from trustifai.structures import TrustLevel
from trustifai.services import ExternalService
from trustifai.config import Config, SOURCE_ID_FIELDS
import numpy as np
from numpy.linalg import norm
from typing import List, Tuple
import hashlib

class CosineSimCalculator:
    @staticmethod
    def calculate(emb1, emb2) -> float:
        if emb1 is None or emb2 is None:
            raise ValueError("Missing embeddings for similarity computation.")

        emb1 = np.atleast_2d(np.array(emb1))
        emb2 = np.atleast_2d(np.array(emb2))

        dot_product = np.dot(emb1, emb2.T)
        magnitude_vec1 = norm(emb1, axis=1)
        magnitude_vec2 = norm(emb2, axis=1)
        
        if np.any(magnitude_vec1 == 0) or np.any(magnitude_vec2 == 0):
            return 0.0

        similarity_matrix = dot_product / (np.outer(magnitude_vec1, magnitude_vec2))
        similarity = similarity_matrix.item()

        return 0.0 if (np.isnan(similarity) or np.isinf(similarity)) else similarity

class DocumentExtractor:
    def __init__(self, service: ExternalService):
        self.service = service
    
    def extract_all(self, documents: List) -> List[str]:
        return [self.service.extract_document(doc) for doc in documents]
    
    def extract_single(self, document) -> str:
        return self.service.extract_document(document)

class SourceIdentifier:
    @staticmethod
    def resolve_source_id(doc, service: ExternalService) -> str:
        metadata = getattr(doc, "metadata", {}) or {}
        for key in SOURCE_ID_FIELDS:
            if key in metadata and metadata[key]:
                return f"{key}:{metadata[key]}"
        
        content = service.extract_document(doc)
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return f"content_hash:{content_hash}"

class ThresholdEvaluator:
    def __init__(self, config: Config):
        self.thresholds = config.thresholds
    
    def evaluate_grounding(self, score: float) -> Tuple[str, str]:
        if score >= self.thresholds.STRONG_GROUNDING:
            return "Strong Grounding", "Fully supported by source documents."
        elif score >= self.thresholds.PARTIAL_GROUNDING:
            return "Partial Grounding", "Some claims may not be fully supported."
        return "Likely Hallucinated Answer", "Many claims lack support from source documents."
    
    def evaluate_drift(self, score: float) -> Tuple[str, str]:
        if score >= self.thresholds.STRONG_ALIGNMENT:
            return "Strong Alignment", "Answer semantically aligned with source documents."
        elif score >= self.thresholds.PARTIAL_ALIGNMENT:
            return "Partial Alignment", "Some claims may not be aligned with source documents."
        return "Likely Hallucinated Answer", "Answer diverges significantly from source documents."
    
    def evaluate_consistency(self, score: float) -> Tuple[str, str]:
        if score >= self.thresholds.STABLE_CONSISTENCY:
            return TrustLevel.STABLE.value, "Model produces highly consistent responses."
        elif score >= self.thresholds.FRAGILE_CONSISTENCY:
            return TrustLevel.FRAGILE.value, "Model shows some variation but maintains core consistency."
        return TrustLevel.UNRELIABLE.value, "Model produces highly inconsistent responses."
    
    def evaluate_diversity(self, score: float) -> Tuple[str, str]:
        if score >= self.thresholds.HIGH_DIVERSITY:
            return TrustLevel.HIGH.value, "Multiple independent sources used for answer."
        elif score >= self.thresholds.MODERATE_DIVERSITY:
            return TrustLevel.MODERATE.value, "Limited corroboration from multiple sources."
        return TrustLevel.LOW.value, "Single/limited source(s) used for answer."
    
    def evaluate_confidence(self, score: float) -> Tuple[str, str]:
        if score >= self.thresholds.HIGH_CONFIDENCE:
            return "High Confidence", "Model is highly confident in its response based on logprobs."
        elif score >= self.thresholds.MEDIUM_CONFIDENCE:
            return "Medium Confidence", "Model shows moderate uncertainty."
        return "Low Confidence", "Model is uncertain about its output."