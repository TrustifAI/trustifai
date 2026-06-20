# structures.py
"""
Shared data structures and types.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np
from pydantic import BaseModel, Field

# --- Enums ---

class TrustLevel(Enum):
    """Enumeration for trust levels across metrics"""
    STRONG = "Strong"
    PARTIAL = "Partial"
    WEAK = "Weak"
    RELIABLE = "Reliable"
    ACCEPTABLE = "Acceptable"
    UNRELIABLE = "Unreliable"
    HIGH = "High Trust"
    MODERATE = "Moderate Trust"
    LOW = "Low Trust"
    STABLE = "Stable Consistency"
    FRAGILE = "Fragile Consistency"
    NA = "N/A"

# --- Pydantic Models (Validation) ---

class SpanItem(BaseModel):
    index: int = Field(description="Index of the answer span")
    reason: str = Field(description="brief reason for support/unsupport for span")
    supported: bool = Field(description="true or false")

class SpanSchema(BaseModel):
    spans: List[SpanItem]

# --- Data Classes ---

@dataclass
class MetricContext:
    query: str
    answer: str
    documents: List[Any]
    query_embeddings: np.ndarray = None
    answer_embeddings: np.ndarray = None
    document_embeddings: np.ndarray = None

@dataclass
class MetricResult:
    """Standardized result structure for all metrics"""
    score: float
    label: str
    details: Dict
    execution_metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        res = {
            "score": round(self.score, 2),
            "label": self.label,
            "details": self.details
        }
        if self.execution_metadata:
            res["execution_metadata"] = self.execution_metadata
        return res

@dataclass
class SpanCheckResult:
    reasoning: str
    supported_count: int
    unsupported_spans: List[str]
    failed_count: int
    fail_reason: Optional[str]
    total_count: int
    cost: Optional[float] = None

@dataclass
class RerankerResult:
    mean_score: float
    global_pass: bool
    fully_supported: int
    partially_supported: List[str]
    detailed_results: List[Dict]

# --- Graph Structures ---

@dataclass
class ReasoningNode:
    node_id: str
    node_type: str 
    name: str
    inputs: Dict
    outputs: Dict
    score: Optional[float] = None
    label: Optional[str] = None
    details: Optional[Dict] = None

@dataclass
class ReasoningEdge:
    source: str
    target: str
    relationship: str 

@dataclass
class ReasoningGraph:
    trace_id: str
    nodes: List[ReasoningNode]
    edges: List[ReasoningEdge]

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "nodes": [node.__dict__ for node in self.nodes],
            "edges": [edge.__dict__ for edge in self.edges],
        }