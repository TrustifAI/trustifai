# TrustifAI 
**🛡️Quantify, Visualize, and Explain Trust in AI.**

TrustifAI is a Python-based framework designed to evaluate the trustworthiness of LLM responses and Retrieval-Augmented Generation (RAG) systems. Unlike simple evaluation frameworks that rely on a single "correctness" score, TrustifAI computes a multi-dimensional **Trust Score** based on grounding, consistency, alignment, and diversity.

It also includes **visualizations** to help showcase why a model output was deemed unreliable.

![Build Status](https://github.com/TrustifAI/trustifai/actions/workflows/run-tests.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/trustifai.svg?icon=si%3Apython)](https://badge.fury.io/py/trustifai)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/trustifai?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/trustifai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?color=green)](https://opensource.org/licenses/MIT)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TrustifAI/trustifai)

## 📊 Key Metrics

**TrustifAI** evaluates trustworthiness using four orthogonal vectors or trust signals. The final *Trust Score* is a weighted aggregation of these components.

### Offline Metrics (For already generated RAG response)

| Metric | Definition | Purpose |
|------|------------|---------|
| **Evidence Coverage** | Segment-level entailment check. The answer is tokenized into sentences and each sentence is verified against retrieved documents using an NLI (Natural Language Inference) approach. | Detects hallucinations. Ensures every claim is supported by the provided context. |
| **Epistemic Consistency** | Measures semantic stability ($1 - \sigma$) across $k$ stochastic generations. Samples $k$ responses at high temperature and computes the mean cosine similarity against the original answer. | Detects model inconsistency. Hallucinated answers tend to vary significantly between runs. |
| **Semantic Drift** | Sentence-level maximum similarity between the Answer/Query embedding and embeddings of sentences from the retrieved documents. Also returns the best-matching document sentence(s). | Detect topic drift and alignment with source documents; low scores indicate the answer diverges from the provided documents (possible hallucination). |
| **Source Diversity** | Normalized count of distinct source_id references contributing to the answer, adjusted using an exponential decay penalty. | Measures reliance on a single source while rewarding synthesis across multiple independent sources, without excessively penalizing cases where a single document is sufficient.

### Online Metrics (For Real-time response generation)

`Applicable only for LLMs which supports logprobs. Might not be a good metric in case LLM is not calibrated enough. Newer LLMs are confident enough in hallucinated responses also, might not be suitable for such cases.`

| Metric | Definition | Purpose |
|------|------------|---------|
| **Confidence Score** | Calculated using the log probabilities (logprobs) of the generated tokens. It considers the geometric mean of probabilities penalized by the variance of the generation. | Provides a real-time confidence signal (0.0−1.0) indicating how sure the model is about its own output.
 
## 🚀 Installation

TrustifAI requires Python 3.10+.

```python
pip install trustifai

#to enable tracing
pip install trustifai[trace]

# for tests
pip install trustifai[test]
```

OR

```
# Clone the repository
git clone https://github.com/Trustifai/trustifai.git
cd trustifai

# Install dependencies
pip install -r requirements.txt
```

## Environment Setup
Create a .env file or export your API keys. TrustifAI uses LiteLLM, so it supports OpenAI, Azure, Anthropic, Gemini, Mistral, and more. (check .env.example)

## ⚡ Quick Start

**1. Evaluate an existing RAG Response in a few lines of code.**

  `Use this flow to score a query/answer pair against retrieved documents.`

```python
from trustifai import Trustifai, MetricContext
from langchain_core.documents import Document #langchain is not required, used here for demo only.

# 1. Define your RAG Context
context = MetricContext(
    query="What is the capital of India?",
    answer="The capital is New Delhi.",
    documents=[
        Document(page_content="New Delhi is the capital of India.", metadata={"source": "wiki.txt"})
    ]
)

# 2. Initialize Engine
trust_engine = Trustifai("config_file.yaml")

# 3. Calculate Score
result = trust_engine.get_trust_score(context)
print(f"Trust Score: {result['score']} | Decision: {result['label']}")

# 4. Visualize Logic
graph = trust_engine.build_reasoning_graph(result)
trust_engine.visualize(graph, graph_type="pyvis") # Saves to reasoning_graph.html
```
![alt text](assets/trust_score_snippet.png)

**2. Generate with Confidence**

  `Use TrustifAI to generate a response and immediately get a confidence score based on token log probabilities.`

```python
from trustifai import Trustifai

# Initialize (Context will be None for pure generation)
trust_engine = Trustifai(config_path="config_file.yaml")

# Generate response
result = trust_engine.generate(
    prompt="What is the capital of France?",
    system_prompt="You are a helpful assistant."
)

print(f"Response: {result['response']}")
print(f"Confidence: {result['metadata']['confidence_score']} ({result['metadata']['confidence_label']})")
```

![alt text](assets/generate_snippet.png)

## ⚙️ Configuration
Control the sensitivity of the evaluation using config_file.yaml.

```python
#custom config can be passed on using config_path
trust_engine = Trustifai(config_path="config_file.yaml")
```
Refer: [config_file.yaml](config_file.yaml)

```yaml
# 1. Model Configuration (via LiteLLM)
llm:
  type: "openai"
  params:
    model_name: "gpt-5"

# 2. Thresholds (Strictness)
metrics:
  - type: "evidence_coverage"
    enabled: true #any metric can be disabled (if not suitable for your use-case)
    params:
      STRONG_GROUNDING: 0.85 # Threshold for "Trusted" label
      PARTIAL_GROUNDING: 0.60
  - type: "consistency"
    enabled: true
    params:
      STABLE_CONSISTENCY: 0.90 # Requires 0.9 cosine sim to be "Stable"

# 3. Weighted Aggregation
# Adjust these based on your business priority.
score_weights:
  - type: "evidence_coverage"
    params: { weight: 0.40 } # Highest priority on factual accuracy
  - type: "semantic_drift"
    params: { weight: 0.30 }
  - type: "consistency"
    params: { weight: 0.20 }
  - type: "source_diversity"
    params: { weight: 0.10 }
```


## 🕸️ Reasoning Graphs

TrustifAI doesn't just give you a number; it gives you a map. The Reasoning Graph is a directed acyclic graph (DAG) representing the evaluation logic.
- Nodes: Represent individual metrics (Green=High Trust, Red=Low Trust).
- Edges: Represent the flow of data into the final aggregation.
- Interactive: The generated HTML uses PyVis for physics-based interaction.

To generate a graph:
```python
# Generate interactive HTML
trust_engine.visualize(graph, graph_type="pyvis")
```
![reasoning graph](assets/graph_gif.gif)

```python
# Generate Mermaid syntax for markdown documentation
print(trust_engine.visualize(graph, graph_type="mermaid"))
```
![mermaid diagram](assets/image-1.png)

## 🧩 Extending TrustifAI (Custom Metrics)

You can plug in custom evaluation logic without modifying the core library.

- Inherit from BaseMetric and implement calculate().

- Register the metric with a unique key.

- Configure the weight in your YAML file.

*Example: Adding a "Temporal Consistency" Metric*
```python
from trustifai.metrics import BaseMetric
from trustifai.structures import MetricResult

# 1. Define Metric
class TemporalConsistencyMetric(BaseMetric):
    """Detects temporal hallucinations - when the answer references dates/times
    that don't match the retrieved documents."""
    def calculate(self) -> MetricResult:
        # Extract dates from answer and documents
        answer_dates = self._extract_dates(self.context.answer) #assuming extract_dates logic is already implemented
        doc_dates = set()
        for doc in self.context.documents:
            doc_dates.update(self._extract_dates(doc.page_content))
        
        if not answer_dates:
            return MetricResult(
                score=1.0,
                label="No Temporal Claims",
                details={"answer_dates": [], "doc_dates": list(doc_dates)}
            )
        
        # Check if answer dates are within document date ranges
        supported_dates = [d for d in answer_dates if d in doc_dates]
        unsupported_dates = [d for d in answer_dates if d not in doc_dates]
        
        score = len(supported_dates) / len(answer_dates) if answer_dates else 1.0
        
        high_threshold = getattr(self.config.thresholds, 'TEMPORALLY_CONSISTENT', 0.8)
        low_threshold = getattr(self.config.thresholds, 'PARTIAL_TEMPORAL_ISSUES', 0.5)
        
        if score >= high_threshold:
            label = "Temporally Consistent"
        elif score >= low_threshold:
            label = "Partial Temporal Issues"
        else:
            label = "Temporal Hallucination Detected"
        
        return MetricResult(
            score=score,
            label=label,
            details={
                "answer_dates": answer_dates,
                "supported_dates": supported_dates,
                "unsupported_dates": unsupported_dates,
                "doc_dates": list(doc_dates)
            }
        )

# 2. Register Metric
from Trustifai import Trustifai
Trustifai.register_metric("temporal_consistency", TemporalConsistencyMetric)

# 3. Use in Trust Engine (Make sure to add it to config.yaml score_weights!)
trust_engine = Trustifai("config_file.yaml")
trust_score = trust_engine.get_trust_score(context)
```

*Updated config.yaml:*
```yaml
metrics:
- type: "evidence_coverage"
    enabled: true
    params:

    ... # ... existing metrics ...
    
- type: "temporal_consistency"         # <--- Your new metric
    enabled: true
    params: 
      TEMPORALLY_CONSISTENT: 0.80
      PARTIAL_TEMPORAL_ISSUES: 0.50 

score_weights:
  - type: "evidence_coverage"
    params: { weight: 0.4 }
    # ... other existing metrics ...
  - type: "temporal_consistency"         # <--- Your new metric
    params: { weight: 0.1 }   # Weights must sum to ~1.0
```

## 🛠️ Flow

```
Input (Query, Answer, Documents)
    ↓
Validation & Type Normalization
    ├─ Accepts: Strings, LangChain/LlamaIndex Documents, Lists, Dicts
    └─ Returns: MetricContext
    ↓
Embedding Computation (Cached or Computed)
    ├─ Query embedding
    ├─ Answer embedding
    └─ Document embeddings
    ↓
Initialize Active Metrics
    ├─ Evidence Coverage Metric
    ├─ Semantic Drift Metric
    ├─ Epistemic Consistency Metric
    ├─ Source Diversity Metric
    └─ Confidence Metric (real-time only for response generation)
    ↓
Parallel Metric Execution
    ├─ Each metric: calculate(context) → MetricResult
    └─ Async-safe: Thread isolation for event loop conflicts
    ↓
Result Aggregation & Thresholding
    ├─ Weighted sum: Σ(weight_i × metric_score_i)
    ├─ Threshold classification: RELIABLE | ACCEPTABLE | UNRELIABLE
    └─ Cost accumulation
    ↓
Reasoning Graph Construction
    ├─ Nodes: Metrics, aggregation, decision
    ├─ Edges: Data flow relationships
    └─ Color-coded by confidence: green ≥ STRONG, orange ≥ PARTIAL, red < PARTIAL
    ↓
Output: TrustScore Result + Visualization
```


## 🎯 Benchmarks
- [Amnesty QA](benchmarks/amnesty_qa/benchmark_report.md)

## TODO
- [ ] Improve Tracing
- [ ] Support for GraphRAG
