# Content Moderation Agent — LangGraph

An agentic AI system that extends the [Japanese NLP Abuse Classifier](https://github.com/adityaladi7/japanese-nlp-classifier) into a full decision-making agent using LangGraph.

## Architecture

```
Input Text
    │
    ▼
[Classifier Node] — calls classify_text() tool
    │
    ▼
[Decision Node] — calls check_policy() tool
    │
    ▼
[Router] ─────────────────────────────────
    │              │               │
    ▼              ▼               ▼
 APPROVE         FLAG          ESCALATE
(publish)   (review queue)  (T&S alert +
                             Jira ticket +
                             Slack notify)
```

## 3 Tools
- `classify_text()` — abuse/sentiment/quality classification
- `check_policy()` — maps classification to required action
- `escalate_to_team()` — routes with ticket creation

## Why LangGraph
Stateful graph, conditional routing, tool calling, full audit trail — easy to extend with new nodes (appeal, re-review) without rewriting.

## Quickstart
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
python agent.py
```

## Extends
[japanese-nlp-classifier](https://github.com/adityaladi7/japanese-nlp-classifier) — 22,000+ records, PySpark, 89% F1
