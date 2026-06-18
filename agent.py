"""
Content Moderation Agent — LangGraph
====================================
An agentic system that extends the Japanese NLP Abuse Classifier into a
full decision-making agent. The agent:

1. Receives raw text input
2. Calls a classifier tool to detect abuse/sentiment/quality
3. Calls an escalation decision tool based on severity
4. Routes to the correct handler: auto-approve / flag for review / escalate
5. Logs the decision with full audit trail

Architecture:
    Input → [Classifier Node] → [Decision Node] → [Router Node] → Output
                                        ↑
                              Tools: classify_text, check_policy, escalate

This directly extends: github.com/adityaladi7/japanese-nlp-classifier
"""

import json
import os
from typing import TypedDict, Annotated, Literal
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool


# ── State ─────────────────────────────────────────────────────────────────────

class ModerationState(TypedDict):
    """Shared state passed between all nodes in the graph."""
    input_text: str                          # Raw text to moderate
    messages: list                           # LLM conversation history
    classification: dict                     # Output from classifier tool
    decision: str                            # approve | flag | escalate
    reason: str                              # Human-readable explanation
    audit_log: list                          # Full decision trail
    iteration_count: int                     # Safety counter


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def classify_text(text: str) -> dict:
    """
    Classify input text for abuse, sentiment, and content quality.
    Simulates the Japanese NLP classifier pipeline output.
    In production: calls the actual PySpark classifier via FastAPI endpoint.

    Args:
        text: Raw input text to classify

    Returns:
        Classification result with severity tier and harm taxonomy
    """
    # Production: this would call → http://classifier-api/predict
    # For demo: rule-based simulation matching real classifier output schema

    text_lower = text.lower()

    # Abuse signals
    abuse_keywords = ["hate", "kill", "stupid", "idiot", "garbage", "terrible",
                      "awful", "worst", "disgusting", "harassment", "threat"]
    abuse_score = sum(1 for kw in abuse_keywords if kw in text_lower) / len(abuse_keywords)

    # Negative sentiment signals
    negative_keywords = ["bad", "poor", "wrong", "fail", "broken", "slow",
                         "useless", "disappointed", "frustrated", "annoyed"]
    neg_score = sum(1 for kw in negative_keywords if kw in text_lower) / len(negative_keywords)

    # Adversarial text detection (from the original classifier)
    has_fullwidth = any(ord(c) > 0xFF00 for c in text)
    has_mixed_script = any(ord(c) > 0x3000 for c in text) and any(c.isascii() for c in text)

    # Determine class and severity
    if abuse_score > 0.05:
        predicted_class = "abusive_content"
        confidence = min(0.95, 0.7 + abuse_score * 2)
        severity = "CRITICAL" if abuse_score > 0.1 else "HIGH"
    elif neg_score > 0.05:
        predicted_class = "negative_sentiment"
        confidence = min(0.92, 0.65 + neg_score * 2)
        severity = "MEDIUM" if neg_score > 0.1 else "LOW"
    else:
        predicted_class = "clean"
        confidence = 0.91
        severity = "NONE"

    return {
        "text_preview": text[:100] + "..." if len(text) > 100 else text,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 3),
        "severity_tier": severity,
        "adversarial_flags": {
            "fullwidth_detected": has_fullwidth,
            "code_switching_detected": has_mixed_script,
        },
        "taxonomy": {
            "category": "Direct Harassment" if predicted_class == "abusive_content" else
                        "Negative Feedback" if predicted_class == "negative_sentiment" else
                        "Clean Content",
            "subcategory": "Personal Attack" if abuse_score > 0.1 else "General",
        },
        "model_version": "japanese-nlp-classifier-v1.2",
        "classifier_endpoint": "http://classifier-api/predict"  # production endpoint
    }


@tool
def check_policy(predicted_class: str, severity: str, confidence: float) -> dict:
    """
    Check content against moderation policy rules.
    Determines required action based on classification output.

    Args:
        predicted_class: Class from classifier (abusive_content, negative_sentiment, clean)
        severity: Severity tier (CRITICAL, HIGH, MEDIUM, LOW, NONE)
        confidence: Model confidence score (0-1)

    Returns:
        Policy decision with required action and SLA
    """
    policy_rules = {
        ("abusive_content", "CRITICAL"): {
            "action": "escalate",
            "sla_hours": 1,
            "notify_team": "trust_and_safety",
            "auto_hide": True,
            "reason": "Critical abuse detected — immediate human review required"
        },
        ("abusive_content", "HIGH"): {
            "action": "escalate",
            "sla_hours": 4,
            "notify_team": "trust_and_safety",
            "auto_hide": True,
            "reason": "High-severity abuse — escalate to T&S team within 4 hours"
        },
        ("negative_sentiment", "MEDIUM"): {
            "action": "flag",
            "sla_hours": 24,
            "notify_team": "content_review",
            "auto_hide": False,
            "reason": "Negative sentiment flagged for review — no immediate action"
        },
        ("negative_sentiment", "LOW"): {
            "action": "flag",
            "sla_hours": 72,
            "notify_team": "content_review",
            "auto_hide": False,
            "reason": "Low-risk negative content — logged for batch review"
        },
        ("clean", "NONE"): {
            "action": "approve",
            "sla_hours": None,
            "notify_team": None,
            "auto_hide": False,
            "reason": "Content passes moderation — approved for publication"
        }
    }

    # Low confidence → always flag regardless of class
    if confidence < 0.75:
        return {
            "action": "flag",
            "sla_hours": 24,
            "notify_team": "content_review",
            "auto_hide": False,
            "reason": f"Low model confidence ({confidence}) — human review required"
        }

    key = (predicted_class, severity)
    return policy_rules.get(key, {
        "action": "flag",
        "sla_hours": 24,
        "notify_team": "content_review",
        "auto_hide": False,
        "reason": "Unknown classification — defaulting to manual review"
    })


@tool
def escalate_to_team(team: str, content_summary: str, severity: str, sla_hours: int) -> dict:
    """
    Route escalation to the appropriate team with full context.
    In production: sends Slack alert + creates Jira ticket + logs to audit DB.

    Args:
        team: Target team (trust_and_safety, content_review)
        content_summary: Brief summary of the flagged content
        severity: Severity tier for prioritisation
        sla_hours: Required response time in hours

    Returns:
        Escalation confirmation with ticket ID and routing details
    """
    ticket_id = f"MOD-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Production: POST to internal ticketing API
    # requests.post("https://internal-api/tickets", json={...})

    return {
        "ticket_id": ticket_id,
        "routed_to": team,
        "severity": severity,
        "sla_deadline": f"{sla_hours}h from now",
        "content_summary": content_summary,
        "channels_notified": ["slack", "email", "jira"],
        "status": "escalation_created",
        "timestamp": datetime.now().isoformat()
    }


# ── Nodes ─────────────────────────────────────────────────────────────────────

TOOLS = [classify_text, check_policy, escalate_to_team]

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1000)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are a content moderation agent for a Japanese-language platform.
Your job is to moderate user-generated content using a structured pipeline:

1. First, call classify_text() with the input text
2. Then call check_policy() using the classification results  
3. If action is "escalate", call escalate_to_team() with full context
4. Finally, summarise your decision clearly

Always use the tools in order. Never skip classification.
Be precise and objective — base all decisions on tool outputs, not assumptions."""


def classifier_node(state: ModerationState) -> ModerationState:
    """Node 1: LLM decides to call classify_text tool."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Please moderate this content: {state['input_text']}")
    ]
    response = llm_with_tools.invoke(messages)
    state["messages"] = messages + [response]
    state["audit_log"].append({
        "node": "classifier_node",
        "timestamp": datetime.now().isoformat(),
        "action": "invoked_llm_with_classify_tool"
    })
    return state


def decision_node(state: ModerationState) -> ModerationState:
    """Node 2: After tool results, LLM makes policy decision."""
    response = llm_with_tools.invoke(state["messages"])
    state["messages"].append(response)
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    state["audit_log"].append({
        "node": "decision_node",
        "timestamp": datetime.now().isoformat(),
        "iteration": state["iteration_count"]
    })
    return state


def final_node(state: ModerationState) -> ModerationState:
    """Node 3: Extract final decision and generate audit summary."""
    # Get last AI message text
    last_msg = state["messages"][-1]
    final_text = last_msg.content if isinstance(last_msg.content, str) else \
                 " ".join(b.get("text", "") for b in last_msg.content if isinstance(b, dict))

    # Parse decision keyword from response
    text_lower = final_text.lower()
    if "escalat" in text_lower:
        state["decision"] = "escalate"
    elif "flag" in text_lower:
        state["decision"] = "flag"
    else:
        state["decision"] = "approve"

    state["reason"] = final_text
    state["audit_log"].append({
        "node": "final_node",
        "timestamp": datetime.now().isoformat(),
        "final_decision": state["decision"]
    })
    return state


# ── Routing ───────────────────────────────────────────────────────────────────

def should_continue(state: ModerationState) -> Literal["tools", "decision", "final"]:
    """Router: decides next node based on last message content."""
    last_msg = state["messages"][-1]

    # Safety limit
    if state.get("iteration_count", 0) >= 5:
        return "final"

    # If LLM called tools → execute them
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    # If we have a classification result → move to decision
    messages_text = str(state["messages"])
    if "predicted_class" in messages_text and "check_policy" not in messages_text:
        return "decision"

    # Otherwise wrap up
    return "final"


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_agent():
    graph = StateGraph(ModerationState)

    tool_node = ToolNode(TOOLS)

    graph.add_node("classifier", classifier_node)
    graph.add_node("tools", tool_node)
    graph.add_node("decision", decision_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("classifier")

    graph.add_conditional_edges("classifier", should_continue, {
        "tools": "tools",
        "decision": "decision",
        "final": "final"
    })
    graph.add_conditional_edges("tools", should_continue, {
        "tools": "tools",
        "decision": "decision",
        "final": "final"
    })
    graph.add_conditional_edges("decision", should_continue, {
        "tools": "tools",
        "decision": "decision",
        "final": "final"
    })
    graph.add_edge("final", END)

    return graph.compile()


# ── Run ───────────────────────────────────────────────────────────────────────

def moderate(text: str, verbose: bool = True) -> dict:
    """
    Run the moderation agent on a single piece of text.

    Args:
        text: Content to moderate
        verbose: Print step-by-step output

    Returns:
        Final moderation result with decision and audit trail
    """
    agent = build_agent()

    initial_state = ModerationState(
        input_text=text,
        messages=[],
        classification={},
        decision="",
        reason="",
        audit_log=[],
        iteration_count=0
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"INPUT: {text[:80]}...")
        print(f"{'='*60}")

    result = agent.invoke(initial_state)

    if verbose:
        print(f"\nDECISION: {result['decision'].upper()}")
        print(f"AUDIT STEPS: {len(result['audit_log'])}")
        print(f"\nREASON:\n{result['reason'][:400]}")
        print(f"\nAUDIT LOG:")
        for entry in result["audit_log"]:
            print(f"  [{entry['timestamp']}] {entry['node']} → {entry.get('action') or entry.get('final_decision', '')}")

    return {
        "decision": result["decision"],
        "reason": result["reason"],
        "audit_log": result["audit_log"],
        "input_text": text
    }


if __name__ == "__main__":
    # Test cases matching real-world moderation scenarios
    test_cases = [
        "This product is absolutely terrible and I hate everyone who made it. Garbage!",
        "The response time is slow and the UI is confusing. Disappointed with this.",
        "Great experience overall, the assistant was very helpful and accurate.",
    ]

    print("CONTENT MODERATION AGENT — LangGraph Demo")
    print("Extending: github.com/adityaladi7/japanese-nlp-classifier\n")

    for text in test_cases:
        result = moderate(text, verbose=True)
        print(f"\n{'─'*60}\n")
