import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from agent.agents.intent.agent import IntentDecisionLogger
from agent.agents.intent.prompt import INTENT_PROMPT
from agent.agents.intent.tools import build_tools
from agent.agents.stylist.prompt import STYLIST_PROMPT


def test_prompt_delegates_styling_and_forbids_fabrication():
    assert "not a stylist" in INTENT_PROMPT
    assert "Never invent item IDs" in INTENT_PROMPT
    assert "search_wardrobe" in INTENT_PROMPT
    assert "count_wardrobe" in INTENT_PROMPT
    assert "list_wardrobe" in INTENT_PROMPT
    assert "only completed items" in INTENT_PROMPT
    assert "restore_previous_outfit" in INTENT_PROMPT
    assert "select_wardrobe_item" in INTENT_PROMPT
    assert "Never tell the user to wait" in INTENT_PROMPT
    assert "most recent user message" in INTENT_PROMPT
    assert "Reply in English to an English" in INTENT_PROMPT
    assert "language of REQUEST.user_message" in STYLIST_PROMPT


def test_search_tool_exposes_only_top_k_five_to_model():
    search_tool = next(tool for tool in build_tools(None, None) if tool.name == "search_wardrobe")
    schema = search_tool.tool_call_schema
    top_k = schema.model_json_schema()["properties"]["top_k"]
    assert top_k["default"] == 5
    assert top_k["minimum"] == 5
    assert top_k["maximum"] == 5
    schema.model_validate({"top_k": 5})
    with pytest.raises(ValidationError):
        schema.model_validate({"top_k": 100})


def test_intent_logs_decision_without_content_by_default(monkeypatch, caplog):
    monkeypatch.delenv("AGENT_LOG_CONTENT", raising=False)
    middleware = IntentDecisionLogger()
    state = {
        "messages": [
            HumanMessage(content="private wardrobe request"),
            AIMessage(content="", tool_calls=[{
                "name": "consult_stylist",
                "args": {"feedback": "private feedback"},
                "id": "call-1",
            }]),
        ],
        "outfit_version": 2,
    }

    with caplog.at_level(logging.INFO):
        middleware.before_model(state, None)
        middleware.after_model(state, None)

    assert "intent_model_start" in caplog.text
    assert "intent_decision outcome=tool_calls" in caplog.text
    assert "consult_stylist" in caplog.text
    assert "private wardrobe request" not in caplog.text
    assert "private feedback" not in caplog.text


def test_intent_can_log_content_when_enabled(monkeypatch, caplog):
    monkeypatch.setenv("AGENT_LOG_CONTENT", "true")
    middleware = IntentDecisionLogger()
    state = {
        "messages": [
            HumanMessage(content="Suggest a blue outfit"),
            AIMessage(content="", tool_calls=[{
                "name": "consult_stylist",
                "args": {"feedback": "make it blue"},
                "id": "call-1",
            }]),
        ],
    }

    with caplog.at_level(logging.INFO):
        middleware.before_model(state, None)
        middleware.after_model(state, None)

    assert "intent_model_input latest_user=\"Suggest a blue outfit\"" in caplog.text
    assert "intent_model_output" in caplog.text
    assert "make it blue" in caplog.text
