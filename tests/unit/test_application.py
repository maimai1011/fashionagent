import pytest
from langgraph.types import Command

from fashion_agent.application import FashionApplication


class FakeAgent:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, payload, config):
        self.calls.append((payload, config))
        return {"messages": [], "current_outfit": None}


class StatefulFakeAgent(FakeAgent):
    def __init__(self, pending):
        super().__init__()
        self.pending = pending

    async def aget_state(self, _config):
        interrupt = (object(),) if self.pending else ()
        task = type("Task", (), {"interrupts": interrupt})()
        return type("Snapshot", (), {"tasks": (task,)})()


@pytest.mark.asyncio
async def test_clarification_resume_uses_command():
    agent = FakeAgent()
    app = FashionApplication(agent)
    await app.chat("user-1", "thread-1", "navy", resume=True)
    assert isinstance(agent.calls[0][0], Command)
    assert agent.calls[0][1]["configurable"]["thread_id"] == "user-1::thread-1"


@pytest.mark.asyncio
async def test_ordinary_feedback_is_new_message():
    agent = FakeAgent()
    app = FashionApplication(agent)
    await app.chat("user-1", "thread-1", "Đổi áo", resume=False)
    assert agent.calls[0][0]["messages"][0]["content"] == "Đổi áo"


@pytest.mark.asyncio
async def test_pending_clarification_requires_resume_flag():
    app = FashionApplication(StatefulFakeAgent(pending=True))
    with pytest.raises(ValueError, match="resume=true"):
        await app.chat("user-1", "thread-1", "navy", resume=False)


@pytest.mark.asyncio
async def test_session_isolation():
    agent = FakeAgent()
    app = FashionApplication(agent)
    await app.chat("user-a", "same-thread", "hello")
    await app.chat("user-b", "same-thread", "hello")
    ids = [call[1]["configurable"]["thread_id"] for call in agent.calls]
    assert ids == ["user-a::same-thread", "user-b::same-thread"]


@pytest.mark.asyncio
async def test_real_interrupt_and_resume_same_thread():
    from typing import TypedDict
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import START, StateGraph
    from langgraph.types import interrupt

    class State(TypedDict, total=False):
        answer: str

    def clarify(_state):
        return {"answer": interrupt({"question": "Which color?"})}

    builder = StateGraph(State)
    builder.add_node("clarify", clarify)
    builder.add_edge(START, "clarify")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "user::thread"}}
    first = await graph.ainvoke({}, config=config)
    assert first["__interrupt__"][0].value["question"] == "Which color?"
    resumed = await graph.ainvoke(Command(resume="navy"), config=config)
    assert resumed["answer"] == "navy"


@pytest.mark.asyncio
async def test_checkpoint_persists_state_and_isolates_threads():
    import operator
    from typing import Annotated, TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import START, StateGraph

    class State(TypedDict):
        values: Annotated[list[str], operator.add]

    async def append_value(state):
        return {"values": [state["values"][-1].upper()]}

    builder = StateGraph(State)
    builder.add_node("append", append_value)
    builder.add_edge(START, "append")
    graph = builder.compile(checkpointer=InMemorySaver())
    first = {"configurable": {"thread_id": "user-a::thread"}}
    second = {"configurable": {"thread_id": "user-b::thread"}}
    await graph.ainvoke({"values": ["a"]}, config=first)
    await graph.ainvoke({"values": ["b"]}, config=second)
    first_state = await graph.aget_state(first)
    second_state = await graph.aget_state(second)
    assert first_state.values["values"] == ["a", "A"]
    assert second_state.values["values"] == ["b", "B"]
