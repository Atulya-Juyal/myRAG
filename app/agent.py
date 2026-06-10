"""
Production-Ready FastAPI + LangGraph Agent with isolated RAG.
"""

from typing import Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langsmith import traceable

from app.config import get_settings
from app.rag import RAGManager


class AgentState(TypedDict):
    """
    State for the production agent.
    Uses Annotated with add_messages reducer for message accumulation.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str
    chat_id: str
    context: list[dict]


class ProductionAgent:
    """
    Production LangGraph agent with:
    - isolated workspace document retrieval (RAG)
    - Retry on failure (model fallback)
    - Graceful error handling
    - LangSmith tracing
    """

    def __init__(self):
        settings = get_settings()

        # Initialize the primary Gemini model
        self.primary_llm = ChatGoogleGenerativeAI(
            model=settings.primary_llm,
            temperature=0,
            timeout=30,
            max_retries=0,  # We handle retries ourselves via graph routing
            api_key=settings.gemini_api_key,
        )

        # Initialize the fallback Gemini model
        self.fallback_llm = ChatGoogleGenerativeAI(
            model=settings.fallback_llm,
            temperature=0,
            timeout=30,
            max_retries=0,
            api_key=settings.gemini_api_key,
        )

        self.max_retries = settings.max_retries
        self.rag_manager = RAGManager()
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Build the LangGraph state machine.
        """

        def retrieve_context(state: AgentState) -> dict:
            """
            Retrieve relevant document chunks for the user's message.
            """
            last_message = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_message = msg.content
                    break
            if not last_message:
                last_message = state["messages"][-1].content

            chunks = self.rag_manager.retrieve(state["chat_id"], last_message, k=4)
            return {"context": chunks}

        def process_message(state: AgentState) -> dict:
            """
            Try to process the message with the primary model.
            """
            try:
                messages = list(state["messages"])
                context_list = state.get("context", [])
                
                if context_list:
                    context_text = "\n\n".join([
                        f"Source: {c['source']} (Match Confidence: {c['score']}%)\nContent: {c['content']}"
                        for c in context_list
                    ])
                    system_prompt = (
                        "You are a helpful assistant. Use the following retrieved context to answer the user's question. "
                        "If the context doesn't contain the answer, answer based on your knowledge but indicate that the answer wasn't found in the documents.\n\n"
                        f"Retrieved Context:\n{context_text}"
                    )
                    messages.insert(0, SystemMessage(content=system_prompt))

                response = self.primary_llm.invoke(messages)
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "primary",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        def try_fallback(state: AgentState) -> dict:
            """
            Fallback to secondary model.
            """
            try:
                messages = list(state["messages"])
                context_list = state.get("context", [])
                
                if context_list:
                    context_text = "\n\n".join([
                        f"Source: {c['source']} (Match Confidence: {c['score']}%)\nContent: {c['content']}"
                        for c in context_list
                    ])
                    system_prompt = (
                        "You are a helpful assistant. Use the following retrieved context to answer the user's question. "
                        "If the context doesn't contain the answer, answer based on your knowledge but indicate that the answer wasn't found in the documents.\n\n"
                        f"Retrieved Context:\n{context_text}"
                    )
                    messages.insert(0, SystemMessage(content=system_prompt))

                response = self.fallback_llm.invoke(messages)
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "fallback",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "model_used": "",
                }

        def handle_error(state: AgentState) -> dict:
            """
            Return a graceful error message.
            """
            return {
                "messages": [
                    AIMessage(
                        content="I'm sorry, I'm having trouble processing your request right now. Please try again in a moment."
                    )
                ],
                "model_used": "error_handler",
            }

        def route_after_process(state: AgentState) -> str:
            """
            Decide what to do after primary model attempt.
            """
            if state.get("error") is None:
                return "done"
            elif state["retry_count"] < self.max_retries:
                return "fallback"
            else:
                return "error"

        def route_after_fallback(state: AgentState) -> str:
            """
            Decide what to do after fallback attempt.
            """
            if state.get("error") is None:
                return "done"
            else:
                return "error"

        # Build the graph topology
        graph = StateGraph(AgentState)

        # Add executable pipeline nodes
        graph.add_node("retrieve", retrieve_context)
        graph.add_node("process", process_message)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)

        # Connect graph entry edge
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "process")

        # Set up dynamic conditional routing criteria
        graph.add_conditional_edges(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"},
        )

        graph.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {"done": END, "error": "error"},
        )

        # Connect error handling node to complete workflow sequence
        graph.add_edge("error", END)

        return graph.compile()

    @traceable(name="production_agent_invoke")
    def invoke(self, message: str, chat_id: str = "default") -> dict:
        """
        Invoke the agent with a user message.
        """
        # Load history
        history = self.rag_manager.load_history(chat_id)
        
        # Build messages list starting with history
        messages = []
        for msg in history:
            if msg.get("sender") == "user":
                messages.append(HumanMessage(content=msg["text"]))
            elif msg.get("sender") == "bot":
                messages.append(AIMessage(content=msg["text"]))
                
        # Append current message
        messages.append(HumanMessage(content=message))

        result = self.graph.invoke({
            "messages": messages,
            "error": None,
            "retry_count": 0,
            "model_used": "",
            "chat_id": chat_id,
            "context": [],
        })

        return {
            "response": result["messages"][-1].content,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
            "sources": result.get("context", []),
        }