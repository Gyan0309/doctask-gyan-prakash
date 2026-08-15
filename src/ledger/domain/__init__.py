"""Domain logic — deliberately free of FastAPI, LangGraph and transport concerns.

Everything here is directly testable with a session and nothing else, which is why the
hardest guarantees (citation resolution, dependency-aware composition) live at this
layer rather than inside graph nodes.
"""
