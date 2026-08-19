
from State import AgentPrepState
def route_after_decision(state: AgentPrepState):

    question_count = state.get(
        "question_count",
        0
    )

    max_questions = state.get(
        "max_questions",
        5
    )

    if question_count >= max_questions:
        return "end"

    return "continue"