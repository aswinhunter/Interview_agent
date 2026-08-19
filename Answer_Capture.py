from State import AgentPrepState
from Speech_To_Text import speech_to_text


def answer_capture_agent(state: AgentPrepState) -> dict:
    """
    Captures the candidate's spoken answer
    and stores the transcript in LangGraph state.
    """

    

    answer = speech_to_text()

    print("\nCandidate Answer:")
    print(answer)

    return {
        "current_answer": answer
    }