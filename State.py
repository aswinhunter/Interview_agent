from typing import List, TypedDict


class AgentPrepState(TypedDict, total=False):
    # Resume input
    resume_path: str

    # Information extracted from resume
    skills: List[str]
    project_summary: str
    experience_level: str
    domain: str

    # Interview configuration
    difficulty_level: int

    # Question generation
    question_history: List[str]
    current_question: str
    current_topic: str
    current_difficulty: str

    # Future answer evaluation
    current_answer: str
    scores: List[float]
    session_score: float