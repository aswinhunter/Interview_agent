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

    # Candidate answer
    current_answer: str
    answer_history: List[str]

    # Answer evaluation
    score: float
    feedback: str
    strengths: List[str]
    weaknesses: List[str]

    # Overall interview score
    scores: List[float]
    session_score: float