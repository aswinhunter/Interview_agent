from typing import List, TypedDict


class AgentPrepState(TypedDict, total=False):

    # Resume input
    resume_path: str

    # Resume information
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

    # Candidate answers
    current_answer: str
    answer_history: List[str]

    # Answer evaluation
    score: float
    feedback: str
    strengths: List[str]
    weaknesses: List[str]

    # Overall scores
    scores: List[float]
    session_score: float

    # Follow-up / next topic
    topic_history: List[str]
    next_action: str
    next_topic: str
    follow_up_count: int
    
    # --------------------------------------------------
    # Interview configuration
    # --------------------------------------------------

    difficulty_level: int
    max_questions: int
    question_count: int