import json

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from State import AgentPrepState


# Load .env
load_dotenv()


# Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)


# --------------------------------------------------
# Difficulty mapping
# --------------------------------------------------

DIFFICULTY_MAP = {
    1: "Easy",
    2: "Medium",
    3: "Hard"
}


# --------------------------------------------------
# Difficulty instructions
# --------------------------------------------------

DIFFICULTY_GUIDANCE = {

    "Easy":
        """
Ask a straightforward conceptual question.
It should be suitable for a fresher.
Do not require deep implementation details.
""",

    "Medium":
        """
Ask a question that requires understanding and
some practical knowledge.

The candidate may need to explain:
- a design decision
- implementation
- tradeoffs
- how a technology works
""",

    "Hard":
        """
Ask a deep technical question.

Probe:
- internals
- edge cases
- architecture
- performance
- tradeoffs
- implementation decisions
"""
}


def question_generator_agent(state: AgentPrepState) -> dict:
    """
    LangGraph node.

    Reads from state:
        skills
        project_summary
        domain
        difficulty_level
        question_history

    Writes to state:
        current_question
        current_topic
        current_difficulty
        question_history
    """

    # --------------------------------------------------
    # 1. Read information from shared state
    # --------------------------------------------------

    project_summary = state.get(
        "project_summary",
        ""
    )

    skills = state.get(
        "skills",
        []
    )

    domain = state.get(
        "domain",
        "DSA"
    )

    difficulty_level = state.get(
        "difficulty_level",
        1
    )

    question_history = state.get(
        "question_history",
        []
    )

    # --------------------------------------------------
    # 2. Convert difficulty number to text
    # --------------------------------------------------

    target_difficulty = DIFFICULTY_MAP.get(
        difficulty_level,
        "Easy"
    )

    difficulty_guidance = DIFFICULTY_GUIDANCE[
        target_difficulty
    ]

    # --------------------------------------------------
    # 3. Prepare previous questions
    # --------------------------------------------------

    if question_history:

        previous_questions = "\n".join(
            f"- {question}"
            for question in question_history[-5:]
        )

        avoid_section = f"""
Do NOT ask any of these questions again:

{previous_questions}
"""

    else:

        avoid_section = """
There are no previously asked questions.
"""

    # --------------------------------------------------
    # 4. Build prompt
    # --------------------------------------------------

    prompt = f"""
You are an experienced technical interviewer.

You are conducting a {domain} interview.

Candidate skills:

{", ".join(skills)}


Candidate project information:

{project_summary}


Your task:

Generate ONE interview question for this candidate .

Difficulty:

{target_difficulty}


Difficulty guidance:

{difficulty_guidance}


Important rules:

1. Ask ONE short interview question based directly on the
   candidate's project or technical skills.

2. The question must be only ONE sentence.

3. Keep the question between 8 and 18 words.

4. Do NOT summarize or paraphrase the candidate's project
   before asking the question.

5. Do NOT mention multiple technologies in the same question
   unless absolutely necessary.

6. Ask about ONE specific concept, technology, or decision.

7. The question should sound like a real interviewer asking
   a direct question.

8. Do not ask a generic question.

9. Do not repeat previous questions.

10. Generate ONLY the question, not an explanation.



{avoid_section}


Return ONLY valid JSON.

Return exactly:

{{
    "question": "the interview question",
    "topic": "the main technical topic",
    "reasoning": "one sentence explaining why this question was selected"
}}
"""

    # --------------------------------------------------
    # 5. Call LLM
    # --------------------------------------------------

    response = llm.invoke(prompt)

    # --------------------------------------------------
    # 6. Parse JSON
    # --------------------------------------------------

    generated = json.loads(response.content)

    # --------------------------------------------------
    # 7. Update question history
    # --------------------------------------------------

    updated_history = (
        question_history
        + [generated["question"]]
    )

    # --------------------------------------------------
    # 8. Return updated state
    # --------------------------------------------------

    return {
        "current_question": generated["question"],
        "current_topic": generated["topic"],
        "current_difficulty": target_difficulty,
        "question_history": updated_history,
    }