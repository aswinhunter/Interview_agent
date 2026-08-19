import json

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from State import AgentPrepState


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# --------------------------------------------------
# Answer Evaluator
# --------------------------------------------------

def answer_evaluator_agent(state: AgentPrepState) -> dict:
    """
    Evaluates the candidate's answer using the LLM.

    Reads from state:
        current_question
        current_answer
        current_topic
        current_difficulty
        skills
        project_summary

    Writes to state:
        score
        feedback
        strengths
        weaknesses
        scores
        session_score
    """

    # --------------------------------------------------
    # 1. Read information from state
    # --------------------------------------------------

    question = state.get(
        "current_question",
        ""
    )

    answer = state.get(
        "current_answer",
        ""
    )

    topic = state.get(
        "current_topic",
        ""
    )

    difficulty = state.get(
        "current_difficulty",
        "Easy"
    )

    skills = state.get(
        "skills",
        []
    )

    project_summary = state.get(
        "project_summary",
        ""
    )

    previous_scores = state.get(
        "scores",
        []
    )


    # --------------------------------------------------
    # 2. Build evaluation prompt
    # --------------------------------------------------

    prompt = f"""
You are a technical interview evaluator.

Evaluate the candidate's answer to the interview question.

Candidate skills:
{", ".join(skills)}

Relevant project context:
{project_summary}

Question topic:
{topic}

Difficulty:
{difficulty}

Interview question:
{question}

Candidate answer:
{answer}


Evaluate the answer based on:

1. Technical correctness
2. Relevance to the question
3. Understanding of the concept
4. Completeness
5. Clarity

Important rules:

- Evaluate ONLY what the candidate actually said.
- Do not assume knowledge that the candidate did not demonstrate.
- Do not penalize minor grammar mistakes.
- Focus on technical knowledge.
- Give a fair score suitable for a technical interview.
- If the answer is completely unrelated, give a very low score.
- If the candidate gives a partially correct answer, give partial credit.
- Do not expect the candidate to use exactly the same wording as a model answer.


Give a score from 0 to 10.

Score guidelines:

9-10 = Excellent
7-8 = Good
5-6 = Average / partially correct
3-4 = Weak
0-2 = Incorrect or irrelevant


Return ONLY valid JSON.

Return exactly:

{{
    "score": 0,
    "feedback": "Short explanation of the evaluation.",
    "strengths": [
        "strength 1",
        "strength 2"
    ],
    "weaknesses": [
        "weakness 1",
        "weakness 2"
    ]
}}
"""


    # --------------------------------------------------
    # 3. Call LLM
    # --------------------------------------------------

    response = llm.invoke(prompt)


    # --------------------------------------------------
    # 4. Parse JSON
    # --------------------------------------------------

    evaluation = json.loads(response.content)


    # --------------------------------------------------
    # 5. Convert score to float
    # --------------------------------------------------

    score = float(evaluation["score"])


    # --------------------------------------------------
    # 6. Update score history
    # --------------------------------------------------

    updated_scores = previous_scores + [score]


    # --------------------------------------------------
    # 7. Calculate session average
    # --------------------------------------------------

    session_score = (
        sum(updated_scores) / len(updated_scores)
    )


    # --------------------------------------------------
    # 8. Print evaluation
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("ANSWER EVALUATION")
    print("=" * 60)

    print("\nScore:")
    print(f"{score}/10")

    print("\nFeedback:")
    print(evaluation["feedback"])

    print("\nStrengths:")

    for strength in evaluation["strengths"]:
        print(f"- {strength}")

    print("\nWeaknesses:")

    for weakness in evaluation["weaknesses"]:
        print(f"- {weakness}")

    print("\nSession Score:")
    print(f"{session_score:.2f}/10")


    # --------------------------------------------------
    # 9. Return updated state
    # --------------------------------------------------

    return {

        "score": score,

        "feedback": evaluation["feedback"],

        "strengths": evaluation["strengths"],

        "weaknesses": evaluation["weaknesses"],

        "scores": updated_scores,

        "session_score": session_score,

    }