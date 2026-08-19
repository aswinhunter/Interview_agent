from langgraph.graph import StateGraph, START, END

from State import AgentPrepState
from Resume_Parser_Aagent import resume_parser_agent
from Question_Generate import question_generator_agent
from Answer_Capture import answer_capture_agent


def build_graph():

    # --------------------------------------------------
    # Create StateGraph
    # --------------------------------------------------

    graph = StateGraph(AgentPrepState)

    # --------------------------------------------------
    # Add nodes
    # --------------------------------------------------

    graph.add_node("resume_parser",resume_parser_agent)

    graph.add_node("question_generator",question_generator_agent)
    
    graph.add_node("answer_capture",answer_capture_agent)
    

    # --------------------------------------------------
    # Define flow
    # --------------------------------------------------

    graph.add_edge(START,"resume_parser")

    graph.add_edge("resume_parser","question_generator")
    graph.add_edge("question_generator","answer_capture")
    graph.add_edge("answer_capture",END)

    # --------------------------------------------------
    # Compile graph
    # --------------------------------------------------

    app = graph.compile()

    return app


# ------------------------------------------------------
# Run graph
# ------------------------------------------------------

if __name__ == "__main__":

    # Build application
    app = build_graph()

    # --------------------------------------------------
    # Initial state
    # --------------------------------------------------

    initial_state: AgentPrepState = {

        "resume_path": "sample_resume.pdf",

        "difficulty_level": 2,

        "question_history": [],

        "current_answer": "",

        "scores": [],

        "session_score": 0.0,
    }

    # --------------------------------------------------
    # Execute graph
    # --------------------------------------------------

    result = app.invoke(initial_state)

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RESUME ANALYSIS")
    print("=" * 60)

    print("\nSkills:")
    print(result["skills"])

    print("\nExperience:")
    print(result["experience_level"])

    print("\nDomain:")
    print(result["domain"])

    print("\nProject Summary:")
    print(result["project_summary"])

    

    print("\n" + "=" * 60)