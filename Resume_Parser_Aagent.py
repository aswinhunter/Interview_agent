

import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
 
from State import AgentPrepState

load_dotenv()  # reads GROQ_API_KEY from your .env file

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0  # 0 = consistent, factual output — not creative
)


def resume_parser_agent(state: AgentPrepState) -> dict:
    """
    This is the LangGraph NODE function.

    Input:  state  -> the shared state dict (must contain 'resume_path')
    Output: a dict with the new fields to merge into state
    """

    # --- 1. read the PDF text ---
    pdf_path = state["resume_path"]
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    resume_text = "\n".join(page.page_content for page in pages)

    # --- 2. ask the LLM to extract structured info ---
    prompt = f"""
You are a resume analysis assistant.
Read the resume below and return a JSON object. No text before or after the JSON.
 
Return exactly this shape:
{{
  "skills": ["skill1", "skill2", ...],
  "experience_level": "fresher" or "1-2 years" or "2+ years",
  "domain": "one of: DSA, ML, DBMS, OS, OOP — pick the strongest match",
  "project_summary": "Write sentences describing the candidate's projects in detail. Include: what they built, what technologies they used, what problem it solved, and any interesting technical decisions they made. This will be used to generate interview questions about their work — so be specific."
}}
 
Resume:
{resume_text}
"""
    response = llm.invoke(prompt)
    extracted = json.loads(response.content)

    # --- 3. return only the fields we want to update in shared state ---
    return {
        "skills":          extracted["skills"],
        "experience_level":extracted["experience_level"],
        "domain":          extracted["domain"],
        "project_summary": extracted["project_summary"],
    }