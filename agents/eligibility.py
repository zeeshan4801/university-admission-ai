
from crewai import Agent


def create_eligibility_agent(llm):
    return Agent(
        role="Student Admission Eligibility Evaluator",
        goal=(
            "Evaluate student eligibility accurately "
            "against the supplied admission criteria."
        ),
        backstory=(
            "You are an admissions eligibility specialist. "
            "You compare marks, qualifications, and "
            "mathematics prerequisites objectively. "
            "You do not assume that eligibility "
            "guarantees admission."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )
