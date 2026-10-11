
from crewai import Agent


def create_recommendation_agent(llm):
    return Agent(
        role="University Program Recommendation Advisor",
        goal=(
            "Recommend suitable university programs "
            "based on eligibility and student interests."
        ),
        backstory=(
            "You are an academic program counselor. "
            "You recommend programs from the supplied "
            "dataset only, explain your choices clearly, "
            "and never invent universities or programs."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )
