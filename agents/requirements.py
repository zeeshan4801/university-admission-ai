
from crewai import Agent


def create_requirements_agent(llm):
    return Agent(
        role="University Admission Requirement Checker",
        goal=(
            "Extract accurate admission requirements "
            "from the supplied university dataset."
        ),
        backstory=(
            "You are a university admissions assistant "
            "who carefully reads published program rules. "
            "You never invent missing requirements."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )
