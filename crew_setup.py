
import json
from pathlib import Path

from crewai import Crew, Process, Task

from config import get_llm

from agents.requirements import create_requirements_agent
from agents.eligibility import create_eligibility_agent
from agents.recommendations import create_recommendation_agent


DATA_PATH = (
    Path(__file__).resolve().parent
    / "university_data.json"
)


def load_university_data():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def run_admission_crew(student):
    # Load local sample admission requirements
    university_data = load_university_data()

    # Configure the Groq language model
    llm = get_llm()

    # Create three independent agents
    requirements_agent = create_requirements_agent(llm)
    eligibility_agent = create_eligibility_agent(llm)
    recommendation_agent = create_recommendation_agent(llm)

    # Task 1: Read admission requirements
    requirements_task = Task(
        description=(
            "Read the following university dataset carefully:\n"
            f"{json.dumps(university_data, indent=2)}\n\n"
            "Extract every university and program with its "
            "minimum percentage, accepted qualifications, "
            "mathematics requirement and description.\n"
            "Use ONLY this dataset. Do not invent any "
            "university, program or admission rule."
        ),
        expected_output=(
            "A clear list of all programs with "
            "their exact admission requirements."
        ),
        agent=requirements_agent,
    )

    # Task 2: Evaluate student eligibility
    eligibility_task = Task(
        description=(
            "Evaluate this student:\n"
            f"{json.dumps(student, indent=2)}\n\n"
            "Use the admission requirements provided "
            "by the previous task.\n\n"
            "For EACH program, check:\n"
            "1. Student percentage >= minimum percentage\n"
            "2. Qualification is in accepted_qualifications\n"
            "3. Mathematics prerequisite is satisfied\n\n"
            "A program is Eligible only if ALL conditions "
            "pass. If a required detail is unknown, "
            "mark Needs Review rather than guessing.\n"
            "Otherwise mark Not Eligible.\n"
            "Explain the reasons clearly.\n"
            "These are demonstration rules, not official "
            "university admission decisions."
        ),
        expected_output=(
            "An eligibility assessment for every program, "
            "including marks comparison, qualification "
            "check, mathematics check and final status."
        ),
        agent=eligibility_agent,
        context=[requirements_task],
    )

    # Task 3: Recommend programs
    recommendation_task = Task(
        description=(
            "Student details:\n"
            f"{json.dumps(student, indent=2)}\n\n"
            "Use the eligibility evaluation provided "
            "by the previous task.\n\n"
            "Recommend up to THREE suitable programs "
            "marked Eligible that best match the student's "
            "interests.\n"
            "For each recommendation, provide:\n"
            "- University name\n"
            "- Program name\n"
            "- Why it matches the student's interests\n"
            "- Eligibility explanation\n\n"
            "Also mention any programs that were "
            "Not Eligible or Need Review.\n"
            "Do not invent new universities or programs.\n"
            "If no programs are eligible, say so clearly.\n"
            "Use Markdown headings and a concise table.\n"
            "Finish with a short note that the dataset "
            "is fictional and actual requirements must "
            "be verified on official university websites."
        ),
        expected_output=(
            "A clear Markdown admission report "
            "with up to three recommended programs, "
            "eligibility reasons and a disclaimer."
        ),
        agent=recommendation_agent,
        context=[eligibility_task],
    )

    # Run all agents one after another
    crew = Crew(
        agents=[
            requirements_agent,
            eligibility_agent,
            recommendation_agent,
        ],
        tasks=[
            requirements_task,
            eligibility_task,
            recommendation_task,
        ],
        process=Process.sequential,
        verbose=False,
        memory=False,
    )

    result = crew.kickoff()

    return str(result)
