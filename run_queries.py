import asyncio
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from multi_agents.agents import ChiefEditorAgent
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

load_dotenv(find_dotenv(".env"))

DATA_PATH = Path("./data")
INTERIM_DATA_PATH = DATA_PATH / "interim"

tracer_provider = register(
  project_name="gpt-researcher",
  endpoint="http://localhost:6006/v1/traces",
)

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


async def run_queries(questions):
    task = {
        "max_sections": 3,
        "publish_formats": {
            "markdown": True,
            "pdf": True,
            "docx": True,
        },
        "include_human_feedback": False,
        "follow_guidelines": False,
        "model": "gigachat:GigaChat-2-Max",
        "guidelines": [],
        "language": "russian",
        "verbose": True,
    }

    research_reports = []

    for question in questions:
        task["query"] = question
        chief_editor = ChiefEditorAgent(task)
        research_report = await chief_editor.run_research_task()
        research_reports.append(research_report)

    return research_reports


async def run_query(question):
    task = {
        "max_sections": 3,
        "publish_formats": {
            "markdown": True,
            "pdf": True,
            "docx": True,
        },
        "include_human_feedback": False,
        "follow_guidelines": False,
        "model": "gigachat:GigaChat-2-Max",
        "guidelines": [],
        "language": "russian",
        "verbose": True,
    }

    research_reports = []

    task["query"] = question
    chief_editor = ChiefEditorAgent(task)
    research_report = await chief_editor.run_research_task()
    research_reports.append(research_report)

    return research_reports


async def main():
    df = pd.read_json(INTERIM_DATA_PATH / "selected_questions.json", lines=True)
    research_reports = await run_queries(df["question"].tolist())
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"data/processed/research_reports_{current_datetime}.json", "w", encoding="utf-8") as file:
        json.dump(research_reports, file, ensure_ascii=False, indent=4)

    # user_input = "Проведи исследование и дай мне ответ сколько в стране людей – близнецов."
    # await run_query(user_input)


if __name__ == "__main__":
    asyncio.run(main())
