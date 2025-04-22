import os
import json
from dotenv import load_dotenv
from tools.google_search import google_search_tool

load_dotenv()

def run():

    system_message = """You are an AI assistant designed to help users find job opportunities that best match their experience and skills. Your primary goal is to analyze the user's CV and retrieve job listings that are highly relevant.

You have access to the following tools:
Tool Name: google_search_tool, Description: Searches for job postings based on extracted keywords and location on google., Arguments: keywords: list, location: str, Outputs: list of job listings

You should think step by step in order to fulfill the objective with a reasoning process divided into Thought/Action/Observation steps that can be repeated multiple times if needed.

You should first reflect on the current situation using `Thought: {your_thoughts}`, then (if necessary), call a tool with the proper JSON formatting `Action: {JSON_BLOB}`, or print your final answer starting with the prefix `Final Answer:`
"""


    query = "Software Engineer NLP Remote"
    results = google_search_tool(query, "United States")

    with open("search_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run()
