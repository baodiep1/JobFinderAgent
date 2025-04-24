from smolagents import tool


@tool
def ranking_results(results: list[dict]) -> list[dict]:
    """
    Rank the results based on the relevance to the query.
    """
    return results


"""
ranking_results_tool = Tool(
    name="ranking_results",
    description="Rank the results based on the relevance to the query.",
    func=ranking_results,
    arguments=[dict],
    outputs=list[dict]
)
"""