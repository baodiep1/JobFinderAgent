import json 
from .tool import Tool


def save_serach_results(results):
    
    with open("search_results.json", "w") as f:

        json.dump(results, f, indent= 2)


save_search_resutls_tool = Tool(
        name = "Save search results tool",
        description = "Saves the results of goolge search",
        func = save_serach_results,
        arguments = [("results", "dict")] 
        outputs= "None"


)
