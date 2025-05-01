from smolagents import tool

@tool
def ranking_results(job_listings: list, candidate_skills: list = None) -> list:
    """
    Rank job listings based on relevance to candidate skills.
    
    Args:
        job_listings: List of job listing dictionaries
        candidate_skills: List of candidate skills to prioritize in ranking
        
    Returns:
        List of ranked job listings with added relevance scores
    """
    if not candidate_skills:
        candidate_skills = ["accounting", "financial", "quickbooks", "microsoft", "excel", "sql"]
    
    # Create a copy of job listings to add ranking scores
    ranked_jobs = []
    
    for job in job_listings:
        score = 0
        job_title = job.get('title', '').lower()
        job_snippet = job.get('snippet', '').lower()
        job_description = job_title + " " + job_snippet
        
        # Score based on skills matches
        for skill in candidate_skills:
            if skill.lower() in job_description:
                score += 10
        
        # Bonus for title matches
        if "account" in job_title or "finance" in job_title:
            score += 20
        
        # Add job with score to ranked list
        ranked_jobs.append({
            **job,
            'relevance_score': score
        })
    
    # Sort by relevance score (highest first)
    ranked_jobs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    return ranked_jobs