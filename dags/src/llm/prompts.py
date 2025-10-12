def generate_jira_prompt(module: str, risk_score: float, context: dict) -> str:
    """
    Generate a precise, code-aware prompt for the LLM to create a Jira ticket.
    The output is guaranteed to be a plain string safe to send to the OpenAI API.
    """
    recent_churn = context.get("recent_churn", "N/A")
    bug_ratio = context.get("bug_ratio", "N/A")
    recent_prs = context.get("recent_prs", "N/A")
    code_snippet = context.get("code_snippet", "")

    prompt = f"""
You are a senior software engineer specialized in code quality and refactoring.
Analyze the following code and generate a specific, actionable Jira ticket suggesting exact changes.

Module: {module}
Predicted Risk Score: {risk_score:.2f}
Recent Churn: {recent_churn}
Bug PR Ratio: {bug_ratio}
Recent PR Count: {recent_prs}

Code Snippet:
{code_snippet.strip()}

Instructions:
1. Identify exact maintainability issues, e.g., nested logic, unclear naming,
   hardcoded constants, inefficient loops, missing error handling, or lack of modularization.
2. Propose a concrete fix referring to specific functions, classes, or code blocks.
3. Return ONLY a **plain text ticket** with:

Title: Short, action-oriented summary.
Description: What is wrong, what should be done, which part of the code is affected.
*Acceptance Criteria*: 2-3 measurable bullet points.

Do NOT include JSON markers, code fences, or extra formatting.
"""
    return prompt.strip()
