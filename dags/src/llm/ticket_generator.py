from typing import List, Dict
from openai import OpenAI

from src.llm.prompts import generate_jira_prompt
from src.utils.config import Config
import logging
import time
from openai import OpenAIError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class JiraTicketGenerator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key)


    def generate_ticket(self, module, risk_score, context, max_retries=3):
        prompt = generate_jira_prompt(module, risk_score, context)

        delay = 1  # initial delay in seconds

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a Jira assistant."},
                        {"role": "user", "content": prompt}
                    ],
                )

                ticket_text = response.choices[0].message.content.strip()

                lines = ticket_text.splitlines()
                title, description = "", ""
                if lines:
                    title = lines[0].replace("Title:", "").strip()
                    description = "\n".join(lines[1:]).strip()

                return {"title": title, "description": description, "module": module, "risk_score": risk_score}

            except (OpenAIError, Exception) as e:
                if attempt < max_retries:
                    print(f"Attempt {attempt} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # exponential backoff
                    return None
                else:
                    print(f"All {max_retries} attempts failed.")
                    raise
        return None

    def generate_tickets_bulk(self, modules: List[Dict], num_of_tickets=None) -> List[Dict]:
        """
        modules: list of dicts with keys: module, risk_score, context
        """
        sorted_modules = sorted(modules, key=lambda x: x['risk_score'], reverse=True)
        if num_of_tickets is not None:
            sorted_modules = sorted_modules[:num_of_tickets]
        tickets = []
        for mod in sorted_modules:
            ticket = self.generate_ticket(mod['module'], mod['risk_score'], mod.get('context', {}))
            tickets.append(ticket)
        return tickets
