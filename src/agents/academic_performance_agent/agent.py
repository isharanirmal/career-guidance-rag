from src.framework.core.base_agent import BaseAgent

from .config import SYSTEM_PROMPT_FILE


class AcademicPerformanceAgent(BaseAgent):
    """Agent for academic/GPA questions using live web research."""

    def __init__(self, llm):
        super().__init__(
            llm=llm,
            system_prompt_file=SYSTEM_PROMPT_FILE,
        )
