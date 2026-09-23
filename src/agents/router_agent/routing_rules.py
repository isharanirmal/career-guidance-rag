ROUTER_SYSTEM_PROMPT = """
You are RouterAgent for CareerGuide AI.
Choose exactly one label:
- academic_performance_agent: GPA, grades, transcript, result sheet, modules, exams, academic progress, scholarships, university academic rules.
- career_path_agent: CV/resume, careers, jobs, internships, skills, experience, hiring, vacancies, roles, career roadmaps.
- general_fallback: anything else.
Return ONLY JSON: {"next_agent":"agent_name"}
Never explain or answer the question.
"""

VALID_LABELS = {"academic_performance_agent", "career_path_agent", "general_fallback"}


class KeywordRoutingRule:
    __slots__ = ("keywords", "label")

    def __init__(self, keywords, label: str):
        self.keywords = keywords
        self.label = label

    def match_count(self, query_clean: str) -> int:
        return sum(1 for keyword in self.keywords if keyword in query_clean)


KEYWORD_RULES = (
    KeywordRoutingRule(
        ("gpa", "result sheet", "results sheet", "transcript", "grades", "semester", "subject", "subjects", "cgpa", "scholarship", "second class", "first class", "cutoff", "exam", "module", "marks", "academic"),
        "academic_performance_agent",
    ),
    KeywordRoutingRule(
        ("cv", "resume", "job", "jobs", "career", "internship", "experience", "skills", "hiring", "vacancy", "role", "position", "roadmap", "portfolio"),
        "career_path_agent",
    ),
)
