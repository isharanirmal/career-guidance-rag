# CareerGuide AI – Professional

A professional Career & Academic Guidance Assistant with a FastAPI backend and responsive web UI.

## Main features

- Professional chatbot with concise, structured answers
- Career recommendations and skill-gap analysis
- CV PDF upload and CV analysis
- Result sheet / academic guidance
- 30/60/90-day career roadmap
- ATS-friendly CV generator
- Current public information through web-grounded Gemini when Gemini is configured
- Useful source links from grounded searches
- Light / dark mode
- Mobile-responsive interface
- PDF validation, page/size limits, request limits and rate limiting
- Restricted CORS, trusted hosts and security headers
- Uploaded documents are processed temporarily and are not intentionally persisted
- Multi-provider LLM support with automatic fallback

## Supported providers

Configure **one or more** of:

- OpenAI (`OPENAI_API_KEY`)
- Anthropic (`ANTHROPIC_API_KEY`)
- Gemini (`GEMINI_API_KEY`)
- Groq (`GROQ_API_KEY`)
- Claude/Anthropic alias (`CLAUDE_API_KEY`)

Empty keys are skipped. You do **not** need every provider key.

Set `LLM_PROVIDER=auto` to use the order in `LLM_FALLBACK_ORDER`. If a configured provider fails, the app automatically tries the next configured provider.

> Live Google Search grounding is currently available through Gemini when `WEB_SEARCH_ENABLED=true` and a Gemini key is configured. Other providers still work normally without it.

## Setup on Windows

From the `career-guidance-rag` directory:

```powershell
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add at least one API key.

Example:

```env
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
LLM_PROVIDER=auto
LLM_FALLBACK_ORDER=openai,gemini,groq,anthropic
WEB_SEARCH_ENABLED=true
```

Start the server:

```powershell
python -m uvicorn web.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Security notes

- Never commit `.env` or API keys to GitHub.
- The frontend never receives provider API keys.
- Do not upload passwords, identity documents, payment-card details or other unnecessary sensitive files.
- Rate limiting is in-memory and intended for a single local/demo instance. Use a shared store and reverse proxy controls for production.


## Provider troubleshooting

Run `python diagnose_providers.py` to check which provider keys are configured without printing any key values.

The default models are selected for currently active provider model families. You can override them in `.env`.

If one provider is rate-limited, unavailable, or has an invalid model, the app continues through `LLM_FALLBACK_ORDER`. A 429/rate-limit response cannot be fixed by application code; another configured provider must have usable quota.

## Secure login & professional dashboard (v3)

This build includes a local account system and a redesigned professional dashboard while preserving the existing AI features.

### First run

1. Install dependencies: `pip install -r requirements.txt`
2. Configure at least one AI provider in `.env` as before.
3. Start the web app: `uvicorn web.main:app --reload --host 127.0.0.1 --port 8000`
4. Open `http://127.0.0.1:8000`.
5. On the first launch, create the administrator username and a strong password. No default password is shipped.

### Added features

- Secure username/password login and logout.
- PBKDF2-SHA256 salted password hashing in a local SQLite database.
- HttpOnly SameSite session cookie, CSRF protection, login throttling and security headers.
- Change username and password from **Account & Security**.
- Password changes invalidate active sessions and require sign-in again.
- Professional dashboard and custom CareerGuide AI logo.
- CV Generator profile photo upload/preview (JPG/PNG/WEBP, validated server-side).
- Professional PDF CV download, including the optional profile photo.
- Existing AI chat, CV analysis, academic guidance, career roadmap and CV generation are retained.

### Production note

When deployed behind HTTPS, set `COOKIE_SECURE=true` in `.env`. Keep `.env` and `data/careerguide_auth.db*` private and out of source control.
