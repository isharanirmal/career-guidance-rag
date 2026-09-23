import io
import logging
import os
import re
import tempfile
import time
from collections import defaultdict, deque
from html import escape
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader

from src.agents.router_agent.router_main import RouterAgent
from src.framework.llm.llm_factory import build_llm_client
from src.framework.loaders.pdf_loader import PDFLoader
from web.security import (
    authenticate, change_password, change_username, create_session, create_user,
    get_session, get_user, init_db, revoke_session, user_count
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("careerguide")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "3"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "15"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "30000"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
MAX_QUESTION_CHARS = 2500
MAX_NAME_CHARS = 100
MAX_FIELD_CHARS = 3000
MAX_REQUEST_BYTES = max(MAX_UPLOAD_MB, MAX_IMAGE_MB) * 1024 * 1024 + 500_000
SESSION_COOKIE = "careerguide_session"
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"

app = FastAPI(title="CareerGuide AI", version="3.0.0", docs_url="/docs" if os.getenv("APP_ENV") == "development" else None)
init_db()

allowed_origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",") if x.strip()]
allowed_hosts = [x.strip() for x in os.getenv("ALLOWED_HOSTS", "127.0.0.1,localhost").split(",") if x.strip()]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-CSRF-Token"])

_request_log: dict[str, deque[float]] = defaultdict(deque)
_login_log: dict[str, deque[float]] = defaultdict(deque)


def check_rate_limit(client_ip: str, log=None, requests=RATE_LIMIT_REQUESTS, window=RATE_LIMIT_WINDOW_SECONDS):
    store = log or _request_log
    now = time.time()
    q = store[client_ip]
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= requests:
        raise HTTPException(status_code=429, detail="Too many requests. Please try again shortly.")
    q.append(now)


def validate_question(question: str) -> str:
    q = (question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Please enter a question.")
    if len(q) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=413, detail=f"Question is too long. Maximum is {MAX_QUESTION_CHARS} characters.")
    return q


def validate_pdf_header(data: bytes):
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Only valid PDF files are accepted.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"PDF is too large. Maximum is {MAX_UPLOAD_MB} MB.")


def read_pdf_upload(document: UploadFile) -> str:
    filename = os.path.basename(document.filename or "document.pdf")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    data = document.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    validate_pdf_header(data)
    try:
        reader = PdfReader(io.BytesIO(data))
        if len(reader.pages) > MAX_PDF_PAGES:
            raise HTTPException(status_code=413, detail=f"PDF has too many pages. Maximum is {MAX_PDF_PAGES}.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="The PDF could not be read safely.")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            temp_path = tmp.name
        return PDFLoader().load(temp_path, max_pages=MAX_PDF_PAGES, max_chars=MAX_TEXT_CHARS)
    finally:
        if temp_path:
            try: os.remove(temp_path)
            except OSError: pass


try:
    llm = build_llm_client()
    router = RouterAgent(llm=llm)
    pdf_loader = PDFLoader()
    startup_error = None
except Exception as exc:
    llm = None
    router = None
    pdf_loader = PDFLoader()
    startup_error = str(exc)
    logger.error("AI configuration error: %s", exc)


class ChatResponse(BaseModel):
    answer: str
    routed_agent: str
    sources: list[dict] = Field(default_factory=list)

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)

class SetupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=10, max_length=256)

class UsernameChangeRequest(BaseModel):
    new_username: str = Field(..., min_length=3, max_length=32)
    current_password: str = Field(..., min_length=1, max_length=256)

class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=256)
    new_password: str = Field(..., min_length=10, max_length=256)

class RoadmapRequest(BaseModel):
    target_role: str = Field(..., min_length=2, max_length=100)
    current_skills: str = Field(default="", max_length=MAX_FIELD_CHARS)
    experience: str = Field(default="", max_length=MAX_FIELD_CHARS)
    goal: str = Field(default="", max_length=MAX_FIELD_CHARS)

class CVRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=MAX_NAME_CHARS)
    email: str = Field(default="", max_length=150)
    phone: str = Field(default="", max_length=60)
    location: str = Field(default="", max_length=150)
    education: str = Field(default="", max_length=MAX_FIELD_CHARS)
    skills: str = Field(default="", max_length=MAX_FIELD_CHARS)
    projects: str = Field(default="", max_length=MAX_FIELD_CHARS)
    experience: str = Field(default="", max_length=MAX_FIELD_CHARS)
    certifications: str = Field(default="", max_length=MAX_FIELD_CHARS)
    target_role: str = Field(default="", max_length=100)


def current_user(request: Request):
    session = get_session(request.cookies.get(SESSION_COOKIE))
    if not session:
        raise HTTPException(status_code=401, detail="Authentication required.")
    user = get_user(session["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    request.state.session = session
    return user


def verify_csrf(request: Request, user=Depends(current_user)):
    session = getattr(request.state, "session", None) or get_session(request.cookies.get(SESSION_COOKIE))
    token = request.headers.get("X-CSRF-Token", "")
    if not session or not token or token != session["csrf"]:
        raise HTTPException(status_code=403, detail="Security token is missing or invalid. Refresh the page and try again.")
    return user


def set_session_cookie(response: Response, token: str):
    response.set_cookie(SESSION_COOKIE, token, httponly=True, secure=COOKIE_SECURE, samesite="strict", max_age=8*60*60, path="/")


@app.middleware("http")
async def security_and_limits(request: Request, call_next):
    if request.method == "POST":
        client_ip = request.client.host if request.client else "unknown"
        check_rate_limit(client_ip)
        content_length = request.headers.get("content-length")
        try:
            if content_length and int(content_length) > MAX_REQUEST_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Request is too large."})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid request length."})
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    if request.url.path.startswith("/api/") or request.url.path in {"/chat", "/analyze-cv", "/career-roadmap", "/generate-cv", "/download-cv"}:
        response.headers["Cache-Control"] = "no-store"
    return response


def require_ai():
    if router is None:
        raise HTTPException(status_code=503, detail="AI service is not configured. Add at least one provider API key to .env and restart the server.")


@app.get("/health")
def health():
    return {"status": "ok", "ai_configured": router is not None, "providers": getattr(llm, "provider_names", []), "web_search_enabled": getattr(llm, "web_search_enabled", False)}

@app.get("/api/auth/status")
def auth_status(request: Request):
    setup_required = user_count() == 0
    session = get_session(request.cookies.get(SESSION_COOKIE))
    user = get_user(session["user_id"]) if session else None
    return {"authenticated": bool(user), "setup_required": setup_required, "user": {"username": user.username} if user else None, "csrf_token": session["csrf"] if user else None}

@app.post("/api/auth/setup")
def auth_setup(payload: SetupRequest, request: Request):
    if user_count() != 0:
        raise HTTPException(status_code=409, detail="Initial setup has already been completed.")
    try:
        user = create_user(payload.username, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    token, csrf = create_session(user.id)
    response = JSONResponse({"ok": True, "user": {"username": user.username}, "csrf_token": csrf})
    set_session_cookie(response, token)
    return response

@app.post("/api/auth/login")
def auth_login(payload: LoginRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip, _login_log, requests=8, window=300)
    user = authenticate(payload.username, payload.password)
    if not user:
        time.sleep(0.15)
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token, csrf = create_session(user.id)
    response = JSONResponse({"ok": True, "user": {"username": user.username}, "csrf_token": csrf})
    set_session_cookie(response, token)
    return response

@app.post("/api/auth/logout")
def auth_logout(request: Request, user=Depends(verify_csrf)):
    revoke_session(request.cookies.get(SESSION_COOKIE))
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response

@app.get("/api/account")
def account(user=Depends(current_user)):
    return {"username": user.username, "created_at": user.created_at, "updated_at": user.updated_at}

@app.post("/api/account/username")
def account_username(payload: UsernameChangeRequest, request: Request, user=Depends(verify_csrf)):
    try:
        updated = change_username(user.id, payload.current_password, payload.new_username)
        return {"ok": True, "username": updated.username}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/api/account/password")
def account_password(payload: PasswordChangeRequest, request: Request, user=Depends(verify_csrf)):
    try:
        change_password(user.id, payload.current_password, payload.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    response = JSONResponse({"ok": True, "reauthenticate": True})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@app.post("/chat", response_model=ChatResponse)
async def chat(question: str = Form(...), document: UploadFile | None = File(None), user=Depends(verify_csrf)):
    require_ai(); q = validate_question(question)
    user_document = read_pdf_upload(document) if document is not None else ""
    try:
        response = router.route_and_execute(q, user_document=user_document)
        return ChatResponse(answer=response.answer, routed_agent=router.last_routed_label or "unknown", sources=response.sources or [])
    except Exception as exc:
        logger.exception("AI chat request failed")
        raise HTTPException(status_code=502, detail="The AI service could not complete the request. Check the server terminal for the provider error.") from exc

@app.post("/analyze-cv", response_model=ChatResponse)
async def analyze_cv(document: UploadFile = File(...), user=Depends(verify_csrf)):
    require_ai(); user_document = read_pdf_upload(document)
    prompt = "Analyze this CV for career guidance. Identify strengths, missing/weak areas, 3 suitable career roles, a concise skill-gap list, and concrete next steps. Use current web research for current role requirements. Keep the answer structured and concise."
    try:
        response = router.career_agent.process_query(prompt, user_document=user_document)
        return ChatResponse(answer=response.answer, routed_agent="career_path_agent", sources=response.sources or [])
    except Exception as exc:
        logger.exception("CV analysis failed")
        raise HTTPException(status_code=502, detail="CV analysis could not be completed. Check the server terminal for the provider error.") from exc

@app.post("/career-roadmap", response_model=ChatResponse)
def career_roadmap(payload: RoadmapRequest, user=Depends(verify_csrf)):
    require_ai(); target = payload.target_role.strip()
    prompt = f"""Create a practical career roadmap for the target role: {target}.
Current skills: {payload.current_skills.strip() or 'Not provided'}
Experience: {payload.experience.strip() or 'Not provided'}
Goal: {payload.goal.strip() or 'Not provided'}
Use current web research for relevant skills/tools/job requirements. Return:
1. Target role overview
2. Skill gaps
3. 30-day plan
4. 60-day plan
5. 90-day plan
6. 2 portfolio projects
7. Interview preparation
8. Useful current resources/sources
Keep it concise and realistic."""
    try:
        response = router.career_agent.process_query(prompt)
        return ChatResponse(answer=response.answer, routed_agent="career_path_agent", sources=response.sources or [])
    except Exception as exc:
        logger.exception("Roadmap generation failed")
        raise HTTPException(status_code=502, detail="Career roadmap could not be generated. Check the server terminal for the provider error.") from exc

@app.post("/generate-cv", response_model=ChatResponse)
def generate_cv(payload: CVRequest, user=Depends(verify_csrf)):
    require_ai()
    prompt = f"""Create an ATS-friendly professional CV draft using ONLY the information provided below. Do not invent employers, dates, grades, technologies, achievements or contact details. Use clean plain text with sections: NAME/CONTACT, PROFESSIONAL SUMMARY, EDUCATION, SKILLS, PROJECTS, EXPERIENCE, CERTIFICATIONS. If a section has no data, omit it. Keep the summary tailored to the target role.
Name: {payload.name.strip()}
Email: {payload.email.strip()}
Phone: {payload.phone.strip()}
Location: {payload.location.strip()}
Target role: {payload.target_role.strip()}
Education: {payload.education.strip()}
Skills: {payload.skills.strip()}
Projects: {payload.projects.strip()}
Experience: {payload.experience.strip()}
Certifications: {payload.certifications.strip()}"""
    try:
        response = router.career_agent.process_query(prompt)
        return ChatResponse(answer=response.answer, routed_agent="career_path_agent", sources=response.sources or [])
    except Exception as exc:
        logger.exception("CV generation failed")
        raise HTTPException(status_code=502, detail="CV generation could not be completed. Check the server terminal for the provider error.") from exc


def _safe_lines(text: str):
    return [line.strip() for line in (text or "").replace("\r", "").split("\n") if line.strip()]

@app.post("/download-cv")
async def download_cv(
    name: str = Form(...), email: str = Form(""), phone: str = Form(""), location: str = Form(""),
    target_role: str = Form(""), cv_text: str = Form(...), photo: UploadFile | None = File(None),
    user=Depends(verify_csrf)
):
    if not name.strip() or len(name) > MAX_NAME_CHARS or len(cv_text) > 20000:
        raise HTTPException(status_code=400, detail="Invalid CV content.")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
        from reportlab.lib.units import mm
        from PIL import Image as PILImage
    except Exception as exc:
        raise HTTPException(status_code=500, detail="PDF dependencies are missing. Run: pip install -r requirements.txt") from exc

    photo_path = None
    try:
        if photo is not None and photo.filename:
            raw = await photo.read(MAX_IMAGE_MB * 1024 * 1024 + 1)
            if len(raw) > MAX_IMAGE_MB * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"Photo is too large. Maximum is {MAX_IMAGE_MB} MB.")
            try:
                img = PILImage.open(io.BytesIO(raw)); img.verify()
                img = PILImage.open(io.BytesIO(raw)).convert("RGB")
                if img.width < 100 or img.height < 100 or img.width > 5000 or img.height > 5000:
                    raise ValueError()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                img.thumbnail((900, 900)); img.save(tmp.name, format="JPEG", quality=90); tmp.close(); photo_path = tmp.name
            except Exception:
                raise HTTPException(status_code=400, detail="Please upload a valid JPG, PNG or WEBP photo.")

        out = io.BytesIO(); doc = SimpleDocTemplate(out, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)
        styles = getSampleStyleSheet()
        name_style = ParagraphStyle("cvName", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=25, textColor=colors.HexColor("#172554"), spaceAfter=4)
        role_style = ParagraphStyle("cvRole", parent=styles["Normal"], fontSize=10.5, textColor=colors.HexColor("#4F46E5"), leading=14)
        contact_style = ParagraphStyle("cvContact", parent=styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#475569"), leading=12)
        body_style = ParagraphStyle("cvBody", parent=styles["BodyText"], fontSize=9.5, leading=13, textColor=colors.HexColor("#1E293B"), spaceAfter=3)
        section_style = ParagraphStyle("cvSection", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#3730A3"), spaceBefore=7, spaceAfter=4, borderWidth=0, borderPadding=0)
        story=[]
        contact = "  •  ".join(escape(x.strip()) for x in [email, phone, location] if x.strip())
        header_text=[Paragraph(escape(name.strip()), name_style)]
        if target_role.strip(): header_text.append(Paragraph(escape(target_role.strip()), role_style))
        if contact: header_text.append(Paragraph(contact, contact_style))
        if photo_path:
            photo_img=Image(photo_path, width=28*mm, height=28*mm)
            header=Table([[header_text, photo_img]], colWidths=[140*mm, 30*mm])
            header.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('ALIGN',(1,0),(1,0),'RIGHT'),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            story.append(header)
        else:
            story.extend(header_text)
        story.append(Spacer(1, 5*mm))
        section_names={"NAME/CONTACT","PROFESSIONAL SUMMARY","EDUCATION","SKILLS","PROJECTS","EXPERIENCE","CERTIFICATIONS","SUMMARY"}
        for line in _safe_lines(cv_text):
            cleaned = re.sub(r"^[#*\-\s]+", "", line).strip()
            upper = cleaned.rstrip(":").upper()
            if upper in section_names:
                if upper == "NAME/CONTACT": continue
                story.append(Paragraph(escape(cleaned.rstrip(":")), section_style))
            elif re.match(r"^[•\-*]", line):
                story.append(Paragraph("• " + escape(re.sub(r"^[•\-*]\s*", "", line)), body_style))
            else:
                story.append(Paragraph(escape(cleaned), body_style))
        doc.build(story)
        filename = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())[:50] or "CareerGuide_CV"
        return Response(out.getvalue(), media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{filename}_CV.pdf"'})
    finally:
        if photo_path:
            try: os.remove(photo_path)
            except OSError: pass


app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/")
def index():
    return FileResponse("web/static/index.html")
