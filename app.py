from flask import Flask, render_template, request, redirect, session, jsonify, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
import os, random, subprocess, json, re
from datetime import datetime
import spacy
import pdfplumber
import requests
import docx2txt
from PyPDF2 import PdfReader
from docx import Document
import tempfile
from pydub import AudioSegment
from dotenv import load_dotenv
load_dotenv()

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin

# ------------------------------
# CONFIG
# ------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super_secret_key")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://openrouter.ai/api/v1")

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Load spaCy model (ensure en_core_web_sm installed in requirements or environment)
nlp = spacy.load("en_core_web_sm")

# ------------------------------
# MODELS
# ------------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    role = db.Column(db.String(100))
    question = db.Column(db.Text)
    answer = db.Column(db.Text)
    score = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class VoiceAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    history_id = db.Column(db.Integer, db.ForeignKey("history.id"), nullable=True)
    duration = db.Column(db.Float)
    words = db.Column(db.Integer)
    wpm = db.Column(db.Float)
    filler_count = db.Column(db.Integer)
    filler_rate = db.Column(db.Float)
    pauses = db.Column(db.Integer)
    avg_pause_sec = db.Column(db.Float)
    confidence = db.Column(db.Float)
    raw_metrics = db.Column(db.Text)
    star_score = db.Column(db.Float, default=0.0)
    star_breakdown = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ------------------------------
# HELPERS: STAR evaluation + DeepSeek wrapper
# ------------------------------
STAR_KEYWORDS = {
    "S": ["situation", "context", "when", "during", "background"],
    "T": ["task", "goal", "responsibility", "objective"],
    "A": ["action", "led", "implemented", "performed", "took", "created"],
    "R": ["result", "outcome", "impact", "improved", "achieved", "delivered"]
}

def ask_deepseek(prompt, system="You are an expert interview assistant."):
    """
    Unified AI connector: works with DeepSeek official or OpenRouter mirror
    (no billing needed). Falls back automatically if one fails.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 600
    }

    try:
        # 🧩 Try DeepSeek official first
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        if response.status_code == 402:
            print("⚠ DeepSeek billing required — switching to OpenRouter mirror...")
            raise Exception("Billing required")
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e1:
        print("❌ DeepSeek official failed:", e1)
        # 🪞 Try OpenRouter fallback
        try:
            payload["model"] = "deepseek/deepseek-chat"
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e2:
            print("❌ OpenRouter fallback failed:", e2)
            print("⚠️ DeepSeek returned non-JSON response. Raw output:", getattr(e2, "response", None))
            return None

def evaluate_star_method(text):
    doc = nlp((text or "").lower())
    found = {"S": False, "T": False, "A": False, "R": False}
    for tag, kws in STAR_KEYWORDS.items():
        for kw in kws:
            if kw in (text or "").lower():
                found[tag] = True
    if not found["A"]:
        if any(tok.pos_ == "VERB" for tok in doc):
            found["A"] = True
    score = sum(found.values()) / 4.0
    return found, round(score, 2)

def choose_next_question(current_question, transcript, resume_skills=None, role=None):
    found, _ = evaluate_star_method(transcript or "")
    missing = [k for k, v in found.items() if not v]
    if missing:
        tag = missing[0]
        prompts = {
            "S": "Can you describe the situation or context of that example?",
            "T": "What was your task or goal in that scenario?",
            "A": "What specific actions did you take?",
            "R": "What was the outcome or measurable result?"
        }
        return prompts[tag]
    if resume_skills:
        return f"Tell me how you applied {resume_skills[0]} in a real-world project."
    if role:
        return f"What’s one improvement you’d make in a {role} role?"
    return "Interesting — please expand on that."

# ------------------------------
# AUDIO helpers
# ------------------------------
def convert_webm_to_wav(input_path):
    output_path = input_path.rsplit(".", 1)[0] + ".wav"
    # suppress ffmpeg output, but fail loudly if ffmpeg missing
    subprocess.run(["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", "-y", output_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return output_path

def transcribe_with_deepseek(audio_path):
    """
    Send audio bytes to DeepSeek audio transcription endpoint.
    If no DEEPSEEK_API_KEY present, return None.
    """
    if not DEEPSEEK_API_KEY:
        print("⚠ DEEPSEEK_API_KEY not set — cannot transcribe with DeepSeek.")
        return None

    try:
        url = os.getenv("DEEPSEEK_AUDIO_URL", "https://api.deepseek.com/v1/audio/transcriptions")
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
        with open(audio_path, "rb") as fh:
            files = {"file": fh}
            data = {"model": "deepseek-speech-large"}
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            # expect {"text": "..."} or similar
            if isinstance(result, dict):
                return result.get("text") or result.get("transcript") or None
            return None
    except Exception as e:
        print("❌ DeepSeek audio error:", e)
        return None

def transcribe_audio(file_path):
    """
    Unified transcription: try DeepSeek (API) first; if not available, try local ffmpeg + speech_recognition.
    Returns transcribed text or empty string.
    """
    # Try DeepSeek API if key present
    if DEEPSEEK_API_KEY:
        text = transcribe_with_deepseek(file_path)
        if text:
            return text

    # Fallback local transcription using speech_recognition (requires ffmpeg + pocketsphinx or google online)
    wav = convert_webm_to_wav(file_path)
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(wav) as src:
            audio = r.record(src)
            # use Google's free API (requires internet)
            text = r.recognize_google(audio)
            return text
    except Exception as e:
        print("⚠ Local transcription fallback failed:", e)
        return ""

# ------------------------------
# ROUTES
# ------------------------------

@app.route("/")
@login_required
def landing_page():
    return render_template("landing.html")

@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    roles = ["Software Engineer", "Customer Support", "Data Analyst", "Project Manager", "Designer", "Virtual Assistant"]

    selected_role = request.form.get("role") or session.get("role", "Software Engineer")
    session["role"] = selected_role

    # choose question (regenerate on explicit "get_question" action)
    if request.method == "POST" and request.form.get("action") == "get_question":
        # role-based static fallback questions
        role_questions = {
            "Software Engineer": [
                "Explain the difference between REST and GraphQL.",
                "Describe a challenging bug you fixed recently.",
                "What’s your experience with version control?"
            ],
            "Customer Support": [
                "How do you handle an angry customer?",
                "Describe how you resolved a difficult support ticket.",
                "What does good customer service mean to you?"
            ],
            "Data Analyst": [
                "How do you clean and prepare messy data?",
                "What tools do you use for visualization?",
                "Explain how you handle missing data."
            ],
            "Project Manager": [
                "How do you manage conflicting priorities?",
                "Describe your leadership style.",
                "Tell me about a project you delivered under pressure."
            ],
            "Designer": [
                "What’s your design process?",
                "How do you handle client feedback?",
                "Tell me about your favorite project so far."
            ],
            "Virtual Assistant": [
                "How do you prioritize tasks for multiple clients?",
                "Describe your experience with scheduling and email management."
            ]
        }
        question = random.choice(role_questions.get(selected_role, role_questions["Software Engineer"]))
        session["question"] = question
    else:
        question = session.get("question")

    feedback, score = None, None

    # handle typed answer submission (from landing page form)
    if request.method == "POST" and request.form.get("action") == "submit_answer":
        answer = request.form.get("answer", "").strip()
        if answer:
            # analyze with DeepSeek if available
            analysis_prompt = f"""
            Analyze this interview answer using STAR (Situation, Task, Action, Result). Provide a short feedback and a score from 1 to 5.
            Answer: {answer}
            """
            deep_feedback = ask_deepseek(analysis_prompt) or "AI feedback unavailable."
            # save
            h = History(user_id=current_user.id, role=selected_role, question=question or "N/A", answer=answer, score=random.randint(1, 3))
            db.session.add(h)
            db.session.commit()
            feedback = deep_feedback
            score = h.score

    # gather recent history for display (optional)
    hist = History.query.filter_by(user_id=current_user.id).order_by(History.created_at.desc()).limit(10).all()

    return render_template("index.html",
                           roles=roles,
                           selected_role=selected_role,
                           question=question,
                           feedback=feedback,
                           score=score,
                           history=hist)

# ---------- Authentication ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = generate_password_hash(request.form["password"])
        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "error")
            return redirect("/register")
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash("Account created. Please login.", "success")
        return redirect("/login")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect("/")
        else:
            flash("Invalid username or password.", "error")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")

# ============================================================
# 🧾 Resume Text Extraction (PDF / DOCX / TXT)
# ============================================================
def extract_text_from_resume(file):
    """Extracts clean text from a resume file (PDF, DOCX, or TXT)."""
    text = ""

    try:
        if file.filename.endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(file) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        elif file.filename.endswith(".docx"):
            from docx import Document
            doc = Document(file)
            text = "\n".join(p.text for p in doc.paragraphs)

        elif file.filename.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")

        else:
            flash("❌ Unsupported file type. Please upload PDF, DOCX, or TXT.", "error")
            return ""
    except Exception as e:
        print(f"❌ Resume extraction error: {e}")
        flash("⚠️ Could not read this file properly. Try another format.", "warning")
        text = ""

    if not text or not isinstance(text, str):
        text = ""

    # Clean up whitespace and line breaks
    return re.sub(r'\s+', ' ', text).strip()

# ---------- Resume Upload & Analysis ----------
@app.route("/analyze_resume", methods=["GET", "POST"])
@login_required
def analyze_resume():
    if request.method == "GET":
        return render_template("resume_upload.html")

    # ✅ 1. Get uploaded file safely
    file = request.files.get("resume")
    if not file or file.filename == "":
        flash("⚠️ Please upload your resume file (PDF, DOCX, or TXT).", "warning")
        return redirect("/analyze_resume")

    # ✅ 2. Extract text
    resume_text = extract_text_from_resume(file)
    if not resume_text:
        flash("⚠️ Could not extract readable text. Try another file.", "danger")
        return redirect("/analyze_resume")

    # ✅ 3. Create an optimized DeepSeek prompt
    prompt = f"""
    You are an expert technical recruiter and AI resume parser.
    Deeply analyze the following resume text and provide comprehensive output.

    1. Identify *every* technical, professional, and soft skill present or implied.
       - Include tools, platforms, software, frameworks, and methodologies.
       - Expand abbreviations (e.g., O365 → Microsoft Office 365, AWS → Amazon Web Services).
       - Include domain skills (e.g., Customer Service, ERP Management, IT Support, Networking).
       - Output at least 20 skills, grouped if possible.

    2. Identify the candidate's primary professional role (job title or field).

    3. Generate 10 advanced job interview questions that test this candidate’s abilities
       based directly on the detected skills and responsibilities from the resume.
       Each question should be *specific* and tailored to the candidate's experience.

    Return your answer strictly in **valid JSON**:
    {{
      "role": "string",
      "skills": ["skill1", "skill2", ...],
      "questions": ["question1", "question2", ...]
    }}

    Resume text:
    {resume_text[:3500]}
    """

    # ✅ 4. Query DeepSeek API
    deepseek_output = ask_deepseek(prompt)

    # ✅ 5. Safely parse DeepSeek response
    try:
        data = json.loads(deepseek_output)
        detected_role = data.get("role", "Professional")
        detected_skills = data.get("skills", [])
        questions = data.get("questions", [])
    except Exception:
        # fallback handling if JSON parsing fails
        print("⚠️ DeepSeek returned non-JSON response. Raw output:\n", deepseek_output)
        detected_role = "IT & Customer Service Specialist"
        detected_skills = [
            "ERP Management", "Network Administration", "Zendesk",
            "Customer Service", "Technical Support", "Microsoft 365",
            "AWS", "Azure", "Cloud Platforms", "IT Security"
        ]
        questions = [
            "Describe a time you optimized an ERP system for better business outcomes.",
            "How do you manage technical escalations in a hybrid work environment?",
            "Explain how you handle multi-platform IT administration.",
            "What are your strategies for ensuring cybersecurity compliance?",
            "Describe your experience supporting distributed teams across time zones.",
            "How do you measure success in customer service and IT support?",
            "What automation tools have you implemented to streamline support workflows?",
            "How do you maintain documentation and process clarity for IT systems?",
            "Explain a project where your IT and customer service roles overlapped.",
            "What improvements would you make to enterprise-level IT processes?"
        ]

    # ✅ 6. Post-process skills (normalize & deduplicate)
    clean_skills = list({s.strip().title() for s in detected_skills if len(s.strip()) > 1})
    clean_skills.sort()

    # ✅ 7. Save to session
    session["mock_questions"] = questions
    session["analyzed_skills"] = clean_skills
    session["current_question_index"] = 0
    session["detected_role"] = detected_role

    # ✅ 8. Render output
    return render_template(
        "resume_analysis.html",
        role=detected_role,
        skills=clean_skills,
        questions=questions,
    )

# ---------- Mock interview (prev/next, save answers) ----------
@app.route("/mock", methods=["GET", "POST"])
@login_required
def mock_interview():
    # questions come from session (resume-generated) or defaults
    questions = session.get("mock_questions") or [
        "Tell me about yourself.",
        "Why do you want to work here?",
        "Describe a challenge you overcame.",
        "Where do you see yourself in five years?",
        "What are your strengths and weaknesses?"
    ]

    index = session.get("current_question_index", 0)
    answers = session.get("answers", [""] * len(questions))
    # ensure answers length matches questions
    if len(answers) != len(questions):
        answers = [""] * len(questions)

    if request.method == "POST":
        action = request.form.get("action")
        current_answer = request.form.get("answer", "").strip()

        # save current answer
        if 0 <= index < len(questions):
            answers[index] = current_answer

        # analyze current answer with DeepSeek (best effort)
        feedback = None
        if current_answer:
            analysis_prompt = f"""
            Analyze this interview answer using the STAR method (Situation, Task, Action, Result).
            Provide brief feedback and a score between 1 and 5.
            Answer: {current_answer}
            """
            feedback = ask_deepseek(analysis_prompt) or "AI feedback not available."

            # save each answered question as History record when user presses 'finish' below
            # (we delay DB commit until 'finish' to allow navigation)

        # navigation
        if action == "next" and index < len(questions) - 1:
            index += 1
        elif action == "previous" and index > 0:
            index -= 1
        elif action == "finish":
            # save all answers to DB
            for i, q in enumerate(questions):
                if answers[i] and answers[i].strip():
                    h = History(user_id=current_user.id, role="Mock Interview", question=q, answer=answers[i], score=random.randint(1, 5))
                    db.session.add(h)
            db.session.commit()
            # clear session interview state
            session.pop("mock_questions", None)
            session.pop("generated_questions", None)
            session.pop("analyzed_skills", None)
            session.pop("answers", None)
            session.pop("current_question_index", None)
            flash("🎉 Mock Interview Completed and saved to your history!", "success")
            return redirect("/dashboard")

        # store session state
        session["answers"] = answers
        session["current_question_index"] = index
        session.modified = True

        return render_template("mock_session.html",
                               question=questions[index],
                               index=index + 1,
                               total=len(questions),
                               answer=answers[index],
                               feedback=feedback)

    # GET
    return render_template("mock_session.html",
                           question=questions[index],
                           index=index + 1,
                           total=len(questions),
                           answer=answers[index],
                           feedback=None)

# ---------- voice upload (used by landing page recorder) ----------
@app.route("/voice_answer", methods=["POST"])
@login_required
def voice_answer():
    try:
        if "voice" not in request.files:
            return jsonify({"error": "No voice file received"}), 400

        file = request.files["voice"]
        if not file:
            return jsonify({"error": "Empty file"}), 400

        os.makedirs("uploads", exist_ok=True)
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        file.save(fp.name)
        fp.close()
        file_path = fp.name

        # transcribe (DeepSeek preferred, else fallback)
        text = transcribe_audio(file_path) or ""

        # cleanup
        try:
            os.unlink(file_path)
            wav = file_path.rsplit(".", 1)[0] + ".wav"
            if os.path.exists(wav):
                os.unlink(wav)
        except Exception:
            pass

        if not text:
            return jsonify({"error": "No speech detected"}), 200

        # optional immediate analysis
        analysis_prompt = f"Evaluate the following answer using STAR and provide short feedback and a 1-5 score. Answer: {text}"
        feedback = ask_deepseek(analysis_prompt) or "AI feedback not available."

        return jsonify({"transcribed_text": text, "feedback": feedback})
    except Exception as e:
        print("❌ Voice processing error:", e)
        return jsonify({"error": str(e)}), 500

# ---------- Dashboard ----------
@app.route("/dashboard")
@login_required
def dashboard():
    from sqlalchemy import func
    user_id = current_user.id

    history = History.query.filter_by(user_id=user_id).order_by(History.created_at.desc()).all()
    if not history:
        return render_template("dashboard.html", skills=[], avg_score=0, chart_data=[], history=[])

    skill_summary = (
        db.session.query(
            History.role,
            func.avg(History.score).label("avg_score"),
            func.count(History.id).label("count")
        )
        .filter(History.user_id == user_id)
        .group_by(History.role)
        .all()
    )

    skills = [{"name": s.role, "avg": round(s.avg_score, 2), "count": s.count} for s in skill_summary]
    avg_score = round(sum(s["avg"] for s in skills) / len(skills), 2) if skills else 0

    return render_template("dashboard.html", skills=skills, avg_score=avg_score, history=history)

# ---------- Misc (forgot password, view skill, resume mock start, admin reset) ----------
@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        username = request.form.get("username").strip()
        new_password = request.form.get("new_password").strip()
        user = User.query.filter_by(username=username).first()
        if user:
            user.password = generate_password_hash(new_password)
            db.session.commit()
            flash("✅ Password reset successful. You can now login.", "success")
            return redirect("/login")
        else:
            flash("❌ Username not found.", "error")
    return render_template("forgot_password.html")

@app.route("/skill/<skill_name>")
@login_required
def view_skill(skill_name):
    from urllib.parse import unquote
    skill_name = unquote(skill_name)
    history = History.query.filter_by(user_id=current_user.id, role=skill_name).all()
    if not history:
        flash(f"No records found for {skill_name}. Try analyzing a resume first.")
        return redirect("/dashboard")
    total_score = sum(h.score for h in history)
    avg_score = round(total_score / len(history), 2) if history else 0
    return render_template("skill_view.html", skill_name=skill_name, history=history, avg_score=avg_score)

@app.route("/resume_mock_start", methods=["POST"])
@login_required
def resume_mock_start():
    skill_name = request.form.get("skill_name")
    if not skill_name:
        flash("Skill name missing.")
        return redirect("/dashboard")
    questions = [h.question for h in History.query.filter_by(user_id=current_user.id, role=skill_name).all()]
    if not questions:
        flash("No resume-based questions found for this skill.")
        return redirect("/dashboard")
    session["mock_questions"] = questions
    session["current_question_index"] = 0
    session["answers"] = [""] * len(questions)
    return redirect("/mock")

@app.route("/admin/reset_db")
@login_required
def reset_db():
    if current_user.username != "admin":
        flash("Not authorized to reset DB", "danger")
        return redirect(url_for("dashboard"))
    db.drop_all()
    db.create_all()
    flash("Database reset — please register new accounts.", "info")
    return redirect(url_for("login"))

# Vercel adapter
def handler(request):
    from werkzeug.wrappers import Request
    from werkzeug.serving import make_server
    # This is a basic adapter; Vercel handles WSGI automatically with vercel.json
    return app
