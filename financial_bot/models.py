# models.py
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
db = SQLAlchemy()

class User(UserMixin,db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    portfolios = db.relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Portfolio(db.Model):
    __tablename__ = "portfolios"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), default="My Portfolio")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    user = db.relationship("User", back_populates="portfolios")
    holdings = db.relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")

class Holding(db.Model):
    __tablename__ = "holdings"
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(32), nullable=False, index=True)
    quantity = db.Column(db.Float, nullable=False, default=0.0)
    avg_price = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    portfolio_id = db.Column(db.Integer, db.ForeignKey("portfolios.id"), nullable=False)

    portfolio = db.relationship("Portfolio", back_populates="holdings")


class IpoReport(db.Model):
    __tablename__ = "ipo_reports"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    # What the UI shows
    title = db.Column(db.String(512), nullable=True)
    filename = db.Column(db.String(512), nullable=True)       # new: store uploaded filename if any
    content_md = db.Column(db.Text, nullable=True)            # markdown shown in the modal
    raw_text = db.Column(db.Text, nullable=True)              # optional stored text
    excerpt = db.Column(db.String(512), nullable=True)

    # Scores & structured fields
    overall_score = db.Column(db.Float, nullable=True)        # general overall score
    sebi_score = db.Column(db.Integer, nullable=True)         # heuristic / SEBI checks score
    llm_score = db.Column(db.Integer, nullable=True)          # optional LLM derived score

    # Heuristic metadata
    heuristic_meta = db.Column(db.Text, nullable=True)        # JSON/text of heuristics (stringified)
    checklist = db.Column(db.Text, nullable=True)             # checklist (JSON/text)

    # SEBI checks (store as JSON when DB supports it, else Text)
    try:
        # if the DB dialect supports JSON (Postgres), SQLAlchemy provides db.JSON
        sebi_checks = db.Column(db.JSON, nullable=True)
    except Exception:
        sebi_checks = db.Column(db.Text, nullable=True)

    # Quick-extracted financials (string versions kept for display)
    revenue = db.Column(db.String(256), nullable=True)
    profit  = db.Column(db.String(256), nullable=True)
    debt    = db.Column(db.String(256), nullable=True)
    promoter_mentioned = db.Column(db.Boolean, default=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
