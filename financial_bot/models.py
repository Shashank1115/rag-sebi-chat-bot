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
    content_md = db.Column(db.Text, nullable=False)   # markdown shown in the modal
    raw_text = db.Column(db.Text, nullable=True)      # optional stored text

    # For the list + compare UI
    sebi_score = db.Column(db.Integer, nullable=True)
    llm_score = db.Column(db.Integer, nullable=True)

    revenue = db.Column(db.String(128), nullable=True)
    profit  = db.Column(db.String(128), nullable=True)
    debt    = db.Column(db.String(128), nullable=True)
    promoter_mentioned = db.Column(db.Boolean, default=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

