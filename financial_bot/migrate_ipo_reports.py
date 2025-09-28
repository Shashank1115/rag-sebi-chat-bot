# migrate_ipo_reports.py
from backend import app, db
from sqlalchemy import text

stmts = [
    "ALTER TABLE ipo_reports ADD COLUMN revenue VARCHAR(128);",
    "ALTER TABLE ipo_reports ADD COLUMN profit VARCHAR(128);",
    "ALTER TABLE ipo_reports ADD COLUMN debt VARCHAR(128);",
    "ALTER TABLE ipo_reports ADD COLUMN promoter_mentioned BOOLEAN DEFAULT 0;",
]

with app.app_context():
    print("DB URL:", app.config["SQLALCHEMY_DATABASE_URI"])
    conn = db.engine.connect()
    for s in stmts:
        try:
            conn.execute(text(s))
            print("OK:", s)
        except Exception as e:
            print("SKIP/ERR:", s, "=>", e)
    conn.close()
