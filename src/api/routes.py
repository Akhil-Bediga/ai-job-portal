"""REST endpoints: job search + stats over MySQL, parsing + recommendations over the engine."""
from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, request
from sqlalchemy import text

from src.api.service import service
from src.db.mysql import get_engine
from src.nlp.skills import extract_skills

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _query(sql: str, params: dict | None = None) -> list[dict[str, Any]]:
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params or {}).fetchall()
    return [dict(r._mapping) for r in rows]


@api_bp.get("/health")
def health():
    return {"status": "ok", "recommender_loaded": service.is_ready()}


@api_bp.get("/jobs")
def jobs():
    q = request.args.get("q", "").strip()
    family = request.args.get("family", "").strip()
    limit = min(int(request.args.get("limit", 20)), 100)

    where, params = [], {"limit": limit}
    if q:
        where.append("(title LIKE :q OR company LIKE :q)")
        params["q"] = f"%{q}%"
    if family:
        where.append("family = :family")
        params["family"] = family
    clause = "WHERE " + " AND ".join(where) if where else ""

    rows = _query(f"""
        SELECT job_id, title, company, family, city, state, remote,
               seniority, salary_min, salary_max, salary_avg, posted_date
        FROM jobs
        {clause}
        ORDER BY posted_date DESC
        LIMIT :limit
    """, params)
    return jsonify({"count": len(rows), "jobs": rows})


@api_bp.get("/jobs/<job_id>")
def job_detail(job_id: str):
    rows = _query("SELECT * FROM jobs WHERE job_id = :id", {"id": job_id})
    if not rows:
        return {"error": "job not found"}, 404
    job = rows[0]
    skills = [r["skill"] for r in _query(
        "SELECT skill FROM job_skills WHERE job_id = :id ORDER BY skill", {"id": job_id})]
    job["skills"] = skills
    return jsonify(job)


@api_bp.post("/resume/parse")
def parse_resume():
    body = request.get_json(silent=True) or {}
    resume_text = (body.get("text") or "").strip()
    if not resume_text:
        return {"error": "missing 'text'"}, 400
    skills = extract_skills(resume_text)
    return jsonify({
        "skills": skills,
        "skill_count": len(skills),
        "char_count": len(resume_text),
    })


@api_bp.post("/recommend")
def recommend():
    body = request.get_json(silent=True) or {}
    resume_text = (body.get("text") or "").strip()
    top_n = min(int(body.get("top_n", 10)), 50)
    if not resume_text:
        return {"error": "missing 'text'"}, 400

    matches = service.recommend(resume_text, top_n=top_n)
    return jsonify({
        "resume_skills": extract_skills(resume_text),
        "count": len(matches),
        "matches": matches,
    })


@api_bp.get("/stats/skills")
def stats_skills():
    limit = min(int(request.args.get("limit", 15)), 50)
    rows = _query("""
        SELECT skill, COUNT(*) AS demand
        FROM job_skills
        GROUP BY skill
        ORDER BY demand DESC
        LIMIT :limit
    """, {"limit": limit})
    return jsonify({"skills": rows})


@api_bp.get("/stats/salary")
def stats_salary():
    rows = _query("""
        SELECT family,
               ROUND(AVG(salary_avg)) AS avg_salary,
               MIN(salary_min)        AS min_salary,
               MAX(salary_max)        AS max_salary,
               COUNT(*)               AS postings
        FROM jobs
        GROUP BY family
        ORDER BY avg_salary DESC
    """)
    return jsonify({"salary_by_family": rows})


@api_bp.get("/stats/families")
def stats_families():
    rows = _query("SELECT family, COUNT(*) AS postings FROM jobs GROUP BY family ORDER BY postings DESC")
    return jsonify({"families": rows})


@api_bp.post("/reload")
def reload_model():
    n = service.reload()
    return {"reloaded": True, "jobs_indexed": n}
