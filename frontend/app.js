const $ = (id) => document.getElementById(id);

const SAMPLE = `Data Engineer with 5 years of experience building and shipping production systems.
Comfortable working in a fast paced agile environment and owning projects end to end.
Experience:
- You will design, build, and maintain solutions using apache spark and apache airflow.
- Own the end to end lifecycle of systems built on snowflake and dbt.
- Partner with the team to scale our use of apache kafka and aws.
Skills: python, sql, apache spark, apache airflow, apache kafka, etl, snowflake, dbt, aws.`;

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

function pct(x) { return Math.round(x * 100); }

function chips(skills, cls) {
  if (!skills || !skills.length) return "";
  return `<div class="chips">${skills.map(s => `<span class="chip ${cls}">${s}</span>`).join("")}</div>`;
}

function renderMatches(data) {
  $("status").textContent = `${data.count} matches · resume skills: ${data.resume_skills.join(", ") || "none found"}`;
  $("results").innerHTML = data.matches.map(m => `
    <div class="match">
      <div class="match-top">
        <div>
          <div class="match-title">${m.title || "Untitled"}</div>
          <div class="match-company">${m.company || ""} · ${m.city || ""}, ${m.state || ""}${m.remote ? " · remote" : ""}</div>
        </div>
        <div class="score">${pct(m.score)}%</div>
      </div>
      <div class="meta">
        <span class="badge">${m.family}</span>
        <span class="badge">${m.seniority}</span>
        ${m.salary_avg ? `<span class="badge">~$${Number(m.salary_avg).toLocaleString()}</span>` : ""}
        &nbsp; text ${pct(m.text_score)}% · skills ${pct(m.skill_score)}%
      </div>
      ${m.matched_skills.length ? `<div class="chip-label">matched</div>${chips(m.matched_skills, "match")}` : ""}
      ${m.missing_skills.length ? `<div class="chip-label">in the role, not on your resume</div>${chips(m.missing_skills, "miss")}` : ""}
    </div>
  `).join("");
}

async function findMatches() {
  const text = $("resume").value.trim();
  if (!text) { $("status").textContent = "Paste a resume first."; return; }
  $("status").textContent = "Scoring...";
  $("results").innerHTML = "";
  try {
    const data = await api("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, top_n: Number($("top-n").value) }),
    });
    renderMatches(data);
  } catch (e) {
    $("status").textContent = "Error: " + e.message;
  }
}

function renderBars(el, rows, labelKey, valueKey, fmt) {
  const max = Math.max(...rows.map(r => r[valueKey]), 1);
  el.innerHTML = rows.map(r => `
    <div class="row">
      <div class="label"><span>${r[labelKey]}</span><span>${fmt(r[valueKey])}</span></div>
      <div class="bar-wrap"><div class="bar-fill" style="width:${(r[valueKey] / max) * 100}%"></div></div>
    </div>
  `).join("");
}

async function loadStats() {
  try {
    const s = await api("/api/stats/skills?limit=12");
    renderBars($("skill-stats"), s.skills, "skill", "demand", v => v);
    const sal = await api("/api/stats/salary");
    renderBars($("salary-stats"), sal.salary_by_family, "family", "avg_salary",
      v => "$" + Number(v).toLocaleString());
  } catch (e) {
    $("skill-stats").textContent = "Stats unavailable (run the pipeline first).";
  }
}

$("match-btn").addEventListener("click", findMatches);
$("sample-btn").addEventListener("click", () => { $("resume").value = SAMPLE; });
loadStats();
