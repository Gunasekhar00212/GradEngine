import { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

function Upload({ onUploaded }) {
  const [student, setStudent] = useState();
  const [rubric, setRubric] = useState();
  const [status, setStatus] = useState("");
  async function submit(event) {
    event.preventDefault();
    if (!student || !rubric) return setStatus("Choose both files to continue.");
    setStatus("Rendering PDF pages…");
    try {
      const form = new FormData();
      form.append("student_file", student); form.append("rubric_file", rubric); form.append("mode", "manual");
      const upload = await api("/api/upload", { method: "POST", body: form });
      onUploaded(await api(`/api/sessions/${upload.session_id}`));
    } catch (error) { setStatus(error.message); }
  }
  return <main className="landing"><form className="card upload" onSubmit={submit}><p className="eyebrow">GRADEENGINE</p><h1>Upload an answer sheet.</h1><p>Next, you will click each question start across the rendered pages.</p><label>Student PDF<input type="file" accept="application/pdf" onChange={(e) => setStudent(e.target.files?.[0])} /></label><label>Rubric JSON<input type="file" accept="application/json" onChange={(e) => setRubric(e.target.files?.[0])} /></label>{status && <p className="status">{status}</p>}<button>Continue to splitting</button></form></main>;
}

function Split({ session, onComplete, onBack }) {
  const [page, setPage] = useState(0), [clicks, setClicks] = useState([]), [height, setHeight] = useState(0), [status, setStatus] = useState("");
  const marks = clicks.filter((click) => click.page_index === page);
  const image = `/api/sessions/${session.session_id}/pages/${page}`;
  function renumber(all) { return all.map((click, index) => ({ ...click, question_index: index + 1 })); }
  function mark(event) { if (!height) return; const box = event.currentTarget.getBoundingClientRect(); setClicks((all) => renumber([...all, { page_index: page, y: Math.round((event.clientY - box.top) / box.height * height), question_index: all.length + 1 }])); }
  async function finish() {
    try {
      setStatus("Saving question starts…");
      await api(`/api/sessions/${session.session_id}/manual-clicks`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(clicks) });
      setStatus("Cropping question images…");
      await api(`/api/sessions/${session.session_id}/split`, { method: "POST" });
      setStatus("Reading handwriting with OCR…");
      await api(`/api/sessions/${session.session_id}/extract`, { method: "POST" });
      setStatus("Grading answers against the rubric…");
      onComplete((await api(`/api/sessions/${session.session_id}/grade`, { method: "POST" })).evaluation);
      setStatus("");
    } catch (error) { setStatus(error.message); }
  }
  return <main className="split"><header><div><p className="eyebrow">STEP 2 OF 3</p><h1>Mark question starts</h1></div><div><button type="button" className="secondary" onClick={onBack}>Start over</button><button type="button" onClick={finish}>Finish and grade</button></div></header><section><aside><p>Click once where each new question begins. The next click ends the previous question.</p>{session.page_paths.map((_, index) => <button type="button" key={index} className={index === page ? "tab selected" : "tab"} onClick={() => setPage(index)}>Page {index + 1}<small>{clicks.filter((point) => point.page_index === index).length} starts</small></button>)}<button type="button" className="secondary wide" onClick={() => setClicks((all) => renumber(all.filter((point) => point.page_index !== page)))}>Clear this page</button><button type="button" className="secondary wide" onClick={() => setClicks((all) => renumber(all.slice(0, -1)))}>Undo last mark</button><div className="status" style={{ color: "#d9e8fb" }}>Questions marked: {clicks.length}</div>{status && <p className="status">{status}</p>}</aside><div className="stage"><div className="canvas" onClick={mark}><img src={image} onLoad={(e) => setHeight(e.currentTarget.naturalHeight)} />{height > 0 && marks.map((point) => <i key={`${point.question_index}-${point.page_index}-${point.y}`} style={{ top: `${point.y / height * 100}%` }}>Q{point.question_index}</i>)}</div></div></section></main>;
}

function Results({ result, reset }) {
  const questions = useMemo(() => result?.questions || (result ? [result] : []), [result]);
  return <main className="results"><header><div><p className="eyebrow">STEP 3 OF 3</p><h1>Evaluation complete</h1></div><button className="secondary" onClick={reset}>Grade another PDF</button></header>{questions.map((question) => <article className="card score" key={question.question_id}><div className="number">{question.evaluation.marks}<small>/ {question.evaluation.max_marks}</small></div><div><p className="eyebrow">{question.question_id}</p><h2>{question.evaluation.needs_human_review ? "Needs teacher review" : "Ready for review"}</h2><p>{question.evaluation.feedback}</p><span>Confidence: {Math.round(question.evaluation.evaluation_confidence * 100)}%</span><details><summary>Criterion feedback</summary><pre>{JSON.stringify(question.evaluation.criteria_results, null, 2)}</pre></details></div></article>)}</main>;
}

function App() { const [screen, setScreen] = useState("upload"), [session, setSession] = useState(), [result, setResult] = useState(); if (screen === "split") return <Split session={session} onBack={() => setScreen("upload")} onComplete={(next) => { setResult(next); setScreen("results"); }} />; if (screen === "results") return <Results result={result} reset={() => setScreen("upload")} />; return <Upload onUploaded={(next) => { setSession(next); setScreen("split"); }} />; }
createRoot(document.getElementById("root")).render(<App />);
