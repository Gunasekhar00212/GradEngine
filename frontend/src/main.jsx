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
  const [page, setPage] = useState(0);
  const [clicks, setClicks] = useState([]);
  const [height, setHeight] = useState(0);
  const [status, setStatus] = useState("");
  const [preview, setPreview] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const marks = clicks.filter((click) => click.page_index === page);
  const image = `/api/sessions/${session.session_id}/pages/${page}`;

  function renumber(all) {
    return [...all]
      .sort((a, b) => a.page_index - b.page_index || a.y - b.y)
      .map((click, index) => ({ ...click, question_index: index + 1 }));
  }

  function mark(event) {
    if (!height) return;
    const box = event.currentTarget.getBoundingClientRect();
    const y = Math.round((event.clientY - box.top) / box.height * height);
    setClicks((all) => renumber([...all, { page_index: page, y, question_index: all.length + 1 }]));
  }

  async function createSplitPreview() {
    try {
      setStatus("Saving question starts…");
      await api(`/api/sessions/${session.session_id}/manual-clicks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(clicks),
      });
      setStatus("Creating question crops…");
      await api(`/api/sessions/${session.session_id}/split`, { method: "POST" });
      setStatus("Checking split…");
      const nextPreview = await api(`/api/sessions/${session.session_id}/split-preview`);
      setPreview(nextPreview);
      setShowPreview(true);
      setStatus("");
    } catch (error) {
      setStatus(error.message);
    }
  }

  function applyBoundaryChange(questionIndex, field, value) {
    setPreview((current) => {
      if (!current) return current;
      return {
        ...current,
        previews: current.previews.map((item, index) => index === questionIndex
          ? { ...item, boundary: { ...item.boundary, [field]: value === "" ? null : Number(value) } }
          : item),
      };
    });
  }

  async function adjustAndRefresh() {
    if (!preview?.previews?.length) return;
    try {
      setStatus("Applying boundary changes…");
      const boundaries = preview.previews.map((item, index) => ({
        ...(item.boundary || {}),
        question_index: index + 1,
      }));
      await api(`/api/sessions/${session.session_id}/adjust-boundary`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ boundaries }),
      });
      const nextPreview = await api(`/api/sessions/${session.session_id}/split-preview`);
      setPreview(nextPreview);
      setStatus("");
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function confirmAndGrade() {
    try {
      setStatus("Reading handwriting with OCR…");
      await api(`/api/sessions/${session.session_id}/extract`, { method: "POST" });
      setStatus("Grading answers against the rubric…");
      onComplete((await api(`/api/sessions/${session.session_id}/grade`, { method: "POST" })).evaluation);
      setStatus("");
    } catch (error) {
      setStatus(error.message);
    }
  }

  if (showPreview) {
    return <main className="split"><header><div><p className="eyebrow">STEP 2 OF 3</p><h1>Verify question split</h1></div><div><button type="button" className="secondary" onClick={() => setShowPreview(false)}>Back to pages</button><button type="button" onClick={confirmAndGrade} disabled={preview?.validation?.valid === false}>Confirm and grade</button></div></header><section className="preview-layout"><aside><p>Check each crop before OCR and grading. A warning means the crop may start in the middle of an answer.</p>{preview?.validation?.errors?.map((error) => <p className="validation-error" key={error}>{error}</p>)}{preview?.validation?.warnings?.map((warning) => <p className="validation-warning" key={warning}>{warning}</p>)}<button type="button" className="secondary wide" onClick={adjustAndRefresh}>Apply boundary edits</button>{status && <p className="status">{status}</p>}</aside><div className="preview-grid">{preview?.previews?.map((item, index) => <article className={`card crop-card ${item.looks_like_continuation ? "crop-warning" : ""}`} key={item.question_id}><div className="crop-heading"><div><p className="eyebrow">{item.question_id}</p><strong>{item.warning || "Split looks consistent"}</strong></div><img src={item.crop_path} alt={`${item.question_id} crop preview`} /></div><label>Start page<input type="number" value={item.boundary?.start_page ?? ""} onChange={(e) => applyBoundaryChange(index, "start_page", e.target.value)} /></label><label>Start Y<input type="number" value={item.boundary?.start_y ?? ""} onChange={(e) => applyBoundaryChange(index, "start_y", e.target.value)} /></label><label>End page<input type="number" value={item.boundary?.end_page ?? ""} onChange={(e) => applyBoundaryChange(index, "end_page", e.target.value)} /></label><label>End Y<input type="number" value={item.boundary?.end_y ?? ""} onChange={(e) => applyBoundaryChange(index, "end_y", e.target.value)} /></label><pre>{item.ocr_start || "No OCR preview available."}</pre></article>)}</div></section></main>;
  }

  return <main className="split"><header><div><p className="eyebrow">STEP 2 OF 3</p><h1>Mark question starts</h1></div><div><button type="button" className="secondary" onClick={onBack}>Start over</button><button type="button" onClick={createSplitPreview} disabled={clicks.length === 0}>Preview split</button></div></header><section><aside><p>Click once where each new question begins. The next click ends the previous question. Then preview the crops before grading.</p>{session.page_paths.map((_, index) => <button type="button" key={index} className={index === page ? "tab selected" : "tab"} onClick={() => setPage(index)}>Page {index + 1}<small>{clicks.filter((point) => point.page_index === index).length} starts</small></button>)}<button type="button" className="secondary wide" onClick={() => setClicks((all) => renumber(all.filter((point) => point.page_index !== page)))}>Clear this page</button><button type="button" className="secondary wide" onClick={() => setClicks((all) => renumber(all.slice(0, -1)))}>Undo last mark</button><div className="status" style={{ color: "#d9e8fb" }}>Questions marked: {clicks.length}</div>{status && <p className="status">{status}</p>}</aside><div className="stage"><div className="canvas" onClick={mark}><img src={image} onLoad={(e) => setHeight(e.currentTarget.naturalHeight)} />{height > 0 && marks.map((point) => <i key={`${point.question_index}-${point.page_index}-${point.y}`} style={{ top: `${point.y / height * 100}%` }}>Q{point.question_index}</i>)}</div></div></section></main>;
}

function Results({ result, reset }) {
  const questions = useMemo(() => result?.questions || (result ? [result] : []), [result]);
  return <main className="results"><header><div><p className="eyebrow">STEP 3 OF 3</p><h1>Evaluation complete</h1></div><button className="secondary" onClick={reset}>Grade another PDF</button></header>{questions.map((question) => <article className="card score" key={question.question_id}><div className="number">{question.evaluation.marks ?? "—"}<small>/ {question.evaluation.max_marks}</small></div><div><p className="eyebrow">{question.question_id}</p><h2>{question.evaluation.needs_human_review ? "Needs teacher review" : "Ready for review"}</h2><p>{question.evaluation.feedback}</p><span>Confidence: {Math.round(question.evaluation.evaluation_confidence * 100)}%</span><details><summary>Criterion feedback</summary><pre>{JSON.stringify(question.evaluation.criteria_results, null, 2)}</pre></details></div></article>)}</main>;
}

function App() { const [screen, setScreen] = useState("upload"), [session, setSession] = useState(), [result, setResult] = useState(); if (screen === "split") return <Split session={session} onBack={() => setScreen("upload")} onComplete={(next) => { setResult(next); setScreen("results"); }} />; if (screen === "results") return <Results result={result} reset={() => setScreen("upload")} />; return <Upload onUploaded={(next) => { setSession(next); setScreen("split"); }} />; }
createRoot(document.getElementById("root")).render(<App />);