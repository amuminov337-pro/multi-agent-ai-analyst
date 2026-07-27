"use client";

import { useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

const DEFAULT_QUESTION =
  "Engineering bo'limida nechta xodim bor va ularning o'rtacha oylik maoshi qancha? " +
  "Shu o'rtacha maoshni Python bilan yillik summaga aylantirib ko'rsat.";

function detailOf(event) {
  const parts = [];
  if (typeof event.documents === "number") parts.push(`${event.documents} hujjat`);
  if (event.sql_result) parts.push("SQL natijasi olindi");
  if (event.code_result) parts.push("kod bajarildi");
  if (event.critic_ok === true) parts.push("critic: tasdiqlandi");
  if (event.critic_ok === false && event.critic_reason) parts.push(`critic: ${event.critic_reason}`);
  if (Array.isArray(event.steps) && event.steps.length) parts.push(event.steps[event.steps.length - 1]);
  return parts.join(" · ");
}

export default function Page() {
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [events, setEvents] = useState([]);
  const [answer, setAnswer] = useState("");
  const [traceUrl, setTraceUrl] = useState("");
  const [elapsed, setElapsed] = useState(null);
  const [criticOk, setCriticOk] = useState(null);
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);
  const sourceRef = useRef(null);

  useEffect(() => {
    return () => {
      if (sourceRef.current) sourceRef.current.close();
    };
  }, []);

  function closeStream() {
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
    setRunning(false);
  }

  function ask() {
    if (!question.trim() || running) return;

    setEvents([]);
    setAnswer("");
    setTraceUrl("");
    setElapsed(null);
    setCriticOk(null);
    setError("");
    setRunning(true);

    const url = `${API_BASE}/api/stream?question=${encodeURIComponent(question)}`;
    const source = new EventSource(url);
    sourceRef.current = source;

    source.addEventListener("start", () => {
      setEvents((prev) => [...prev, { node: "start", elapsed: 0 }]);
    });

    source.addEventListener("step", (message) => {
      const data = JSON.parse(message.data);
      setEvents((prev) => [...prev, data]);
      if (data.critic_ok !== undefined && data.critic_ok !== null) setCriticOk(data.critic_ok);
    });

    source.addEventListener("done", (message) => {
      const data = JSON.parse(message.data);
      setAnswer(data.answer || "");
      setTraceUrl(data.trace_url || "");
      setElapsed(data.elapsed ?? null);
      if (data.critic_ok !== undefined && data.critic_ok !== null) setCriticOk(data.critic_ok);
      // Server oqimni yopadi; yopmasak EventSource qayta ulanib savolni takrorlaydi.
      closeStream();
    });

    source.addEventListener("error", (message) => {
      // Serverdan kelgan xato hodisasida data bo'ladi; ulanish uzilishida bo'lmaydi.
      if (message.data) {
        try {
          setError(JSON.parse(message.data).message || "Noma'lum xato");
        } catch {
          setError("Noma'lum xato");
        }
      } else if (sourceRef.current) {
        setError("Backend bilan ulanish uzildi. Server ishlab turganini tekshiring.");
      }
      closeStream();
    });
  }

  return (
    <main className="wrap">
      <h1>Multi-Agent AI Analyst</h1>
      <p className="sub">
        LangGraph supervisor · retriever / web / data(SQL) / code agentlari · critic ·
        Langfuse tracing
      </p>

      <div className="ask">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Savolingizni yozing..."
          disabled={running}
        />
        <button onClick={ask} disabled={running || !question.trim()}>
          {running ? "Ishlayapti..." : "So'rash"}
        </button>
      </div>

      {events.length > 0 && (
        <div className="panel">
          <h2>
            {running && <span className="pulse" />}
            Agent qadamlari (jonli)
          </h2>
          {events.map((event, index) => {
            const node = event.node || "—";
            const detail = detailOf(event);
            return (
              <div className="step" key={index}>
                <span className="time">{Number(event.elapsed ?? 0).toFixed(2)}s</span>
                <span className={`badge ${node}`}>{node}</span>
                <span className="detail">{detail}</span>
              </div>
            );
          })}
        </div>
      )}

      {error && (
        <div className="panel">
          <h2>Xato</h2>
          <div className="err">{error}</div>
        </div>
      )}

      {answer && (
        <div className="panel">
          <h2>Javob</h2>
          <div className="answer">{answer}</div>
          <div className="meta">
            {elapsed !== null && <span>{elapsed}s</span>}
            {criticOk === true && <span className="ok">critic tasdiqladi</span>}
            {criticOk === false && <span className="err">critic rad etdi</span>}
            {traceUrl && (
              <a href={traceUrl} target="_blank" rel="noreferrer">
                Langfuse trace
              </a>
            )}
          </div>
        </div>
      )}
    </main>
  );
}