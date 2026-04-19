import { FormEvent, useEffect, useMemo, useState } from "react";

type Paper = {
  paper_id: string;
  title: string;
  ingest_status: string;
};

type EventItem = {
  event_type: string;
  message: string;
  payload?: Record<string, unknown>;
};

const API_BASE = "http://127.0.0.1:8000";

async function readStream(response: Response, onEvent: (event: EventItem) => void) {
  const reader = response.body?.getReader();
  if (!reader) {
    return;
  }
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const pieces = buffer.split("\n\n");
    buffer = pieces.pop() ?? "";
    for (const piece of pieces) {
      if (!piece.startsWith("data: ")) {
        continue;
      }
      const payload = piece.replace("data: ", "");
      onEvent(JSON.parse(payload));
    }
  }
}

export default function App() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selectedPaperId, setSelectedPaperId] = useState("");
  const [pdfPath, setPdfPath] = useState("");
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [question, setQuestion] = useState("What is the main contribution of this paper?");
  const [events, setEvents] = useState<EventItem[]>([]);
  const [answer, setAnswer] = useState("");
  const [pptResult, setPptResult] = useState("");
  const [busy, setBusy] = useState(false);

  const selectedPaper = useMemo(
    () => papers.find((paper) => paper.paper_id === selectedPaperId) ?? null,
    [papers, selectedPaperId],
  );

  useEffect(() => {
    void loadPapers();
  }, []);

  async function loadPapers() {
    const response = await fetch(`${API_BASE}/papers`);
    const data = (await response.json()) as Paper[];
    setPapers(data);
    if (!selectedPaperId && data.length > 0) {
      setSelectedPaperId(data[0].paper_id);
    }
  }

  async function handleIngest(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setPptResult("");
    try {
      let response: Response;
      if (pdfFile) {
        const formData = new FormData();
        formData.append("file", pdfFile);
        response = await fetch(`${API_BASE}/ingest/upload`, {
          method: "POST",
          body: formData,
        });
      } else {
        response = await fetch(`${API_BASE}/ingest`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            pdf: pdfPath || null,
            url: url || null,
          }),
        });
      }
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "Ingest failed");
      }
      await loadPapers();
      setSelectedPaperId(payload.paper_id);
    } finally {
      setBusy(false);
    }
  }

  async function handleChat(event: FormEvent) {
    event.preventDefault();
    if (!selectedPaperId) {
      return;
    }
    setBusy(true);
    setEvents([]);
    setAnswer("");
    setPptResult("");
    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          paper_id: selectedPaperId,
          question,
          style: "beginner",
        }),
      });
      await readStream(response, (item) => {
        setEvents((current) => [...current, item]);
        if (item.event_type === "final_answer_stream") {
          setAnswer((current) => current + item.message);
        }
        if (item.event_type === "ppt_generated") {
          const pptPath = String(item.payload?.ppt_path ?? "");
          setPptResult(pptPath);
        }
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">PaperAgent</p>
          <h1>论文讲解与 PPT 生成助手</h1>
          <p className="subtitle">
            先导入论文，再通过自然语言对话触发检索讲解或讲解型 PPT 生成。
          </p>
        </div>
      </header>

      <main className="grid">
        <section className="panel">
          <h2>1. 导入论文</h2>
          <form onSubmit={handleIngest} className="stack">
            <label>
              上传 PDF 文件
              <input
                type="file"
                accept=".pdf"
                onChange={(event) => setPdfFile(event.target.files?.[0] ?? null)}
              />
            </label>
            <label>
              本地 PDF 路径
              <input value={pdfPath} onChange={(event) => setPdfPath(event.target.value)} placeholder="D:\\papers\\demo.pdf" />
            </label>
            <label>
              或 PDF URL
              <input value={url} onChange={(event) => setUrl(event.target.value)} placeholder="https://example.com/paper.pdf" />
            </label>
            <button disabled={busy} type="submit">
              导入并建库
            </button>
          </form>
        </section>

        <section className="panel">
          <h2>2. 论文列表</h2>
          <div className="paper-list">
            {papers.map((paper) => (
              <button
                key={paper.paper_id}
                className={paper.paper_id === selectedPaperId ? "paper-card active" : "paper-card"}
                onClick={() => setSelectedPaperId(paper.paper_id)}
                type="button"
              >
                <strong>{paper.title}</strong>
                <span>{paper.paper_id}</span>
                <span>{paper.ingest_status}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="panel wide">
          <h2>3. 对话讲解</h2>
          <form onSubmit={handleChat} className="stack">
            <label>
              当前论文
              <input value={selectedPaper?.title ?? ""} readOnly />
            </label>
            <label>
              问题
              <textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={4} />
            </label>
            <button disabled={busy || !selectedPaperId} type="submit">
              开始流式讲解
            </button>
          </form>
          <p className="ppt-result">
            {pptResult || "如果想生成 PPT，请直接在问题里输入“给这篇论文做个 PPT”。"}
          </p>
          <div className="chat-grid">
            <div className="chat-box">
              <h3>最终回答</h3>
              <pre>{answer || "等待回答..."}</pre>
            </div>
            <div className="chat-box">
              <h3>过程事件</h3>
              <ul>
                {events.map((item, index) => (
                  <li key={`${item.event_type}-${index}`}>
                    <strong>{item.event_type}</strong>: {item.message}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

      </main>
    </div>
  );
}
