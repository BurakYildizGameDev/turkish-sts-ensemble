import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import { Ensemble } from "./sts.js";

const $ = (id) => document.getElementById(id);

const FEATURES = [
  ["minilm_sim", "MiniLM kosinüs benzerliği", 3],
  ["lstm_prob", "Bi-LSTM paraphrase olasılığı", 3],
  ["tfidf_sim", "TF-IDF kosinüs benzerliği", 3],
  ["jaccard_sim", "Jaccard (kelime kümesi) benzerliği", 3],
  ["token_overlap", "Ortak kelime sayısı", 0],
  ["levenshtein_dist", "Karakter düzenleme mesafesi", 0],
];
const EXAMPLES = {
  "Paraphrase": ["Derin öğrenme, yapay sinir ağlarını kullanan bir makine öğrenmesi yöntemidir.",
                 "Yapay sinir ağlarına dayanan makine öğrenmesi yöntemine derin öğrenme denir."],
  "Çıkarım (NLI)": ["Bir grup insan maraton koşuyor.", "İnsanlar koşuyor."],
  "Farklı anlam": ["Hava bugün çok sıcak.", "Kediler çok tatlı hayvanlardır."],
  "Ortak kelimeler, farklı anlam": ["Adam köpeği parkta gezdiriyor.", "Köpek parkta adamı kovalıyor."],
};

let ensemble = null;
let last = null;

function setStatus(text, progress) {
  $("status").textContent = text;
  $("bar").style.display = progress === undefined ? "none" : "block";
  if (progress !== undefined) $("bar").firstElementChild.style.width = `${progress}%`;
}

async function load() {
  const [config, lstm, lightgbm] = await Promise.all([
    fetch("model/config.json").then((r) => r.json()),
    fetch("model/lstm.bin").then((r) => r.arrayBuffer()),
    fetch("model/lightgbm.json").then((r) => r.json()),
  ]);

  const files = new Map();
  const extractor = await pipeline("feature-extraction", config.minilm, {
    dtype: "q8",
    progress_callback: (p) => {
      if (p.status === "progress" && p.total) {
        files.set(p.file, [p.loaded, p.total]);
        let loaded = 0, total = 0;
        for (const [l, t] of files.values()) { loaded += l; total += t; }
        setStatus(`MiniLM indiriliyor… ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`, (100 * loaded) / total);
      }
    },
  });
  const embedSimilarity = async (a, b) => {
    const [u, v] = (await extractor([a, b], { pooling: "mean", normalize: true })).tolist();
    return u.reduce((s, x, i) => s + x * v[i], 0);
  };
  ensemble = new Ensemble(config, lstm, lightgbm, embedSimilarity);
}

function render() {
  if (!last) return;
  const { prob, features, ms } = last;
  const t = parseFloat($("t").value);
  const yes = prob >= t;
  $("result").style.display = "block";
  $("verdict").textContent = yes ? "✅ Paraphrase" : "❌ Paraphrase değil";
  $("verdict").className = `badge ${yes ? "yes" : "no"}`;
  $("m-prob").textContent = prob.toFixed(3);
  $("m-minilm").textContent = features.minilm_sim.toFixed(3);
  $("m-time").textContent = `${ms.toFixed(0)} ms`;
  $("gv").textContent = prob.toFixed(3);
  $("arc").setAttribute("stroke-dasharray", `${(prob * 100).toFixed(1)} 100`);
  $("arc").setAttribute("stroke", yes ? "var(--good)" : "var(--bad)");
  const angle = Math.PI * (1 - t);
  const [cx, cy] = [100, 100];
  $("tick").setAttribute("x1", cx + 70 * Math.cos(angle)); $("tick").setAttribute("y1", cy - 70 * Math.sin(angle));
  $("tick").setAttribute("x2", cx + 92 * Math.cos(angle)); $("tick").setAttribute("y2", cy - 92 * Math.sin(angle));
  $("features").innerHTML = FEATURES.map(([key, label, digits]) =>
    `<tr><td>${label}<br><span class="hint">${key}</span></td><td class="num">${features[key].toFixed(digits)}</td></tr>`).join("");
}

async function run() {
  const a = $("a").value.trim(), b = $("b").value.trim();
  if (!a || !b) { setStatus("Lütfen iki cümleyi de girin."); return; }
  $("run").disabled = true;
  setStatus("Hesaplanıyor…");
  const start = performance.now();
  const { prob, features } = await ensemble.predict(a, b);
  last = { prob, features, ms: performance.now() - start };
  render();
  setStatus("");
  $("run").disabled = false;
}

for (const [label, [a, b]] of Object.entries(EXAMPLES)) {
  const btn = document.createElement("button");
  btn.textContent = label;
  btn.onclick = () => { $("a").value = a; $("b").value = b; if (ensemble) run(); };
  $("examples").appendChild(btn);
}
$("t").oninput = () => { $("tv").textContent = parseFloat($("t").value).toFixed(2); render(); };
$("run").onclick = run;
for (const id of ["a", "b"]) $(id).addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && ensemble) run(); });

load().then(() => {
  $("run").disabled = false;
  $("run").textContent = "Analiz et";
  setStatus("Hazır. Cümleleri girip Analiz et'e basın (Ctrl+Enter).");
}).catch((err) => {
  console.error(err);
  setStatus(`Model yüklenemedi: ${err.message}. Sayfayı yenileyip tekrar deneyin.`);
});
