import { pipeline, env } from "./vendor/transformers.min.js";
import { Ensemble } from "./sts.js";

// Everything is served from this Space: no CDN, no remote model repositories.
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL("./models/", location.href).href;
const vendor = new URL("./vendor/", location.href).href;
env.backends.onnx.wasm.wasmPaths = { mjs: `${vendor}ort-wasm-simd-threaded.jsep.js`, wasm: `${vendor}ort-wasm-simd-threaded.jsep.wasm` };

const $ = (id) => document.getElementById(id);
const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

// ------------------------------------------------------------------ text

const T = {
  tr: {
    brand: "Türkçe Paraphrase Tespiti",
    title: "İki Türkçe cümle aynı şeyi mi söylüyor?",
    lead: "Bir transformer (MiniLM), sıfırdan eğitilmiş bir Siamese Bi-LSTM ve dört sözcüksel özniteliği LightGBM ile birleştiren hibrit bir model. Tamamen tarayıcınızda çalışır.",
    v_f1: "0,880", v_gain: "+4,7",
    stat_f1: "Test F1 (LightGBM)", stat_gain: "F1 puanı, tek başına MiniLM'e göre", stat_pairs: "benzersiz cümle çifti, 3 kaynak",
    tab_demo: "Demo", tab_results: "Sonuçlar", tab_data: "Veri seti", tab_how: "Nasıl çalışır",
    sent_a: "Birinci cümle", sent_b: "İkinci cümle",
    ph_a: "Örnek: Bir grup insan maraton koşuyor.", ph_b: "Örnek: İnsanlar koşuyor.",
    examples: "Örnek dene", threshold: "Karar eşiği",
    loading_btn: "Model yükleniyor…", run: "Analiz et", running: "Hesaplanıyor…",
    load_start: "Model dosyaları yükleniyor…",
    load_minilm: (a, b) => `MiniLM yükleniyor… ${a} / ${b} MB (yalnızca ilk ziyarette; sonra önbellekten gelir)`,
    ready: "Hazır. Cümleleri girin ve Analiz et'e basın (Ctrl+Enter).",
    load_fail: (m) => `Model yüklenemedi: ${m}. Sayfayı yenileyip tekrar deneyin.`,
    need_both: "Lütfen iki cümleyi de girin.",
    yes: "✓ Aynı anlam", no: "✕ Farklı anlam",
    sub_yes: (p) => `Model, cümlelerin aynı anlamı taşıdığını %${p} olasılıkla düşünüyor.`,
    sub_no: (p) => `Model, cümlelerin aynı anlamı taşıma olasılığını %${p} buluyor.`,
    cmp_ens: "Ensemble", cmp_minilm: "Sadece MiniLM", cmp_time: "Süre",
    same: "aynı", diff: "farklı",
    why: "Model neden böyle karar verdi?",
    why_hint: "Her çubuk, özniteliğin kararı ne kadar ittiğini gösterir (log-odds). Mavi: aynı anlam yönüne, kırmızı: farklı anlam yönüne.",
    toward_same: "aynı anlam →", toward_diff: "← farklı anlam",
    tt_value: "Öznitelik değeri", tt_contrib: "Karara etkisi",
    r_models: "Model karşılaştırması",
    r_models_hint: "Ayrılmış test kümesi, 12.383 çift. Mavi noktalar ensemble modelleri. Ayrıntı için noktaların üzerine gelin.",
    show_table: "Tabloyu göster",
    r_boot: "Kazanç gerçek mi?",
    r_boot_hint: "Her modelin MiniLM'e göre F1 farkı ve %95 güven aralığı (anchor gruplarıyla küme bootstrap, 1000 tekrar). Aralık sıfırı içermiyorsa fark anlamlıdır.",
    boot_axis: "MiniLM'e göre F1 farkı (puan)",
    r_source: "Kaynağa göre", r_source_hint: "Asıl kazanç NLI verisinde; STS-B'de ensemble geride (yalnızca 225 çift).",
    r_ablation: "Ablation", r_ablation_hint: "Bir öznitelik çıkarılınca test F1'deki değişim (puan, XGBoost).",
    r_leak: "Sızıntı düzeltmesi",
    r_leak_hint: "İlk sürüm rastgele bölme kullanıyordu; test çiftlerinin %23'ü eğitimde de vardı. Aynı kod iki protokolle çalıştırıldı: skorlar sızıntı yüzünden şişmemiş.",
    col_model: "Model", col_type: "Tür", col_source: "Kaynak", col_pairs: "Test çifti", col_metric: "Ölçüt",
    col_orig: "Eski protokol", col_fixed: "Yeni protokol",
    leak_pairs: "Eğitimde de olan test çiftleri", leak_anchor: "Eğitimde görülen test anchor'ları",
    d_title: "Üç açık veri kümesi",
    d_hint: "620.089 ham çift; tekrarlar ve çelişkili etiketler temizlenince 326.279 benzersiz çift. 62.000 çiftlik örneklem, anchor cümlesine göre gruplanarak %80/%20 bölündü.",
    d_stats: "İstatistikler",
    col_raw: "Ham çift", col_unique: "Benzersiz", col_dup: "Tekrar", col_pos: "Pozitif", col_words: "Kelime (A / B)",
    lic: "Lisans", pairs: "Benzersiz çift", positive: "Pozitif oran", label: "Etiket",
    h_arch: "Mimari", h_in: "Cümle A + Cümle B", h_minilm: "kosinüs benzerliği (hazır model)", h_lstm: "+ attention, sıfırdan eğitildi",
    h_lex_t: "Sözcüksel", h_meta: "6 öznitelik → olasılık",
    h_protocol: "Değerlendirme protokolü", h_limits: "Kısıtlar", h_browser: "Tarayıcıda nasıl çalışıyor?",
    types: { "Ensemble": "Ensemble", "Pre-trained": "Hazır model", "Simple ensemble": "Basit ortalama", "Deep learning": "Derin öğrenme" },
    sources: { nli_triplet: "NLI üçlüleri", ml_paraphrase: "ML paraphrase", stsb_tr: "STS-B (tr)", all: "Tümü" },
    features: {
      minilm_sim: "MiniLM benzerliği", lstm_prob: "Bi-LSTM olasılığı", tfidf_sim: "TF-IDF benzerliği",
      jaccard_sim: "Jaccard benzerliği", token_overlap: "Ortak kelime", levenshtein_dist: "Levenshtein mesafesi",
    },
    ablation: (c) => c === "lexical only" ? "yalnızca sözcüksel" : c === "minilm_sim only" ? "yalnızca minilm_sim"
      : c.startsWith("without ") ? `${c.slice(8)} olmadan` : c,
    model: (m) => m.replace("Score average", "Skor ortalaması"),
    examples_list: [
      ["Paraphrase", "Derin öğrenme, yapay sinir ağlarını kullanan bir makine öğrenmesi yöntemidir.", "Yapay sinir ağlarına dayanan makine öğrenmesi yöntemine derin öğrenme denir."],
      ["Çıkarım (NLI)", "Bir grup insan maraton koşuyor.", "İnsanlar koşuyor."],
      ["Çelişki: MiniLM yanılıyor", "İki köpek çimenli bir alanda koşuyor.", "İki köpek çimenli bir alanda yatıyor."],
      ["Kelime sırası tuzağı", "Adam köpeği parkta gezdiriyor.", "Köpek parkta adamı kovalıyor."],
    ],
    source_cards: {
      nli_triplet: ["SNLI ve MultiNLI'nin Türkçe makine çevirisi. Her üçlü: öncül, ondan çıkan bir cümle (1) ve onunla çelişen bir cümle (0). Gerçek paraphrase değil, çıkarım ilişkisi; en zor kaynak.", "SNLI/MultiNLI lisansları"],
      ml_paraphrase: ["Makine öğrenmesi konulu Türkçe cümle çiftleri. Negatifler rastgele eşleştirilmiş ilgisiz cümleler, bu yüzden kolay.", "Apache 2.0"],
      stsb_tr: ["STS Benchmark'ın Google Translate ile Türkçe çevirisi (GEM 2021). İnsanların verdiği 0-5 benzerlik puanı; ≥ 3 ise 1.", "Belirtilmemiş (araştırma amaçlı)"],
    },
    protocol: [
      "<b>Tekrarlar temizlendi:</b> 620 bin çiftin %47'si birebir tekrardı; 62 çelişkili etiket atıldı.",
      "<b>Gruplu bölme:</b> aynı anchor cümlesi hem eğitimde hem testte olamaz. Eğitimde de bulunan test çiftleri %23'ten %0'a indi.",
      "<b>Out-of-fold stacking:</b> eğitim satırlarının lstm_prob değeri, o satırları görmemiş modellerden (5 katlı) gelir.",
      "<b>Adil eşikler:</b> TF-IDF ve bütün eşikler yalnızca eğitim verisinde belirlendi; meta-model çapraz doğrulamayla seçildi.",
      "<b>Anlamlılık:</b> MiniLM'e göre fark, anchor gruplarını yeniden örnekleyen küme bootstrap ile test edildi.",
    ],
    limits: [
      "<b>Çıkarım ≠ paraphrase:</b> verinin %80'i NLI; model \"B, A'dan çıkar\" ilişkisini daha iyi öğreniyor.",
      "<b>Kelime sırası:</b> Bi-LSTM dışındaki öznitelikler sıraya duyarsız; \"Adam köpeği gezdiriyor\" / \"Köpek adamı kovalıyor\" yüksek skor alır.",
      "<b>Makine çevirisi:</b> NLI ve STS-B cümleleri İngilizceden çevrildi; doğal metinde sonuç farklı olabilir.",
      "<b>Tek örneklem:</b> sonuçlar tek bir 62 bin çiftlik örneklem ve tek bölmeden (seed 42) geliyor.",
    ],
    browser: "Sunucu yok: her şey bu sayfada, sizin cihazınızda hesaplanır ve cümleler hiçbir yere gönderilmez. MiniLM, transformers.js ile 8-bit ONNX modeli olarak çalışır; Bi-LSTM, TF-IDF, sözcüksel öznitelikler ve LightGBM ağaçları JavaScript'e taşındı. 303 test çiftinde JavaScript öznitelikleri Python ile 10⁻⁷ hassasiyetle aynı; 8-bit MiniLM yüzünden kararların %99,3'ü aynı. Bütün model dosyaları bu Space'te barındırılır.",
  },
  en: {
    brand: "Turkish Paraphrase Detection",
    title: "Do two Turkish sentences say the same thing?",
    lead: "A hybrid model that stacks a transformer (MiniLM), a Siamese Bi-LSTM trained from scratch and four lexical features with LightGBM. Runs entirely in your browser.",
    v_f1: "0.880", v_gain: "+4.7",
    stat_f1: "Test F1 (LightGBM)", stat_gain: "F1 points over MiniLM alone", stat_pairs: "unique sentence pairs, 3 sources",
    tab_demo: "Demo", tab_results: "Results", tab_data: "Dataset", tab_how: "How it works",
    sent_a: "First sentence", sent_b: "Second sentence",
    ph_a: "e.g. Bir grup insan maraton koşuyor.", ph_b: "e.g. İnsanlar koşuyor.",
    examples: "Try an example", threshold: "Decision threshold",
    loading_btn: "Loading model…", run: "Analyse", running: "Computing…",
    load_start: "Loading model files…",
    load_minilm: (a, b) => `Loading MiniLM… ${a} / ${b} MB (first visit only; cached afterwards)`,
    ready: "Ready. Enter two sentences and press Analyse (Ctrl+Enter).",
    load_fail: (m) => `Could not load the model: ${m}. Reload the page and try again.`,
    need_both: "Please enter both sentences.",
    yes: "✓ Same meaning", no: "✕ Different meaning",
    sub_yes: (p) => `The model is ${p}% confident the sentences mean the same thing.`,
    sub_no: (p) => `The model gives only a ${p}% probability that the sentences mean the same thing.`,
    cmp_ens: "Ensemble", cmp_minilm: "MiniLM alone", cmp_time: "Time",
    same: "same", diff: "different",
    why: "Why did the model decide this?",
    why_hint: "Each bar shows how far a feature pushed the decision (log-odds). Blue: towards same meaning, red: towards different meaning.",
    toward_same: "same meaning →", toward_diff: "← different meaning",
    tt_value: "Feature value", tt_contrib: "Effect on decision",
    r_models: "Model comparison",
    r_models_hint: "Held-out test set, 12,383 pairs. Blue dots are the ensembles. Hover a dot for details.",
    show_table: "Show table",
    r_boot: "Is the gain real?",
    r_boot_hint: "F1 difference to MiniLM with a 95% interval (cluster bootstrap over anchor groups, 1,000 resamples). An interval that excludes zero is a significant difference.",
    boot_axis: "F1 difference to MiniLM (points)",
    r_source: "By source", r_source_hint: "The gain comes from NLI; on STS-B the ensemble trails (only 225 pairs).",
    r_ablation: "Ablation", r_ablation_hint: "Change in test F1 when a feature is removed (points, XGBoost).",
    r_leak: "Leakage fix",
    r_leak_hint: "The first version used a random split, and 23% of test pairs were also in training. Running the same code under both protocols shows the scores were not inflated by the leak.",
    col_model: "Model", col_type: "Type", col_source: "Source", col_pairs: "Test pairs", col_metric: "Metric",
    col_orig: "Original protocol", col_fixed: "Fixed protocol",
    leak_pairs: "Test pairs also in training", leak_anchor: "Test anchors seen in training",
    d_title: "Three public datasets",
    d_hint: "620,089 raw pairs; 326,279 unique pairs after removing duplicates and conflicting labels. A 62,000-pair sample is split 80/20, grouped by anchor sentence.",
    d_stats: "Statistics",
    col_raw: "Raw pairs", col_unique: "Unique", col_dup: "Duplicates", col_pos: "Positive", col_words: "Words (A / B)",
    lic: "License", pairs: "Unique pairs", positive: "Positive rate", label: "Label",
    h_arch: "Architecture", h_in: "Sentence A + Sentence B", h_minilm: "cosine similarity (pre-trained)", h_lstm: "+ attention, trained from scratch",
    h_lex_t: "Lexical", h_meta: "6 features → probability",
    h_protocol: "Evaluation protocol", h_limits: "Limitations", h_browser: "How does it run in the browser?",
    types: { "Ensemble": "Ensemble", "Pre-trained": "Pre-trained", "Simple ensemble": "Simple average", "Deep learning": "Deep learning" },
    sources: { nli_triplet: "NLI triplets", ml_paraphrase: "ML paraphrase", stsb_tr: "STS-B (tr)", all: "All" },
    features: {
      minilm_sim: "MiniLM similarity", lstm_prob: "Bi-LSTM probability", tfidf_sim: "TF-IDF similarity",
      jaccard_sim: "Jaccard similarity", token_overlap: "Shared words", levenshtein_dist: "Levenshtein distance",
    },
    ablation: (c) => c,
    model: (m) => m,
    examples_list: [
      ["Paraphrase", "Derin öğrenme, yapay sinir ağlarını kullanan bir makine öğrenmesi yöntemidir.", "Yapay sinir ağlarına dayanan makine öğrenmesi yöntemine derin öğrenme denir."],
      ["Entailment (NLI)", "Bir grup insan maraton koşuyor.", "İnsanlar koşuyor."],
      ["Contradiction: MiniLM is fooled", "İki köpek çimenli bir alanda koşuyor.", "İki köpek çimenli bir alanda yatıyor."],
      ["Word-order trap", "Adam köpeği parkta gezdiriyor.", "Köpek parkta adamı kovalıyor."],
    ],
    source_cards: {
      nli_triplet: ["Turkish machine translation of SNLI and MultiNLI. Each triplet: a premise, a sentence it entails (1) and one it contradicts (0). Entailment rather than true paraphrase; the hardest source.", "SNLI/MultiNLI licenses"],
      ml_paraphrase: ["Turkish sentence pairs about machine learning. Negatives are random unrelated sentences, so they are easy.", "Apache 2.0"],
      stsb_tr: ["The STS Benchmark translated to Turkish with Google Translate (GEM 2021). Human 0-5 similarity scores; ≥ 3 means 1.", "Not stated (research use)"],
    },
    protocol: [
      "<b>Deduplicated:</b> 47% of the 620K pairs were exact duplicates; 62 conflicting labels were dropped.",
      "<b>Grouped split:</b> an anchor sentence never appears in both train and test. Test pairs also in training fell from 23% to 0%.",
      "<b>Out-of-fold stacking:</b> lstm_prob for training rows comes from models that never saw those rows (5 folds).",
      "<b>Fair thresholds:</b> TF-IDF and all thresholds are fit on training data only; the meta-model is chosen by cross-validation.",
      "<b>Significance:</b> the gain over MiniLM is tested with a cluster bootstrap that resamples anchor groups.",
    ],
    limits: [
      "<b>Entailment ≠ paraphrase:</b> 80% of the data is NLI, so the model learns \"B follows from A\" better.",
      "<b>Word order:</b> every feature except the Bi-LSTM ignores order; \"Adam köpeği gezdiriyor\" / \"Köpek adamı kovalıyor\" scores high.",
      "<b>Machine translation:</b> NLI and STS-B sentences are translated from English; natural text may score differently.",
      "<b>Single sample:</b> results come from one 62K-pair sample and one split (seed 42).",
    ],
    browser: "No server: everything is computed on this page, on your device, and the sentences are never sent anywhere. MiniLM runs as an 8-bit ONNX model through transformers.js; the Bi-LSTM, TF-IDF, lexical features and LightGBM trees are ported to JavaScript. On 303 test pairs the JavaScript features match Python to 10⁻⁷; with the 8-bit MiniLM, 99.3% of the decisions agree. All model files are hosted in this Space.",
  },
};

let lang = store.get("lang") || (navigator.language?.startsWith("tr") ? "tr" : "en");
const t = (k) => T[lang][k];
const num = (x, d = 3) => x.toLocaleString(lang === "tr" ? "tr-TR" : "en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (x, d = 1) => `${num(x * 100, d)}%`.replace(/^(.*)%$/, lang === "tr" ? "%$1" : "$1%");
const signed = (x, d = 1) => (x > 0 ? "+" : x < 0 ? "−" : "") + num(Math.abs(x), d);

// ------------------------------------------------------------------ tooltip

const tip = $("tooltip");
function showTip(html, ev) {
  tip.innerHTML = html;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}
const hideTip = () => { tip.hidden = true; };
const ttRows = (pairs) => pairs.map(([k, v]) => `<div class="tt-row"><span>${k}</span><span>${v}</span></div>`).join("");

// ------------------------------------------------------------------ charts (plain SVG)

const NS = "http://www.w3.org/2000/svg";
function el(name, attrs = {}, parent) {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function text(parent, x, y, str, cls, anchor = "start") {
  const e = el("text", { x, y, class: cls, "text-anchor": anchor, "dominant-baseline": "middle" }, parent);
  e.textContent = str;
  return e;
}
// horizontal bar from x0 to x1, 4px rounded at the data end, square at the baseline
function barPath(x0, x1, y, h, r = 4) {
  const w = Math.abs(x1 - x0);
  r = Math.min(r, w, h / 2);
  if (x1 >= x0) return `M${x0},${y}H${x1 - r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 - r},${y + h}H${x0}Z`;
  return `M${x0},${y}H${x1 + r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 + r},${y + h}H${x0}Z`;
}
function ticksFor(min, max, n = 5) {
  const step = [0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10].find((s) => (max - min) / s <= n) ?? 10;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

// rows: [{label, value, lo?, hi?, color, tip}]; draws a dot (with optional whisker) per row
function dotChart(container, rows, { min, max, fmt, ref, axisLabel, rowH = 34 }) {
  container.innerHTML = "";
  const W = container.clientWidth || 600;
  const compact = W < 520;
  if (compact) rowH = 42;
  const LW = compact ? 12 : Math.min(250, W * 0.45), R = 56, top = 8, H = top + rows.length * rowH + 34;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
  const x = (v) => LW + ((v - min) / (max - min)) * (W - LW - R);
  for (const v of ticksFor(min, max)) {
    el("line", { x1: x(v), x2: x(v), y1: top, y2: top + rows.length * rowH, class: "gridline" }, svg);
    text(svg, x(v), top + rows.length * rowH + 14, fmt(v, true), "tick-label", "middle");
  }
  if (axisLabel) text(svg, LW + (W - LW - R) / 2, H - 6, axisLabel, "tick-label", "middle");
  if (ref !== undefined) el("line", { x1: x(ref), x2: x(ref), y1: top, y2: top + rows.length * rowH, class: "refline" }, svg);
  rows.forEach((r, i) => {
    const y0 = top + i * rowH, cy = compact ? y0 + 29 : y0 + rowH / 2;
    const g = el("g", { class: "row" }, svg);
    el("rect", { x: 0, y: y0, width: W, height: rowH, class: "hover-band", rx: 6 }, g);
    text(g, 8, compact ? y0 + 11 : cy, r.label, "row-label");
    if (r.lo !== undefined) {
      el("line", { x1: x(r.lo), x2: x(r.hi), y1: cy, y2: cy, stroke: r.color, "stroke-width": 2, "stroke-linecap": "round" }, g);
      for (const v of [r.lo, r.hi]) el("line", { x1: x(v), x2: x(v), y1: cy - 5, y2: cy + 5, stroke: r.color, "stroke-width": 2 }, g);
    }
    el("circle", { cx: x(r.value), cy, r: 6, fill: r.color, stroke: "var(--surface)", "stroke-width": 2 }, g);
    text(g, (r.hi !== undefined ? x(r.hi) : x(r.value)) + 12, cy, fmt(r.value), "value-label");
    const hit = el("rect", { x: 0, y: y0, width: W, height: rowH, class: "hit" }, g);
    hit.addEventListener("mousemove", (ev) => showTip(r.tip, ev));
    hit.addEventListener("mouseleave", hideTip);
  });
}

// rows: [{label, value, color, tip}]; bars grow from zero, negative to the left
function barChart(container, rows, { min, max, fmt, rowH = 32, leftNote, rightNote }) {
  container.innerHTML = "";
  const W = container.clientWidth || 600;
  const compact = W < 520;
  if (compact) rowH = 44;
  const LW = compact ? 0 : Math.min(200, W * 0.4), PAD = 44, top = leftNote ? 22 : 6, H = top + rows.length * rowH + 26;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
  const x = (v) => LW + PAD + ((v - min) / (max - min)) * (W - LW - 2 * PAD);
  for (const v of ticksFor(min, max, 4)) {
    el("line", { x1: x(v), x2: x(v), y1: top, y2: top + rows.length * rowH, class: v === 0 ? "zeroline" : "gridline" }, svg);
    text(svg, x(v), top + rows.length * rowH + 14, fmt(v, true), "tick-label", "middle");
  }
  if (leftNote) text(svg, x(0) - 6, 9, leftNote, "tick-label", "end");
  if (rightNote) text(svg, x(0) + 6, 9, rightNote, "tick-label", "start");
  rows.forEach((r, i) => {
    const y = top + i * rowH, h = compact ? 14 : Math.min(18, rowH - 10);
    const by = compact ? y + 22 : y + (rowH - h) / 2, cy = by + h / 2;
    const g = el("g", { class: "row" }, svg);
    el("rect", { x: 0, y, width: W, height: rowH, class: "hover-band", rx: 6 }, g);
    text(g, 8, compact ? y + 11 : cy, r.label, "row-label");
    if (Math.abs(x(r.value) - x(0)) > 0.5) el("path", { d: barPath(x(0), x(r.value), by, h), fill: r.color }, g);
    const end = x(r.value);
    text(g, r.value >= 0 ? end + 6 : end - 6, cy, fmt(r.value), "value-label", r.value >= 0 ? "start" : "end");
    const hit = el("rect", { x: 0, y, width: W, height: rowH, class: "hit" }, g);
    hit.addEventListener("mousemove", (ev) => showTip(r.tip, ev));
    hit.addEventListener("mouseleave", hideTip);
  });
}

function tableHTML(head, rows, numericFrom = 1, highlight = () => false) {
  const th = head.map((h, i) => `<th${i >= numericFrom ? ' class="num"' : ""}>${h}</th>`).join("");
  const tr = rows.map((r) => `<tr${highlight(r) ? ' class="hl"' : ""}>${r.map((c, i) => `<td${i >= numericFrom ? ' class="num"' : ""}>${c}</td>`).join("")}</tr>`).join("");
  return `<div class="table-wrap"><table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`;
}

// ------------------------------------------------------------------ static content

let results = null;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function renderResults() {
  if (!results) return;
  const accent = "var(--accent)", neutral = "var(--neutral-mark)", neg = "var(--neg)";

  const models = [...results.models].sort((a, b) => b.F1 - a.F1);
  dotChart($("chart-models"), models.map((m) => ({
    label: T[lang].model(m.Model), value: m.F1, color: m.Type === "Ensemble" ? accent : neutral,
    tip: `<b>${T[lang].model(m.Model)}</b>${ttRows([[t("col_type"), t("types")[m.Type]], ["F1", num(m.F1)], ["Accuracy", num(m.Accuracy)],
      ["Precision", num(m.Precision)], ["Recall", num(m.Recall)], ["AUC", num(m.AUC)]])}`,
  })), { min: 0.7, max: 0.9, fmt: (v, tick) => num(v, tick ? 2 : 3) });
  $("table-models").innerHTML = tableHTML([t("col_model"), t("col_type"), "Acc", "Prec", "Rec", "F1", "AUC"],
    models.map((m) => [T[lang].model(m.Model), t("types")[m.Type], num(m.Accuracy), num(m.Precision), num(m.Recall), num(m.F1), num(m.AUC)]), 2,
    (r) => r[1] === t("types").Ensemble);

  const boot = results.bootstrap.filter((b) => !b.Model.startsWith("MiniLM")).sort((a, b) => b.Diff_vs_MiniLM - a.Diff_vs_MiniLM);
  dotChart($("chart-boot"), boot.map((b) => ({
    label: T[lang].model(b.Model), value: b.Diff_vs_MiniLM * 100, lo: b.Diff_CI_low * 100, hi: b.Diff_CI_high * 100,
    color: b.Diff_CI_low > 0 ? accent : neg,
    tip: `<b>${T[lang].model(b.Model)}</b>${ttRows([["F1", num(b.F1)], ["Δ", signed(b.Diff_vs_MiniLM * 100)],
      ["95% CI", `${signed(b.Diff_CI_low * 100)} … ${signed(b.Diff_CI_high * 100)}`]])}`,
  })), { min: -4, max: 6, ref: 0, fmt: (v, tick) => signed(v, tick ? 0 : 1), axisLabel: t("boot_axis") });

  $("table-source").innerHTML = tableHTML([t("col_source"), t("col_pairs"), "MiniLM F1", "Ensemble F1", "Δ"],
    results.per_source.map((s) => {
      const overall = results.bootstrap.find((b) => b.Model === "LightGBM").Diff_vs_MiniLM * 100;
      const d = s.Source === "all" ? overall : (s.Ensemble_F1 - s.MiniLM_F1) * 100;
      return [t("sources")[s.Source], s.Test_pairs.toLocaleString(lang === "tr" ? "tr-TR" : "en-US"), num(s.MiniLM_F1), num(s.Ensemble_F1),
        `<span class="${d >= 0 ? "delta-up" : "delta-down"}">${d >= 0 ? "▲" : "▼"} ${signed(d)}</span>`];
    }), 1, (r) => r[0] === t("sources").all);

  const abl = results.ablation.filter((a) => a.Configuration !== "All features").sort((a, b) => b.Delta_vs_all - a.Delta_vs_all);
  barChart($("chart-ablation"), abl.map((a) => ({
    label: T[lang].ablation(a.Configuration), value: a.Delta_vs_all * 100, color: accent,
    tip: `<b>${T[lang].ablation(a.Configuration)}</b>${ttRows([["Test F1", num(a.Test_F1)], ["Δ", signed(a.Delta_vs_all * 100)]])}`,
  })), { min: -16, max: 0, fmt: (v, tick) => (tick ? num(v, 0) : signed(v)) });

  const legacy = Object.fromEntries(results.legacy_models.map((m) => [m.Model, m.F1]));
  const fixed = Object.fromEntries(results.models.map((m) => [m.Model, m.F1]));
  $("table-leak").innerHTML = tableHTML([t("col_metric"), t("col_orig"), t("col_fixed")], [
    [t("leak_pairs"), pct(results.legacy_split.pair_seen_in_train), pct(results.split.pair_seen_in_train)],
    [t("leak_anchor"), pct(results.legacy_split.anchor_seen_in_train), pct(results.split.anchor_seen_in_train)],
    ...["LightGBM", "Random Forest", "MiniLM-L12 (zero-shot)"].map((m) => [`F1 · ${m}`, num(legacy[m]), num(fixed[m])]),
  ]);

  const byName = Object.fromEntries(results.datasets.map((d) => [d.Source, d]));
  const links = {
    nli_triplet: "mertcobanov/all-nli-triplets-turkish", ml_paraphrase: "dogukanvzr/ml-paraphrase-tr", stsb_tr: "figenfikri/stsb_tr",
  };
  const pairs = {
    nli_triplet: ["Bir grup insan maraton koşuyor.", "İnsanlar koşuyor.", "1"],
    ml_paraphrase: ["MSE, bir tahmin modelinin doğruluğunu ölçmek için kullanılan önemli bir metriktir.", "MSE, tahmin modelinin ne kadar doğru olduğunu belirlemek için kullanılır.", "1"],
    stsb_tr: ["Bir kadın brokoli pişiriyor.", "Bir adam yemeğini yiyor.", "0 (0.5)"],
  };
  $("sources").innerHTML = Object.keys(links).map((k) => {
    const d = byName[k], [desc, lic] = t("source_cards")[k], [a, b, lab] = pairs[k];
    return `<div class="source"><h4>${t("sources")[k]}</h4>
      <a href="https://huggingface.co/datasets/${links[k]}" target="_blank" rel="noopener"><code>${links[k]}</code></a>
      <p>${desc}</p>
      <dl><dt>${t("pairs")}</dt><dd>${d.Unique_pairs.toLocaleString(lang === "tr" ? "tr-TR" : "en-US")}</dd>
      <dt>${t("positive")}</dt><dd>${pct(d.Positive_rate)}</dd><dt>${t("lic")}</dt><dd>${lic}</dd></dl>
      <div class="example-pair">“${a}”<br>“${b}” → ${t("label")} ${lab}</div></div>`;
  }).join("");
  $("table-data").innerHTML = tableHTML([t("col_source"), t("col_raw"), t("col_unique"), t("col_dup"), t("col_pos"), t("col_words")],
    results.datasets.map((d) => [t("sources")[d.Source], d.Raw_pairs.toLocaleString(lang === "tr" ? "tr-TR" : "en-US"),
      d.Unique_pairs.toLocaleString(lang === "tr" ? "tr-TR" : "en-US"), pct(d.Duplicate_share), pct(d.Positive_rate),
      `${num(d.Mean_words_a, 1)} / ${num(d.Mean_words_b, 1)}`]), 1, (r) => r[0] === t("sources").all);
}

function applyLanguage() {
  document.documentElement.lang = lang;
  $("lang").textContent = lang === "tr" ? "EN" : "TR";
  for (const e of document.querySelectorAll("[data-i18n]")) {
    const v = t(e.dataset.i18n);
    if (typeof v === "string" && !(e.id === "run" && ensemble)) e.textContent = v;
  }
  for (const e of document.querySelectorAll("[data-i18n-placeholder]")) e.placeholder = t(e.dataset.i18nPlaceholder);
  if (ensemble) $("run").textContent = t("run");
  $("examples").innerHTML = "";
  for (const [label, a, b] of t("examples_list")) {
    const btn = document.createElement("button");
    btn.className = "example";
    btn.innerHTML = `<b>${label}</b><span>${a}<br>${b}</span>`;
    btn.onclick = () => { $("a").value = a; $("b").value = b; if (ensemble) run(); };
    $("examples").appendChild(btn);
  }
  $("list-protocol").innerHTML = t("protocol").map((x) => `<li>${x}</li>`).join("");
  $("list-limits").innerHTML = t("limits").map((x) => `<li>${x}</li>`).join("");
  $("browser-text").textContent = t("browser");
  if (statusKey) setStatus(statusKey, statusArgs, statusProgress);
  renderResults();
  renderResult();
}

// ------------------------------------------------------------------ tabs, theme, language

for (const btn of document.querySelectorAll("[data-tab]")) {
  btn.addEventListener("click", () => {
    for (const b of document.querySelectorAll("[data-tab]")) b.setAttribute("aria-selected", b === btn);
    for (const p of document.querySelectorAll(".panel")) p.hidden = p.id !== `panel-${btn.dataset.tab}`;
    renderResults();
    renderResult();
  });
}
$("lang").onclick = () => { lang = lang === "tr" ? "en" : "tr"; store.set("lang", lang); applyLanguage(); };
const savedTheme = store.get("theme");
if (savedTheme) document.documentElement.dataset.theme = savedTheme;
$("theme").onclick = () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  store.set("theme", document.documentElement.dataset.theme);
};
let resizeTimer;
addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { renderResults(); renderResult(); }, 150); });

// ------------------------------------------------------------------ demo

let ensemble = null, last = null;
let statusKey = "load_start", statusArgs = [], statusProgress;

function setStatus(key, args = [], progress) {
  statusKey = key; statusArgs = args; statusProgress = progress;
  const v = key ? t(key) : "";
  $("status").textContent = typeof v === "function" ? v(...args) : v;
  $("loader").classList.toggle("done", progress === undefined);
  if (progress !== undefined) $("bar-fill").style.width = `${progress}%`;
}

function renderResult() {
  if (!last || $("panel-demo").hidden) return;
  const { prob, features, contributions, minilmParaphrase, ms } = last;
  const th = parseFloat($("t").value), yes = prob >= th;
  $("result").hidden = false;
  $("verdict").textContent = yes ? t("yes") : t("no");
  $("verdict").className = `badge ${yes ? "yes" : "no"}`;
  $("verdict-sub").textContent = (yes ? t("sub_yes") : t("sub_no"))(num(prob * 100, 1));
  $("cmp-ens").textContent = `${yes ? "✓" : "✕"} ${yes ? t("same") : t("diff")} · ${num(prob)}`;
  $("cmp-minilm").textContent = `${minilmParaphrase ? "✓" : "✕"} ${minilmParaphrase ? t("same") : t("diff")} · ${num(features.minilm_sim)}`;
  $("cmp-time").textContent = `${Math.round(ms)} ms`;
  $("gv").textContent = num(prob);
  $("arc").setAttribute("stroke-dasharray", `${(prob * 100).toFixed(1)} 100`);
  $("arc").setAttribute("stroke", yes ? "var(--good)" : "var(--bad)");
  const ang = Math.PI * (1 - th);
  $("tick").setAttribute("x1", 100 + 68 * Math.cos(ang)); $("tick").setAttribute("y1", 100 - 68 * Math.sin(ang));
  $("tick").setAttribute("x2", 100 + 92 * Math.cos(ang)); $("tick").setAttribute("y2", 100 - 92 * Math.sin(ang));

  const names = t("features");
  const rows = Object.entries(contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).map(([k, c]) => ({
    label: names[k], value: c, color: c >= 0 ? "var(--accent)" : "var(--neg)",
    tip: `<b>${names[k]}</b>${ttRows([[t("tt_value"), num(features[k], Number.isInteger(features[k]) ? 0 : 3)], [t("tt_contrib"), signed(c, 2)]])}`,
  }));
  const m = Math.max(1, ...rows.map((r) => Math.abs(r.value))) * 1.1;
  barChart($("contrib"), rows, { min: -m, max: m, fmt: (v, tick) => signed(v, tick ? 0 : 2), leftNote: t("toward_diff"), rightNote: t("toward_same") });
}

async function run() {
  const a = $("a").value.trim(), b = $("b").value.trim();
  if (!a || !b) { setStatus("need_both"); return; }
  $("run").disabled = true;
  $("run").textContent = t("running");
  const start = performance.now();
  const out = await ensemble.predict(a, b);
  last = { ...out, ms: performance.now() - start };
  $("run").disabled = false;
  $("run").textContent = t("run");
  setStatus(null);
  renderResult();
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
      if (p.status !== "progress" || !p.total) return;
      files.set(p.file, [p.loaded, p.total]);
      let loaded = 0, total = 0;
      for (const [l, tt] of files.values()) { loaded += l; total += tt; }
      setStatus("load_minilm", [Math.round(loaded / 1e6), Math.round(total / 1e6)], (100 * loaded) / total);
    },
  });
  const embedSimilarity = async (a, b) => {
    const [u, v] = (await extractor([a, b], { pooling: "mean", normalize: true })).tolist();
    return u.reduce((s, x, i) => s + x * v[i], 0);
  };
  ensemble = new Ensemble(config, lstm, lightgbm, embedSimilarity);
}

$("t").oninput = () => { $("tv").textContent = parseFloat($("t").value).toFixed(2); renderResult(); };
$("run").onclick = run;
for (const id of ["a", "b"]) $(id).addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && ensemble) run(); });

applyLanguage();
setStatus("load_start", [], 2);
fetch("data/results.json").then((r) => r.json()).then((r) => { results = r; renderResults(); });
load().then(() => {
  $("run").disabled = false;
  $("run").textContent = t("run");
  setStatus("ready");
}).catch((err) => {
  console.error(err);
  setStatus("load_fail", [err.message]);
});
