// In-browser port of the ensemble in src/sts (features.py, lstm.py, ensemble.py).
// Works in the browser and in Node (scripts/check_web.mjs compares it with Python).

const WORD = /[\p{L}\p{N}_]+/gu;          // Python re \w+ (lstm.tokenize)
const TFIDF_TOKEN = /[\p{L}\p{N}_]{2,}/gu; // sklearn token_pattern (?u)\b\w\w+\b

// ---------------------------------------------------------------- lexical features

const whitespaceTokens = (s) => new Set(s.toLowerCase().split(/\s+/).filter(Boolean));

export function jaccard(a, b) {
  const sa = whitespaceTokens(a), sb = whitespaceTokens(b);
  let inter = 0;
  for (const t of sa) if (sb.has(t)) inter++;
  const union = sa.size + sb.size - inter;
  return union ? inter / union : 0;
}

export function tokenOverlap(a, b) {
  const sa = whitespaceTokens(a), sb = whitespaceTokens(b);
  let inter = 0;
  for (const t of sa) if (sb.has(t)) inter++;
  return inter;
}

export function levenshtein(a, b) {
  const x = Array.from(a), y = Array.from(b); // code points, like Python str
  let prev = Array.from({ length: y.length + 1 }, (_, j) => j);
  for (let i = 1; i <= x.length; i++) {
    const cur = [i];
    for (let j = 1; j <= y.length; j++) {
      cur.push(Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x[i - 1] === y[j - 1] ? 0 : 1)));
    }
    prev = cur;
  }
  return prev[y.length];
}

// ---------------------------------------------------------------- TF-IDF

function tfidfVector(text, tfidf) {
  const counts = new Map();
  for (const tok of text.toLowerCase().match(TFIDF_TOKEN) || []) {
    const idx = tfidf.vocabulary[tok];
    if (idx !== undefined) counts.set(idx, (counts.get(idx) || 0) + 1);
  }
  let norm = 0;
  for (const [idx, c] of counts) {
    const v = c * tfidf.idf[idx];
    counts.set(idx, v);
    norm += v * v;
  }
  norm = Math.sqrt(norm);
  if (norm > 0) for (const [idx, v] of counts) counts.set(idx, v / norm);
  return counts;
}

export function tfidfSimilarity(a, b, tfidf) {
  const va = tfidfVector(a, tfidf), vb = tfidfVector(b, tfidf);
  let dot = 0;
  for (const [idx, v] of va) if (vb.has(idx)) dot += v * vb.get(idx);
  return dot;
}

// ---------------------------------------------------------------- Siamese Bi-LSTM + attention

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

class BiLSTM {
  constructor(cfg, buffer) {
    const data = new Float32Array(buffer);
    this.t = {};
    for (const [name, { offset, shape }] of Object.entries(cfg.tensors)) {
      const size = shape.reduce((p, d) => p * d, 1);
      this.t[name] = { data: data.subarray(offset, offset + size), shape };
    }
    this.stoi = new Map(cfg.vocab.map((w, i) => [w, i]));
    this.maxLen = cfg.max_len;
    this.embDim = this.t["embed.weight"].shape[1];
    this.hidden = this.t["rnn.weight_hh_l0"].shape[1];
  }

  encodeIds(text) {
    const ids = (text.toLowerCase().match(WORD) || []).map((w) => this.stoi.get(w) ?? 1);
    return ids.slice(0, this.maxLen);
  }

  // one direction of the LSTM over the given embeddings; PyTorch gate order i, f, g, o
  run(embs, suffix) {
    const H = this.hidden, E = this.embDim;
    const Wih = this.t[`rnn.weight_ih_l0${suffix}`].data, Whh = this.t[`rnn.weight_hh_l0${suffix}`].data;
    const bih = this.t[`rnn.bias_ih_l0${suffix}`].data, bhh = this.t[`rnn.bias_hh_l0${suffix}`].data;
    let h = new Float64Array(H), c = new Float64Array(H);
    const outs = [];
    for (const x of embs) {
      const gates = new Float64Array(4 * H);
      for (let r = 0; r < 4 * H; r++) {
        let s = bih[r] + bhh[r];
        const wi = r * E, wh = r * H;
        for (let k = 0; k < E; k++) s += Wih[wi + k] * x[k];
        for (let k = 0; k < H; k++) s += Whh[wh + k] * h[k];
        gates[r] = s;
      }
      const nh = new Float64Array(H), nc = new Float64Array(H);
      for (let k = 0; k < H; k++) {
        const i = sigmoid(gates[k]), f = sigmoid(gates[H + k]);
        const g = Math.tanh(gates[2 * H + k]), o = sigmoid(gates[3 * H + k]);
        nc[k] = f * c[k] + i * g;
        nh[k] = o * Math.tanh(nc[k]);
      }
      h = nh; c = nc;
      outs.push(h);
    }
    return outs;
  }

  // sentence vector: Bi-LSTM outputs pooled with additive attention (lstm.SiameseLSTM.encode)
  encode(text) {
    let ids = this.encodeIds(text);
    const empty = ids.length === 0;
    if (empty) ids = [0]; // an empty sentence is one padding step, as in the packed PyTorch batch
    const E = this.embDim, emb = this.t["embed.weight"].data;
    const embs = ids.map((id) => emb.subarray(id * E, id * E + E));
    const fwd = this.run(embs, "");
    const bwd = this.run([...embs].reverse(), "_reverse").reverse();
    const D = 2 * this.hidden;
    const outs = fwd.map((f, t) => { const o = new Float64Array(D); o.set(f, 0); o.set(bwd[t], this.hidden); return o; });

    const aw = this.t["attn.weight"].data, ab = this.t["attn.bias"].data[0];
    // padded positions (outputs are zero) take part in the softmax only when every position is masked
    const T = this.maxLen;
    const scores = new Float64Array(T).fill(-1e9);
    if (!empty) {
      outs.forEach((o, t) => { let s = ab; for (let k = 0; k < D; k++) s += aw[k] * Math.tanh(o[k]); scores[t] = s; });
    }
    const m = Math.max(...scores);
    const w = scores.map((s) => Math.exp(s - m));
    const z = w.reduce((p, q) => p + q, 0);
    const vec = new Float64Array(D);
    outs.forEach((o, t) => { for (let k = 0; k < D; k++) vec[k] += (w[t] / z) * o[k]; });
    return vec;
  }

  prob(a, b) {
    const u = this.encode(a), v = this.encode(b), D = u.length;
    const z = new Float64Array(4 * D);
    for (let k = 0; k < D; k++) {
      z[k] = u[k]; z[D + k] = v[k]; z[2 * D + k] = Math.abs(u[k] - v[k]); z[3 * D + k] = u[k] * v[k];
    }
    const W1 = this.t["head.1.weight"].data, b1 = this.t["head.1.bias"].data;
    const W2 = this.t["head.4.weight"].data, b2 = this.t["head.4.bias"].data[0];
    const n1 = b1.length, n0 = z.length;
    let out = b2;
    for (let r = 0; r < n1; r++) {
      let s = b1[r];
      for (let k = 0; k < n0; k++) s += W1[r * n0 + k] * z[k];
      out += W2[r] * Math.max(0, s);
    }
    return sigmoid(out);
  }
}

// ---------------------------------------------------------------- LightGBM

function treeValue(node, x) {
  while (node.leaf_value === undefined) {
    const v = x[node.split_feature];
    let left;
    if (Number.isNaN(v)) left = node.default_left;
    else left = node.decision_type === "<=" ? v <= node.threshold : v === node.threshold;
    node = left ? node.left_child : node.right_child;
  }
  return node.leaf_value;
}

export function lightgbmProb(model, x) {
  let raw = 0;
  for (const tree of model.tree_info) raw += treeValue(tree.tree_structure, x);
  return sigmoid(raw);
}

// ---------------------------------------------------------------- full ensemble

export class Ensemble {
  // embed(textA, textB) -> cosine similarity of MiniLM embeddings (supplied by the caller)
  constructor(config, lstmBuffer, lightgbm, embedSimilarity) {
    this.config = config;
    this.lstm = new BiLSTM(config.lstm, lstmBuffer);
    this.lightgbm = lightgbm;
    this.embedSimilarity = embedSimilarity;
  }

  async features(a, b) {
    return {
      minilm_sim: await this.embedSimilarity(a, b),
      lstm_prob: this.lstm.prob(a, b),
      tfidf_sim: tfidfSimilarity(a, b, this.config.tfidf),
      jaccard_sim: jaccard(a, b),
      token_overlap: tokenOverlap(a, b),
      levenshtein_dist: levenshtein(a, b),
    };
  }

  async predict(a, b) {
    const f = await this.features(a, b);
    const x = this.config.features.map((name) => f[name]);
    return { prob: lightgbmProb(this.lightgbm, x), features: f };
  }
}
