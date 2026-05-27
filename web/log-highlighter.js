/**
 * Log syntax highlighter — VS Code log grammar (emilast/vscode-logfile-highlighter)
 * plus structured parsers for spdlog / batch-runner lines.
 *
 * Token pipeline (same idea as TextMate / VS Code):
 *  1. Line-level structure (spdlog prefix, batch tags, sections)
 *  2. Protect string regions so numbers inside JSON are not re-tokenized
 *  3. Apply scoped patterns with priority (level keywords beat constants)
 */

export function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

const PRIORITY = {
  "log-error": 90,
  "log-warning": 85,
  "log-info": 80,
  "log-debug": 75,
  "log-trace": 70,
  "log-exception-type": 65,
  "log-url": 60,
  "log-date": 55,
  "log-json-key": 52,
  "log-string": 50,
  "log-namespace": 45,
  "log-constant": 40,
};

/** VS Code extensions/log/syntaxes/log.tmLanguage.json — ordered patterns. */
const VSCODE_LOG_PATTERNS = [
  { re: /\b([Tt]race|TRACE)\b:?/gi, cls: "log-trace" },
  { re: /\[(verbose|verb|vrb|vb|v)\]/gi, cls: "log-trace" },
  { re: /\b(DEBUG|Debug)\b|\bdebug:/gi, cls: "log-debug" },
  { re: /\[(debug|dbug|dbg|de|d)\]/gi, cls: "log-debug" },
  { re: /\b(HINT|INFO|INFORMATION|Info|NOTICE|II)\b|\b(info|information):/gi, cls: "log-info" },
  { re: /\[(information|info|inf|in|i)\]/gi, cls: "log-info" },
  { re: /\b(WARNING|WARN|Warn|WW)\b|\bwarning:/gi, cls: "log-warning" },
  { re: /\[(warning|warn|wrn|wn|w)\]/gi, cls: "log-warning" },
  {
    re: /\b(ALERT|CRITICAL|EMERGENCY|ERROR|FAILURE|FAIL|Fatal|FATAL|Error|EE)\b|\berror:/gi,
    cls: "log-error",
  },
  { re: /\[(error|eror|err|er|e|fatal|fatl|ftl|fa|f)\]/gi, cls: "log-error" },
  { re: /\b\d{4}-\d{2}-\d{2}(?=T|\b)/g, cls: "log-date" },
  { re: /(?<=(^|\s))\d{2}[^\w\s]\d{2}[^\w\s]\d{4}\b/g, cls: "log-date" },
  { re: /T?\d{1,2}:\d{2}(:\d{2}([.,]\d+)?)?(Z| ?[+-]\d{1,2}:\d{2})?\b/g, cls: "log-date" },
  { re: /T\d{2}\d{2}(\d{2}([.,]\d+)?)?(Z| ?[+-]\d{1,2}\d{2})?\b/g, cls: "log-date" },
  { re: /\b([0-9a-fA-F]{40}|[0-9a-fA-F]{10}|[0-9a-fA-F]{7})\b/g, cls: "log-constant" },
  { re: /\b[0-9a-fA-F]{8}-?([0-9a-fA-F]{4}-?){3}[0-9a-fA-F]{12}\b/gi, cls: "log-constant" },
  // Oniguruma {2,}+ is invalid in JS; match MAC-like hex pairs instead.
  { re: /\b(?:[0-9a-fA-F]{2}(?:[:-][0-9a-fA-F]{2}){2,})\b/gi, cls: "log-constant" },
  { re: /\b(0x[a-fA-F0-9]+)\b/g, cls: "log-constant" },
  { re: /\b([0-9]+|true|false|null)\b/gi, cls: "log-constant" },
  { re: /\b([a-zA-Z.]*Exception)\b/g, cls: "log-exception-type" },
  { re: /\b[a-z]+:\/\/\S+\b\/?/gi, cls: "log-url" },
  { re: /(?<![\w/\\])([\w-]+\.)+([\w-])+(?![\w/\\])/g, cls: "log-namespace" },
];

const SPDLOG_LINE =
  /^(\[\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\])\s+(\[(trace|debug|info|warn(?:ing)?|error|critical|fatal)\])\s+(.*)$/i;

const LEVEL_CLASS = {
  trace: "log-trace",
  debug: "log-debug",
  info: "log-info",
  warn: "log-warning",
  warning: "log-warning",
  error: "log-error",
  critical: "log-error",
  fatal: "log-error",
};

const LINE_LEVEL_CLASS = {
  trace: "level-trace",
  debug: "level-debug",
  info: "level-info",
  warn: "level-warning",
  warning: "level-warning",
  error: "level-error",
  critical: "level-error",
  fatal: "level-error",
};

function wrapAll(text, cls, extraClass = "") {
  const extra = extraClass ? ` ${extraClass}` : "";
  return `<span class="${cls}${extra}">${escapeHtml(text)}</span>`;
}

function createTokenState(length) {
  return {
    classes: new Array(length).fill(null),
    priorities: new Array(length).fill(-1),
    masked: new Uint8Array(length),
  };
}

function assignRange(state, start, end, cls) {
  const pri = PRIORITY[cls] ?? 0;
  for (let i = start; i < end; i++) {
    if (state.masked[i]) continue;
    if (pri >= state.priorities[i]) {
      state.classes[i] = cls;
      state.priorities[i] = pri;
    }
  }
}

function maskRange(state, start, end, cls) {
  for (let i = start; i < end; i++) {
    state.masked[i] = 1;
    state.classes[i] = cls;
    state.priorities[i] = PRIORITY[cls] ?? 100;
  }
}

function markQuotedStrings(line, state) {
  for (const re of [/"(?:\\.|[^"\\])*"/g, /(?<![\w])'(?:\\.|[^'\\])*'/g]) {
    re.lastIndex = 0;
    let m;
    while ((m = re.exec(line)) !== null) {
      maskRange(state, m.index, m.index + m[0].length, "log-string");
    }
  }
}

function markJsonKeys(line, state) {
  const re = /"([^"\\]|\\.)*"(?=\s*:)/g;
  let m;
  while ((m = re.exec(line)) !== null) {
    if (state.masked[m.index]) continue;
    assignRange(state, m.index, m.index + m[0].length, "log-json-key");
  }
}

function applyVscodePatterns(line, state) {
  for (const { re, cls } of VSCODE_LOG_PATTERNS) {
    const regex = new RegExp(re.source, re.flags.includes("g") ? re.flags : `${re.flags}g`);
    let m;
    while ((m = regex.exec(line)) !== null) {
      assignRange(state, m.index, m.index + m[0].length, cls);
    }
  }
}

function classesToHtml(line, classes) {
  let out = "";
  let i = 0;
  while (i < line.length) {
    const cls = classes[i];
    let j = i + 1;
    while (j < line.length && classes[j] === cls) j++;
    const slice = escapeHtml(line.slice(i, j));
    out += cls ? `<span class="${cls}">${slice}</span>` : slice;
    i = j;
  }
  return out;
}

function highlightSegment(text, { skipStrings = true, skipPatterns = false } = {}) {
  if (!text) return "";
  const state = createTokenState(text.length);
  if (skipStrings) markQuotedStrings(text, state);
  if (!skipPatterns) {
    markJsonKeys(text, state);
    applyVscodePatterns(text, state);
  }
  return classesToHtml(text, state.classes);
}

function detectLineLevel(line) {
  const spd = line.match(SPDLOG_LINE);
  if (spd) return LINE_LEVEL_CLASS[spd[3].toLowerCase()] || null;
  if (/\[(error|eror|err|fatal|critical)\]/i.test(line)) return "level-error";
  if (/\[(warning|warn|wrn)\]/i.test(line)) return "level-warning";
  if (/\b(ERROR|FAIL|FATAL|CRITICAL)\b/i.test(line)) return "level-error";
  if (/\b(WARN|WARNING)\b/i.test(line)) return "level-warning";
  return null;
}

function highlightSpdlogLine(line) {
  const m = line.match(SPDLOG_LINE);
  if (!m) return null;
  const [, ts, levelToken, levelWord, message] = m;
  const levelCls = LEVEL_CLASS[levelWord.toLowerCase()] || "log-info";
  const tsEnd = ts.length;
  const gap1End = tsEnd + 1;
  const levelEnd = gap1End + levelToken.length;
  const gap2End = levelEnd + 1;
  let out = wrapAll(ts, "log-date");
  out += escapeHtml(line.slice(tsEnd, gap1End));
  out += wrapAll(levelToken, levelCls, "log-level-badge");
  out += escapeHtml(line.slice(levelEnd, gap2End));
  out += highlightSegment(message);
  return out;
}

function highlightBatchTagLine(line) {
  const m = line.match(/^(\[(batch|viz|sim|grading|web)\])([\s\S]*)$/i);
  if (!m) return null;
  return wrapAll(m[1], "log-tag") + highlightSegment(m[3]);
}

/** Highlight one log line (VS Code log grammar + batch-runner conveniences). */
export function highlightLogLine(line) {
  if (/^===/.test(line)) return wrapAll(line, "log-section");
  if (/^\$/.test(line)) return wrapAll(line, "log-command");
  if (/^[\t ]*at[\t ]/.test(line)) return wrapAll(line, "log-exception");

  const spd = highlightSpdlogLine(line);
  if (spd) return spd;

  const batch = highlightBatchTagLine(line);
  if (batch) return batch;

  const state = createTokenState(line.length);
  markQuotedStrings(line, state);
  markJsonKeys(line, state);
  applyVscodePatterns(line, state);
  return classesToHtml(line, state.classes);
}

/** Render log lines into a container element. */
export function renderLogPanel(viewEl, metaEl, lines) {
  if (!lines || !lines.length) {
    viewEl.innerHTML = '<div class="log-empty">（无日志）</div>';
    if (metaEl) metaEl.textContent = "0 行";
    return;
  }
  const frag = document.createDocumentFragment();
  lines.forEach((line, idx) => {
    const row = document.createElement("div");
    const levelCls = detectLineLevel(line);
    row.className = levelCls ? `log-line ${levelCls}` : "log-line";
    const ln = document.createElement("span");
    ln.className = "log-ln";
    ln.textContent = String(idx + 1);
    const txt = document.createElement("span");
    txt.className = "log-text";
    txt.innerHTML = highlightLogLine(line);
    row.appendChild(ln);
    row.appendChild(txt);
    frag.appendChild(row);
  });
  viewEl.innerHTML = "";
  viewEl.appendChild(frag);
  if (metaEl) metaEl.textContent = `${lines.length} 行`;
  viewEl.scrollTop = viewEl.scrollHeight;
}
