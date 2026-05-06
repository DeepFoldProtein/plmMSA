// MSA viewer — pure module. Exposes window.MsaViewer.create(opts).
//
// No host-environment dependencies (no acquireVsCodeApi, no postMessage).
// Persistence is delegated to opts.storage (a Storage-shaped object;
// localStorage works directly). The same module ships in:
//   • this VS Code extension via media/viewer.adapter.js
//   • plmMSA's public/ via a 5-line wrapper in app.js
//
// Multiple instances may co-exist on a page — every piece of mutable
// state is captured in the create() closure. The tooltip element and
// document/window listeners are owned per-instance and removed on
// destroy().

(function () {
  "use strict";

  const FONT_MIN = 6;
  const FONT_MAX = 32;
  const FONT_DEFAULT = 10;
  const HIST_MIN = 30;
  const HIST_MAX = 240;
  const HIST_DEFAULT = 30;
  const LABEL_MIN = 0;
  const LABEL_MAX = 600;
  const LABEL_DEFAULT = 192;
  const LABEL_HIDE_THRESHOLD = 24;
  const VLIST_OVERSCAN = 8;
  // Horizontal virtualization: extra match-columns rendered on each
  // side of the visible window. Sized to absorb typical scroll deltas
  // between two `scroll`-event ticks at moderate fling speeds, so most
  // horizontal scrolls don't trigger a row rebuild.
  const COL_OVERSCAN = 64;
  const ROW_LINE_HEIGHT = 1.25;
  const VALID_PALETTES = new Set(["chemical", "clustalx", "taylor", "zappo"]);
  // Coverage histogram coloring: off | entropy | blosum.
  const VALID_COVER_COLORS = new Set(["off", "entropy", "blosum"]);
  const COVER_COLOR_DEFAULT = "blosum";

  // BLOSUM62 — used to color coverage bars by per-column conservation.
  // Self-pair max = 11 (W↔W); off-diagonal min = -4.
  const BLOSUM62_AA = "ARNDCQEGHILKMFPSTWYV";
  const BLOSUM62_ROWS = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
  ];
  const BLOSUM62_INDEX = new Map();
  for (let i = 0; i < BLOSUM62_AA.length; i++) BLOSUM62_INDEX.set(BLOSUM62_AA[i], i);
  // Practical normalization range for mean SP scores on protein columns:
  // [-2, +6]. Clamped on both sides for the histogram color ramp.
  const BLOSUM62_NORM_LO = -2;
  const BLOSUM62_NORM_HI = 6;
  const LN20 = Math.log(20);

  // Per-row HTML batching. Building hundreds of cells via createElement +
  // appendChild is the dominant cost when rendering wide alignments;
  // assembling one HTML string per row and assigning innerHTML once is
  // 3–5× faster. Residue chars in our inputs are ASCII letters / `-` /
  // `.` / `*` / `?` / digits — none need escaping in practice — but we
  // escape defensively so a stray `<` or `&` in a synthetic track can't
  // inject markup.
  const HTML_ESC_MAP = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
  function htmlEscape(s) {
    return /[&<>"]/.test(s) ? String(s).replace(/[&<>"]/g, (c) => HTML_ESC_MAP[c]) : String(s);
  }
  function htmlEscapeChar(ch) {
    return HTML_ESC_MAP[ch] || ch;
  }

  // Hoisted to module scope so the precomputed residue-HTML tables
  // below can reference it. Re-used (via closure capture) inside
  // `create()` for one-off slow-path calls on non-ASCII residues.
  function residueClass(residue) {
    switch (String(residue).toUpperCase()) {
      case "A": case "I": case "L": case "M":
      case "F": case "W": case "V":
        return "hydrophobic";
      case "D": case "E": return "acidic";
      case "K": case "R": return "basic";
      case "H": case "Y": return "aromatic";
      case "S": case "T": case "N": case "Q": return "polar";
      case "C": return "cysteine";
      case "G": return "glycine";
      case "P": return "proline";
      case "-": case ".": return "gap";
      default: return "other";
    }
  }

  // Pre-baked per-character HTML for residue cells. Residues are ASCII;
  // we precompute the full `<span …>X</span>` string once per char code,
  // for both match and insert variants. matchHtml/insertHtml become a
  // single array index — no class lookup, no escape, no template-string
  // interpolation per cell. Hover column lookup no longer relies on
  // data-col; it walks siblings within the row body and offsets by the
  // row's recorded colStart (set as a JS expando on the row element).
  const RESIDUE_HTML_MATCH = new Array(128);
  const RESIDUE_HTML_INSERT = new Array(128);
  (function precomputeResidueHtml() {
    for (let cc = 0; cc < 128; cc++) {
      const ch = String.fromCharCode(cc);
      const cls = residueClass(ch);
      const upper = ch >= "a" && ch <= "z" ? String.fromCharCode(cc - 32) : ch;
      const aa = (upper >= "A" && upper <= "Z") ? ` data-aa="${upper}"` : "";
      const escaped = HTML_ESC_MAP[ch] || ch;
      RESIDUE_HTML_MATCH[cc] = `<span class="msa-residue msa-aa-${cls}"${aa}>${escaped}</span>`;
      RESIDUE_HTML_INSERT[cc] = `<span class="msa-residue msa-aa-${cls} msa-insert"${aa}>${escaped}</span>`;
    }
  })();
  function matchCellHtml(ch) {
    const cc = ch.charCodeAt(0);
    if (cc < 128) return RESIDUE_HTML_MATCH[cc];
    // Slow path: non-ASCII residue character. Should not happen in real
    // alignments but we keep it correct.
    const cls = residueClass(ch);
    return `<span class="msa-residue msa-aa-${cls}">${htmlEscapeChar(ch)}</span>`;
  }
  function insertCellHtml(ch) {
    const cc = ch.charCodeAt(0);
    if (cc < 128) return RESIDUE_HTML_INSERT[cc];
    const cls = residueClass(ch);
    return `<span class="msa-residue msa-aa-${cls} msa-insert">${htmlEscapeChar(ch)}</span>`;
  }
  /** breakAfter = 0 → auto-fit to window width on every resize. Manual
   *  override clamps to [_MIN, _MAX]. */
  const BREAK_AFTER_MIN = 30;
  const BREAK_AFTER_MAX = 240;
  const BREAK_AFTER_AUTO = 0;
  const BREAK_AFTER_DEFAULT = BREAK_AFTER_AUTO;
  // Hit-table columns we let the user sort by (clicked TH).
  const HHR_SORT_KEYS = new Set([
    "num", "desc", "prob", "evalue", "score", "cols", "qStart", "tStart",
  ]);

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  /**
   * UniProt/UniRef FASTA header parsers — pure, return null on miss.
   *
   * UniProtKB:
   *   >db|UniqueIdentifier|EntryName ProteinName OS=… [OX=…] [GN=…] [PE=…] [SV=…]
   *   db ∈ {sp, tr}. Archived entries keep the same head with "Deleted, …"
   *   in the protein name slot — head still parses + links.
   *
   * UniRef:
   *   >UniRef{100,90,50}_UniqueIdentifier ClusterName n=Members Tax=Taxon
   *                                                  TaxID=ID RepID=Rep
   *
   * Both tails use a generic tag-aware splitter so values may contain
   * whitespace, parentheses, slashes, etc. — the only delimiter is the
   * next known KEY= token.
   */
  function parseTaggedTail(tail, keys) {
    const set = new Set(keys);
    const out = { __name: "" };
    if (!tail) return out;
    const tokens = String(tail).split(/\s+/).filter((t) => t.length > 0);
    const nameTokens = [];
    let curKey = null;
    let curVal = [];
    function flush() {
      if (curKey != null) out[curKey] = curVal.join(" ").trim();
      curKey = null;
      curVal = [];
    }
    for (const tok of tokens) {
      const m = /^([A-Za-z]+)=(.*)$/.exec(tok);
      if (m && set.has(m[1])) {
        flush();
        curKey = m[1];
        if (m[2] !== "") curVal.push(m[2]);
      } else if (curKey != null) {
        curVal.push(tok);
      } else {
        nameTokens.push(tok);
      }
    }
    flush();
    out.__name = nameTokens.join(" ").trim();
    return out;
  }

  // HHsuite convention: many headers carry a "/<start>-<end>" subseq
  // range after the accession. Strip and capture it separately so the
  // accession used for linking is the bare ID.
  function splitAccessionRange(s) {
    const m = /^(.+?)(?:\/(\d+)-(\d+))?$/.exec(String(s));
    if (!m) return { acc: s, range: null };
    return {
      acc: m[1],
      range: m[2] ? { start: Number(m[2]), end: Number(m[3]) } : null,
    };
  }

  // Canonical UniProtKB accession regex from the UniProt FASTA spec.
  // Two alternative shapes; optional "-<N>" isoform suffix. Source string
  // form is reused inside other patterns via UNIPROT_ACC_SRC.
  const UNIPROT_ACC_SRC =
    "(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\\d+)?";
  const UNIPROT_ACC_RE = new RegExp(`^${UNIPROT_ACC_SRC}$`);
  const UNIPROT_KB_HEADER_RE = new RegExp(
    `^(sp|tr)\\|(${UNIPROT_ACC_SRC})\\|(\\S+?)(?:\\/(\\d+)-(\\d+))?(?:\\s+(.+))?$`,
    "i",
  );

  function parseUniProtKbHeader(text) {
    if (!text) return null;
    const m = UNIPROT_KB_HEADER_RE.exec(String(text).trim());
    if (!m) return null;
    const tail = parseTaggedTail(m[6] || "", ["OS", "OX", "GN", "PE", "SV"]);
    return {
      kind: "uniprotkb",
      db: m[1].toLowerCase(),
      accession: m[2].toUpperCase(),
      entryName: m[3],
      range: m[4] ? { start: Number(m[4]), end: Number(m[5]) } : null,
      proteinName: tail.__name,
      os: tail.OS, ox: tail.OX, gn: tail.GN, pe: tail.PE, sv: tail.SV,
    };
  }

  // UniParc identifiers: "UPI" followed by exactly 10 uppercase hex
  // digits (e.g. UPI00012E1B92). FASTA tail may carry "status=active".
  const UNIPARC_ID_RE = /^UPI[0-9A-F]{10}$/i;
  function parseUniParcHeader(text) {
    if (!text) return null;
    const m = /^(UPI[0-9A-F]{10})(?:\/(\d+)-(\d+))?(?:\s+(.+))?$/i.exec(String(text).trim());
    if (!m) return null;
    const tail = parseTaggedTail(m[4] || "", ["status"]);
    return {
      kind: "uniparc",
      accession: m[1].toUpperCase(),
      range: m[2] ? { start: Number(m[2]), end: Number(m[3]) } : null,
      status: tail.status,
    };
  }

  function parseUniRefHeader(text) {
    if (!text) return null;
    const m = /^(UniRef(?:100|90|50))_(\S+?)(?:\s+(.+))?$/.exec(String(text).trim());
    if (!m) return null;
    const { acc, range } = splitAccessionRange(m[2]);
    const tail = parseTaggedTail(m[3] || "", ["n", "Tax", "TaxID", "RepID"]);
    return {
      kind: "uniref",
      cluster: m[1],
      accession: acc,
      range,
      fullId: `${m[1]}_${acc}`,
      clusterName: tail.__name,
      n: tail.n, tax: tail.Tax, taxId: tail.TaxID, repId: tail.RepID,
    };
  }

  // Secondary chip rendered after the primary head link. Single AFDB
  // affordance — the primary click already goes to the canonical entry
  // (UniProt entry / UniRef cluster), so this chip just exposes the
  // structural alternative.
  function secondaryHeaderLinks(parsed, link) {
    if (link && link.kind === "uniprot" && link.id) {
      return [{
        url: `https://alphafold.ebi.ac.uk/entry/${link.id}`,
        label: "AFDB",
        tip: `AlphaFold DB → ${link.id}`,
      }];
    }
    if (link && link.kind === "uniref"
        && parsed && parsed.kind === "uniref"
        && parsed.accession
        && UNIPROT_ACC_RE.test(parsed.accession)) {
      return [{
        url: `https://alphafold.ebi.ac.uk/entry/${parsed.accession}`,
        label: "AFDB",
        tip: `AlphaFold DB → ${parsed.accession} (UniRef rep)`,
      }];
    }
    return [];
  }

  function formatHeaderTip(parsed, link) {
    const lines = [];
    if (link && link.url) lines.push(`${link.label || "open"} → ${link.url}`);
    if (parsed.kind === "uniprotkb") {
      const dbLabel = parsed.db === "sp" ? "Swiss-Prot" : "TrEMBL";
      const accLine = parsed.range
        ? `${dbLabel} · ${parsed.accession}/${parsed.range.start}-${parsed.range.end}`
        : `${dbLabel} · ${parsed.accession}`;
      lines.push(accLine);
      if (parsed.entryName) lines.push(`Entry: ${parsed.entryName}`);
      if (parsed.proteinName) lines.push(`Protein: ${parsed.proteinName}`);
      if (parsed.os) {
        lines.push(parsed.ox
          ? `Organism: ${parsed.os} (taxID ${parsed.ox})`
          : `Organism: ${parsed.os}`);
      }
      if (parsed.gn) lines.push(`Gene: ${parsed.gn}`);
      const meta = [];
      if (parsed.pe) meta.push(`PE=${parsed.pe}`);
      if (parsed.sv) meta.push(`SV=${parsed.sv}`);
      if (meta.length) lines.push(meta.join(" · "));
    } else if (parsed.kind === "uniparc") {
      const accLine = parsed.range
        ? `UniParc · ${parsed.accession}/${parsed.range.start}-${parsed.range.end}`
        : `UniParc · ${parsed.accession}`;
      lines.push(accLine);
      if (parsed.status) lines.push(`Status: ${parsed.status}`);
    } else if (parsed.kind === "uniref") {
      const repLine = parsed.range
        ? `${parsed.cluster} · rep ${parsed.accession}/${parsed.range.start}-${parsed.range.end}`
        : `${parsed.cluster} · rep ${parsed.accession}`;
      lines.push(repLine);
      if (parsed.clusterName) lines.push(`Cluster: ${parsed.clusterName}`);
      if (parsed.n) lines.push(`Members: ${parsed.n}`);
      if (parsed.tax) {
        lines.push(parsed.taxId
          ? `Tax: ${parsed.tax} (taxID ${parsed.taxId})`
          : `Tax: ${parsed.tax}`);
      }
      if (parsed.repId) lines.push(`RepID: ${parsed.repId}`);
    }
    return lines.join("\n");
  }

  /**
   * Default DB link resolver — accession-shape heuristics. Returns
   * { url, label } for the first matching database, or null.
   *
   * Patterns are intentionally conservative: a token like "P12345"
   * matches UniProt, but "Endo-1,4" does not match anything. The
   * caller can override via opts.linkResolver(name) to integrate
   * with internal directories.
   */
  function defaultLinkResolver(name) {
    if (!name) return null;
    const text = String(name).trim();
    let m;
    // PDB with chain (e.g. "2DFB_A"). 4-char alnum starting with a digit.
    if ((m = /^([0-9][A-Za-z0-9]{3})_[A-Za-z0-9]+$/.exec(text))) {
      const id = m[1].toUpperCase();
      return { url: `https://www.rcsb.org/structure/${id}`, label: "PDB", kind: "pdb", id };
    }
    // PDB without chain ("2dfb").
    if ((m = /^([0-9][A-Za-z0-9]{3})$/.exec(text))) {
      const id = m[1].toUpperCase();
      return { url: `https://www.rcsb.org/structure/${id}`, label: "PDB", kind: "pdb", id };
    }
    // UniProtKB pipe-delimited form: "sp|P12345|FOO_HUMAN" / "tr|...|...".
    // Primary click → canonical UniProt entry; AFDB lives in a chip.
    if ((m = /^(sp|tr)\|([A-Z0-9]+(?:-\d+)?)\|(\S+)$/i.exec(text))) {
      const acc = m[2].toUpperCase();
      return { url: `https://www.uniprot.org/uniprotkb/${acc}/entry`, label: "UniProt", kind: "uniprot", id: acc };
    }
    // UniProt accession (Swiss-Prot + TrEMBL), optionally with HHsuite
    // "/start-end" subseq suffix.
    {
      const bare = text.replace(/\/\d+-\d+$/, "");
      if (UNIPROT_ACC_RE.test(bare)) {
        return { url: `https://www.uniprot.org/uniprotkb/${bare}/entry`, label: "UniProt", kind: "uniprot", id: bare };
      }
    }
    // UniRef cluster (e.g. "UniRef90_P12345" or "UniRef90_A0A409W2S0/241-416").
    if ((m = /^(UniRef\d+)_([\w.-]+?)(?:\/(\d+)-(\d+))?$/.exec(text))) {
      const fullId = `${m[1]}_${m[2]}`;
      return { url: `https://www.uniprot.org/uniref/${fullId}`, label: "UniRef", kind: "uniref", id: fullId };
    }
    // UniParc: "UPI" + 10 hex digits, optional "/start-end" subseq.
    {
      const bare = text.replace(/\/\d+-\d+$/, "");
      if (UNIPARC_ID_RE.test(bare)) {
        const id = bare.toUpperCase();
        return { url: `https://www.uniprot.org/uniparc/${id}/entry`, label: "UniParc", kind: "uniparc", id };
      }
    }
    // MGnify protein (e.g. "MGYP000510094044", optionally with a /start-end suffix).
    if ((m = /^(MGYP\d+)(?:\/.*)?$/.exec(text))) {
      return { url: `https://www.ebi.ac.uk/metagenomics/sequence/${m[1]}`, label: "MGnify" };
    }
    // Pfam family.
    if ((m = /^(PF\d{5})$/.exec(text))) {
      return { url: `https://www.ebi.ac.uk/interpro/entry/pfam/${m[1]}`, label: "Pfam" };
    }
    // InterPro entry.
    if ((m = /^(IPR\d{6})$/.exec(text))) {
      return { url: `https://www.ebi.ac.uk/interpro/entry/InterPro/${m[1]}`, label: "InterPro" };
    }
    // NCBI RefSeq protein (e.g. NP_001234.1, XP_004567890).
    if ((m = /^([NXY]P_\d+(?:\.\d+)?)$/.exec(text))) {
      return { url: `https://www.ncbi.nlm.nih.gov/protein/${m[1]}`, label: "NCBI" };
    }
    return null;
  }

  /**
   * Create an MSA viewer instance bound to `opts.container`.
   *
   *   opts.container  HTMLElement (required)
   *   opts.storage    Storage-shaped object (default: localStorage)
   *   opts.storageKey string (default: 'msa-viewer.state')
   *
   * Returns { load(viewerMsa), destroy() }.
   */
  function create(opts) {
    if (!opts || !opts.container) throw new Error("MsaViewer.create: opts.container is required");
    const container = opts.container;
    const storage = opts.storage || (typeof localStorage !== "undefined" ? localStorage : null);
    const storageKey = opts.storageKey || "msa-viewer.state";
    // Optional caller-supplied link resolver (Phase 3, item 3.4). Takes
    // a single string (a sequence name / hit accession) and returns
    // {url, label} or null. The built-in fallback handles common
    // protein databases — PDB / UniProt / UniRef / MGYP / Pfam /
    // InterPro / NCBI — based on accession-shape heuristics.
    const linkResolver = typeof opts.linkResolver === "function"
      ? opts.linkResolver
      : defaultLinkResolver;
    // Optional: open a PDB structure inside the host (VS Code download
    // + open, or whatever the host wires up). When unset, the inline
    // "Code" button next to PDB links is hidden.
    const onPdbOpen = typeof opts.onPdbOpen === "function" ? opts.onPdbOpen : null;

    // ---- per-instance state ----
    let state = loadState();
    if (typeof state.showInserts !== "boolean") state.showInserts = false;
    if (typeof state.fontPx !== "number") state.fontPx = FONT_DEFAULT;
    if (typeof state.histPx !== "number") state.histPx = HIST_DEFAULT;
    if (typeof state.filter !== "string") state.filter = "";
    if (typeof state.labelWidth !== "number") state.labelWidth = LABEL_DEFAULT;
    if (typeof state.palette !== "string") state.palette = "chemical";
    if (!VALID_PALETTES.has(state.palette)) state.palette = "chemical";
    if (typeof state.msaColorOn !== "boolean") state.msaColorOn = true;
    if (typeof state.hhrColorOn !== "boolean") state.hhrColorOn = false;
    if (typeof state.breakAfter !== "number") state.breakAfter = BREAK_AFTER_DEFAULT;
    // Migration: a previous version defaulted to 80 (manual). Anyone
    // who never explicitly set a value should now flip to auto.
    if (state.breakAfter === 80) state.breakAfter = BREAK_AFTER_AUTO;
    if (typeof state.hhrSortKey !== "string") state.hhrSortKey = "num";
    if (typeof state.hhrSortDir !== "string") state.hhrSortDir = "asc";
    if (typeof state.msaTemplatesMode !== "boolean") state.msaTemplatesMode = false;
    if (typeof state.msaCoverColor !== "string") state.msaCoverColor = COVER_COLOR_DEFAULT;
    if (!VALID_COVER_COLORS.has(state.msaCoverColor)) state.msaCoverColor = COVER_COLOR_DEFAULT;
    if (typeof state.firstRowIsQuery !== "boolean") state.firstRowIsQuery = true;
    if (typeof state.scrollTop !== "number" || !isFinite(state.scrollTop)) state.scrollTop = 0;
    if (typeof state.scrollLeft !== "number" || !isFinite(state.scrollLeft)) state.scrollLeft = 0;
    state.scrollTop = Math.max(0, state.scrollTop);
    state.scrollLeft = Math.max(0, state.scrollLeft);
    state.fontPx = clamp(state.fontPx, FONT_MIN, FONT_MAX);
    state.histPx = clamp(state.histPx, HIST_MIN, HIST_MAX);
    state.labelWidth = clamp(state.labelWidth, LABEL_MIN, LABEL_MAX);
    if (state.breakAfter !== BREAK_AFTER_AUTO) {
      state.breakAfter = clamp(state.breakAfter, BREAK_AFTER_MIN, BREAK_AFTER_MAX);
    }
    if (!HHR_SORT_KEYS.has(state.hhrSortKey)) state.hhrSortKey = "num";
    if (state.hhrSortDir !== "asc" && state.hhrSortDir !== "desc") state.hhrSortDir = "asc";

    // Per-column info populated by load() and read by the tooltip handler.
    let currentColumnInfo = null;

    // Per-entry memo for header parsing + link resolution. The same
    // header is rendered on every scroll-driven row redraw, so caching
    // the regex+resolver work eliminates a lot of string scanning.
    // Keyed by entry; invalidated when the cached `text` differs.
    const headerCache = new WeakMap();
    function getHeaderInfo(entry, text) {
      if (entry) {
        const c = headerCache.get(entry);
        if (c && c.text === text) return c;
      }
      const m = /^(\S+)(\s.*)?$/.exec(text);
      const head = m ? m[1] : text;
      const tail = m && m[2] ? m[2] : "";
      const link = linkResolver(head);
      const parsed = parseUniProtKbHeader(text)
        || parseUniRefHeader(text)
        || parseUniParcHeader(text);
      const info = { text, head, tail, link, parsed };
      if (entry) headerCache.set(entry, info);
      return info;
    }

    // Error display lives inside the container so the module owns its DOM.
    container.innerHTML = "";
    const errorEl = document.createElement("div");
    errorEl.className = "msa-error";
    errorEl.hidden = true;
    container.appendChild(errorEl);

    // Floating tooltip — one per instance, parented to document.body so it
    // can escape overflow:hidden ancestors.
    const tipEl = document.createElement("div");
    tipEl.className = "msa-tooltip";
    tipEl.hidden = true;
    document.body.appendChild(tipEl);
    let currentTipTarget = null;

    // Listeners registered per-instance; tracked for destroy().
    const disposers = [];
    function on(target, type, fn, options) {
      target.addEventListener(type, fn, options);
      disposers.push(() => target.removeEventListener(type, fn, options));
    }

    // ---- drag-and-drop ----
    // The module shows a visual highlight on dragover and calls
    // opts.onFileDrop(file) on drop. The host owns parsing — for VS
    // Code the adapter forwards file bytes to the extension via
    // postMessage; for plmMSA the wrapper can do anything (parse
    // client-side, submit as a new job, etc.). If onFileDrop is not
    // provided we still show feedback but no-op on drop.
    let dragDepth = 0;
    on(container, "dragenter", (ev) => {
      // Only react to file drags — drag of a text selection from
      // within the page should not trigger the import overlay.
      if (!hasFiles(ev)) return;
      ev.preventDefault();
      dragDepth++;
      if (dragDepth === 1) container.classList.add("msa-drag-over");
    });
    on(container, "dragover", (ev) => {
      if (!hasFiles(ev)) return;
      ev.preventDefault();
      if (ev.dataTransfer) ev.dataTransfer.dropEffect = "copy";
    });
    on(container, "dragleave", (ev) => {
      if (!hasFiles(ev)) return;
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) container.classList.remove("msa-drag-over");
    });
    on(container, "drop", async (ev) => {
      if (!hasFiles(ev)) return;
      ev.preventDefault();
      dragDepth = 0;
      container.classList.remove("msa-drag-over");
      const file = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
      if (!file) return;
      if (typeof opts.onFileDrop !== "function") return;
      try {
        await opts.onFileDrop(file);
      } catch (e) {
        showError(`Failed to load dropped file: ${e && e.message ? e.message : e}`);
      }
    });

    function hasFiles(ev) {
      const dt = ev.dataTransfer;
      if (!dt) return false;
      // `types` is a DOMStringList in older browsers; coerce to array.
      const types = Array.from(dt.types || []);
      return types.indexOf("Files") !== -1;
    }

    // ---- tooltip wiring (scoped to container; document for blur) ----
    //
    // Three tooltip sources, in priority order:
    //   1. data-tip="..." anywhere — explicit string.
    //   2. data-col="N" on histogram bars — col-info card.
    //   3. .msa-residue match cells inside a row — col-info card,
    //      derived from the row's recorded _colStart + sibling count.
    //      Match cells no longer carry data-col; computing on demand
    //      saves ~10 bytes per cell across thousands of cells per row.
    function residueColTarget(node) {
      if (!node || !node.classList) return null;
      if (!node.classList.contains("msa-residue")) return null;
      if (node.classList.contains("msa-insert")) return null;
      if (node.classList.contains("msa-spacer")) return null;
      return node;
    }
    function colForResidue(cell) {
      const row = cell.closest(".static-msa-row");
      if (!row || row._colStart == null) return -1;
      let count = 0;
      let prev = cell.previousElementSibling;
      while (prev) {
        if (prev.classList.contains("msa-residue") &&
            !prev.classList.contains("msa-insert") &&
            !prev.classList.contains("msa-spacer")) {
          count++;
        }
        prev = prev.previousElementSibling;
      }
      return row._colStart + count;
    }
    on(container, "mouseover", (ev) => {
      const t = ev.target.closest("[data-tip], [data-col], .msa-residue");
      if (!t || !container.contains(t)) return;
      let text = t.dataset && t.dataset.tip;
      if (!text && t.dataset && t.dataset.col != null) {
        text = formatColInfo(+t.dataset.col);
      }
      if (!text) {
        const residue = residueColTarget(t);
        if (residue) {
          const col = colForResidue(residue);
          if (col >= 0) text = formatColInfo(col);
        }
      }
      if (!text) return;
      tipEl.textContent = text;
      tipEl.hidden = false;
      positionTip(t);
      currentTipTarget = t;
    });
    on(container, "mouseout", (ev) => {
      const t = ev.target.closest("[data-tip], [data-col], .msa-residue");
      if (t && currentTipTarget === t) {
        tipEl.hidden = true;
        currentTipTarget = null;
      }
    });
    on(document, "scroll", () => {
      if (currentTipTarget) positionTip(currentTipTarget);
    }, true);
    on(window, "blur", () => {
      if (currentTipTarget) {
        tipEl.hidden = true;
        currentTipTarget = null;
      }
    });

    function positionTip(target) {
      const r = target.getBoundingClientRect();
      const tipR = tipEl.getBoundingClientRect();
      let left = r.left;
      let top = r.bottom + 6;
      const maxLeft = window.innerWidth - tipR.width - 8;
      if (left > maxLeft) left = Math.max(8, maxLeft);
      if (top + tipR.height > window.innerHeight - 8) {
        top = Math.max(8, r.top - tipR.height - 6);
      }
      tipEl.style.left = `${left}px`;
      tipEl.style.top = `${top}px`;
    }

    function formatColInfo(idx) {
      if (!currentColumnInfo) return null;
      const info = currentColumnInfo[idx];
      if (!info) return null;
      const cov = info.total > 0 ? (info.nonGap / info.total) * 100 : 0;
      const consE = info.consEntropy == null ? "—" : info.consEntropy.toFixed(2);
      const blosumStr = info.blosum == null
        ? "—"
        : `${info.blosum.toFixed(2)} (norm ${info.consBlosum.toFixed(2)})`;
      // The query residue at this column is visible in the sticky query
      // row above, so we don't repeat it here.
      const head =
        `col ${idx + 1}\n` +
        `coverage: ${info.nonGap} / ${info.total} (${cov.toFixed(1)}%)\n` +
        `entropy: ${info.entropy.toFixed(2)} nats (N_eff = ${Math.exp(info.entropy).toFixed(2)}, cons ${consE})\n` +
        `BLOSUM62 SP: ${blosumStr}`;
      if (!info.top || info.top.length === 0) return head;
      const countWidth = String(info.top[0].count).length;
      const lines = info.top.map((t) => {
        const pct = info.total > 0 ? (t.count / info.total) * 100 : 0;
        const c = String(t.count).padStart(countWidth, " ");
        return `  ${t.res}  ${c}/${info.total} (${pct.toFixed(1)}%)`;
      });
      const label = info.top.length === 1 ? "top residue:" : `top ${info.top.length} residues:`;
      return `${head}\n${label}\n${lines.join("\n")}`;
    }

    // ---- public API ----
    function load(payload) {
      // Clear any prior render except errorEl, which we replace.
      container.innerHTML = "";
      errorEl.hidden = true;
      container.appendChild(errorEl);

      // Discriminated payload (Phase 2). Back-compat: a bare ViewerMsa
      // (no `kind`) is treated as an MSA payload — that shape is what
      // the original VS Code adapter and plmMSA still send.
      if (payload && payload.kind === "pairwise") {
        if (!payload.hhr) {
          const empty = document.createElement("div");
          empty.className = "static-msa-empty";
          empty.textContent = "No HHR data to display.";
          container.appendChild(empty);
          return;
        }
        renderPairwiseInstance(payload.hhr);
        return;
      }

      const viewer =
        payload && payload.kind === "msa" ? payload.viewer : payload;
      if (!viewer || !Array.isArray(viewer.entries) || viewer.entries.length === 0) {
        const empty = document.createElement("div");
        empty.className = "static-msa-empty";
        empty.textContent = "No sequences to display.";
        container.appendChild(empty);
        return;
      }

      renderInstance(viewer);
    }

    function showError(message) {
      // Empty the container except the error element, then surface the message.
      container.innerHTML = "";
      errorEl.hidden = false;
      errorEl.textContent = message || "";
      container.appendChild(errorEl);
    }

    function destroy() {
      while (disposers.length) {
        try { disposers.pop()(); } catch (e) { /* ignore */ }
      }
      if (tipEl.parentNode) tipEl.parentNode.removeChild(tipEl);
      container.innerHTML = "";
    }

    function loadState() {
      if (!storage) return {};
      try {
        const raw = storage.getItem(storageKey);
        return raw ? JSON.parse(raw) : {};
      } catch { return {}; }
    }

    function persist() {
      if (!storage) return;
      try { storage.setItem(storageKey, JSON.stringify(state)); } catch { /* ignore */ }
    }

    // Per-instance render scope. Builds the wrapper, controls, table, list
    // and registers all the inner event listeners (each tracked via `on`).
    function renderInstance(viewer) {
      const supportsInserts = viewer.format === "a2m" || viewer.format === "a3m";
      const query = viewer.entries.find((e) => e.isQuery) || viewer.entries[0];
      const hits = viewer.entries.filter((e) => e !== query);

      const coverage = computeCoverage(viewer.entries, viewer.matchLen);
      const meanH = meanHammingToQuery(viewer.entries, query, viewer.matchLen);
      currentColumnInfo = computeColumnInfo(viewer, query);
      let currentRowStats = computeRowStats(viewer, query, state.firstRowIsQuery);
      const stats = {
        seqLen: query.matchSeq.replace(/[-.]/g, "").length,
        depth: viewer.entries.length,
        nEff: computeNeff(currentColumnInfo, query),
        meanH,
        log10H: meanH > 0 ? Math.log10(meanH) : null,
        log2H: meanH > 0 ? Math.log2(meanH) : null,
      };

      const wrapper = document.createElement("div");
      wrapper.className = "static-msa-viewer";
      applyFont(wrapper);
      applyHistHeight(wrapper);
      applyLabelWidth(wrapper);
      applyPalette(wrapper);

      const controls = document.createElement("div");
      controls.className = "static-msa-controls";

      const prevBtn = makeBtn("msa-page-btn", "↑");
      prevBtn.dataset.tip = "Page up (PgUp)";
      const pageLabel = document.createElement("span");
      pageLabel.className = "msa-page-label";
      const nextBtn = makeBtn("msa-page-btn", "↓");
      nextBtn.dataset.tip = "Page down (PgDn / Space)";

      const filterInput = document.createElement("input");
      filterInput.type = "search";
      filterInput.className = "msa-filter";
      filterInput.placeholder = "filter names…";
      filterInput.value = state.filter || "";
      filterInput.spellcheck = false;
      filterInput.dataset.tip = "Filter by name (Cmd/Ctrl+F)";
      on(filterInput, "input", () => {
        state.filter = filterInput.value.trim();
        // Filtering reshapes the row list — the saved scroll offset
        // would land on a different row. Reset to top.
        state.scrollTop = 0;
        table.scrollTop = 0;
        persist();
        drawAll();
      });

      const gotoInput = document.createElement("input");
      gotoInput.type = "number";
      gotoInput.min = "1";
      gotoInput.max = String(viewer.matchLen);
      gotoInput.className = "msa-goto";
      gotoInput.placeholder = "col";
      gotoInput.dataset.tip = `Jump to column 1–${viewer.matchLen} (Cmd/Ctrl+L · Enter to jump)`;
      on(gotoInput, "keydown", (ev) => {
        if (ev.key !== "Enter") return;
        const n = parseInt(gotoInput.value, 10);
        if (!isFinite(n) || n < 1 || n > viewer.matchLen) return;
        scrollToColumn(n);
      });

      function scrollToColumn(col) {
        // 1-indexed match column; with column virtualization the target
        // cell is typically NOT in the DOM yet, so we compute the scroll
        // target from colOffsets/cellWidthPx, then flash the cell after
        // the smooth-scroll animation lands and drawList puts it back.
        const matchIdx = col - 1;
        if (matchIdx < 0 || matchIdx >= viewer.matchLen) return;
        if (!colOffsets) return;
        if (cellWidthPx <= 0) measureCellWidth();
        if (cellWidthPx <= 0) return;

        const insertWidthsHere = getInsertWidths();
        const insertW = insertWidthsHere ? (insertWidthsHere[matchIdx] || 0) : 0;
        // Match cell sits inside slot at colOffsets[matchIdx], after the
        // pre-insert. Convert to pixels in scroll-content coordinates.
        const cellLeftCh = colOffsets[matchIdx] + insertW;
        const seqStartPx = state.labelWidth + 8; // 0.5rem padding-left
        const cellLeftPx = seqStartPx + cellLeftCh * cellWidthPx;
        const center = cellLeftPx - table.clientWidth / 2 + cellWidthPx / 2;
        table.scrollTo({ left: Math.max(0, center), behavior: "smooth" });

        // Find and flash the cell after the scroll lands. The drawList
        // tied to the final scroll event will have placed it in the DOM.
        setTimeout(() => {
          // Walk the static query row first (always present when query
          // is pinned); fall back to row 0 of the virtual list.
          const seqEl = (staticQueryBody)
            || (list.firstElementChild && list.firstElementChild.querySelector(".static-msa-seq"));
          if (!seqEl) return;
          const row = seqEl.parentElement;
          const cs = (row && row._colStart) || 0;
          if (matchIdx < cs) return;
          let count = 0;
          let target = null;
          for (const c of seqEl.children) {
            if (c.classList.contains("msa-residue") &&
                !c.classList.contains("msa-insert") &&
                !c.classList.contains("msa-spacer")) {
              if (cs + count === matchIdx) { target = c; break; }
              count++;
            }
          }
          if (!target) return;
          target.classList.add("msa-flash");
          setTimeout(() => target.classList.remove("msa-flash"), 900);
        }, 380);
      }

      const paletteSel = document.createElement("select");
      paletteSel.className = "msa-palette-sel";
      paletteSel.dataset.tip = "Color palette";
      for (const [v, lbl] of [
        ["chemical", "Chemical"],
        ["clustalx", "Clustal X"],
        ["taylor", "Taylor"],
        ["zappo", "Zappo"],
      ]) {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = lbl;
        paletteSel.appendChild(opt);
      }
      paletteSel.value = state.palette;
      on(paletteSel, "change", () => {
        state.palette = paletteSel.value;
        applyPalette(wrapper);
        persist();
      });

      const zoomOut = makeBtn("msa-zoom-btn", "−");
      zoomOut.dataset.tip = "Zoom out (Ctrl/Cmd + scroll)";
      zoomOut.setAttribute("aria-label", "Zoom out");
      const zoomLabel = document.createElement("span");
      zoomLabel.className = "msa-zoom-label";
      zoomLabel.dataset.tip = `Click to reset zoom to ${FONT_DEFAULT}px`;
      on(zoomLabel, "click", () => {
        if (state.fontPx === FONT_DEFAULT) return;
        state.fontPx = FONT_DEFAULT;
        applyFont(wrapper);
        refreshZoomLabel();
        measureCellWidth();
        drawList();
        persist();
      });
      const zoomIn = makeBtn("msa-zoom-btn", "+");
      zoomIn.dataset.tip = "Zoom in (Ctrl/Cmd + scroll)";
      zoomIn.setAttribute("aria-label", "Zoom in");

      let toggleBtn = null;
      if (supportsInserts) {
        toggleBtn = makeBtn("msa-toggle-btn", "Inserts");
        toggleBtn.setAttribute("aria-pressed", String(!!state.showInserts));
        toggleBtn.title = "Toggle insert columns (a2m/a3m)";
        on(toggleBtn, "click", () => {
          state.showInserts = !state.showInserts;
          toggleBtn.setAttribute("aria-pressed", String(state.showInserts));
          persist();
          drawAll();
        });
      }

      // Templates view: shown only when most headers look like a
      // hmmsearch / hhsuite "[subseq from]" template manifest.
      // Switches the body from the residue grid to a structured table
      // (accession / range / length / description). The PDB column
      // links into RCSB the same way as everywhere else.
      const templatesMatch = isTemplatesA3m(viewer);
      let templatesBtn = null;
      if (templatesMatch) {
        templatesBtn = makeBtn("msa-toggle-btn", "Templates");
        templatesBtn.setAttribute("aria-pressed", String(!!state.msaTemplatesMode));
        templatesBtn.dataset.tip = "Switch to templates table view";
        on(templatesBtn, "click", () => {
          state.msaTemplatesMode = !state.msaTemplatesMode;
          templatesBtn.setAttribute("aria-pressed", String(state.msaTemplatesMode));
          persist();
          applyMsaMode();
        });
      } else {
        // Force off when the file isn't templates-shaped, even if
        // saved state has a stale `true` from a previous file.
        state.msaTemplatesMode = false;
      }

      // "First row = query" — when checked, row 0 is the reference for
      // Hamming and BLOSUM62 stats in row tooltips. Uncheck to drop
      // query-relative stats everywhere (e.g. when no row is the query).
      const queryToggleLabel = document.createElement("label");
      queryToggleLabel.className = "msa-checkbox-toggle";
      const queryToggleInput = document.createElement("input");
      queryToggleInput.type = "checkbox";
      queryToggleInput.checked = state.firstRowIsQuery;
      queryToggleLabel.appendChild(queryToggleInput);
      queryToggleLabel.appendChild(document.createTextNode(" first = query"));
      queryToggleLabel.dataset.tip =
        "Treat the first row as query.\n" +
        "When checked: row tooltips on the other rows show Hamming and " +
        "BLOSUM62 distance to the query; the query row's own row tooltip " +
        "skips those (self-comparison).\n" +
        "Uncheck to hide query-relative stats everywhere.";
      on(queryToggleInput, "change", () => {
        state.firstRowIsQuery = queryToggleInput.checked;
        persist();
        currentRowStats = computeRowStats(viewer, query, state.firstRowIsQuery);
        drawAll();
      });

      const fullBtn = makeBtn("msa-fullscreen-btn", "");
      fullBtn.title = "Full screen";
      fullBtn.setAttribute("aria-label", "Full screen");
      setFullscreenIcon(fullBtn, false);

      const statsEl = document.createElement("span");
      statsEl.className = "msa-stats";
      appendStat(statsEl, "seqLen", String(stats.seqLen),
        "Ungapped length of the query sequence (gap chars stripped from match columns).");
      appendStat(statsEl, "depth", String(stats.depth),
        "Number of sequences in the alignment, including the query.");
      appendStat(statsEl, "N_eff", stats.nEff.toFixed(2),
        "exp of mean per-column Shannon entropy (nats), gaps excluded — HHblits/HHsuite definition. " +
        "Range ≈ 1 (single sequence) to ~20 (uniform AA usage). " +
        "No Henikoff position weights applied.");
      appendStat(statsEl, "log₁₀H", stats.log10H == null ? "—" : stats.log10H.toFixed(2),
        `log10 of mean Hamming distance from query to each other sequence over match columns.\n` +
        `mean H = ${stats.meanH.toFixed(2)} (over ${Math.max(0, stats.depth - 1)} hits).`);
      appendStat(statsEl, "log₂H", stats.log2H == null ? "—" : stats.log2H.toFixed(2),
        `log2 of mean Hamming distance from query to each other sequence over match columns.\n` +
        `mean H = ${stats.meanH.toFixed(2)} (over ${Math.max(0, stats.depth - 1)} hits).`);

      controls.appendChild(prevBtn);
      controls.appendChild(pageLabel);
      controls.appendChild(nextBtn);
      controls.appendChild(filterInput);
      controls.appendChild(gotoInput);
      if (toggleBtn) controls.appendChild(toggleBtn);
      if (templatesBtn) controls.appendChild(templatesBtn);
      controls.appendChild(queryToggleLabel);
      controls.appendChild(statsEl);
      controls.appendChild(paletteSel);
      controls.appendChild(zoomOut);
      controls.appendChild(zoomLabel);
      controls.appendChild(zoomIn);
      controls.appendChild(fullBtn);

      const table = document.createElement("div");
      table.className = "static-msa-table";

      const labelResize = document.createElement("div");
      labelResize.className = "msa-label-resize";
      labelResize.dataset.tip =
        "Drag to resize the name column · pull left to hide · double-click to reset";
      attachLabelResize(labelResize, wrapper);

      wrapper.appendChild(controls);
      wrapper.appendChild(table);
      wrapper.appendChild(labelResize);
      container.appendChild(wrapper);

      function refreshZoomLabel() {
        zoomLabel.textContent = `${state.fontPx}px`;
        zoomOut.disabled = state.fontPx <= FONT_MIN;
        zoomIn.disabled = state.fontPx >= FONT_MAX;
      }

      function applyZoom(delta) {
        const next = clamp(state.fontPx + delta, FONT_MIN, FONT_MAX);
        if (next === state.fontPx) return;
        state.fontPx = next;
        applyFont(wrapper);
        refreshZoomLabel();
        // Cell width in px tracks --msa-font-size; remeasure before
        // the next col-range computation. Zoom usually shrinks/expands
        // the visible window enough that liveRowsKey will change anyway.
        measureCellWidth();
        drawList();
        persist();
      }

      on(zoomOut, "click", () => applyZoom(-1));
      on(zoomIn, "click", () => applyZoom(+1));
      on(table, "wheel", (ev) => {
        if (!(ev.ctrlKey || ev.metaKey)) return;
        ev.preventDefault();
        applyZoom(ev.deltaY > 0 ? -1 : +1);
      }, { passive: false });

      const list = document.createElement("div");
      list.className = "static-msa-list";

      // Hidden sentinel for measuring `1ch` in pixels at the current
      // font size. Sized via the same .msa-residue rule the real cells
      // use, so getBoundingClientRect().width stays accurate as zoom
      // changes. Lives on the wrapper so it's measured against the
      // viewer's font, not whatever the host page applies.
      const cellProbe = document.createElement("span");
      cellProbe.className = "msa-residue msa-cell-probe";
      cellProbe.textContent = "M";
      cellProbe.setAttribute("aria-hidden", "true");
      wrapper.appendChild(cellProbe);

      // Column-virt state. colOffsets[i] = left edge of slot i in ch
      // (slot i = optional pre-insert + match cell i; trailing insert
      // sits at colOffsets[matchLen]). cellWidthPx is `1ch` measured
      // from the probe; the two together let us derive the visible
      // match-col window from `table.scrollLeft` + `table.clientWidth`.
      let cellWidthPx = 0;
      let colOffsets = null;
      let currentCs = 0;
      let currentCe = viewer.matchLen;
      // Body element of the sticky query row, kept around so horizontal
      // scrolls can re-render its slice without rebuilding the whole row.
      let staticQueryBody = null;

      function rowHeight() {
        return Math.max(1, Math.ceil(state.fontPx * ROW_LINE_HEIGHT));
      }

      function recomputeColLayout(insertWidths) {
        const m = viewer.matchLen | 0;
        const off = new Int32Array(m + 1);
        let acc = 0;
        for (let i = 0; i < m; i++) {
          off[i] = acc;
          acc += (insertWidths ? (insertWidths[i] || 0) : 0) + 1;
        }
        off[m] = acc;
        colOffsets = off;
      }

      function measureCellWidth() {
        const w = cellProbe.getBoundingClientRect().width;
        if (w > 0) cellWidthPx = w;
      }

      // Compute [cs, ce) — the match-column window to render. Full
      // matchLen when content fits in the viewport (no horizontal
      // scroll); a clipped slice with COL_OVERSCAN otherwise.
      function computeColRange() {
        const m = viewer.matchLen | 0;
        if (m === 0 || !colOffsets) return { cs: 0, ce: m };
        if (cellWidthPx <= 0) measureCellWidth();
        if (cellWidthPx <= 0) return { cs: 0, ce: m };

        const seqStartPx = state.labelWidth + 8; // 0.5rem padding-left
        const viewportPx = table.clientWidth;
        const totalContentPx = colOffsets[m] * cellWidthPx;
        // Whole alignment fits — skip virt entirely.
        if (totalContentPx <= viewportPx) return { cs: 0, ce: m };

        const visibleLeftPx = Math.max(0, table.scrollLeft - seqStartPx);
        const visibleRightPx = visibleLeftPx + viewportPx;
        const visibleLeftCh = visibleLeftPx / cellWidthPx;
        const visibleRightCh = visibleRightPx / cellWidthPx;

        let cs = 0;
        while (cs < m && colOffsets[cs + 1] <= visibleLeftCh) cs++;
        let ce = cs;
        while (ce < m && colOffsets[ce] < visibleRightCh) ce++;
        cs = Math.max(0, cs - COL_OVERSCAN);
        ce = Math.min(m, ce + COL_OVERSCAN);
        return { cs, ce };
      }

      function getFiltered() {
        const source = state.firstRowIsQuery ? hits : viewer.entries;
        const needle = state.filter.toLowerCase();
        return needle
          ? source.filter((h) => (h.name || h.id || "").toLowerCase().includes(needle))
          : source;
      }

      function getInsertWidths() {
        return state.showInserts && supportsInserts
          ? computeInsertWidths(viewer.entries, viewer.matchLen)
          : null;
      }

      function cycleCoverColor() {
        const order = ["off", "entropy", "blosum"];
        const i = order.indexOf(state.msaCoverColor);
        state.msaCoverColor = order[(i + 1) % order.length];
        persist();
        drawAll();
      }

      function drawAll() {
        if (state.msaTemplatesMode) {
          drawTemplatesTable();
          return;
        }
        labelResize.style.display = "";
        table.innerHTML = "";
        staticQueryBody = null;
        const insertWidths = getInsertWidths();
        recomputeColLayout(insertWidths);
        // Ruler + histogram stay full-width — they're rebuilt only on
        // drawAll (filter / inserts / palette / zoom), not on scroll,
        // so the cost is amortized and the visible alignment stays
        // visually consistent under horizontal scroll.
        renderRulerRow(table, viewer.matchLen, insertWidths);
        renderHistogramRow(
          table,
          coverage,
          insertWidths,
          viewer.entries.length,
          wrapper,
          currentColumnInfo,
          state.msaCoverColor,
          cycleCoverColor,
        );
        // Pin the query row at the top only when the user has confirmed
        // row 0 (or whatever's marked isQuery) actually is the query.
        // Otherwise it flows into the virtualized list with the rest.
        if (state.firstRowIsQuery) {
          const queryRow = renderRow(
            table, query, "static-msa-query", insertWidths, currentRowStats,
            0, viewer.matchLen,
          );
          // Resolve a quick reference for slice updates on horizontal scroll.
          if (queryRow) staticQueryBody = queryRow.querySelector(".static-msa-seq");
        }
        table.appendChild(list);
        // First measurement happens after the probe has laid out under
        // the active --msa-font-size, which the wrapper now carries.
        measureCellWidth();
        drawList(insertWidths);

        // Restore the persisted scroll position on first open. Done
        // after drawList so list.style.height + the row content's
        // intrinsic width are set — without that, the browser clamps
        // .scrollTop / .scrollLeft to (0, 0) and the position is lost.
        // The .scrollTo here fires a scroll event → drawList re-runs
        // with the correct column window. One wasted render on first
        // open, but it's the cleanest way to avoid showing wrong
        // content at the saved position.
        if (!initialScrollRestored) {
          initialScrollRestored = true;
          const wantTop = state.scrollTop || 0;
          const wantLeft = state.scrollLeft || 0;
          if (wantTop > 0 || wantLeft > 0) {
            if (table.scrollTop !== wantTop) table.scrollTop = wantTop;
            if (table.scrollLeft !== wantLeft) table.scrollLeft = wantLeft;
          }
        }
      }

      let initialScrollRestored = false;

      function applyMsaMode() {
        // The button click already flipped state; just rebuild.
        drawAll();
      }

      function drawTemplatesTable() {
        // The label-resize handle is for the residue grid only; hide
        // it in templates mode so the table can use the full width.
        labelResize.style.display = "none";
        table.innerHTML = "";

        // HitMap above the table — same per-query coverage plot as
        // the HHR view, derived from each row's first/last non-gap
        // residue in matchSeq. Click a rect → scroll matching row.
        const mapSection = document.createElement("div");
        mapSection.className = "hhr-mapsection msa-templates-mapsection";
        renderTemplatesHitMap(mapSection, viewer);
        table.appendChild(mapSection);

        const tableEl = document.createElement("div");
        tableEl.className = "msa-templates-section";
        renderTemplatesTable(tableEl, viewer);
        table.appendChild(tableEl);

        // Wire HitMap click → scroll matching templates row.
        on(mapSection, "click", (ev) => {
          if (ev.target.closest("a, button")) return;
          const rect = ev.target.closest("[data-num]");
          if (!rect) return;
          const target = tableEl.querySelector(
            `tr[data-num="${CSS.escape(String(rect.dataset.num))}"]`,
          );
          if (!target) return;
          target.scrollIntoView({ behavior: "smooth", block: "center" });
          target.classList.add("hhr-row-flash");
          setTimeout(() => target.classList.remove("hhr-row-flash"), 900);
        });
      }

      // Incremental virtualization. liveRows tracks the DOM elements
      // currently in the list, keyed by their index in `filtered`.
      // Each scroll tick: remove rows that left the viewport, add rows
      // that entered. Rows still in viewport are untouched.
      //
      // liveRowsKey invalidates the entire pool when something changes
      // that affects every row (filter / query toggle / inserts /
      // font size).
      const liveRows = new Map();
      let liveRowsKey = null;

      function drawList(insertWidths) {
        if (insertWidths === undefined) insertWidths = getInsertWidths();
        const filtered = getFiltered();
        const rh = rowHeight();
        list.style.height = `${filtered.length * rh}px`;

        // Column window. When it changes, all live rows must be rebuilt
        // (their cached HTML is keyed by [cs, ce]); fold the range into
        // liveRowsKey so the existing invalidation handles it.
        const range = computeColRange();
        const cs = range.cs;
        const ce = range.ce;
        currentCs = cs;
        currentCe = ce;

        // Static query row (rendered in drawAll) re-slices its sequence
        // body to track the visible window. Cache hits inside renderSequence
        // when [cs, ce] hasn't moved.
        if (staticQueryBody) {
          renderSequence(staticQueryBody, query, insertWidths, cs, ce);
          const sqRow = staticQueryBody.parentElement;
          if (sqRow) sqRow._colStart = cs;
        }

        const newKey = `${state.filter}|${state.firstRowIsQuery}|${state.showInserts}|${state.fontPx}|${cs}|${ce}`;
        if (newKey !== liveRowsKey) {
          list.innerHTML = "";
          liveRows.clear();
          liveRowsKey = newKey;
        }

        const listOffsetTop = list.offsetTop;
        const viewportH = table.clientHeight;
        const visibleTop = Math.max(0, table.scrollTop);
        const visibleHeight = Math.max(0, viewportH - listOffsetTop);

        const startRow = Math.max(0, Math.floor(visibleTop / rh) - VLIST_OVERSCAN);
        const endRow = Math.min(
          filtered.length,
          Math.ceil((visibleTop + visibleHeight) / rh) + VLIST_OVERSCAN,
        );

        // Remove rows that scrolled out of the viewport.
        for (const [idx, el] of liveRows) {
          if (idx < startRow || idx >= endRow) {
            el.remove();
            liveRows.delete(idx);
          }
        }

        // Add rows that just entered the viewport.
        let toAdd = null;
        for (let i = startRow; i < endRow; i++) {
          if (liveRows.has(i)) continue;
          const row = buildVirtualRow(filtered[i], i * rh, insertWidths, cs, ce);
          liveRows.set(i, row);
          if (!toAdd) toAdd = document.createDocumentFragment();
          toAdd.appendChild(row);
        }
        if (toAdd) list.appendChild(toAdd);

        const total = state.firstRowIsQuery ? hits.length : viewer.entries.length;
        const f = filtered.length;
        const startIdx = f === 0 ? 0 : startRow + 1;
        const endIdx = Math.min(endRow, f);
        pageLabel.textContent =
          f === 0
            ? state.filter ? `0 of ${total}` : "query only"
            : state.filter
            ? `${startIdx}–${endIdx} of ${f} (of ${total})`
            : `${startIdx}–${endIdx} of ${total}`;

        prevBtn.disabled = table.scrollTop <= 0;
        nextBtn.disabled = table.scrollTop >= list.offsetTop + list.offsetHeight - viewportH - 1;
      }

      function buildVirtualRow(entry, top, insertWidths, cs, ce) {
        const row = document.createElement("div");
        row.className = "static-msa-row";
        row.style.transform = `translateY(${top}px)`;
        // Recorded so the hover handler can derive a match-col index
        // from sibling position without scanning insertWidths.
        row._colStart = cs;
        const label = document.createElement("span");
        label.className = "static-msa-label";
        const nameText = entry.name || entry.id || "";
        renderLinkedHeader(label, nameText, { entry });
        decorateRowLabel(label, entry, nameText);
        const body = document.createElement("span");
        body.className = "static-msa-seq";
        renderSequence(body, entry, insertWidths, cs, ce);
        row.appendChild(label);
        row.appendChild(body);
        return row;
      }

      function decorateRowLabel(label, entry, nameText) {
        decorateRowLabelImpl(label, entry, nameText, currentRowStats);
      }

      let scrollPending = false;
      // Trailing-edge debounce — keep state.scroll* in sync with the
      // live position on every tick (cheap), but only commit to storage
      // a moment after the user stops scrolling. Reopening the file
      // restores the last position; mid-fling persists are wasted I/O.
      let scrollPersistTimer = 0;
      on(table, "scroll", () => {
        state.scrollLeft = table.scrollLeft;
        state.scrollTop = table.scrollTop;
        if (scrollPersistTimer) clearTimeout(scrollPersistTimer);
        scrollPersistTimer = setTimeout(() => {
          scrollPersistTimer = 0;
          persist();
        }, 250);
        if (scrollPending) return;
        scrollPending = true;
        requestAnimationFrame(() => {
          scrollPending = false;
          drawList();
        });
      }, { passive: true });
      disposers.push(() => {
        if (scrollPersistTimer) clearTimeout(scrollPersistTimer);
      });

      on(prevBtn, "click", () => {
        table.scrollBy({ top: -table.clientHeight + 50, behavior: "smooth" });
      });
      on(nextBtn, "click", () => {
        table.scrollBy({ top: table.clientHeight - 50, behavior: "smooth" });
      });
      on(fullBtn, "click", () => toggleFullscreen(wrapper));
      on(document, "fullscreenchange", () => {
        const isFs = document.fullscreenElement === wrapper;
        fullBtn.title = isFs ? "Exit full screen" : "Full screen";
        fullBtn.setAttribute("aria-label", fullBtn.title);
        setFullscreenIcon(fullBtn, isFs);
        drawList();
      });
      on(window, "resize", () => drawList());

      // Catches the case where this webview was hidden (clientHeight = 0,
      // virtualized list rendered an empty viewport) and is now visible
      // again — neither `scroll` nor `window.resize` fires for that, so
      // without this the alignment renders tiny until the user touches
      // it. Coalesced via rAF.
      let observePending = false;
      const tableObserver = typeof ResizeObserver === "function"
        ? new ResizeObserver(() => {
            if (observePending) return;
            observePending = true;
            requestAnimationFrame(() => {
              observePending = false;
              drawList();
              syncLabelResizeTop();
            });
          })
        : null;
      if (tableObserver) {
        tableObserver.observe(table);
        disposers.push(() => tableObserver.disconnect());
      }

      function syncLabelResizeTop() {
        labelResize.style.top = `${controls.offsetHeight}px`;
      }
      syncLabelResizeTop();
      on(window, "resize", syncLabelResizeTop);

      // Keyboard shortcuts. Bound on document so they fire regardless of
      // focus, but ignored when an input *outside* this instance has focus.
      on(document, "keydown", (ev) => {
        const mod = ev.metaKey || ev.ctrlKey;
        const tag = (document.activeElement && document.activeElement.tagName) || "";

        if (mod && !ev.shiftKey && !ev.altKey && ev.key.toLowerCase() === "f") {
          ev.preventDefault();
          filterInput.focus();
          filterInput.select();
          return;
        }
        if (mod && !ev.shiftKey && !ev.altKey && ev.key.toLowerCase() === "l") {
          ev.preventDefault();
          gotoInput.focus();
          gotoInput.select();
          return;
        }
        if (ev.key === "Escape") {
          if (document.activeElement === filterInput) {
            if (filterInput.value) {
              filterInput.value = "";
              state.filter = "";
              state.scrollTop = 0;
              table.scrollTop = 0;
              persist();
              drawAll();
            } else {
              filterInput.blur();
            }
            ev.preventDefault();
            return;
          }
          if (document.activeElement === gotoInput) {
            gotoInput.blur();
            ev.preventDefault();
            return;
          }
        }
        if (tag === "INPUT" || tag === "TEXTAREA") return;

        // No-modifier nav. Vim-ish g/G stays alongside Home/End so both
        // muscle memories work.
        if (!mod && !ev.shiftKey && !ev.altKey) {
          // Step sizes: arrows scroll by ~10 cells horizontally / 1 row
          // vertically, matching what feels like "one motion" in the
          // residue grid. Cmd/Ctrl-modified arrows below scroll a page.
          const cw = cellWidthPx > 0 ? cellWidthPx : 8;
          const stepX = Math.max(8, Math.round(cw * 10));
          const stepY = rowHeight();
          if (ev.key === "ArrowLeft") {
            ev.preventDefault();
            table.scrollBy({ left: -stepX });
            return;
          }
          if (ev.key === "ArrowRight") {
            ev.preventDefault();
            table.scrollBy({ left: stepX });
            return;
          }
          if (ev.key === "ArrowUp") {
            ev.preventDefault();
            table.scrollBy({ top: -stepY });
            return;
          }
          if (ev.key === "ArrowDown") {
            ev.preventDefault();
            table.scrollBy({ top: stepY });
            return;
          }
          if (ev.key === "PageUp") {
            ev.preventDefault();
            table.scrollBy({ top: -table.clientHeight + stepY, behavior: "smooth" });
            return;
          }
          if (ev.key === "PageDown" || ev.key === " ") {
            ev.preventDefault();
            table.scrollBy({ top: table.clientHeight - stepY, behavior: "smooth" });
            return;
          }
          if (ev.key === "Home") {
            ev.preventDefault();
            table.scrollTo({ left: 0, behavior: "smooth" });
            return;
          }
          if (ev.key === "End") {
            ev.preventDefault();
            table.scrollTo({ left: table.scrollWidth, behavior: "smooth" });
            return;
          }
          if (ev.key === "g") {
            ev.preventDefault();
            table.scrollTo({ top: 0, behavior: "smooth" });
            return;
          }
          if (ev.key === "G") {
            ev.preventDefault();
            table.scrollTo({ top: list.offsetHeight, behavior: "smooth" });
            return;
          }
        }

        // Cmd/Ctrl-modified nav: jump to the four corners + page-width
        // horizontal scroll. Mirrors VS Code-editor conventions enough
        // to be intuitive.
        if (mod && !ev.shiftKey && !ev.altKey) {
          if (ev.key === "Home") {
            ev.preventDefault();
            table.scrollTo({ top: 0, left: 0, behavior: "smooth" });
            return;
          }
          if (ev.key === "End") {
            ev.preventDefault();
            table.scrollTo({
              top: table.scrollHeight, left: table.scrollWidth, behavior: "smooth",
            });
            return;
          }
          if (ev.key === "ArrowLeft") {
            ev.preventDefault();
            table.scrollBy({ left: -table.clientWidth, behavior: "smooth" });
            return;
          }
          if (ev.key === "ArrowRight") {
            ev.preventDefault();
            table.scrollBy({ left: table.clientWidth, behavior: "smooth" });
            return;
          }
        }
      });

      refreshZoomLabel();
      drawAll();
    }

    function renderPairwiseInstance(hhr) {
      const wrapper = document.createElement("div");
      wrapper.className = "static-msa-viewer hhr-viewer";
      applyFont(wrapper);
      applyPalette(wrapper);
      applyHhrColor(wrapper);

      const controls = document.createElement("div");
      controls.className = "static-msa-controls";

      const summary = document.createElement("span");
      summary.className = "msa-stats";
      appendStat(summary, "query", hhr.header.query || "—",
        "Query name from the .hhr Query line.");
      appendStat(summary, "matchCols", String(hhr.header.matchColumns || 0),
        "Match_columns from the .hhr header — length of the query HMM.");
      appendStat(summary, "hits", String(hhr.alignments.length),
        "Number of per-hit alignment blocks parsed from the .hhr file.");
      appendStat(summary, "Neff", (hhr.header.neff || 0).toFixed(2),
        "Neff reported by HHsuite for the query MSA. Computed by the profile builder with Henikoff weights — generally higher than our viewer's MSA-side N_eff.");

      // Color-on toggle. Default OFF for pairwise (Toolkit convention —
      // agree-string + consensus rows already convey conservation).
      const colorBtn = makeBtn("msa-toggle-btn", "Color");
      colorBtn.setAttribute("aria-pressed", String(state.hhrColorOn));
      colorBtn.dataset.tip = "Toggle per-residue coloring";
      on(colorBtn, "click", () => {
        state.hhrColorOn = !state.hhrColorOn;
        colorBtn.setAttribute("aria-pressed", String(state.hhrColorOn));
        applyHhrColor(wrapper);
        persist();
      });

      // breakAfter input — wraps each alignment block at this many cols.
      // 0 (or empty) means "auto fit to window width", recomputed on
      // every resize. A typed number overrides auto with a fixed wrap.
      const breakInput = document.createElement("input");
      breakInput.type = "number";
      breakInput.min = "0";
      breakInput.max = String(BREAK_AFTER_MAX);
      breakInput.className = "msa-goto";
      breakInput.value = state.breakAfter === BREAK_AFTER_AUTO ? "" : String(state.breakAfter);
      breakInput.placeholder = "auto";
      breakInput.dataset.tip = `Wrap alignments at N columns (empty / 0 = auto-fit to window)`;
      on(breakInput, "change", () => {
        const raw = breakInput.value.trim();
        let n;
        if (!raw || raw === "0") {
          n = BREAK_AFTER_AUTO;
          breakInput.value = "";
        } else {
          n = clamp(parseInt(raw, 10) || BREAK_AFTER_MIN, BREAK_AFTER_MIN, BREAK_AFTER_MAX);
          breakInput.value = String(n);
        }
        if (n === state.breakAfter) return;
        state.breakAfter = n;
        persist();
        rerenderAlignments();
      });
      const breakLabel = document.createElement("span");
      breakLabel.className = "msa-break-label";
      breakLabel.textContent = "wrap";
      breakLabel.dataset.tip = "Columns per alignment chunk (0 = auto-fit to window)";

      const paletteSel = document.createElement("select");
      paletteSel.className = "msa-palette-sel";
      paletteSel.dataset.tip = "Color palette";
      for (const [v, lbl] of [
        ["chemical", "Chemical"],
        ["clustalx", "Clustal X"],
        ["taylor", "Taylor"],
        ["zappo", "Zappo"],
      ]) {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = lbl;
        paletteSel.appendChild(opt);
      }
      paletteSel.value = state.palette;
      on(paletteSel, "change", () => {
        state.palette = paletteSel.value;
        applyPalette(wrapper);
        persist();
      });

      const zoomOut = makeBtn("msa-zoom-btn", "−");
      zoomOut.dataset.tip = "Zoom out (Ctrl/Cmd + scroll)";
      const zoomLabel = document.createElement("span");
      zoomLabel.className = "msa-zoom-label";
      zoomLabel.dataset.tip = `Click to reset zoom to ${FONT_DEFAULT}px`;
      on(zoomLabel, "click", () => {
        if (state.fontPx === FONT_DEFAULT) return;
        state.fontPx = FONT_DEFAULT;
        applyFont(wrapper);
        refreshZoomLabel();
        persist();
      });
      const zoomIn = makeBtn("msa-zoom-btn", "+");
      zoomIn.dataset.tip = "Zoom in (Ctrl/Cmd + scroll)";

      const fullBtn = makeBtn("msa-fullscreen-btn", "");
      fullBtn.title = "Full screen";
      setFullscreenIcon(fullBtn, false);

      controls.appendChild(summary);
      controls.appendChild(colorBtn);
      controls.appendChild(breakLabel);
      controls.appendChild(breakInput);
      controls.appendChild(paletteSel);
      controls.appendChild(zoomOut);
      controls.appendChild(zoomLabel);
      controls.appendChild(zoomIn);
      controls.appendChild(fullBtn);

      // Body: HitMap on top, then hit table, then alignments. Click in
      // either the HitMap or the hit table scrolls to the alignment.
      const body = document.createElement("div");
      body.className = "hhr-body";

      const mapSection = document.createElement("div");
      mapSection.className = "hhr-mapsection";
      renderHitMap(mapSection, hhr);
      body.appendChild(mapSection);

      const hitsSection = document.createElement("div");
      hitsSection.className = "hhr-hits";
      body.appendChild(hitsSection);
      renderHitTable(hitsSection, hhr);

      const alignmentsSection = document.createElement("div");
      alignmentsSection.className = "hhr-alignments";
      body.appendChild(alignmentsSection);

      wrapper.appendChild(controls);
      wrapper.appendChild(body);
      container.appendChild(wrapper);

      // Initial render uses the now-laid-out body width to compute
      // auto-fit. Subsequent resizes re-render only when the computed
      // breakAfter actually changes (avoids needless DOM churn on
      // sub-column resizes).
      let lastBreak = -1;
      function rerenderIfBreakChanged() {
        const next = computeBreakAfter(body);
        if (next === lastBreak) return;
        lastBreak = next;
        alignmentsSection.innerHTML = "";
        renderAlignments(alignmentsSection, hhr, next);
      }
      rerenderIfBreakChanged();

      function rerenderAlignments() {
        lastBreak = computeBreakAfter(body);
        alignmentsSection.innerHTML = "";
        renderAlignments(alignmentsSection, hhr, lastBreak);
      }
      function rerenderHits() {
        hitsSection.innerHTML = "";
        renderHitTable(hitsSection, hhr);
      }

      function refreshZoomLabel() {
        zoomLabel.textContent = `${state.fontPx}px`;
        zoomOut.disabled = state.fontPx <= FONT_MIN;
        zoomIn.disabled = state.fontPx >= FONT_MAX;
      }
      function applyZoom(delta) {
        const next = clamp(state.fontPx + delta, FONT_MIN, FONT_MAX);
        if (next === state.fontPx) return;
        state.fontPx = next;
        applyFont(wrapper);
        refreshZoomLabel();
        persist();
        // Auto-fit depends on per-char width, which scales with zoom.
        rerenderIfBreakChanged();
      }
      on(zoomOut, "click", () => applyZoom(-1));
      on(zoomIn, "click", () => applyZoom(+1));
      on(body, "wheel", (ev) => {
        if (!(ev.ctrlKey || ev.metaKey)) return;
        ev.preventDefault();
        applyZoom(ev.deltaY > 0 ? -1 : +1);
      }, { passive: false });

      on(fullBtn, "click", () => toggleFullscreen(wrapper));
      on(document, "fullscreenchange", () => {
        const isFs = document.fullscreenElement === wrapper;
        fullBtn.title = isFs ? "Exit full screen" : "Full screen";
        setFullscreenIcon(fullBtn, isFs);
        rerenderIfBreakChanged();
      });

      // Auto-wrap on window resize. Debounced so a slow drag doesn't
      // burn cycles, and we only re-render if the computed breakAfter
      // actually changes (sub-column resizes are no-ops).
      let resizeTimer = null;
      on(window, "resize", () => {
        if (resizeTimer) clearTimeout(resizeTimer);
        resizeTimer = setTimeout(rerenderIfBreakChanged, 80);
      });

      // Click a row in the hit table OR a rect in the HitMap → scroll
      // that alignment block into view. Click "No N." inside an
      // alignment block → scroll the matching row in the hit table.
      function scrollToBlock(num) {
        const target = wrapper.querySelector(
          `.hhr-block[data-num="${CSS.escape(String(num))}"]`,
        );
        if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
      function scrollToHitRow(num) {
        const target = wrapper.querySelector(
          `.hhr-hit-table tr[data-num="${CSS.escape(String(num))}"]`,
        );
        if (!target) return;
        target.scrollIntoView({ behavior: "smooth", block: "center" });
        target.classList.add("hhr-row-flash");
        setTimeout(() => target.classList.remove("hhr-row-flash"), 900);
      }
      on(hitsSection, "click", (ev) => {
        // Don't hijack clicks on inline links/buttons — those have
        // their own action (open in new tab, etc.) and should not
        // also scroll the alignment list.
        if (ev.target.closest("a, button")) return;
        // Header click → sort.
        const th = ev.target.closest("th[data-sortkey]");
        if (th) {
          const k = th.dataset.sortkey;
          if (state.hhrSortKey === k) {
            state.hhrSortDir = state.hhrSortDir === "asc" ? "desc" : "asc";
          } else {
            state.hhrSortKey = k;
            state.hhrSortDir = "asc";
          }
          persist();
          rerenderHits();
          return;
        }
        const tr = ev.target.closest("tr[data-num]");
        if (tr) scrollToBlock(tr.dataset.num);
      });
      on(mapSection, "click", (ev) => {
        if (ev.target.closest("a, button")) return;
        const rect = ev.target.closest("[data-num]");
        if (rect) scrollToBlock(rect.dataset.num);
      });
      on(alignmentsSection, "click", (ev) => {
        if (ev.target.closest("a, button")) return;
        const num = ev.target.closest(".hhr-block-num[data-num]");
        if (num) scrollToHitRow(num.dataset.num);
      });

      refreshZoomLabel();
    }

    function applyHhrColor(wrapper) {
      wrapper.dataset.colorOff = state.hhrColorOn ? "false" : "true";
    }

    /**
     * Heuristic: an a3m / a2m / fas where a majority of the first 50
     * headers carry the `[subseq from]` marker that hmmsearch / hhsuite
     * use for template manifests. False for a regular MSA.
     */
    function isTemplatesA3m(viewer) {
      if (!viewer || !Array.isArray(viewer.entries)) return false;
      if (viewer.format !== "a3m" && viewer.format !== "a2m" && viewer.format !== "fas") {
        return false;
      }
      const sample = viewer.entries.slice(0, 50);
      if (sample.length === 0) return false;
      let matches = 0;
      for (const e of sample) {
        if (/\[subseq from\]/i.test(e.name || "")) matches++;
      }
      return matches >= Math.max(2, Math.ceil(sample.length * 0.5));
    }

    /**
     * Parse a templates-style header into structured fields:
     *   "7sch_A/55-703 [subseq from] mol:protein length:720  Exostosin-1"
     *     → { accession: "7sch_A", start: 55, end: 703,
     *         length: 720, description: "Exostosin-1" }
     */
    function parseTemplateName(name) {
      const text = String(name || "");
      const m = /^(\S+?)(?:\/(\d+)-(\d+))?\s+\[subseq from\](?:\s+mol:\S+)?(?:\s+length:(\d+))?\s*(.*)$/i.exec(text);
      if (!m) return null;
      return {
        accession: m[1],
        start: m[2] ? parseInt(m[2], 10) : undefined,
        end: m[3] ? parseInt(m[3], 10) : undefined,
        length: m[4] ? parseInt(m[4], 10) : undefined,
        description: (m[5] || "").trim(),
      };
    }

    /**
     * Per-query coverage HitMap for templates view. Each row's
     * coverage is the [first, last] non-gap residue index of its
     * `matchSeq`. Templates with no aligned residues are skipped.
     */
    function renderTemplatesHitMap(parent, viewer) {
      const matchCols = viewer.matchLen || 0;
      if (!matchCols || !Array.isArray(viewer.entries)) return;
      const items = [];
      for (let i = 0; i < viewer.entries.length; i++) {
        const e = viewer.entries[i];
        const seq = e.matchSeq || "";
        let first = -1;
        let last = -1;
        const lim = Math.min(seq.length, matchCols);
        for (let c = 0; c < lim; c++) {
          const ch = seq.charCodeAt(c);
          // 45 = '-', 46 = '.'
          if (ch !== 45 && ch !== 46) {
            if (first === -1) first = c;
            last = c;
          }
        }
        if (first === -1) continue;
        const num = i + 1;
        const t = parseTemplateName(e.name || "") || null;
        const acc = (t && t.accession) || (e.name || "").split(/\s+/)[0] || `entry ${num}`;
        const desc = (t && t.description) || e.name || "";
        items.push({
          num,
          start: first + 1,
          end: last + 1,
          label: desc ? `${acc} ${desc}` : acc,
          tip:
            `${num}. ${acc}\n` +
            (desc ? `${desc}\n` : "") +
            `Q ${first + 1}–${last + 1}` +
            (t && t.length != null ? ` · template length ${t.length}` : ""),
        });
      }
      renderHitmapSvg(parent, { matchCols, items });
    }

    function renderTemplatesTable(parent, viewer) {
      const tbl = document.createElement("table");
      tbl.className = "msa-templates-table";

      const thead = document.createElement("thead");
      thead.innerHTML =
        "<tr>" +
        "<th class='r'>#</th>" +
        "<th>Accession</th>" +
        "<th class='r'>Range</th>" +
        "<th class='r'>Length</th>" +
        "<th>Description</th>" +
        "</tr>";
      tbl.appendChild(thead);

      const tbody = document.createElement("tbody");
      for (let i = 0; i < viewer.entries.length; i++) {
        const e = viewer.entries[i];
        const t = parseTemplateName(e.name || e.id || "") || {
          accession: e.id || e.name || "",
          description: e.name || "",
        };
        const tr = document.createElement("tr");
        // 1-based num, matches the HitMap rect's data-num so clicks
        // can find this row from either side.
        tr.dataset.num = String(i + 1);

        const numTd = document.createElement("td");
        numTd.className = "r";
        numTd.textContent = String(i + 1);
        tr.appendChild(numTd);

        const accTd = document.createElement("td");
        renderLinkedHeader(accTd, t.accession || "");
        tr.appendChild(accTd);

        const rangeTd = document.createElement("td");
        rangeTd.className = "r";
        rangeTd.textContent =
          t.start != null && t.end != null ? `${t.start}–${t.end}` : "";
        tr.appendChild(rangeTd);

        const lenTd = document.createElement("td");
        lenTd.className = "r";
        lenTd.textContent = t.length != null ? String(t.length) : "";
        tr.appendChild(lenTd);

        const descTd = document.createElement("td");
        descTd.textContent = t.description || "";
        tr.appendChild(descTd);

        tbody.appendChild(tr);
      }
      tbl.appendChild(tbody);
      parent.appendChild(tbl);
    }

    function renderAlignments(parent, hhr, breakAfter) {
      const eff = breakAfter && breakAfter > 0 ? breakAfter : 80;
      for (const al of hhr.alignments) {
        parent.appendChild(renderAlignmentBlock(al, eff));
      }
    }

    /**
     * Decide how many columns each alignment chunk should hold. If
     * the user pinned a value via the input, use that; otherwise fit
     * to the body's clientWidth based on the current zoom font-size.
     *
     * The math: subtract the static columns of `.hhr-line` (label,
     * start, end, three gaps) plus body padding from the available
     * width. What remains is the seq column. Divide by ~0.6 × font-size
     * (a good monospace estimate; Menlo/Monaco are ~0.55, Consolas ~0.6).
     */
    function computeBreakAfter(bodyEl) {
      if (state.breakAfter && state.breakAfter > 0) return state.breakAfter;
      const root =
        parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
      const labelW = 7 * root;       // .hhr-line grid col 1
      const startW = 2.5 * root;     // .hhr-line grid col 2
      const endW = 2.5 * root;       // .hhr-line grid col 4
      const gaps = 3 * 0.4 * root;   // .hhr-line gap × 3
      const bodyPad = 0.75 * root * 2; // .hhr-body padding-left + right
      const overhead = labelW + startW + endW + gaps + bodyPad + 24; // 24 ≈ scrollbar / safety
      const containerW = (bodyEl && bodyEl.clientWidth) || window.innerWidth;
      const seqW = Math.max(0, containerW - overhead);
      const charW = state.fontPx * 0.6;
      const fit = Math.floor(seqW / Math.max(1, charW));
      // Auto-fit: cap on the low end only. The viewport width is the
      // real limit — clamping to BREAK_AFTER_MAX (a manual-input cap)
      // would leave whitespace on wide windows.
      return Math.max(BREAK_AFTER_MIN, fit);
    }

    /**
     * Sortable hit table. The `Hit` column is text (description); the
     * rest are numeric — we coerce sort keys accordingly. Clicking a
     * header toggles ascending → descending → ascending. The active
     * column gets an indicator (▲ ▼).
     */
    function renderHitTable(parent, hhr) {
      const sorted = sortAlignments(hhr.alignments, hhr.hits, state.hhrSortKey, state.hhrSortDir);

      const table = document.createElement("table");
      table.className = "hhr-hit-table";
      const thead = document.createElement("thead");
      const tr = document.createElement("tr");
      const cols = [
        { key: "num", label: "No", cls: "r" },
        { key: "desc", label: "Hit", cls: "" },
        { key: "prob", label: "Prob", cls: "r" },
        { key: "evalue", label: "E-value", cls: "r" },
        { key: "score", label: "Score", cls: "r" },
        { key: "cols", label: "Cols", cls: "r" },
        { key: "qStart", label: "Q range", cls: "r" },
        { key: "tStart", label: "T range", cls: "r" },
      ];
      for (const c of cols) {
        const th = document.createElement("th");
        th.className = c.cls;
        th.dataset.sortkey = c.key;
        const ind = state.hhrSortKey === c.key
          ? (state.hhrSortDir === "asc" ? " ▲" : " ▼")
          : "";
        th.textContent = c.label + ind;
        if (state.hhrSortKey === c.key) th.classList.add("active");
        tr.appendChild(th);
      }
      thead.appendChild(tr);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      for (const al of sorted) {
        const summary = (hhr.hits && hhr.hits.find((h) => h.num === al.num)) || null;
        const row = document.createElement("tr");
        row.dataset.num = String(al.num);
        const desc = (al.description || (summary && summary.hit) || "").slice(0, 80);
        const qRange = `${al.query.start ?? "?"}–${al.query.end ?? "?"}`;
        const tRange = al.template.ref
          ? `${al.template.start ?? "?"}–${al.template.end ?? "?"} (${al.template.ref})`
          : `${al.template.start ?? "?"}–${al.template.end ?? "?"}`;
        const numTd = document.createElement("td");
        numTd.className = "r";
        numTd.textContent = String(al.num);
        const descTd = document.createElement("td");
        renderLinkedHeader(descTd, desc);
        row.appendChild(numTd);
        row.appendChild(descTd);
        const cells = [
          [al.metrics.probab.toFixed(1), "r"],
          [formatExp(al.metrics.evalue), "r"],
          [al.metrics.score.toFixed(1), "r"],
          [String(al.metrics.alignedCols), "r"],
          [qRange, "r"],
          [tRange, "r"],
        ];
        for (const [text, cls] of cells) {
          const td = document.createElement("td");
          if (cls) td.className = cls;
          td.textContent = text;
          row.appendChild(td);
        }
        tbody.appendChild(row);
      }
      table.appendChild(tbody);
      parent.appendChild(table);
    }

    function sortAlignments(alignments, hits, key, dir) {
      const out = alignments.slice();
      const sign = dir === "desc" ? -1 : 1;
      const accessor = sortAccessor(key);
      out.sort((a, b) => {
        const av = accessor(a);
        const bv = accessor(b);
        if (typeof av === "string" || typeof bv === "string") {
          return sign * String(av).localeCompare(String(bv));
        }
        return sign * (av - bv);
      });
      return out;
    }

    function sortAccessor(key) {
      switch (key) {
        case "num":    return (a) => a.num;
        case "desc":   return (a) => (a.description || a.templateName || "").toLowerCase();
        case "prob":   return (a) => a.metrics.probab;
        case "evalue": return (a) => a.metrics.evalue;
        case "score":  return (a) => a.metrics.score;
        case "cols":   return (a) => a.metrics.alignedCols;
        case "qStart": return (a) => a.query.start ?? 0;
        case "tStart": return (a) => a.template.start ?? 0;
        default:       return (a) => a.num;
      }
    }

    /**
     * HitMap — SVG visualization of every hit's coverage along the
     * query. Rows are packed greedily so non-overlapping hits share a
     * row; overlapping hits push down. Each hit's color encodes its
     * probability (red → muted teal as prob falls).
     */
    function renderHitMap(parent, hhr) {
      const matchCols = hhr.header.matchColumns || 0;
      if (!matchCols || hhr.alignments.length === 0) return;
      renderHitmapSvg(parent, {
        matchCols,
        items: hhr.alignments.map((al) => ({
          num: al.num,
          start: al.query.start ?? 1,
          end: al.query.end ?? matchCols,
          color: probColor(al.metrics.probab),
          label: (al.description && al.description.trim()) || al.templateName || "",
          tip:
            `No ${al.num}: ${(al.description || al.templateName || "").slice(0, 60)}\n` +
            `Prob ${al.metrics.probab.toFixed(1)} · E-value ${formatExp(al.metrics.evalue)} · Score ${al.metrics.score.toFixed(1)}\n` +
            `Q ${al.query.start ?? "?"}–${al.query.end ?? "?"}`,
        })),
      });
    }

    /**
     * Generic per-query coverage HitMap. Items are { num, start, end,
     * color?, label?, tip? }. start/end are 1-based query coords.
     * Greedy row-packing; per-rect clipPath so labels never spill.
     * Click on a rect bubbles a click event with `data-num` set; the
     * caller wires its own scroll-to-target handler.
     */
    function renderHitmapSvg(parent, opts) {
      const SVG_NS = "http://www.w3.org/2000/svg";
      const matchCols = opts.matchCols || 0;
      const items = opts.items || [];
      if (!matchCols || items.length === 0) return;

      // Pack rows: greedy first-fit. Sort by start; place each item in
      // the lowest row whose previous end is strictly below its start.
      const sorted = items.slice().sort((a, b) => (a.start ?? 0) - (b.start ?? 0));
      const rowEnds = [];
      const placement = new Map();
      for (const it of sorted) {
        const s = it.start ?? 1;
        const e = it.end ?? matchCols;
        let placed = -1;
        for (let r = 0; r < rowEnds.length; r++) {
          if (rowEnds[r] < s) { placed = r; break; }
        }
        if (placed === -1) { placed = rowEnds.length; rowEnds.push(0); }
        rowEnds[placed] = e;
        placement.set(it.num, placed);
      }

      const rowH = 8;
      const rowGap = 2;
      const margin = { top: 12, right: 8, bottom: 18, left: 8 };
      const width = 800;
      const usable = width - margin.left - margin.right;
      const trackH = rowEnds.length * (rowH + rowGap);
      const height = margin.top + trackH + margin.bottom;

      const svg = document.createElementNS(SVG_NS, "svg");
      svg.setAttribute("class", "hhr-hitmap");
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      svg.setAttribute("preserveAspectRatio", "none");
      svg.setAttribute("role", "img");
      svg.setAttribute("aria-label",
        `Hit coverage across query of ${matchCols} match columns`);

      const axis = document.createElementNS(SVG_NS, "line");
      axis.setAttribute("x1", String(margin.left));
      axis.setAttribute("x2", String(margin.left + usable));
      axis.setAttribute("y1", String(margin.top - 2));
      axis.setAttribute("y2", String(margin.top - 2));
      axis.setAttribute("class", "hhr-hitmap-axis");
      svg.appendChild(axis);

      const tickStep = matchCols >= 200 ? 50 : matchCols >= 80 ? 25 : 10;
      for (let c = 0; c <= matchCols; c += tickStep) {
        const x = margin.left + (c / matchCols) * usable;
        const tick = document.createElementNS(SVG_NS, "line");
        tick.setAttribute("x1", String(x));
        tick.setAttribute("x2", String(x));
        tick.setAttribute("y1", String(margin.top - 5));
        tick.setAttribute("y2", String(margin.top - 1));
        tick.setAttribute("class", "hhr-hitmap-tick");
        svg.appendChild(tick);
        const txt = document.createElementNS(SVG_NS, "text");
        txt.setAttribute("x", String(x));
        txt.setAttribute("y", String(margin.top - 7));
        txt.setAttribute("class", "hhr-hitmap-ticklabel");
        txt.setAttribute("text-anchor", "middle");
        txt.textContent = String(c || 1);
        svg.appendChild(txt);
      }

      const LABEL_FONT_PX = 7;
      const charWidthEst = LABEL_FONT_PX * 0.55;
      const clipPrefix = `hm-clip-${Math.floor(Math.random() * 1e6).toString(36)}-`;
      const fallbackColor =
        getComputedStyle(document.documentElement)
          .getPropertyValue("--bio-accent")
          .trim() || "#1f66d1";

      for (const it of items) {
        const s = it.start ?? 1;
        const e = it.end ?? matchCols;
        const rowIdx = placement.get(it.num) ?? 0;
        const x = margin.left + ((s - 1) / matchCols) * usable;
        const w = Math.max(1, ((e - s + 1) / matchCols) * usable);
        const y = margin.top + rowIdx * (rowH + rowGap);

        const rect = document.createElementNS(SVG_NS, "rect");
        rect.setAttribute("x", String(x));
        rect.setAttribute("y", String(y));
        rect.setAttribute("width", String(w));
        rect.setAttribute("height", String(rowH));
        rect.setAttribute("rx", "1.5");
        rect.setAttribute("fill", it.color || fallbackColor);
        rect.setAttribute("class", "hhr-hitmap-rect");
        rect.dataset.num = String(it.num);
        if (it.tip) rect.dataset.tip = it.tip;
        svg.appendChild(rect);

        const labelSrc = it.label || "";
        if (labelSrc && w >= charWidthEst * 4 + 4) {
          const clipId = clipPrefix + it.num;
          const clip = document.createElementNS(SVG_NS, "clipPath");
          clip.setAttribute("id", clipId);
          const cr = document.createElementNS(SVG_NS, "rect");
          cr.setAttribute("x", String(x));
          cr.setAttribute("y", String(y));
          cr.setAttribute("width", String(w));
          cr.setAttribute("height", String(rowH));
          clip.appendChild(cr);
          svg.appendChild(clip);

          const txt = document.createElementNS(SVG_NS, "text");
          txt.setAttribute("x", String(x + 3));
          txt.setAttribute("y", String(y + rowH / 2 + 2.5));
          txt.setAttribute("class", "hhr-hitmap-rectlabel");
          txt.setAttribute("clip-path", `url(#${clipId})`);
          txt.textContent = labelSrc.slice(0, 200);
          svg.appendChild(txt);
        }
      }

      parent.appendChild(svg);
    }

    /** Map probability (0..100) to a color: red for high-confidence
     *  hits, fading through orange to muted teal for marginal ones. */
    function probColor(prob) {
      const p = Math.max(0, Math.min(100, prob)) / 100;
      if (p > 0.9) return "#e43d30";
      if (p > 0.7) return "#e8771f";
      if (p > 0.5) return "#d29b00";
      if (p > 0.3) return "#2da44e";
      return "#1f66d1";
    }

    function renderAlignmentBlock(al, breakAfter) {
      const block = document.createElement("div");
      block.className = "hhr-block";
      block.dataset.num = String(al.num);

      const header = document.createElement("div");
      header.className = "hhr-block-header";
      const numSpan = document.createElement("span");
      numSpan.className = "hhr-block-num";
      numSpan.textContent = `No ${al.num}.`;
      // Click → scroll matching row in the hit table into view (the
      // inverse of the existing row-click → block-scroll). Wired by
      // event delegation on alignmentsSection in the parent renderer.
      numSpan.dataset.num = String(al.num);
      numSpan.dataset.tip = `Jump to hit ${al.num} in the table`;
      const descSpan = document.createElement("span");
      descSpan.className = "hhr-block-desc";
      renderLinkedHeader(descSpan, al.description || al.templateName);
      header.appendChild(numSpan);
      header.appendChild(document.createTextNode(" "));
      header.appendChild(descSpan);
      block.appendChild(header);

      const meta = document.createElement("div");
      meta.className = "hhr-block-meta";
      const m = al.metrics;
      const pieces = [
        `Probab=${m.probab.toFixed(2)}`,
        `E-value=${formatExp(m.evalue)}`,
        `Score=${m.score.toFixed(2)}`,
        `Aligned_cols=${m.alignedCols}`,
      ];
      if (m.identities != null) pieces.push(`Identities=${m.identities}%`);
      if (m.similarity != null) pieces.push(`Similarity=${m.similarity}`);
      if (m.templateNeff != null) pieces.push(`Template_Neff=${m.templateNeff}`);
      meta.textContent = pieces.join("  ");
      block.appendChild(meta);

      // Build per-row "lines" from the available tracks. Order mirrors
      // Toolkit's HHpred tab — the most informative tracks go closest
      // to the query/template seqs.
      const lines = [];
      if (al.querySsPred?.seq) lines.push({ label: "Q ss_pred", track: al.querySsPred, kind: "ss" });
      lines.push({ label: `Q ${al.queryName}`, track: al.query, kind: "seq" });
      if (al.queryConsensus?.seq) lines.push({ label: "Q Consensus", track: al.queryConsensus, kind: "consensus" });
      if (al.agree) lines.push({ label: "", track: { seq: al.agree }, kind: "agree" });
      if (al.templateConsensus?.seq) lines.push({ label: "T Consensus", track: al.templateConsensus, kind: "consensus" });
      lines.push({ label: `T ${al.templateName}`, track: al.template, kind: "seq" });
      if (al.templateSsDssp?.seq) lines.push({ label: "T ss_dssp", track: al.templateSsDssp, kind: "ss" });
      if (al.templateSsPred?.seq) lines.push({ label: "T ss_pred", track: al.templateSsPred, kind: "ss" });
      if (al.confidence) lines.push({ label: "Confidence", track: { seq: al.confidence }, kind: "confidence" });
      if (al.pp) lines.push({ label: "PP", track: { seq: al.pp }, kind: "pp" });

      // Wrap by breakAfter columns. A "chunk" is a block of breakAfter
      // columns of every track stacked vertically. Tracks of unequal
      // length pad with spaces so columns stay aligned within a chunk.
      const totalCols = lines.reduce((m, l) => Math.max(m, l.track.seq.length), 0);

      // Pre-compute residue counts per track for the start/end labels.
      // For each "seq" track, the start increments by the number of
      // non-gap residues consumed in prior chunks.
      const seqResidueCounts = lines.map((l) =>
        l.kind === "seq" || l.kind === "consensus"
          ? new Int32Array(Math.ceil(totalCols / breakAfter) + 1)
          : null,
      );
      for (let li = 0; li < lines.length; li++) {
        const line = lines[li];
        const counts = seqResidueCounts[li];
        if (!counts) continue;
        let cursor = 0;
        let chunkIdx = 0;
        const seq = line.track.seq;
        for (let c = 0; c < seq.length; c++) {
          if (c > 0 && c % breakAfter === 0) {
            counts[++chunkIdx] = cursor;
          }
          const ch = seq[c];
          if (ch !== "-" && ch !== ".") cursor++;
        }
        // Final cursor for the last chunk's end-position lookup.
        counts[chunkIdx + 1] = cursor;
      }

      for (let off = 0; off < totalCols; off += breakAfter) {
        const chunk = document.createElement("div");
        chunk.className = "hhr-chunk";
        const chunkIdx = Math.floor(off / breakAfter);

        for (let li = 0; li < lines.length; li++) {
          const line = lines[li];
          const seq = line.track.seq;
          const slice = seq.slice(off, off + breakAfter);
          if (line.kind !== "seq" && !slice) continue;

          const row = document.createElement("div");
          row.className = `hhr-line hhr-line-${line.kind}`;

          const labelEl = document.createElement("span");
          labelEl.className = "hhr-label";
          labelEl.textContent = line.label;
          row.appendChild(labelEl);

          const startEl = document.createElement("span");
          startEl.className = "hhr-pos hhr-start";
          if (line.kind === "seq" || line.kind === "consensus") {
            const baseStart = line.track.start;
            const counts = seqResidueCounts[li];
            startEl.textContent = baseStart != null && counts
              ? String(baseStart + counts[chunkIdx])
              : "";
          } else {
            startEl.textContent = "";
          }
          row.appendChild(startEl);

          const seqEl = document.createElement("span");
          seqEl.className = "hhr-seq";
          renderHhrSeqInline(seqEl, slice, line.kind);
          row.appendChild(seqEl);

          const endEl = document.createElement("span");
          endEl.className = "hhr-pos hhr-end";
          if (line.kind === "seq" || line.kind === "consensus") {
            const baseStart = line.track.start;
            const counts = seqResidueCounts[li];
            const endResidueIdx =
              baseStart != null && counts
                ? baseStart + counts[chunkIdx + 1] - 1
                : null;
            endEl.textContent =
              endResidueIdx != null && endResidueIdx >= 0
                ? String(endResidueIdx)
                : "";
          } else {
            endEl.textContent = "";
          }
          row.appendChild(endEl);

          chunk.appendChild(row);
        }
        block.appendChild(chunk);
      }

      return block;
    }

    function renderHhrSeqInline(container, slice, kind) {
      // ss tracks: each char = h/H/e/E/c/C → distinct color via msa-ss-* class.
      // agree / confidence / pp: plain text in a single span.
      // seq / consensus: per-residue spans (palette, hover, etc. apply).
      let out = "";
      if (kind === "seq" || kind === "consensus") {
        for (let i = 0; i < slice.length; i++) {
          const ch = slice[i];
          const cls = residueClass(ch);
          const upper = ch >= "a" && ch <= "z" ? String.fromCharCode(ch.charCodeAt(0) - 32) : ch;
          const aa = (upper >= "A" && upper <= "Z") ? ` data-aa="${upper}"` : "";
          out += `<span class="msa-residue msa-aa-${cls}"${aa}>${htmlEscapeChar(ch)}</span>`;
        }
      } else if (kind === "ss") {
        for (let i = 0; i < slice.length; i++) {
          const ch = slice[i];
          out += `<span class="msa-residue msa-ss-${ssClass(ch)}">${htmlEscapeChar(ch)}</span>`;
        }
      } else {
        // agree / confidence / pp — plain monospace block
        out = `<span class="hhr-plain hhr-plain-${kind}">${htmlEscape(slice)}</span>`;
      }
      container.innerHTML = out;
    }

    function ssClass(ch) {
      const c = String(ch).toUpperCase();
      if (c === "H") return "h";       // alpha helix
      if (c === "E") return "e";       // beta strand
      if (c === "C") return "c";       // coil
      if (c === "G") return "g";       // 3_10 helix
      if (c === "T") return "t";       // turn
      if (c === "S") return "s";       // bend
      if (c === "B") return "b";       // beta bridge
      if (c === "I") return "i";       // pi helix
      return "other";
    }

    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, (c) => (
        { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
      ));
    }

    /**
     * Render a header/description string into `target` with the leading
     * accession turned into an external link if the resolver knows it.
     * "2DFB_A Endo-1,4-beta-xylanase…" → <a>2DFB_A</a> + " Endo-1,4-…"
     *
     * Uses createElement (not innerHTML) so URL/label values can't
     * inject markup even if a custom resolver returns garbage.
     */
    function renderLinkedHeader(target, text, opts2 = {}) {
      target.textContent = "";
      if (!text) return;
      // Pulls head / tail / link / parsed from the per-entry cache when
      // an entry is provided (typical for MSA row labels), or computes
      // them inline for one-off call sites (HHR descriptions, template
      // table cells).
      const info = getHeaderInfo(opts2.entry, text);
      const { head, tail, link, parsed } = info;
      const tip = parsed
        ? formatHeaderTip(parsed, link)
        : (link && link.url ? `${link.label || "open"} → ${link.url}` : null);
      if (link && link.url) {
        const a = document.createElement("a");
        a.className = "msa-dblink";
        a.href = link.url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = head;
        if (tip) a.dataset.tip = tip;
        target.appendChild(a);
      } else {
        const span = document.createElement("span");
        span.textContent = head;
        if (tip) span.dataset.tip = tip;
        target.appendChild(span);
      }
      // Twin-icon affordances: small chips after the head linking to the
      // canonical entry pages (UniProt entry, AFDB for UniRef rep, …).
      // Replaces the need for a hidden Shift+Click gesture.
      const secondaries = secondaryHeaderLinks(parsed, link);
      for (const s of secondaries) {
        const chip = document.createElement("a");
        chip.className = "msa-dblink-sec";
        chip.href = s.url;
        chip.target = "_blank";
        chip.rel = "noopener noreferrer";
        chip.textContent = s.label;
        if (s.tip) chip.dataset.tip = s.tip;
        target.appendChild(chip);
      }
      if (tail) {
        const rest = document.createElement("span");
        rest.textContent = tail;
        target.appendChild(rest);
      }
    }

    function formatExp(n) {
      if (!isFinite(n)) return String(n);
      if (n === 0) return "0";
      if (Math.abs(n) >= 0.001 && Math.abs(n) < 1000) return n.toPrecision(3);
      return n.toExponential(1);
    }

    function applyFont(wrapper) {
      wrapper.style.setProperty("--msa-font-size", `${state.fontPx}px`);
    }
    function applyHistHeight(wrapper) {
      wrapper.style.setProperty("--msa-hist-height", `${state.histPx}px`);
    }
    function applyLabelWidth(wrapper) {
      wrapper.style.setProperty("--msa-label-width", `${state.labelWidth}px`);
      wrapper.dataset.labelCollapsed = state.labelWidth === 0 ? "true" : "false";
    }
    function applyPalette(wrapper) {
      wrapper.dataset.palette = state.palette;
    }

    function attachLabelResize(handle, wrapper) {
      on(handle, "mousedown", (ev) => {
        ev.preventDefault();
        const startX = ev.clientX;
        const startW = state.labelWidth;
        handle.classList.add("dragging");
        const prevCursor = document.body.style.cursor;
        const prevSelect = document.body.style.userSelect;
        document.body.style.cursor = "col-resize";
        document.body.style.userSelect = "none";

        function onMove(e) {
          const next = clamp(startW + (e.clientX - startX), LABEL_MIN, LABEL_MAX);
          if (next === state.labelWidth) return;
          state.labelWidth = next;
          applyLabelWidth(wrapper);
        }
        function onUp() {
          document.removeEventListener("mousemove", onMove);
          document.removeEventListener("mouseup", onUp);
          handle.classList.remove("dragging");
          document.body.style.cursor = prevCursor;
          document.body.style.userSelect = prevSelect;
          if (state.labelWidth > 0 && state.labelWidth < LABEL_HIDE_THRESHOLD) {
            state.labelWidth = 0;
            applyLabelWidth(wrapper);
          }
          persist();
        }
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
      });
      on(handle, "dblclick", () => {
        state.labelWidth = LABEL_DEFAULT;
        applyLabelWidth(wrapper);
        persist();
      });
    }

    function attachHistResize(handle, wrapper) {
      on(handle, "mousedown", (ev) => {
        ev.preventDefault();
        const startY = ev.clientY;
        const startH = state.histPx;
        handle.classList.add("dragging");
        const prevCursor = document.body.style.cursor;
        const prevSelect = document.body.style.userSelect;
        document.body.style.cursor = "row-resize";
        document.body.style.userSelect = "none";

        function onMove(e) {
          const next = clamp(startH + (e.clientY - startY), HIST_MIN, HIST_MAX);
          if (next === state.histPx) return;
          state.histPx = next;
          applyHistHeight(wrapper);
        }
        function onUp() {
          document.removeEventListener("mousemove", onMove);
          document.removeEventListener("mouseup", onUp);
          handle.classList.remove("dragging");
          document.body.style.cursor = prevCursor;
          document.body.style.userSelect = prevSelect;
          persist();
        }
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
      });
    }

    // ---- pure helpers (no per-instance state) ----

    function appendStat(parent, key, value, definition) {
      const pill = document.createElement("span");
      pill.className = "msa-stat";
      pill.dataset.tip = `${key} — ${definition}`;
      const k = document.createElement("span");
      k.className = "msa-stat-k";
      k.textContent = key;
      const v = document.createElement("span");
      v.className = "msa-stat-v";
      v.textContent = value;
      pill.appendChild(k);
      pill.appendChild(v);
      parent.appendChild(pill);
    }

    function makeBtn(cls, text) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = cls;
      if (text) b.textContent = text;
      return b;
    }

    function meanHammingToQuery(entries, query, matchLen) {
      if (entries.length <= 1 || matchLen === 0) return 0;
      const q = query.matchSeq;
      let total = 0;
      let count = 0;
      for (const e of entries) {
        if (e === query) continue;
        const s = e.matchSeq;
        const lim = Math.min(matchLen, q.length, s.length);
        let d = 0;
        for (let i = 0; i < lim; i++) {
          if (q.charCodeAt(i) !== s.charCodeAt(i)) d++;
        }
        total += d;
        count++;
      }
      return count > 0 ? total / count : 0;
    }

    // Reuses per-column entropy from `columnInfo` so we don't re-walk the
    // entries × columns matrix. Restricted to columns where the query has
    // a residue (HHsuite/HHblits convention).
    function computeNeff(columnInfo, query) {
      if (!columnInfo || columnInfo.length === 0) return 1;
      const qSeq = query.matchSeq;
      let sumH = 0;
      let L = 0;
      for (let i = 0; i < columnInfo.length; i++) {
        const qch = qSeq.charCodeAt(i);
        if (qch === 45 || qch === 46) continue;
        L++;
        const col = columnInfo[i];
        if (col && col.nonGap > 0) sumH += col.entropy;
      }
      return L > 0 ? Math.exp(sumH / L) : 1;
    }

    function computeCoverage(entries, matchLen) {
      const counts = new Array(matchLen).fill(0);
      for (const e of entries) {
        const seq = e.matchSeq || "";
        const lim = Math.min(seq.length, matchLen);
        for (let i = 0; i < lim; i++) {
          const ch = seq.charCodeAt(i);
          if (ch !== 45 && ch !== 46) counts[i]++;
        }
      }
      const n = entries.length || 1;
      return counts.map((c) => c / n);
    }

    // Per-row aggregate stats — surfaced in the row-label tooltip and as
    // an inline coverage glyph. Computed once per renderInstance.
    //   portion = non-gap residues / matchLen
    //   hamming = positions where row differs from query (either side
    //             non-gap counts as a position)
    //   blosum  = mean BLOSUM62 score over positions where both row and
    //             query have recognized AAs
    // Hamming and blosum are null on the query row itself, and across
    // all rows when `queryEnabled` is false.
    function computeRowStats(viewer, query, queryEnabled) {
      const L = viewer.matchLen;
      const stats = new Map();
      const qSeq = query ? query.matchSeq : "";
      for (const e of viewer.entries) {
        const seq = e.matchSeq || "";
        let nonGap = 0;
        let cmpPositions = 0;
        let hamming = 0;
        let blosumSum = 0;
        let blosumCount = 0;
        const compareToQuery = queryEnabled && query && e !== query;
        for (let i = 0; i < L; i++) {
          const c = seq[i];
          const isResC = c && c !== "-" && c !== ".";
          if (isResC) nonGap++;
          if (!compareToQuery) continue;
          const qc = qSeq[i];
          const isResQ = qc && qc !== "-" && qc !== ".";
          if (isResC || isResQ) {
            cmpPositions++;
            if (c !== qc) hamming++;
          }
          if (isResC && isResQ) {
            const upperC = c >= "a" && c <= "z" ? String.fromCharCode(c.charCodeAt(0) - 32) : c;
            const upperQ = qc >= "a" && qc <= "z" ? String.fromCharCode(qc.charCodeAt(0) - 32) : qc;
            const ic = BLOSUM62_INDEX.get(upperC);
            const iq = BLOSUM62_INDEX.get(upperQ);
            if (ic !== undefined && iq !== undefined) {
              blosumSum += BLOSUM62_ROWS[iq][ic];
              blosumCount++;
            }
          }
        }
        stats.set(e, {
          portion: L > 0 ? nonGap / L : 0,
          nonGap,
          matchLen: L,
          hamming: compareToQuery ? hamming : null,
          cmpPositions: compareToQuery ? cmpPositions : null,
          blosumMean: compareToQuery && blosumCount > 0 ? blosumSum / blosumCount : null,
          blosumCount: compareToQuery ? blosumCount : null,
          isQuery: queryEnabled && query ? e === query : false,
        });
      }
      return stats;
    }

    function computeColumnInfo(viewer, query) {
      const L = viewer.matchLen;
      const N = viewer.entries.length;
      const info = new Array(L);
      const counts = new Map();
      const aaCounts = new Int32Array(20);
      const qSeq = query.matchSeq;
      for (let i = 0; i < L; i++) {
        counts.clear();
        aaCounts.fill(0);
        let nonGap = 0;
        let aaTotal = 0;
        for (const e of viewer.entries) {
          const ch = e.matchSeq[i];
          if (ch && ch !== "-" && ch !== ".") {
            nonGap++;
            counts.set(ch, (counts.get(ch) || 0) + 1);
            const upper = ch >= "a" && ch <= "z" ? ch.toUpperCase() : ch;
            const idx = BLOSUM62_INDEX.get(upper);
            if (idx !== undefined) {
              aaCounts[idx]++;
              aaTotal++;
            }
          }
        }
        let entropy = 0;
        if (nonGap > 0) {
          for (const v of counts.values()) {
            const p = v / nonGap;
            entropy -= p * Math.log(p);
          }
        }
        // Mean BLOSUM62 sum-of-pairs over recognized AAs (gaps + X/B/Z/U/O ignored).
        let blosum = null;
        if (aaTotal >= 2) {
          let T = 0;
          let diag = 0;
          for (let a = 0; a < 20; a++) {
            const ca = aaCounts[a];
            if (ca === 0) continue;
            const row = BLOSUM62_ROWS[a];
            diag += ca * row[a];
            for (let b = 0; b < 20; b++) {
              const cb = aaCounts[b];
              if (cb === 0) continue;
              T += ca * cb * row[b];
            }
          }
          const sumPairs = (T - diag) / 2;
          const nPairs = (aaTotal * (aaTotal - 1)) / 2;
          blosum = sumPairs / nPairs;
        }
        const consEntropy = nonGap > 0 ? 1 - entropy / LN20 : null;
        const consBlosum = blosum == null
          ? null
          : Math.max(0, Math.min(1, (blosum - BLOSUM62_NORM_LO) / (BLOSUM62_NORM_HI - BLOSUM62_NORM_LO)));
        const top = Array.from(counts.entries())
          .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
          .slice(0, 5)
          .map(([res, count]) => ({ res, count }));
        info[i] = {
          queryRes: qSeq[i] || "-",
          total: N,
          nonGap,
          entropy,
          consEntropy,
          blosum,
          consBlosum,
          top,
        };
      }
      return info;
    }

    function computeInsertWidths(entries, matchLen) {
      const widths = new Array(matchLen + 1).fill(0);
      for (const e of entries) {
        const inserts = e.inserts || {};
        for (const k of Object.keys(inserts)) {
          const idx = Number(k);
          const len = inserts[k].length;
          if (len > widths[idx]) widths[idx] = len;
        }
      }
      return widths;
    }

    // Adds the inline coverage glyph at the left of the label and a
    // structured row tooltip (name + portion + Hamming + BLOSUM62 vs
    // query). When `rowStats` is missing or doesn't have an entry, the
    // tooltip falls back to the raw name (no glyph).
    function decorateRowLabelImpl(label, entry, nameText, rowStats) {
      const stats = rowStats ? rowStats.get(entry) : null;
      if (!stats) {
        label.title = nameText;
        return;
      }
      const glyph = document.createElement("span");
      glyph.className = "msa-row-glyph";
      glyph.style.setProperty("--p", String(stats.portion));
      label.prepend(glyph);
      label.dataset.tip = formatRowTip(entry, stats, nameText);
    }

    function formatRowTip(entry, stats, nameText) {
      const lines = [nameText];
      const pct = (stats.portion * 100).toFixed(1);
      lines.push(`residue portion: ${stats.nonGap} / ${stats.matchLen} (${pct}%)`);
      if (stats.isQuery) {
        lines.push("(query row — Hamming / BLOSUM62 are self-comparisons, skipped)");
      } else if (stats.hamming != null) {
        const hPct = stats.cmpPositions > 0
          ? (stats.hamming / stats.cmpPositions * 100).toFixed(1)
          : "—";
        lines.push(`Hamming to query: ${stats.hamming} / ${stats.cmpPositions} (${hPct}%)`);
      }
      if (stats.blosumMean != null) {
        lines.push(`BLOSUM62 vs query: ${stats.blosumMean.toFixed(2)} (mean over ${stats.blosumCount} aligned AAs)`);
      }
      return lines.join("\n");
    }

    function renderRow(parent, entry, extraClass, insertWidths, rowStats, colStart, colEnd) {
      const row = document.createElement("div");
      row.className = "static-msa-row";
      if (extraClass) row.classList.add(extraClass);
      // Recorded so the hover handler can derive a match-col index
      // from sibling position. The pinned query row is only rebuilt
      // on drawAll, but its sequence body re-slices on every drawList;
      // the colStart for both is whatever drawAll passed in (initially
      // 0 — drawList updates it via renderSequence on horizontal scroll
      // and also rewrites the row._colStart in syncStaticQueryColStart).
      row._colStart = colStart != null ? colStart : 0;
      const label = document.createElement("span");
      label.className = "static-msa-label";
      const nameText = entry.name || entry.id || "";
      renderLinkedHeader(label, nameText, { entry });
      decorateRowLabelImpl(label, entry, nameText, rowStats);
      const body = document.createElement("span");
      body.className = "static-msa-seq";
      renderSequence(body, entry, insertWidths, colStart, colEnd);
      row.appendChild(label);
      row.appendChild(body);
      parent.appendChild(row);
      return row;
    }

    function coverColorTip(mode) {
      const cycleLine =
        mode === "off"     ? "click → entropy → BLOSUM62 → off" :
        mode === "entropy" ? "click → BLOSUM62 → off → entropy" :
                             "click → off → entropy → BLOSUM62";
      if (mode === "off") {
        return (
          "Coverage bars are uncolored.\n" +
          "Bar height = non-gap residue fraction per match column.\n" +
          cycleLine
        );
      }
      if (mode === "entropy") {
        return (
          "Color: per-column conservation = 1 − H/ln(20)\n" +
          "where H = Shannon entropy in nats (gaps excluded).\n" +
          "0 = uniform (20 residues equally likely); 1 = single residue dominates.\n" +
          cycleLine
        );
      }
      // BLOSUM
      return (
        "Color: per-column mean BLOSUM62 sum-of-pairs (SP)\n" +
        "SP = (Σₐ Σ_b nₐ·n_b·B[a,b] − Σₐ nₐ·B[a,a]) / (2 · N(N−1)/2)\n" +
        "  N = recognized AAs in the column (gaps + X/B/Z/U/O excluded);\n" +
        "  B = BLOSUM62 substitution matrix.\n" +
        "Normalized for the ramp: clamp((SP + 2) / 8, 0, 1).\n" +
        cycleLine
      );
    }

    function renderHistogramRow(parent, coverage, insertWidths, total, wrapper,
                                columnInfo, colorMode, onCycleColor) {
      const row = document.createElement("div");
      row.className = "static-msa-row static-msa-histogram";

      const label = document.createElement("span");
      label.className = "static-msa-label";
      label.title =
        `Non-gap residue frequency per match column (${total} sequences)\n` +
        `Drag bottom edge to resize`;

      // Inline toggle. Cycles off → entropy → BLOSUM. Lives in the row's
      // label slot so the top toolbar stays uncluttered.
      const colorBtn = document.createElement("button");
      colorBtn.type = "button";
      colorBtn.className = "msa-histogram-color-btn";
      colorBtn.dataset.colorMode = colorMode;
      colorBtn.textContent =
        colorMode === "blosum" ? "BLOSUM ↻" :
        colorMode === "entropy" ? "entropy ↻" : "coverage ↻";
      colorBtn.setAttribute("aria-pressed", String(colorMode !== "off"));
      colorBtn.dataset.tip = coverColorTip(colorMode);
      colorBtn.addEventListener("click", (ev) => {
        ev.preventDefault();
        if (typeof onCycleColor === "function") onCycleColor();
      });
      label.appendChild(colorBtn);

      const useColor = colorMode !== "off" && Array.isArray(columnInfo);
      const colorKey = colorMode === "blosum" ? "consBlosum" : "consEntropy";

      const body = document.createElement("span");
      body.className = "static-msa-seq";
      const consAt = (i) => {
        if (!useColor) return null;
        const c = columnInfo[i];
        if (!c) return null;
        const v = c[colorKey];
        return typeof v === "number" ? v : null;
      };
      let out = "";
      if (insertWidths) {
        out += histInsertGapHtml(insertWidths[0] || 0);
        for (let i = 0; i < coverage.length; i++) {
          out += barHtml(coverage[i], total, i, consAt(i));
          out += histInsertGapHtml(insertWidths[i + 1] || 0);
        }
      } else {
        for (let i = 0; i < coverage.length; i++) out += barHtml(coverage[i], total, i, consAt(i));
      }
      body.innerHTML = out;

      const resizer = document.createElement("div");
      resizer.className = "msa-histogram-resize";
      resizer.title = "Drag to resize coverage row";
      attachHistResize(resizer, wrapper);

      row.appendChild(label);
      row.appendChild(body);
      row.appendChild(resizer);
      parent.appendChild(row);
    }

    function renderRulerRow(parent, matchLen, insertWidths) {
      const row = document.createElement("div");
      row.className = "static-msa-row static-msa-ruler";
      const label = document.createElement("span");
      label.className = "static-msa-label";
      const body = document.createElement("span");
      body.className = "static-msa-seq";

      const cells = new Array(matchLen).fill(" ");
      const ticks = new Set();
      for (let i = 9; i < matchLen; i += 10) {
        ticks.add(i);
        const num = String(i + 1);
        for (let j = 0; j < num.length; j++) {
          const pos = i - num.length + 1 + j;
          if (pos >= 0 && pos < matchLen) cells[pos] = num[j];
        }
      }

      let out = "";
      if (insertWidths) {
        out += rulerInsertGapHtml(insertWidths[0] || 0);
        for (let i = 0; i < matchLen; i++) {
          out += rulerCellHtml(cells[i], ticks.has(i));
          out += rulerInsertGapHtml(insertWidths[i + 1] || 0);
        }
      } else {
        for (let i = 0; i < matchLen; i++) out += rulerCellHtml(cells[i], ticks.has(i));
      }
      body.innerHTML = out;
      row.appendChild(label);
      row.appendChild(body);
      parent.appendChild(row);
    }

    function rulerCellHtml(ch, isTick) {
      const cls = isTick ? "msa-ruler-cell msa-ruler-tick" : "msa-ruler-cell";
      return `<span class="${cls}">${htmlEscapeChar(ch)}</span>`;
    }

    function rulerInsertGapHtml(width) {
      return width > 0
        ? `<span class="msa-ruler-cell msa-ruler-insert" style="width:${width}ch"></span>`
        : "";
    }

    function barHtml(frac, total, colIdx, cons) {
      const colAttr = colIdx != null ? ` data-col="${colIdx}"` : "";
      // Avoid CSS `max()` and `rgba(R, G, B, A)` inside inline-style
      // attribute values — both contain commas, which some webview/CSS
      // pipelines silently mis-parse, dropping the entire declaration
      // and leaving bars invisible.
      // - Floor height in JS (Math.max), emit a plain percentage.
      // - Use space-separated `rgb(R G B / A)` for the conservation tint.
      const h = frac > 0 ? Math.max(2, frac * 100) : 0;
      let style = `height:${h.toFixed(2)}%`;
      if (cons != null) {
        const a = (0.25 + 0.65 * cons).toFixed(3);
        style += `;background-color:rgb(70 130 230 / ${a});opacity:1`;
      }
      return `<span class="msa-histogram-cell"${colAttr}><span class="msa-histogram-bar" style="${style}"></span></span>`;
    }

    function histInsertGapHtml(width) {
      return width > 0
        ? `<span class="msa-histogram-cell msa-histogram-insert" style="width:${width}ch"></span>`
        : "";
    }

    // Per-entry sequence-HTML cache. WeakMap so entries that get GC'd
    // take their cache with them. Stores at most one (key, html) pair
    // per entry — vertical scroll keeps the key stable; horizontal
    // scroll / zoom / inserts toggle changes it and forces a rebuild.
    const sequenceHtmlCache = new WeakMap();

    // Build the inner HTML for one row's sequence track.
    //
    // Optional [colStart, colEnd) clips the rendered match cells to a
    // window — column virtualization. Off-window space becomes a single
    // transparent spacer per side so the row's intrinsic width still
    // matches the full alignment (preserves horizontal scroll geometry
    // and keeps ruler/histogram visually aligned).
    //
    // Insert handling: insertRun[i] sits between match[i-1] and match[i]
    // (insertRun[0] = leading inserts before match[0]; insertRun[total]
    // = trailing inserts after match[total-1]). Inserts BETWEEN two
    // visible match cells are emitted inline. insertRun[cs] sits between
    // an off-window cell and a visible cell — it goes into the left
    // spacer (chars are not visible context for the visible window).
    // Symmetric on the right. The exception is the boundary: when cs==0
    // the leading insert is genuinely visible context for match[0], and
    // when ce==total the trailing insert sits after the last visible
    // match cell — both are emitted inline rather than absorbed.
    function buildSequenceHtml(entry, insertWidths, colStart, colEnd) {
      const matchSeq = entry.matchSeq || "";
      const inserts = entry.inserts || {};
      const total = matchSeq.length;
      const cs = colStart != null ? Math.max(0, colStart) : 0;
      const ce = colEnd != null ? Math.min(total, colEnd) : total;

      let leftCh = 0;
      let rightCh = 0;
      if (insertWidths) {
        // Slots [0, cs) — every slot pair (insertRun + match) absorbed.
        for (let i = 0; i < cs; i++) leftCh += 1 + (insertWidths[i] || 0);
        // insertRun[cs] joins the spacer ONLY when cs > 0. At cs == 0
        // it's the leading insert and should render inline.
        if (cs > 0) leftCh += insertWidths[cs] || 0;
        // Right side: when ce < total, absorb insertRun[ce] + slots
        // [ce, total) + trailing insertRun[total].
        if (ce < total) {
          rightCh += insertWidths[ce] || 0;
          for (let i = ce; i < total; i++) rightCh += 1 + (insertWidths[i + 1] || 0);
        }
      } else {
        leftCh = cs;
        rightCh = total - ce;
      }

      let out = "";
      if (leftCh > 0) out += spacerHtml(leftCh);
      if (insertWidths) {
        if (cs === 0 && (insertWidths[0] || 0) > 0) {
          out += insertRunHtml(inserts[0] || "", insertWidths[0] || 0);
        }
        for (let i = cs; i < ce; i++) {
          out += matchCellHtml(matchSeq[i]);
          if (i + 1 < ce) {
            out += insertRunHtml(inserts[i + 1] || "", insertWidths[i + 1] || 0);
          }
        }
        if (ce === total && (insertWidths[total] || 0) > 0) {
          out += insertRunHtml(inserts[total] || "", insertWidths[total] || 0);
        }
      } else {
        for (let i = cs; i < ce; i++) out += matchCellHtml(matchSeq[i]);
      }
      if (rightCh > 0) out += spacerHtml(rightCh);
      return out;
    }

    // Per-entry sequence HTML cache — keyed by the render parameters.
    // Vertical scroll keeps the same key (cache hits, every new row in
    // the viewport pays only the innerHTML parse). Horizontal scroll
    // and zoom change the key (rebuild on miss).
    function renderSequence(container, entry, insertWidths, colStart, colEnd) {
      const total = (entry.matchSeq || "").length;
      const cs = colStart != null ? colStart : 0;
      const ce = colEnd != null ? colEnd : total;
      const key = `${insertWidths ? 1 : 0}|${cs}|${ce}`;
      const cache = sequenceHtmlCache.get(entry);
      if (cache && cache.key === key) {
        container.innerHTML = cache.html;
        return;
      }
      const html = buildSequenceHtml(entry, insertWidths, cs, ce);
      sequenceHtmlCache.set(entry, { key, html });
      container.innerHTML = html;
    }

    function spacerHtml(widthCh) {
      return `<span class="msa-residue msa-spacer msa-aa-gap" style="width:${widthCh}ch"></span>`;
    }

    function insertRunHtml(text, width) {
      if (width <= 0) return "";
      let out = "";
      for (let i = 0; i < text.length; i++) {
        out += insertCellHtml(text[i]);
      }
      const pad = width - text.length;
      if (pad > 0) {
        out += `<span class="msa-residue msa-aa-gap msa-insert msa-insert-pad" style="width:${pad}ch">${".".repeat(pad)}</span>`;
      }
      return out;
    }

    async function toggleFullscreen(wrapper) {
      if (document.fullscreenElement === wrapper) {
        await document.exitFullscreen();
      } else if (wrapper.requestFullscreen) {
        await wrapper.requestFullscreen();
      }
    }

    function setFullscreenIcon(button, isFs) {
      button.innerHTML = isFs
        ? '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 5V3h2M11 3h2v2M13 11v2h-2M5 13H3v-2" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>'
        : '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3H3v3M10 3h3v3M13 10v3h-3M3 10v3h3" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';
    }

    return { load, showError, destroy };
  }

  if (typeof window !== "undefined") {
    window.MsaViewer = { create };
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { create };
  }
})();
