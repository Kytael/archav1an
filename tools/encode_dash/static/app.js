"use strict";

const POLL_MS = 2000;

// The revision of the roster this page last saw. Every write quotes it, and a
// 409 means someone else -- vim over ssh, or another browser -- got there
// first. The page reloads its data rather than retrying: a retry would be the
// clobber this guard exists to prevent.
let rosterRev = null;
// Set while a write is in flight, so the poll does not redraw the switch the
// operator just clicked back to its old position before the write lands.
let writing = false;
// Bumped once per completed write. `writing` alone is not enough to protect
// rosterRev: it is false again the instant the write resolves, so a poll whose
// fetch STARTED before the write and resolved after it would put the pre-write
// revision back. The next click then quotes a revision the file no longer has
// and gets a 409 that blames an external editor who does not exist. The
// snapshot is slow to build -- a a few thousand-line manifest and two scans of
// state.jsonl -- so that overlap is the common case. Each poll remembers the
// counter it started on and drops its revision if the counter has moved.
let writeGen = 0;

async function write(path, method, body) {
  writing = true;
  try {
    const r = await fetch(path, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(Object.assign({ rev: rosterRev }, body || {})),
    });
    const text = await r.text();
    if (r.ok) {
      // Guarded, because not every write answers with a rev. The control
      // routes -- yield's kill request, retry, stop -- answer {ok, id}, and
      // assigning that missing field would set rosterRev to undefined and 409
      // every roster edit from then on. A yield DOES answer with one, because
      // it edits the roster before it asks for the kill.
      const parsed = JSON.parse(text);
      if (typeof parsed.rev === "string") rosterRev = parsed.rev;
      say("");
      return true;
    }
    // 409 is not an error the operator caused, so it reads differently from a
    // rejected edit: the file moved under them and the page is now stale.
    if (r.status === 409) {
      say("The roster changed elsewhere. This page reloaded it; try again.");
      return false;
    }
    // The daemon puts its failure message in a JSON body -- {"error": "..."}
    // -- exactly so this branch can show it instead of an HTML error page.
    // A body that will not parse, or one that parses without an "error"
    // string, must not throw here: an error handler that throws hides the
    // original failure behind a worse one.
    let msg = `HTTP ${r.status}`;
    try {
      const parsed = JSON.parse(text);
      if (parsed && typeof parsed.error === "string") msg = parsed.error;
    } catch (e) { /* not JSON -- fall back to the status */ }
    say(msg);
    return false;
  } catch (e) {
    say("daemon unreachable — the edit was not made");
    return false;
  } finally {
    writing = false;
    // Bumped on every outcome, not only on success: a refused write can still
    // have moved the file (a 409 means someone else did), and in every case a
    // snapshot fetched before this point knows less than this page does.
    writeGen++;
  }
}

function say(text) {
  const p = document.getElementById("roster-msg");
  p.textContent = text;
  p.hidden = !text;
}

// An em dash, not "0". Throughout this design an absent measurement and a
// measured zero are different facts, and the page must not conflate them: a
// lane that has not started looks nothing like a lane that has stalled.
function fps(v) { return v === null || v === undefined ? "—" : v.toFixed(2); }
function int(v) { return (v || 0).toLocaleString(); }

// A clip's frame total, or an em dash. manifest._frames_from records 0 for a
// clip whose count the probe never returned, so a total of 0 is an unknown
// length rather than a length of zero -- and "900 / 0 frames" reads as a
// contradiction rather than as the missing probe it actually is.
function total(v) { return v ? int(v) : "—"; }

function dur(s) {
  if (s === null || s === undefined) return "—";
  if (s < 90) return Math.round(s) + " s";
  if (s < 5400) return Math.round(s / 60) + " min";
  return (s / 3600).toFixed(1) + " h";
}

function when(epoch) {
  if (!epoch) return "—";
  return new Date(epoch * 1000).toLocaleDateString(undefined,
    { month: "short", day: "numeric" });
}

// The quotes matter as much as the angle brackets: escaped values land inside
// attributes here (a lane's state names a CSS class), and a value carrying a
// double quote would close the attribute early. Neither source is a fixed
// vocabulary -- the state comes from a heartbeat file on disk, the lane name
// from a TOML the operator edits, the reason from a GPU driver's log tail.
function esc(s) {
  return String(s === null || s === undefined ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function cell(html, cls) {
  const td = document.createElement("td");
  if (cls) td.className = cls;
  td.innerHTML = html;
  return td;
}

function renderTotals(t) {
  // No manifest, or one that will not parse, gives a frame total of 0. "0.0%"
  // would claim the run has done none of a known amount of work; it has done
  // none of an unknown amount, and the banner above says why.
  const pct = t.frames ? (100 * t.frames_done / t.frames).toFixed(1) + "%" : "—";
  document.getElementById("totals").innerHTML = `
    <div>done<b>${int(t.done)} / ${int(t.clips)}</b></div>
    <div>frames<b>${pct}</b></div>
    <div>now<b>${fps(t.fps_live)} fps</b></div>
    <div>finish<b>${when(t.eta_finish)}</b></div>`;
  document.getElementById("queued").textContent = int(t.queued) + " queued";
  // The Failed heading is renderFailures's, because its text depends on the
  // filter as well as the totals.
}

// replaceChildren destroys every element in a container and the keyboard focus
// with it, so a control reached by Tab stayed usable only until the next poll
// -- two seconds. Remember which control had it, by key and kind, and re-apply
// it to the element that replaced it. Matched on the pair rather than on
// position, because a row added or removed elsewhere shifts everything after.
function keepFocus(body) {
  const was = document.activeElement;
  return was && body.contains(was)
    ? { key: was.dataset.key, control: was.dataset.control } : null;
}

function restoreFocus(body, keep) {
  if (!keep || keep.key === undefined) return;
  // Compared in JavaScript rather than through a [data-key="..."] selector: a
  // key is a lane name or a clip path out of a file the operator edits, and
  // either may hold a quote.
  const again = Array.prototype.find.call(
    body.querySelectorAll("[data-control]"),
    (el) => el.dataset.key === keep.key && el.dataset.control === keep.control);
  if (again) again.focus();
}

function renderLanes(lanes) {
  const body = document.querySelector("#lanes tbody");
  const keep = keepFocus(body);
  body.replaceChildren();
  for (const l of lanes) {
    const tr = document.createElement("tr");

    const sw = document.createElement("span");
    sw.className = "sw" + (l.enabled ? " on" : "");
    sw.dataset.key = l.name;
    sw.dataset.control = "switch";
    sw.tabIndex = 0;
    sw.setAttribute("role", "switch");
    sw.setAttribute("aria-checked", String(l.enabled));
    sw.title = l.enabled
      ? "Disable: the lane finishes its clip, then takes no more"
      : "Enable: the lane takes the next clip";
    const toggle = () =>
      write(`/api/lane/${encodeURIComponent(l.name)}/enabled`, "POST",
            { enabled: !l.enabled });
    sw.onclick = toggle;
    // role="switch" advertises a keyboard contract a <span> does not honour on
    // its own: unlike a <button>, it gets no synthetic click from Enter or
    // Space. Space also scrolls the page by default, so it needs
    // preventDefault; Enter does not scroll and needs none.
    sw.onkeydown = (e) => {
      if (e.key === "Enter") toggle();
      else if (e.key === " ") { e.preventDefault(); toggle(); }
    };
    tr.appendChild(cell("")).appendChild(sw);

    tr.appendChild(cell(
      `<span class="name">${esc(l.name)}</span>
       <span class="sub">${esc(l.host)} · ${esc(l.backend)}` +
      (l.window ? ` · win ${l.window}` : "") + `</span>`));

    if (l.current) {
      const c = l.current;
      const pct = c.progress === null ? 0 : Math.round(c.progress * 100);
      tr.appendChild(cell(
        `<span class="tag ${esc(l.state)}">${esc(l.state)}</span>
         ${esc(c.src.split("/").pop())}
         <span class="sub">${c.frames_done === null ? "—" : int(c.frames_done)}
           / ${total(c.frames)} frames ·
           ${c.eta_s === null ? "elapsed " + dur(c.elapsed_s)
                              : dur(c.eta_s) + " left"}</span>
         <div class="bar"><i style="width:${pct}%"></i></div>`));
    } else {
      tr.appendChild(cell(`<span class="tag ${esc(l.state)}">${esc(l.state)}</span>`));
    }

    tr.appendChild(cell(
      `<b>${fps(l.fps_live)}</b><span class="sub">live fps</span>`, "num"));
    tr.appendChild(cell(
      `<b>${fps(l.fps_recent)}</b>
       <span class="sub">${int(l.clips_done)} clips end-to-end</span>`, "num"));

    const yd = document.createElement("button");
    yd.type = "button";
    yd.className = "rm";
    yd.dataset.key = l.name;
    yd.dataset.control = "yield";
    yd.textContent = "yield";
    yd.title = "Disable this lane and kill the clip it is holding now. The "
             + "clip goes back on the queue with no attempt spent.";
    // Nothing to kill on an idle lane, and offering it would suggest the
    // button does something else -- disabling is what the switch is for.
    yd.disabled = !l.current;
    yd.onclick = () =>
      write(`/api/lane/${encodeURIComponent(l.name)}/yield`, "POST", {});

    const ed = document.createElement("button");
    ed.type = "button";
    ed.className = "rm";
    ed.dataset.key = l.name;
    ed.dataset.control = "edit";
    ed.textContent = "edit";
    ed.title = "Change this lane's settings. Takes effect at the next clip: "
             + "nothing in flight is killed. The name cannot change.";
    // Guarded, not assumed: the form is built on the first snapshot, and a
    // row rendered from that same snapshot could otherwise wire a click to
    // nothing at all.
    ed.disabled = openLaneForm === null;
    ed.onclick = () => openLaneForm(l);

    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "rm";
    rm.dataset.key = l.name;
    rm.dataset.control = "remove";
    rm.textContent = "remove";
    // No confirm(): a modal dialog blocks the poll loop. Removing a lane is
    // reversible -- re-add it -- and it cannot lose work, because a removed
    // lane's worker parks after the clip it holds.
    rm.onclick = () =>
      write(`/api/lane/${encodeURIComponent(l.name)}`, "DELETE", {});

    const actions = cell("", "actions");
    actions.appendChild(yd);
    actions.appendChild(ed);
    actions.appendChild(rm);
    tr.appendChild(actions);

    body.appendChild(tr);
  }

  restoreFocus(body, keep);
}

// Three typed fields, no picker. The archive run is NOT one of the choices
// here: it is driven by manifest-raw.tsv, it denoises, and it is started by
// "Start the run". Listing it as a folder offered a choice and then refused
// half of it, which reads worse than not listing it.
const JOB_FIELDS = [
  ["host", "text", "ssh alias holding the folder, or 'local'"],
  ["path", "text", "absolute path on that host"],
  ["dest", "text", "subpath under encoded/, e.g. SetA/2026/Comp"],
];

function renderJobs(jobs) {
  const body = document.querySelector("#jobs tbody");
  const keep = keepFocus(body);
  body.replaceChildren();
  const rows = jobs || [];
  for (const j of rows) {
    const tr = document.createElement("tr");
    tr.appendChild(cell(
      `<span class="name">${esc(j.encoder || j.name)}</span>
       <span class="sub">slot ${esc(j.name.slice((j.encoder || "").length + 1))}</span>`));
    if (j.src) {
      const pct = j.progress === null || j.progress === undefined
        ? 0 : Math.round(j.progress * 100);
      tr.appendChild(cell(
        `<span class="tag ${esc(j.state)}">${esc(j.state)}</span> ${esc(j.stem)}
         <span class="sub">${j.frames_done === null || j.frames_done === undefined
                              ? "\u2014" : int(j.frames_done)}
           / ${total(j.frames)} frames \u00b7
           ${j.eta_s === null || j.eta_s === undefined
               ? "elapsed " + dur(j.elapsed_s) : dur(j.eta_s) + " left"}</span>
         <div class="bar"><i style="width:${pct}%"></i></div>`));
    } else {
      tr.appendChild(cell(`<span class="tag idle">idle</span>`));
    }
    tr.appendChild(cell(
      `<b>${fps(j.fps_live)}</b><span class="sub">live fps</span>`, "num"));
    tr.appendChild(cell(
      `<b>${fps(j.fps_recent)}</b>
       <span class="sub">${int(j.clips_done)} clips end-to-end</span>`, "num"));
    body.appendChild(tr);
  }
  // The TABLE hides when nothing is running, never the section: the section
  // also holds "Encode a folder", and hiding it made the only control that
  // creates a job appear only once a job already existed.
  const active = rows.some((j) => j.src);
  document.querySelector("#jobs").hidden = !active;
  const empty = document.getElementById("jobs-empty");
  if (empty) empty.hidden = active;
  restoreFocus(body, keep);
}

function option(value, text) {
  const o = document.createElement("option");
  o.value = value;
  o.textContent = text;
  return o;
}

// Fetched once, not polled: the catalogue is the scripts on disk. A failure
// leaves the fleet-settings entry alone in the list, which still submits.
async function loadPresets(select) {
  let data;
  try {
    const res = await fetch("/api/presets");
    if (!res.ok) return;
    data = await res.json();
  } catch (err) {
    return;
  }
  for (const p of data.presets || []) {
    select.appendChild(option(
      p.id, `${p.label} \u2014 crf ${p.quality}, speed ${p.speed}`));
  }
}

function buildJobForm() {
  const form = document.getElementById("add-job");
  if (form.children.length) return;

  const note = document.createElement("p");
  note.className = "note";
  note.textContent = "Every .MOV and .MP4 directly in this folder, encoded "
    + "with no denoise pass. One level only: a folder inside it is not "
    + "included, because its files would be published to a destination chosen "
    + "for their parent.";
  form.appendChild(note);

  for (const [key, type, hint] of JOB_FIELDS) {
    const label = document.createElement("label");
    label.textContent = key;
    const input = document.createElement("input");
    input.name = key;
    input.type = type;
    input.placeholder = hint;
    form.appendChild(label);
    label.appendChild(input);
  }

  // A dropdown, not a typed name: the catalogue is a fixed list of scripts on
  // disk, and a typo in a typed one is only found when the first job runs.
  const plabel = document.createElement("label");
  plabel.textContent = "preset";
  const select = document.createElement("select");
  select.name = "preset";
  select.appendChild(option("", "the fleet settings (crf 27, dance)"));
  form.appendChild(plabel);
  plabel.appendChild(select);
  loadPresets(select);

  const submit = document.createElement("button");
  submit.type = "submit";
  submit.textContent = "Queue this folder";
  form.appendChild(submit);

  form.onsubmit = async (e) => {
    e.preventDefault();
    const body = {};
    for (const [key] of JOB_FIELDS) body[key] = form.elements[key].value.trim();
    body.preset = select.value;
    if (await write("/api/jobs/submit", "POST", body)) {
      form.reset();
      form.hidden = true;
    }
  };
  document.getElementById("add-job-open").onclick = () => {
    form.hidden = !form.hidden;
  };
}


function buildHostForm() {
  const form = document.getElementById("add-host");
  if (form.children.length) return;

  const note = document.createElement("p");
  note.className = "note";
  note.hidden = true;
  form.appendChild(note);

  for (const [key, type, hint] of ENCODER_FIELDS) {
    const label = document.createElement("label");
    label.textContent = key;
    const input = document.createElement("input");
    input.name = key;
    input.type = type;
    input.placeholder = hint;
    label.appendChild(input);
    form.appendChild(label);
  }

  const submit = document.createElement("button");
  submit.type = "submit";
  form.appendChild(submit);

  const fill = (fields, noteText, origin) => {
    for (const [key] of ENCODER_FIELDS) form.elements[key].value = "";
    note.textContent = noteText || "";
    note.hidden = !noteText;
    if (!fields) return;
    for (const [key, value] of Object.entries(fields)) {
      // An absent value is not a zero, the same rule the lane form follows:
      // the daemon sends every writable key and the unset ones arrive as ""
      // or 0, which typed into the boxes would turn "this host needs no
      // checkout path" into a saved empty string.
      if (value === "" || value === null || value === undefined) continue;
      const input = form.elements[key];
      if (input) input.value = String(value);
      else say(`${origin} sets '${key}', which this form cannot show`);
    }
  };

  const setMode = (host) => {
    editingHost = host ? host.name : null;
    // The name keys every lane's allowlist, so the daemon refuses a rename for
    // a reason the lane form does not have: a renamed host leaves each lane
    // that routed to it pointing at nothing.
    form.elements.name.readOnly = editingHost !== null;
    submit.textContent = host ? "Save changes" : "Add, switched off";
    if (host) {
      fill(host.fields, "Takes effect at this host's next clip: nothing in "
           + "flight is killed. Clear a field to put it back on its default. "
           + "The name cannot change — every lane's allowlist names it.",
           `host '${host.name}'`);
    } else {
      form.reset();
      fill(null, "A new host arrives switched off. Give it a port block no "
           + "other host uses: slot N listens on port_base + N.");
    }
    form.hidden = false;
    say("");
  };
  openHostForm = setMode;

  form.onsubmit = async (e) => {
    e.preventDefault();
    const body = {};
    for (const [key, type] of ENCODER_FIELDS) {
      const raw = form.elements[key].value.trim();
      if (!raw) {
        if (editingHost) body[key] = "";
        continue;
      }
      body[key] = type === "number" ? Number(raw) : raw;
    }
    const path = editingHost
      ? `/api/encoder/${encodeURIComponent(editingHost)}`
      : "/api/encoder";
    if (await write(path, "POST", body)) {
      form.reset();
      note.textContent = "";
      note.hidden = true;
      form.hidden = true;
      editingHost = null;
      form.elements.name.readOnly = false;
    }
  };
  document.getElementById("add-host-open").onclick = () => {
    if (!form.hidden && editingHost === null) form.hidden = true;
    else setMode(null);
  };
}


function renderHosts(encode, rates) {
  const body = document.querySelector("#hosts tbody");
  const keep = keepFocus(body);
  body.replaceChildren();
  // A legacy [encode] roster has one shared pool and no per-host rows, so
  // there is nothing to switch. Hide the section rather than show an empty
  // table, which would read as "no encoders" on a fleet that has six.
  const hosts = (encode && encode.hosts) || [];
  document.querySelector("#hosts").closest("section").hidden = !hosts.length;
  for (const e of hosts) {
    const tr = document.createElement("tr");

    const sw = document.createElement("span");
    sw.className = "sw" + (e.enabled ? " on" : "");
    // Prefixed, because keepFocus keys on dataset.key and the roster has a
    // lane and an encoder both called "gpu4" -- an unprefixed key would
    // restore focus to the wrong table's switch after a poll.
    sw.dataset.key = "encoder:" + e.name;
    sw.dataset.control = "switch";
    sw.tabIndex = 0;
    sw.setAttribute("role", "switch");
    sw.setAttribute("aria-checked", String(e.enabled));
    sw.title = e.enabled
      ? "Disable: the host finishes what it is encoding, then takes no more"
      : "Enable: lanes may pick this host again";
    const toggle = () =>
      write(`/api/encoder/${encodeURIComponent(e.name)}/enabled`, "POST",
            { enabled: !e.enabled });
    sw.onclick = toggle;
    // Same keyboard contract as the lane switch: role="switch" on a <span>
    // gets no synthetic click from Enter or Space, and Space also scrolls.
    sw.onkeydown = (ev) => {
      if (ev.key === "Enter") toggle();
      else if (ev.key === " ") { ev.preventDefault(); toggle(); }
    };
    tr.appendChild(cell("")).appendChild(sw);

    tr.appendChild(cell(
      `<span class="name">${esc(e.name)}</span>
       <span class="sub">${esc(e.host)}</span>`));
    tr.appendChild(cell(
      `<b>${e.slots}</b><span class="sub">slots</span>`, "num"));
    tr.appendChild(cell(
      `<b>${e.lp_level === null || e.lp_level === undefined
             ? "\u2014" : e.lp_level}</b><span class="sub">--lp</span>`, "num"));
    // Per host, summed across its slots. A single slot's rate cannot be
    // compared with another host running two, which is the comparison this
    // table exists to make.
    const r = (rates || {})[e.name] || {};
    tr.appendChild(cell(
      `<b>${fps(r.fps_live)}</b><span class="sub">live fps</span>`, "num"));
    tr.appendChild(cell(
      `<b>${fps(r.fps_recent)}</b>
       <span class="sub">${int(r.clips_done || 0)} jobs done</span>`, "num"));

    const actions = cell("", "actions");
    const ed = document.createElement("button");
    ed.type = "button";
    ed.className = "rm";
    ed.dataset.key = "encoder:" + e.name;
    ed.dataset.control = "edit";
    ed.textContent = "edit";
    ed.title = "Change this host's slots, --lp, port block or address.";
    ed.onclick = () => { if (openHostForm) openHostForm(e); };
    actions.appendChild(ed);

    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "rm";
    rm.dataset.key = "encoder:" + e.name;
    rm.dataset.control = "remove";
    rm.textContent = "remove";
    rm.title = "Take this host out of the pool. Refused while any lane's "
             + "allowlist still names it — clear that routing first.";
    rm.onclick = () =>
      write(`/api/encoder/${encodeURIComponent(e.name)}`, "DELETE", {});
    actions.appendChild(rm);
    tr.appendChild(actions);

    body.appendChild(tr);
  }
  restoreFocus(body, keep);
}


function renderLp(encode) {
  const sel = document.getElementById("lp");
  // Disabled too when the pool sets no single lp_level: this control writes
  // one number for the whole pool, so with encoders on different levels there
  // is nothing here it could show without lying about the others.
  //
  // And disabled when lp_editable is false, because rosterio can only write
  // the legacy [encode] table: on a pool roster every save here answered 400,
  // and a control that cannot succeed must not be offered.
  if (!encode || encode.lp_level === null) { sel.disabled = true; return; }
  if (encode.lp_editable === false) {
    sel.disabled = true;
    sel.title = "per-encoder --lp editing is not available yet; "
      + "edit lp_level on the [[encoder]] entry in the roster";
    return;
  }
  sel.disabled = false;
  sel.title = "";
  if (sel.options.length === 0) {
    for (let i = 0; i <= 6; i++) {
      const o = document.createElement("option");
      o.value = String(i);
      o.textContent = String(i);
      sel.appendChild(o);
    }
    sel.onchange = () =>
      write("/api/encode/lp_level", "POST", { lp_level: Number(sel.value) });
  }
  // Never while the select has focus: rewriting the value under an open
  // dropdown snaps it shut mid-choice.
  if (document.activeElement !== sel) sel.value = String(encode.lp_level);
}

// Built once. The fields are roster.py's, and its messages are what the form
// shows on a refusal -- the page states no rules of its own (spec 6).
//
// This is a SECOND copy of rosterio.FIELDS, not the only one: the form is
// built before the page has ever seen a snapshot, so the list cannot come from
// the daemon. A key added to Denoiser has to be added in both places.
const LANE_FIELDS = [
  ["name", "text", "unique, e.g. gpu3"],
  ["host", "text", "an ssh alias, or 'local'"],
  ["backend", "text", "trt or migraphx"],
  ["device", "number", "0"],
  ["tiling", "text", "none, auto, or 1112x992"],
  ["window", "number", "required when tiled"],
  ["margin", "number", "32; at least 16 when tiled"],
  ["root", "text", "checkout path on a remote"],
];

// Opens the lane form: pass a lane row to edit it, or null for a new lane.
// buildAddForm assigns this once the form exists; renderLanes runs on every
// poll and needs a handle that outlives the closure.
let openLaneForm = null;

// Encoder names from the last snapshot, for the lane form's allowlist boxes.
// The form is built once, before any snapshot has arrived, so the boxes are
// rebuilt on open rather than at build time -- a host added while the page was
// up must be routable without a reload.
let encoderNames = [];
// Opens the encode host form: a host row to edit it, null for a new one.
let openHostForm = null;
let editingHost = null;

const ENCODER_FIELDS = [
  ["name", "text", "unique, e.g. gpu1"],
  ["host", "text", "an ssh alias, or 'local'"],
  ["root", "text", "checkout path on a remote"],
  ["stream_ip", "text", "fixed address"],
  ["stream_net", "text", "e.g. 10.0.0.0/24 if it moves"],
  ["port_base", "number", "first port of a free block, e.g. 5360"],
  ["slots", "number", "clips at once, e.g. 2"],
  ["lp_level", "number", "4, or 6 for more memory"],
];
// The name of the lane being edited, or null while the form is adding one.
// The submit handler reads it to choose the route AND to decide whether an
// empty field means "leave it out" or "unset it", which are not the same.
let editing = null;

function buildAddForm() {
  const form = document.getElementById("add-lane");
  if (form.children.length) return;

  // The picker is built first so it reads as where you start, not as an
  // afterthought under ten empty boxes. It stays hidden until the catalogue
  // arrives, and a catalogue that will not load leaves it hidden for good:
  // typing every field is how this form worked before, and it still works.
  let hasPresets = false;
  const pick = document.createElement("label");
  pick.className = "preset";
  pick.hidden = true;
  pick.textContent = "known lane";
  const sel = document.createElement("select");
  sel.id = "lane-preset";
  pick.appendChild(sel);
  form.appendChild(pick);

  // Its own row, NOT inside the picker's label. Edit mode hides the picker
  // and still has something to say -- that the change lands at the next clip
  // and that the name cannot move -- and a note nested in the hidden element
  // goes with it.
  const note = document.createElement("p");
  note.className = "note";
  note.hidden = true;
  form.appendChild(note);

  for (const [key, type, hint] of LANE_FIELDS) {
    const label = document.createElement("label");
    label.textContent = key;
    const input = document.createElement("input");
    input.name = key;
    input.type = type;
    input.placeholder = hint;
    label.appendChild(input);
    form.appendChild(label);
  }
  const stage = document.createElement("label");
  stage.textContent = "stage_source";
  const box = document.createElement("input");
  box.name = "stage_source";
  box.type = "checkbox";
  stage.appendChild(box);
  form.appendChild(stage);

  // Routing. A checkbox per encode host rather than a text field, because the
  // roster validates these names against the [[encoder]] table and a typo
  // there is refused only after the operator has typed the whole list.
  const routing = document.createElement("fieldset");
  routing.className = "routing";
  const legend = document.createElement("legend");
  legend.textContent = "encoders";
  routing.appendChild(legend);
  const routingNote = document.createElement("p");
  routingNote.className = "note";
  routingNote.textContent = "Which hosts this lane may encode on. None ticked "
    + "means any enabled host, which is the default.";
  routing.appendChild(routingNote);
  const routingBoxes = document.createElement("div");
  routingBoxes.className = "boxes";
  routing.appendChild(routingBoxes);
  form.appendChild(routing);

  // Rebuilt from the latest snapshot each time the form opens, and the ticks
  // restored from `chosen`. A lane may name a host that has since been removed
  // from the pool -- that lane is already refused by the validator, so the
  // stale name is shown ticked and disabled rather than dropped in silence,
  // which would let a save quietly rewrite the routing the operator came to
  // look at.
  const buildRouting = (chosen) => {
    routingBoxes.replaceChildren();
    const picked = new Set(chosen || []);
    const known = new Set(encoderNames);
    const names = encoderNames.concat(
      [...picked].filter((n) => !known.has(n)).sort());
    if (!names.length) {
      routing.hidden = true;
      return;
    }
    routing.hidden = false;
    for (const name of names) {
      const label = document.createElement("label");
      label.className = "box";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.dataset.encoder = name;
      cb.checked = picked.has(name);
      label.appendChild(cb);
      label.appendChild(document.createTextNode(" " + name));
      if (!known.has(name)) {
        label.classList.add("gone");
        label.title = "This host is not in the pool any more. Untick it to "
                    + "clear the routing; the roster refuses a lane that "
                    + "names a host that does not exist.";
      }
      routingBoxes.appendChild(label);
    }
  };
  const chosenEncoders = () =>
    [...routingBoxes.querySelectorAll("input:checked")]
      .map((cb) => cb.dataset.encoder);

  const submit = document.createElement("button");
  submit.type = "submit";
  submit.textContent = "Add, switched off";
  form.appendChild(submit);

  // Fills every field a preset names and CLEARS every field it does not.
  // Without the clear, choosing an untiled lane after a tiled one leaves the
  // window behind, and the roster refuses "window without tiling" over a value
  // the operator never typed.
  const fill = (fields, noteText, origin) => {
    for (const [key] of LANE_FIELDS) form.elements[key].value = "";
    box.checked = false;
    buildRouting(fields && fields.encoders);
    note.textContent = noteText || "";
    note.hidden = !noteText;
    if (!fields) return;
    for (const [key, value] of Object.entries(fields)) {
      if (key === "stage_source") { box.checked = Boolean(value); continue; }
      // Handled by buildRouting above, which fill() has already called. It has
      // no input in LANE_FIELDS, so without this it would fall through to the
      // "this form cannot show it" warning on every edit.
      if (key === "encoders") continue;
      // An absent value is not a zero. The daemon sends every writable key on
      // every lane, and the unset ones come back as "" or 0 -- writing those
      // into the boxes would turn "this lane has no window" into "window 0"
      // the moment anyone saved.
      if (value === "" || value === null || value === undefined) continue;
      const input = form.elements[key];
      // A key this form has no input for would otherwise be dropped in
      // silence and the lane written without it. LANE_FIELDS is already a
      // second copy of rosterio.FIELDS and the catalogue is a third;
      // tests/test_lane_presets.py pins all three together, and this is what
      // the operator sees if that ever slips anyway.
      if (input) input.value = String(value);
      else say(`${origin} sets '${key}', which this form cannot show`);
    }
  };

  // Fetched once, not polled: the catalogue is a file in the checkout, and it
  // only changes when the daemon is redeployed.
  fetch("/static/lane-presets.json", { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : null))
    .then((cat) => {
      const list = cat && Array.isArray(cat.presets) ? cat.presets : [];
      if (!list.length) return;
      const blank = document.createElement("option");
      blank.value = "";
      blank.textContent = "start from a benchmarked lane…";
      sel.appendChild(blank);
      for (const preset of list) {
        const option = document.createElement("option");
        option.value = preset.id;
        option.textContent = preset.label;
        sel.appendChild(option);
      }
      sel.onchange = () => {
        const preset = list.find((item) => item.id === sel.value) || null;
        fill(preset && preset.fields,
             preset && `${preset.note} (${preset.source})`,
             preset && `the '${preset.id}' preset`);
      };
      hasPresets = true;
      pick.hidden = editing !== null;
    })
    .catch(() => { /* every field is still there to type into */ });

  // null adds a lane; a lane row edits that one.
  const setMode = (lane) => {
    editing = lane ? lane.name : null;
    // A preset is a starting point for a lane that does not exist yet.
    // Offering it over an edit would invite one click to overwrite a host's
    // settings with another host's.
    pick.hidden = !hasPresets || editing !== null;
    // The name keys the lane's heartbeat, its temp directory and the
    // scheduler's bookkeeping, so the daemon refuses to rename. Saying that
    // with a read-only box beats saying it in an error after the fact.
    form.elements.name.readOnly = editing !== null;
    submit.textContent = lane ? "Save changes" : "Add, switched off";
    if (lane) {
      fill(lane.fields, "Takes effect at the next clip: nothing in flight " +
           "is killed. Clear a field to put it back on its default. The " +
           "name cannot change — remove the lane and add it again.",
           `lane '${lane.name}'`);
    } else {
      form.reset();
      fill(null);
    }
    form.hidden = false;
    say("");
  };
  openLaneForm = setMode;

  form.onsubmit = async (e) => {
    e.preventDefault();
    const body = {};
    for (const [key, type] of LANE_FIELDS) {
      const raw = form.elements[key].value.trim();
      // An edit sends every key, the empty ones included: "" is how this
      // page says "unset this", and skipping it would make clearing a field
      // a silent no-op. An add sends only what was filled in, because there
      // an absent key and an empty one already mean the same thing.
      if (!raw) {
        if (editing) body[key] = "";
        continue;
      }
      body[key] = type === "number" ? Number(raw) : raw;
    }
    // Same split: on an edit the box's OFF position has to travel, or a
    // stage_source that is switched off never comes off the lane.
    if (editing) body.stage_source = box.checked;
    else if (box.checked) body.stage_source = true;
    // Same split as stage_source: on an edit an EMPTY list has to travel, or
    // clearing every tick could never put the lane back on "any encoder".
    const picked = chosenEncoders();
    if (editing) body.encoders = picked;
    else if (picked.length) body.encoders = picked;
    const path = editing
      ? `/api/lane/${encodeURIComponent(editing)}`
      : "/api/lane";
    if (await write(path, "POST", body)) {
      // reset() puts the select back on the blank option, so the note it
      // belongs to has to go with it.
      form.reset();
      note.textContent = "";
      note.hidden = true;
      form.hidden = true;
      editing = null;
      form.elements.name.readOnly = false;
    }
  };
  document.getElementById("add-lane-open").onclick = () => {
    // Closing only when it is already open in ADD mode. Clicking this while
    // an edit is open switches to adding, which is what the label promises;
    // toggling it shut instead would leave the operator clicking twice.
    if (!form.hidden && editing === null) form.hidden = true;
    else setMode(null);
  };
}

function renderList(id, rows, build) {
  const body = document.querySelector(`#${id} tbody`);
  const keep = keepFocus(body);
  body.replaceChildren();
  for (const row of rows) body.appendChild(build(row));
  restoreFocus(body, keep);
}

// Whether the stop button is one click from firing. Module scope, because the
// button is rebuilt on nothing and the state has to outlive a poll.
let stopArmed = false;
let stopTimer = null;

// Starting a run is the opposite of stopping one, so it is one click, not two:
// the daemon answers one run at a time, and a second click while one is live
// gets a 409 the page shows. The button disables itself the moment the batch
// is reported running, which covers the normal case anyway.
function renderStart(batch) {
  const b = document.getElementById("start-run");
  b.disabled = batch.running;
  if (b.onclick) return;
  b.onclick = postStart;
}

async function postStart() {
  // No rev: starting a run is not an edit of the roster file, and the
  // roster-rev machinery in write() must not see it. The Content-Type header
  // is still required -- it is the CSRF gate every write route uses.
  try {
    const r = await fetch("/api/run/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const text = await r.text();
    if (r.ok) {
      const parsed = JSON.parse(text);
      say(`started, pid ${parsed.pid}`);
      return true;
    }
    // 409 here is the daemon saying a batch is already running, which is not
    // the operator's mistake -- say the daemon's own words instead of a
    // generic status line.
    let msg = `HTTP ${r.status}`;
    try {
      const parsed = JSON.parse(text);
      if (parsed && typeof parsed.error === "string") msg = parsed.error;
    } catch (e) { /* not JSON -- fall back to the status */ }
    say(msg);
    return false;
  } catch (e) {
    say("daemon unreachable — the run was not started");
    return false;
  }
}

function renderStop(batch) {
  const b = document.getElementById("stop-run");
  b.disabled = !batch.running;
  if (b.onclick) return;
  // Two clicks rather than confirm(): a modal dialog blocks the poll loop, and
  // this button ends a run that can last fifteen days. The armed state expires
  // by itself so a stray first click cannot sit waiting for an accidental
  // second one minutes later.
  b.onclick = async () => {
    if (!stopArmed) {
      stopArmed = true;
      b.textContent = "Stop the run — click again";
      b.classList.add("armed");
      stopTimer = setTimeout(disarmStop, 5000);
      return;
    }
    disarmStop();
    await write("/api/run/stop", "POST", {});
  };
}

function disarmStop() {
  const b = document.getElementById("stop-run");
  stopArmed = false;
  if (stopTimer) clearTimeout(stopTimer);
  stopTimer = null;
  b.textContent = "Stop the run";
  b.classList.remove("armed");
}

// What the batch did with each request. The daemon writes a request file and
// the batch answers it up to a second later, so without this the buttons look
// like they did nothing.
function renderAcks(rows) {
  const host = document.getElementById("acks");
  host.replaceChildren();
  for (const a of (rows || []).slice(-5).reverse()) {
    const li = document.createElement("li");
    li.className = a.accepted ? "ack" : "ack no";
    const when = new Date(a.at * 1000).toLocaleTimeString();
    li.textContent = `${when} · ${a.note}`;
    host.appendChild(li);
  }
  host.hidden = host.children.length === 0;
}

function renderBanners(snap) {
  const host = document.getElementById("banners");
  host.replaceChildren();
  const add = (text) => {
    const d = document.createElement("div");
    d.className = "banner";
    d.textContent = text;
    host.appendChild(d);
  };
  if (snap.roster_error) {
    add("The roster will not parse, so every lane parks with no other " +
        "warning: " + snap.roster_error);
  }
  if (snap.batch_roster_error) {
    add("The batch cannot read the roster, so every lane is parked: " +
        snap.batch_roster_error);
  }
  if (snap.manifest_error) {
    add("The manifest will not parse, so the clip list and every frame total " +
        "are unavailable: " + snap.manifest_error);
  }
}

// The history link is deployment configuration, so it arrives with the
// snapshot rather than being written into the page. Only http and https are
// accepted: the value reaches an href, and a javascript: URL there would run
// on click. It comes from the operator's own command line, which was once an
// argument for not checking at all. That argument is gone: this page writes
// the roster, so script running in it can change the run.
function renderHistoryLink(url) {
  const a = document.getElementById("grafana");
  let ok = false;
  if (url) {
    try {
      const p = new URL(url, window.location.href).protocol;
      ok = p === "http:" || p === "https:";
    } catch (e) {
      ok = false;
    }
  }
  if (ok) a.href = url; else a.removeAttribute("href");
  a.hidden = !ok;
}

// `gen` is the value of writeGen when this snapshot's fetch was issued.
// The last failures drawn, so the filter checkbox can redraw this panel alone.
// Re-running apply() would be the obvious alternative and is wrong: it takes a
// write generation, and replaying an old one would let a stale snapshot
// overwrite a roster edit made since.
let lastFailures = [];
let lastExhaustedTotal = 0;
let retryAllArmed = false;
let retryAllTimer = null;

function renderFailures(failures, exhaustedTotal) {
  lastFailures = failures;
  lastExhaustedTotal = exhaustedTotal;
  const only = document.getElementById("failed-exhausted-only");
  if (only && !only.onchange) {
    only.onchange = () => renderFailures(lastFailures, lastExhaustedTotal);
  }
  const shown = only && only.checked
    ? failures.filter((f) => f.exhausted) : failures;
  // Both numbers, because they disagree and this panel showed only the first:
  // a run whose failures all retried successfully reported "0 exhausted" over
  // a list of fifty rows, which reads as a broken page rather than a healthy
  // run. The list is the last 50 attempts, exhausted or not; the count is how
  // many clips the run has actually given up on.
  document.getElementById("failed").textContent =
    `${int(exhaustedTotal)} exhausted \u00b7 ${int(shown.length)} listed`;
  renderRetryAll(exhaustedTotal);

  renderList("failures", shown, (f) => {
    const tr = document.createElement("tr");
    tr.appendChild(cell(
      `${esc(f.src.split("/").pop())}
       <span class="sub">${esc(f.lane)} · attempt ${f.attempts}` +
      (f.exhausted ? ", no retries left" : ", retried") + `</span>
       <div class="reason">${esc(f.reason)}</div>`));
    const actions = cell("", "actions");
    if (f.exhausted) {
      // Only when the run has given up. A clip with an attempt left is
      // already coming back on its own, and a button that re-queues it would
      // look like it did something.
      const rt = document.createElement("button");
      rt.type = "button";
      rt.className = "rm";
      rt.dataset.key = f.src;
      rt.dataset.control = "retry";
      rt.textContent = "retry";
      rt.title = "Put this clip back on the queue and reset its failure "
               + "count, in this run and the next.";
      rt.onclick = () =>
        write(`/api/clip/${encodeURIComponent(f.src)}/retry`, "POST", {});
      actions.appendChild(rt);
    }
    tr.appendChild(actions);
    return tr;
  });
}

// Bulk, and it acts on every abandoned clip in state.jsonl rather than on the
// rows this panel happens to carry -- the preview is capped, so a button driven
// from the rendered rows would quietly do less than its own label says.
function renderRetryAll(exhaustedTotal) {
  const b = document.getElementById("retry-all");
  if (!b) return;
  b.disabled = !exhaustedTotal;
  b.title = exhaustedTotal
    ? `Put all ${exhaustedTotal} clip(s) that are out of attempts back on the `
      + `queue and reset their failure counts. Takes effect on the next start `
      + `if no run is going.`
    : "No clip is out of attempts.";
  if (!retryAllArmed) b.textContent = "Retry all";
  if (b.onclick) return;
  // Two clicks rather than confirm(), for the same two reasons as #stop-run: a
  // modal blocks the poll loop, and this one re-queues work measured in days.
  // The armed state expires so a stray first click cannot wait for an
  // accidental second one minutes later.
  b.onclick = async () => {
    if (!retryAllArmed) {
      retryAllArmed = true;
      // lastExhaustedTotal, not a captured argument: this handler is installed
      // once and the count moves under it every poll.
      b.textContent = `Retry ${int(lastExhaustedTotal)} — click again`;
      b.classList.add("armed");
      retryAllTimer = setTimeout(disarmRetryAll, 5000);
      return;
    }
    disarmRetryAll();
    await write("/api/clips/retry-all", "POST", {});
  };
}

function disarmRetryAll() {
  const b = document.getElementById("retry-all");
  retryAllArmed = false;
  if (retryAllTimer) clearTimeout(retryAllTimer);
  retryAllTimer = null;
  if (b) {
    b.textContent = "Retry all";
    b.classList.remove("armed");
  }
}

function apply(snap, gen) {
  // Two conditions, and both are needed. `writing` covers a write still in
  // flight; `gen === writeGen` covers a write that started and finished while
  // this snapshot was on the wire, whose result is newer than anything this
  // snapshot can know. Either way the revision in hand is the better one.
  if (!writing && gen === writeGen) rosterRev = snap.roster_rev;
  renderHistoryLink(snap.grafana_url);
  renderBanners(snap);
  renderTotals(snap.totals);
  buildAddForm();
  renderLp(snap.encode);
  renderLanes(snap.lanes);
  // Before renderHosts, so the lane form's allowlist boxes and the host table
  // are built from the same snapshot.
  encoderNames = ((snap.encode && snap.encode.hosts) || []).map((h) => h.name);
  buildHostForm();
  renderHosts(snap.encode, snap.encoder_rates);
  buildJobForm();
  renderJobs(snap.jobs);
  renderStart(snap.batch);
  renderStop(snap.batch);
  renderAcks(snap.acks);

  renderList("queue", snap.queue, (q) => {
    const tr = document.createElement("tr");
    // Both job types share one queue, so a row has to say which it is or a
    // folder of encode jobs is indistinguishable from archive clips.
    tr.appendChild(cell(esc(q.src) +
      (q.denoise === false ? ' <span class="tag encode">encode</span>' : "")));
    tr.appendChild(cell(total(q.frames) + " fr", "num"));
    return tr;
  });

  renderFailures(snap.failures, snap.totals.failed);

  // lp_level is null when the pool does not agree on one, and there is then no
  // single --lp to name. Saying nothing beats naming one host's level as if it
  // were the pool's.
  const enc = snap.encode
    ? ` · ${snap.encode.slots} encode slots`
      + (snap.encode.lp_level === null ? "" : ` at --lp ${snap.encode.lp_level}`)
    : "";
  document.getElementById("stamp").textContent =
    (snap.batch.running ? `batch running, pid ${snap.batch.pid}`
                        : "batch not running") + enc +
    ` · updated ${new Date().toLocaleTimeString()}`;
}

async function poll() {
  const stamp = document.getElementById("stamp");
  let snap = null;
  // Read before the fetch is issued, not after it resolves: that is what makes
  // it say "no write has completed since this request left".
  const gen = writeGen;

  // Fetching and rendering are caught separately on purpose. Wrapping both in
  // one try makes a renderer that throws -- a field this page has not been
  // taught about yet -- report "daemon unreachable", so the page freezes on
  // stale data and blames a daemon that is answering perfectly. Sending someone
  // to the wrong machine is worse than saying nothing.
  try {
    const r = await fetch("/api/status", { cache: "no-store" });
    if (r.ok) snap = await r.json();
    else stamp.textContent = `HTTP ${r.status}`;
  } catch (e) {
    // A daemon restart mid-run is expected and must not need a page reload.
    stamp.textContent = "daemon unreachable — retrying";
  }

  if (snap) {
    try {
      apply(snap, gen);
    } catch (e) {
      // The data arrived; this page could not draw it. Say so, and keep
      // polling: the next snapshot may be renderable.
      stamp.textContent = `page cannot render this snapshot: ${e.message}`;
    }
  }
  setTimeout(poll, POLL_MS);
}

poll();
