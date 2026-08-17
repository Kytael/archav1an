"use strict";

const POLL_MS = 2000;

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
  document.getElementById("failed").textContent = int(t.failed) + " exhausted";
}

function renderLanes(lanes) {
  const body = document.querySelector("#lanes tbody");
  body.replaceChildren();
  for (const l of lanes) {
    const tr = document.createElement("tr");

    // Disabled on purpose in part 1: the switch does nothing until the
    // scheduler can give a lane added mid-run a worker.
    tr.appendChild(cell(`<span class="sw ${l.enabled ? "on" : ""}"
      title="Read-only until part 2"></span>`));

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
    body.appendChild(tr);
  }
}

function renderList(id, rows, build) {
  const body = document.querySelector(`#${id} tbody`);
  body.replaceChildren();
  for (const row of rows) body.appendChild(build(row));
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
  if (snap.manifest_error) {
    add("The manifest will not parse, so the clip list and every frame total " +
        "are unavailable: " + snap.manifest_error);
  }
}

// The history link is deployment configuration, so it arrives with the
// snapshot rather than being written into the page. Only http and https are
// accepted: the value reaches an href, and a javascript: URL there would run
// on click. It comes from the operator's own command line today, which is an
// argument for not checking that no longer holds the day part 2 lets the page
// write anything back.
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

function apply(snap) {
  renderHistoryLink(snap.grafana_url);
  renderBanners(snap);
  renderTotals(snap.totals);
  renderLanes(snap.lanes);

  renderList("queue", snap.queue, (q) => {
    const tr = document.createElement("tr");
    tr.appendChild(cell(esc(q.src)));
    tr.appendChild(cell(total(q.frames) + " fr", "num"));
    return tr;
  });

  renderList("failures", snap.failures, (f) => {
    const tr = document.createElement("tr");
    tr.appendChild(cell(
      `${esc(f.src.split("/").pop())}
       <span class="sub">${esc(f.lane)} · attempt ${f.attempts}` +
      (f.exhausted ? ", no retries left" : ", retried") + `</span>
       <div class="reason">${esc(f.reason)}</div>`));
    return tr;
  });

  const enc = snap.encode
    ? ` · ${snap.encode.slots} encode slots at --lp ${snap.encode.lp_level}`
    : "";
  document.getElementById("stamp").textContent =
    (snap.batch.running ? `batch running, pid ${snap.batch.pid}`
                        : "batch not running") + enc +
    ` · updated ${new Date().toLocaleTimeString()}`;
}

async function poll() {
  const stamp = document.getElementById("stamp");
  let snap = null;

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
      apply(snap);
    } catch (e) {
      // The data arrived; this page could not draw it. Say so, and keep
      // polling: the next snapshot may be renderable.
      stamp.textContent = `page cannot render this snapshot: ${e.message}`;
    }
  }
  setTimeout(poll, POLL_MS);
}

poll();
