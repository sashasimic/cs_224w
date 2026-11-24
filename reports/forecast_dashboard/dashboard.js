"use strict";

const content = document.querySelector("#content");
const button = document.querySelector("#refresh");
const pages = {live: "Live results", backtest: "Backtest", feed: "Feed readiness"};
const models = {
  weekly: "Weekly persistence", pooled_xgb: "Pooled XGBoost", zone_xgb: "Per-zone XGBoost",
  lear: "LEAR", factor: "Market factor", sequence: "Pooled sequence", blend: "Blend",
  timesfm_joint: "TimesFM joint", timesfm_independent: "TimesFM independent",
};
const publications = {
  unknown_at_cutoff: "Unknown at cutoff", published_at_cutoff: "Published at cutoff",
  publication_unverified: "Publication unverified",
};
const levels = {
  overall: "All zones", country_or_market: "Country or market",
  zone: "Bidding zone", equal_weight_country_or_market: "Equal-weight markets",
};
const states = {success: "Succeeded", fallback: "Used fallback", late: "Late", missing: "Missing",
  pending: "Pending", running: "Running", ok: "Available", empty: "No data",
  fx_unavailable: "FX unavailable", http_error: "HTTP error", invalid_or_unavailable: "Unavailable"};
const scopes = {core: "Core models", private: "TimesFM research"};
const study = {pilot: "Pilot", experiment: "14-day experiment"};
const zone = {timeZone: "Europe/Brussels"};
const dayShort = new Intl.DateTimeFormat("en-GB", {...zone, weekday: "short", day: "numeric", month: "short"});
const dayLong = new Intl.DateTimeFormat("en-GB", {...zone, day: "numeric", month: "short", year: "numeric"});
const time = new Intl.DateTimeFormat("en-GB", {...zone, hour: "2-digit", minute: "2-digit", hourCycle: "h23"});
const integer = new Intl.NumberFormat("en-GB");
const state = {phase: "pilot", scope: "core", publication: "unknown_at_cutoff", level: "overall",
  group: "all", backtestScope: "core", fold: "", query: ""};
let data;
let error = false;
let loading = false;

function el(tag, attrs = {}, ...children) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key.startsWith("on")) element.addEventListener(key.slice(2), value);
    else element.setAttribute(key, value);
  }
  element.append(...children.filter(child => child != null && child !== false));
  return element;
}

function heading(title, detail, badge) {
  return el("div", {class: "page-head"}, el("div", {}, el("h1", {}, title), el("p", {}, detail)), badge);
}
function pill(label, color = "") {
  return el("span", {class: "pill " + color}, label);
}
function status(value) {
  const color = ["ok", "success"].includes(value) ? "good" :
    ["late", "missing", "fallback", "http_error", "invalid_or_unavailable", "fx_unavailable"].includes(value) ? "warn" : "";
  return pill(states[value] || value, color);
}
function panel(title, detail, children, action) {
  return el("section", {class: "panel"}, el("header", {class: "panel-head"},
    el("div", {}, el("h2", {}, title), detail && el("p", {}, detail)), action), ...children);
}
function empty(title, detail, link) {
  return el("div", {class: "empty"}, el("h3", {}, title), el("p", {}, detail), link);
}
function kpi(label, value, detail) {
  return el("div", {class: "kpi"}, el("p", {}, label), el("strong", {}, value),
    detail && el("p", {class: "small"}, detail));
}
function table(headers, rows, name) {
  return el("div", {class: "scroll", tabindex: "0", role: "region", "aria-label": name},
    el("table", {}, el("thead", {}, el("tr", {}, ...headers.map(h => el("th", {scope: "col"}, h)))),
      el("tbody", {}, ...rows.map(row => el("tr", {}, ...row.map(value => el("td", {}, value)))))));
}
function select(label, id, key, choices) {
  const change = e => {
    state[key] = e.target.value;
    if (key === "level") state.group = "all";
    render(id);
  };
  const menu = el("select", {id, onchange: change},
    ...Object.entries(choices).map(([value, text]) => el("option", {value}, text)));
  menu.value = state[key];
  return el("label", {for: id}, label, menu);
}
function controls(...items) {
  return el("div", {class: "controls"}, ...items);
}
function num(value, digits) {
  return value == null ? "—" : digits == null ? integer.format(value) : Number(value).toFixed(digits);
}
function pct(value) {
  return value == null ? "—" : (value * 100).toFixed(1) + "%";
}
function fraction(value, total) {
  return num(value) + " / " + num(total);
}
function skill(value) {
  const tone = value > 0 ? "positive" : value < 0 ? "negative" : "";
  return el("span", {class: tone}, value > 0 ? "+" + pct(value) : pct(value));
}
function dayText(value, format = dayShort) {
  return format.format(new Date(value + "T12:00:00Z"));
}

function timetable(phase) {
  const labels = {capture: "Collect inputs", freeze: "Freeze snapshot", commit: "Commit forecasts", outcomes: "Check outcomes"};
  return el("div", {class: "timeline"}, ...phase.days.map(day => {
    const summary = {
      capture: day.receipts ? fraction(day.receipts, data.zones.length) + " zones checked" : null,
      freeze: day.frozen ? "Snapshot frozen" : null,
      commit: day.committed ? fraction(day.committed, phase.models.length) + " jobs recorded" : null,
      outcomes: day.checked ? num(day.observed) + " prices observed" : null,
    };
    return el("article", {class: "delivery"},
      el("div", {}, el("h3", {}, dayText(day.date)), el("p", {class: "small"}, "Delivery day")),
      el("div", {class: "steps"}, ...Object.entries(labels).map(([key, label]) => {
        const stamp = new Date(day.times[key]);
        const expired = stamp < new Date(data.updated);
        return el("div", {class: "step" + (summary[key] ? " done" : "")}, el("b", {}, label),
          el("time", {datetime: day.times[key]}, time.format(stamp)),
          el("span", {}, summary[key] || (expired ? "Awaiting record" : "Scheduled")));
      })));
  }));
}

function liveScores(phase, report, name) {
  const actions = el("div", {class: "toolbar-actions"},
    select("Model scope", "live-scope", "scope", scopes),
    report && el("a", {class: "button", href: "/data/" + name + ".csv", download: ""}, "Download CSV"));
  let children;
  if (!report) {
    const first = phase && new Date(phase.days[0].times.outcomes);
    const when = first && first > new Date(data.updated)
      ? "The first outcome check is scheduled for " + dayLong.format(first) + " at " + time.format(first) + " Brussels time."
      : "Scores appear here after the outcome service writes its first report.";
    children = [empty("Awaiting scored outcomes", when, el("a", {href: "#backtest"}, "Explore historical backtest"))];
  } else {
    const filtered = report.scores.filter(row => row.publication === state.publication && row.level === state.level);
    const groups = [...new Set(filtered.map(row => row.group))].sort();
    if (!groups.includes(state.group)) state.group = groups[0] || "all";
    const choices = controls(
      select("Publication status", "publication", "publication", publications),
      select("Geography", "level", "level", levels),
      ["country_or_market", "zone"].includes(state.level) && groups.length > 0 && select("Market", "group", "group", Object.fromEntries(groups.map(v => [v, v]))));
    const rows = filtered.filter(row => row.group === state.group).sort((a, b) => (a.mae ?? Infinity) - (b.mae ?? Infinity));
    children = [choices, !rows.some(row => row.mae != null) &&
      empty("No scored hours in this view", "Try a different publication status or wait for published outcomes."),
    rows.length ? table(["Model", "MAE · €/MWh", "Matched skill", "Forecast coverage", "Outcome coverage", "Fallback hours", "Median worker · s"],
      rows.map(r => [models[r.model] || r.model, num(r.mae, 2), skill(r.skill), pct(r.coverage),
        pct(r.outcome_coverage), num(r.fallback), num(report.operations?.[r.model]?.worker_seconds?.median, 1)]), "Market scores") : null];
  }
  return panel("Market scores", "Skill uses the same scored hours for the model and weekly persistence.", children, actions);
}

function live() {
  const phase = data.phases.find(p => p.name === state.phase);
  const name = state.phase + (state.scope === "private" ? "-private" : "");
  const report = data.reports[name];
  const days = phase?.days || [];
  const checked = days.filter(day => day.checked).length;
  const jobs = data.jobs.filter(job => days.some(day => day.date === job.day));
  const badge = report?.complete ? pill("Reporting complete", "good") :
    jobs.some(j => j.status === "running") ? pill("Forecasts running", "good") :
    checked ? pill("Collecting results", "good") : phase ? pill(study[state.phase] + " scheduled", "good") : null;
  const tabs = el("div", {class: "tabs", role: "group", "aria-label": "Study phase"},
    ...Object.entries(study).map(([key, label]) => el("button", {
      id: "tab-" + key, type: "button", "aria-pressed": key === state.phase,
      onclick: () => { state.phase = key; render("tab-" + key); },
    }, label)));
  const jobsPanel = jobs.length ? panel("Forecast jobs", null, [
    table(["Delivery", "Model", "Status", "Committed", "Detail"], jobs.map(j => [
      dayText(j.day), models[j.model] || j.model, status(j.status),
      j.committed ? time.format(new Date(j.committed)) : "—", j.detail || "—",
    ]), "Forecast jobs"),
  ]) : null;
  return [heading("Live results", "Track day-ahead forecasts and scored prices across European bidding zones.", badge), tabs,
    el("div", {class: "kpis"},
      kpi("Bidding zones", num(data.zones.length), "Benchmark roster"),
      kpi("Delivery days checked", phase ? fraction(checked, days.length) : "—", "Outcome checks"),
      kpi("Forecast jobs recorded", phase ? fraction(jobs.filter(j => j.committed).length, days.length * phase.models.length) : "—", "All scheduled models")),
    panel("Forecast timetable", "Schedules use Brussels time on the day before delivery.",
      [phase ? timetable(phase) : empty(state.phase === "pilot" ? "Pilot not scheduled" : "Experiment awaits the pilot",
        state.phase === "pilot" ? "The timetable appears after the pilot is armed." : "The schedule opens automatically after both pilot days pass.")]),
    liveScores(phase, report, name), jobsPanel];
}

function backtest() {
  const core = data.backtests.backtest;
  const privateScores = data.backtests["backtest-private"]?.scores || {};
  if (!core?.folds?.length) return [
    heading("Historical backtest", "Compare models across archived evaluation windows."),
    panel("Model performance", null, [empty("Backtest unavailable", "A report will appear here when historical evaluation produces results.")]),
  ];
  const folds = core.folds;
  if (!folds.some(f => f.start === state.fold)) state.fold = folds.at(-1).start;
  const fold = folds.find(item => item.start === state.fold);
  const source = state.backtestScope === "core" ? fold.scores :
    Object.fromEntries(Object.entries(privateScores).map(([model, reports]) => [model, reports[state.fold]]));
  const results = Object.entries(source).filter(([, row]) => row)
    .sort(([, a], [, b]) => (a.matched_mae ?? Infinity) - (b.matched_mae ?? Infinity));
  const best = results[0];
  const highest = Math.max(1, ...results.map(([, row]) => row.matched_mae || 0));
  const rows = results.map(([model, row]) => {
    const step = Math.round(10 * (row.matched_mae || 0) / highest);
    const chart = el("span", {class: "metric"}, el("span", {class: "bar n" + step, "aria-hidden": "true"}, el("span")),
      num(row.matched_mae, 2));
    return [models[model] || model, chart, skill(row.skill), num(row.matched)];
  });
  const days = core.validation_days_per_fold;
  const end = new Date(new Date(fold.start + "T12:00:00Z").getTime() + (days - 1) * 86400000);
  return [
    heading("Historical backtest", "Compare archived evaluation windows with unverified publication timing.", pill("Exploratory")),
    el("div", {class: "kpis"},
      kpi("Matched baseline MAE", num(best?.[1].weekly_mae, 2), "EUR/MWh"),
      kpi("Lowest matched MAE", num(best?.[1].matched_mae, 2), best && models[best[0]]),
      kpi("Evaluation window", num(days) + " days", dayText(fold.start, dayLong) + " to " + dayLong.format(end))),
    panel("Model performance", "Matched MAE and skill use the same hours as weekly persistence.",
      [controls(select("Evaluation window", "fold", "fold", Object.fromEntries(folds.map(f => [f.start, dayText(f.start, dayLong)]))),
        select("Model scope", "backtest-scope", "backtestScope", scopes)),
      rows.length ? table(["Model", "Matched MAE · €/MWh", "Matched skill", "Matched hours"], rows, "Historical scores") :
        empty("No research results for this window", "Choose the core models to see available scores.")]),
  ];
}

function feed() {
  const check = data.preflight;
  const zones = data.zones.map(z => ({...z, ...check?.zones?.[z.zone]}));
  const results = el("div");
  function update() {
    const query = state.query.trim().toLowerCase();
    const filtered = zones.filter(z => [z.name, z.zone, z.group, z.eic].join(" ").toLowerCase().includes(query));
    const rows = filtered.map(z => [z.zone, z.name, z.eic, z.status ? status(z.status) : "—",
      num(z.http), num(z.complete_hours)]);
    results.replaceChildren(rows.length ?
      table(["Zone", "Market", "ENTSO-E EIC", "Status", "HTTP", "Usable price hours"], rows, "Feed readiness by zone") :
      empty("No matching zones", "Try a zone name, market, or EIC."));
  }
  const search = el("input", {id: "feed-query", type: "search", placeholder: "Zone, market, or EIC", oninput: e => {
    state.query = e.target.value;
    update();
  }});
  search.value = state.query;
  update();
  const stamp = check && new Date(check.checked);
  return [heading("Feed readiness", "Check market identifiers and usable ENTSO-E price coverage.", check &&
    pill(check.passed ? "Preflight passed" : "Preflight failed", check.passed ? "good" : "warn")),
    el("div", {class: "kpis"},
      kpi("Zones checked", check ? fraction(Object.keys(check.zones).length, zones.length) : "—"),
      kpi("Zones with prices", check ? fraction(zones.filter(z => z.complete_hours > 0).length, zones.length) : "—"),
      kpi("Last preflight", stamp ? time.format(stamp) : "—", stamp && dayLong.format(stamp))),
    panel("Bidding zones", null, [controls(el("label", {for: "feed-query"}, "Find a zone", search)), results])];
}

function render(focus) {
  const requested = location.hash.slice(1);
  const page = Object.hasOwn(pages, requested) ? requested : "live";
  for (const link of document.querySelectorAll("nav a")) {
    if (link.dataset.page === page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  const views = {live, backtest, feed};
  const notice = error ? el("p", {class: "banner", role: "status"}, data ?
    "Refresh failed. The last available data is shown." : "The forecast data is temporarily unavailable. Use Refresh to try again.") : null;
  content.replaceChildren(...[notice, ...(data ? views[page]() : [])].filter(Boolean));
  if (focus) document.getElementById(focus)?.focus({preventScroll: true});
}

async function refresh(manual = false) {
  if (loading) return;
  loading = true;
  button.setAttribute("aria-busy", "true");
  try {
    const response = await fetch("/api/dashboard", {cache: "no-store"});
    if (!response.ok) throw new Error("Dashboard request failed");
    data = await response.json();
    error = false;
    const updated = new Date(data.updated);
    document.querySelector("#dashboard-date").textContent = dayLong.format(updated);
    document.querySelector("#updated").textContent = "Updated " + time.format(updated) + " Brussels";
  } catch {
    error = true;
  } finally {
    loading = false;
    button.removeAttribute("aria-busy");
    const focused = document.activeElement;
    if (manual || error || !content.contains(focused) || focused === content) render();
  }
}

window.addEventListener("hashchange", () => { render(); content.focus({preventScroll: true}); });
button.addEventListener("click", () => refresh(true));
refresh();
window.setInterval(refresh, 30000);
