"""Design system for Plant Doctor: deep forest green, off-white, muted sage."""

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');

:root, .gradio-container {
  --pd-forest: #1f4d2e;
  --pd-forest-dark: #163a22;
  --pd-sage: #8aa88f;
  --pd-sage-light: #d9e5db;
  --pd-sage-pale: #eef3ee;
  --pd-cream: #f7f6f1;
  --pd-ink: #1c2420;
  --pd-muted: #5b6b5e;
  --pd-amber: #b7791f;
  --pd-amber-pale: #fbf1dc;
  --pd-rust: #a4452d;
  --pd-rust-pale: #f7e4de;
  --pd-radius: 16px;
  --pd-shadow: 0 1px 2px rgba(28,36,32,.04), 0 8px 24px rgba(28,36,32,.06);
  --body-background-fill: var(--pd-cream);
  --background-fill-primary: #ffffff;
  --background-fill-secondary: var(--pd-sage-pale);
  --block-background-fill: #ffffff;
  --block-border-color: var(--pd-sage-light);
  --block-radius: var(--pd-radius);
  --block-shadow: var(--pd-shadow);
  --button-primary-background-fill: var(--pd-forest);
  --button-primary-background-fill-hover: var(--pd-forest-dark);
  --button-primary-text-color: #ffffff;
  --button-primary-border-color: var(--pd-forest);
  --button-secondary-background-fill: #ffffff;
  --button-secondary-background-fill-hover: var(--pd-sage-pale);
  --button-secondary-text-color: var(--pd-forest);
  --button-secondary-border-color: var(--pd-sage-light);
  --button-large-radius: 12px; --button-small-radius: 10px;
  --body-text-color: var(--pd-ink);
  --body-text-color-subdued: var(--pd-muted);
  --border-color-primary: var(--pd-sage-light);
  --color-accent: var(--pd-forest);
  --color-accent-soft: var(--pd-sage-pale);
  --link-text-color: var(--pd-forest);
  --input-background-fill: #ffffff;
  --checkbox-background-color-selected: var(--pd-forest);
  --slider-color: var(--pd-forest);
  --font: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --body-text-size: 15px;
}
.gradio-container { font-family: var(--font) !important; background: var(--pd-cream) !important; max-width: 1240px !important; margin: 0 auto !important; }
footer { display: none !important; }
.gradio-container .tabs > .tab-nav, .gradio-container .tab-nav { border-bottom: 1px solid var(--pd-sage-light) !important; gap: 4px; }
.gradio-container .tab-nav button { font-weight: 600 !important; color: var(--pd-muted) !important; border: none !important; border-radius: 10px 10px 0 0 !important; padding: 10px 16px !important; }
.gradio-container .tab-nav button.selected { color: var(--pd-forest) !important; background: #fff !important; box-shadow: inset 0 -2px 0 var(--pd-forest); }
button.primary { font-weight: 600 !important; letter-spacing: .01em; box-shadow: 0 6px 16px rgba(31,77,46,.18); }
button.secondary { font-weight: 600 !important; }
.pd-card { background: #fff; border: 1px solid var(--pd-sage-light); border-radius: var(--pd-radius); box-shadow: var(--pd-shadow); padding: 22px 24px; }
.pd-card + .pd-card { margin-top: 14px; }

/* ---- Masthead ---- */
.pd-masthead { display:flex; align-items:center; justify-content:space-between; gap:16px; padding: 18px 4px 10px; }
.pd-brand { display:flex; align-items:center; gap:12px; }
.pd-brand svg { width:42px; height:42px; }
.pd-brand .name { font-family: 'Fraunces', Georgia, serif; font-size: 24px; font-weight: 600; color: var(--pd-forest); letter-spacing:-.01em; }
.pd-brand .tag { font-size: 12.5px; color: var(--pd-muted); margin-top: -2px; }
.pd-status { font-size: 12.5px; color: var(--pd-muted); display:flex; align-items:center; gap:8px; }
.pd-status .dot { width:9px; height:9px; border-radius:50%; background:#2f9e5b; box-shadow:0 0 0 3px rgba(47,158,91,.18); }
.pd-status .dot.err { background: var(--pd-rust); box-shadow:0 0 0 3px rgba(164,69,45,.18); }

/* ---- Hero ---- */
.pd-hero { background: linear-gradient(135deg, var(--pd-forest) 0%, #2c6a3e 60%, #3f7d50 100%); color:#fff; border-radius: 22px; padding: 46px 48px; box-shadow: 0 16px 40px rgba(31,77,46,.22); position:relative; overflow:hidden; }
.pd-hero:before { content:""; position:absolute; right:-80px; top:-80px; width:360px; height:360px; border-radius:50%; background: radial-gradient(circle, rgba(255,255,255,.14), rgba(255,255,255,0) 70%); }
.pd-hero .eyebrow { text-transform: uppercase; letter-spacing:.14em; font-size:11.5px; font-weight:600; opacity:.8; color:#fff; }
.pd-hero h1 { font-family:'Fraunces', Georgia, serif; font-size: 40px; line-height:1.1; margin: 10px 0 12px; font-weight: 600; letter-spacing:-.015em; color:#fff !important; }
.pd-hero p { font-size: 16.5px; line-height:1.6; max-width: 620px; opacity:.94; margin:0; color:#fff; }
.pd-hero .chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:22px; }
.pd-hero .chip { background: rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.22); border-radius: 999px; padding: 6px 12px; font-size: 12.5px; font-weight:500; }
.pd-steps { display:grid; grid-template-columns: repeat(3, 1fr); gap:14px; margin-top: 18px; }
.pd-step { background:#fff; border:1px solid var(--pd-sage-light); border-radius: var(--pd-radius); padding: 20px 20px 18px; box-shadow: var(--pd-shadow); }
.pd-step .num { width:32px; height:32px; border-radius: 10px; background: var(--pd-sage-pale); color: var(--pd-forest); display:flex; align-items:center; justify-content:center; font-weight:700; font-size:14px; margin-bottom: 12px; }
.pd-step h3 { margin:0 0 6px; font-size: 15.5px; font-weight: 600; color: var(--pd-ink); }
.pd-step p { margin:0; color: var(--pd-muted); font-size: 13.5px; line-height: 1.55; }
.pd-crops { display:grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 10px; }
.pd-crop { border:1px solid var(--pd-sage-light); border-radius: 12px; padding: 12px 14px; background: #fff; }
.pd-crop b { display:block; color: var(--pd-forest); font-size: 14px; margin-bottom: 4px; }
.pd-crop span { font-size: 12.5px; color: var(--pd-muted); line-height: 1.45; }
.pd-disclaimer { border-left: 4px solid var(--pd-amber); background: var(--pd-amber-pale); border-radius: 12px; padding: 14px 18px; font-size: 13.5px; color:#5a4310; line-height: 1.5; }
.pd-section-title { font-family:'Fraunces', Georgia, serif; font-size: 22px; color: var(--pd-forest); font-weight: 600; margin: 26px 0 10px; letter-spacing:-.01em; }
.pd-sub { color: var(--pd-muted); font-size: 14px; margin: -4px 0 14px; }

/* ---- Results ---- */
.pd-result { display:flex; gap: 22px; align-items: stretch; flex-wrap: wrap; }
.pd-result .main { flex: 1 1 320px; }
.pd-result .eyebrow { text-transform: uppercase; letter-spacing:.12em; font-size: 11px; font-weight: 600; color: var(--pd-muted); }
.pd-result h2 { font-family:'Fraunces', Georgia, serif; font-size: 30px; line-height:1.15; margin: 6px 0 4px; color: var(--pd-ink); font-weight: 600; letter-spacing:-.01em; }
.pd-result .crop { color: var(--pd-muted); font-size: 15px; }
.pd-badge { display:inline-flex; align-items:center; gap:7px; border-radius: 999px; padding: 6px 12px; font-size: 12.5px; font-weight: 600; margin-top: 12px; }
.pd-badge.ok { background: #e3f1e7; color: #1d6b38; }
.pd-badge.warn { background: var(--pd-amber-pale); color: #8a5a10; }
.pd-badge.bad { background: var(--pd-rust-pale); color: var(--pd-rust); }
.pd-badge.neutral { background: var(--pd-sage-pale); color: var(--pd-forest); }
.pd-conf { flex: 0 0 200px; display:flex; flex-direction:column; align-items:center; justify-content:center; background: var(--pd-sage-pale); border-radius: 16px; padding: 16px; }
.pd-conf .val { font-size: 40px; font-weight: 700; color: var(--pd-forest); letter-spacing: -.02em; line-height:1; }
.pd-conf .lbl { font-size: 12px; color: var(--pd-muted); margin-top: 6px; text-align:center; }
.pd-ring { width: 110px; height: 110px; border-radius: 50%; display:flex; align-items:center; justify-content:center; margin-bottom: 8px; }
.pd-ring .inner { width: 90px; height: 90px; border-radius: 50%; background: var(--pd-sage-pale); display:flex; align-items:center; justify-content:center; }
.pd-bars { margin-top: 10px; }
.pd-bar { display:grid; grid-template-columns: 1fr 120px 56px; align-items:center; gap: 12px; padding: 9px 0; border-top: 1px solid var(--pd-sage-pale); font-size: 14px; }
.pd-bar:first-child { border-top: none; }
.pd-bar .name b { color: var(--pd-ink); }
.pd-bar .name span { color: var(--pd-muted); font-size: 12.5px; display:block; }
.pd-bar .track { height: 8px; border-radius: 999px; background: var(--pd-sage-pale); overflow:hidden; }
.pd-bar .fill { height: 100%; border-radius: 999px; background: var(--pd-forest); }
.pd-bar.alt .fill { background: var(--pd-sage); }
.pd-bar .pct { text-align:right; font-weight: 600; font-variant-numeric: tabular-nums; }
.pd-abstain { background: var(--pd-amber-pale); border: 1px solid #ecd9ad; border-radius: var(--pd-radius); padding: 22px 24px; }
.pd-abstain h2 { font-family:'Fraunces', Georgia, serif; font-size: 24px; margin: 0 0 8px; color: #6b4a0b; }
.pd-abstain p { margin: 0 0 10px; color: #5a4310; font-size: 14.5px; line-height:1.55; }
.pd-abstain ul { margin: 0; padding-left: 18px; color: #5a4310; font-size: 13.5px; }
.pd-note { font-size: 12.5px; color: var(--pd-muted); margin-top: 10px; line-height: 1.5; }
.pd-info h3 { font-size: 13px; text-transform: uppercase; letter-spacing: .1em; color: var(--pd-muted); margin: 18px 0 6px; font-weight: 600; }
.pd-info h3:first-of-type { margin-top: 10px; }
.pd-info p { margin: 0 0 6px; line-height: 1.6; font-size: 14.5px; }
.pd-info ul { margin: 0; padding-left: 18px; line-height: 1.6; font-size: 14px; }
.pd-info .sci { font-style: italic; color: var(--pd-muted); font-size: 13.5px; }
.pd-info .refs { font-size: 12.5px; color: var(--pd-muted); margin-top: 14px; }
.pd-info .refs a { color: var(--pd-forest); }
.pd-kv { display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-top: 10px; }
.pd-kv div { background: var(--pd-sage-pale); border-radius: 12px; padding: 10px 12px; }
.pd-kv b { display:block; font-size: 18px; color: var(--pd-forest); }
.pd-kv span { font-size: 12px; color: var(--pd-muted); }
.pd-compare { display:grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.pd-compare .col { border: 1px solid var(--pd-sage-light); border-radius: 12px; padding: 14px 16px; }
.pd-compare .col h4 { margin: 0 0 6px; font-size: 12px; text-transform: uppercase; letter-spacing:.1em; color: var(--pd-muted); }
.pd-compare .col .big { font-size: 17px; font-weight: 600; color: var(--pd-ink); }
.pd-compare .col .pct { color: var(--pd-forest); font-weight: 700; }
.pd-warnlist { margin: 8px 0 0; padding-left: 18px; font-size: 13px; color: #8a5a10; }

/* ---- History ---- */
.pd-history { display:flex; flex-direction:column; gap: 8px; }
.pd-hist { display:grid; grid-template-columns: 56px 1fr auto; gap: 12px; align-items:center; background:#fff; border:1px solid var(--pd-sage-light); border-radius: 12px; padding: 8px 10px; }
.pd-hist img { width: 56px; height: 56px; border-radius: 9px; object-fit: cover; }
.pd-hist .t { font-size: 14px; font-weight: 600; color: var(--pd-ink); }
.pd-hist .m { font-size: 12.5px; color: var(--pd-muted); }
.pd-hist .c { font-size: 13px; font-weight: 600; color: var(--pd-forest); text-align:right; }
.pd-hist .c.abst { color: var(--pd-amber); }
.pd-empty { color: var(--pd-muted); font-size: 14px; padding: 18px; text-align:center; border: 1px dashed var(--pd-sage-light); border-radius: 12px; }

/* ---- About ---- */
.pd-about h2 { font-family:'Fraunces', Georgia, serif; font-size: 20px; color: var(--pd-forest); margin: 18px 0 6px; }
.pd-about p, .pd-about li { font-size: 14.5px; line-height: 1.65; }
.pd-about table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin: 8px 0; }
.pd-about th, .pd-about td { border-bottom: 1px solid var(--pd-sage-light); padding: 7px 8px; text-align: left; }
.pd-about th { color: var(--pd-muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
.pd-about code { background: var(--pd-sage-pale); padding: 1px 5px; border-radius: 5px; font-size: 12.5px; }

/* ---- Components ---- */
.gradio-container .block.pd-upload { border: 2px dashed var(--pd-sage) !important; background: #fff !important; }
.gradio-container label span { color: var(--pd-muted) !important; font-weight: 600 !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: .08em; }
.gradio-container .progress-text, .gradio-container .progress-level { color: var(--pd-forest) !important; }
.gradio-container .error { border-color: var(--pd-rust) !important; }
.pd-error { background: var(--pd-rust-pale); border: 1px solid #e3b7ab; color: #7a2f1d; border-radius: 12px; padding: 14px 16px; font-size: 14px; }

/* ---- Responsive ---- */
@media (max-width: 900px) {
  .pd-hero { padding: 32px 26px; }
  .pd-hero h1 { font-size: 30px; }
  .pd-steps { grid-template-columns: 1fr; }
  .pd-compare { grid-template-columns: 1fr; }
  .pd-conf { flex-basis: 100%; }
  .pd-bar { grid-template-columns: 1fr 80px 52px; }
  .pd-masthead { flex-direction: column; align-items: flex-start; }
}
@media (max-width: 600px) {
  .pd-card { padding: 16px; }
  .pd-result h2 { font-size: 24px; }
  .pd-hero h1 { font-size: 26px; }
  .pd-hist { grid-template-columns: 48px 1fr; }
  .pd-hist .c { grid-column: 2; text-align: left; }
}
"""

LOGO_SVG = """<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<rect x="2" y="2" width="44" height="44" rx="12" fill="#1f4d2e"/>
<path d="M24 38c0-9 3-17 14-22-1 11-5 19-14 22Z" fill="#8fc49c"/>
<path d="M24 38c0-9-3-17-14-22 1 11 5 19 14 22Z" fill="#d9e5db"/>
<path d="M24 38V22" stroke="#163a22" stroke-width="2" stroke-linecap="round"/>
<path d="M30 12h6M33 9v6" stroke="#fff" stroke-width="2.4" stroke-linecap="round"/>
</svg>"""

# Gradio scopes `css=` rules under its own container, so page-level rules go in the <head>.
HEAD_HTML = """<meta name="theme-color" content="#1f4d2e">
<style>
@media (max-width: 600px) {
  html, body { max-width: 100vw; overflow-x: hidden; }
  gradio-app .gradio-container .main.app { padding-left: 12px !important; padding-right: 12px !important; }
}
</style>"""
