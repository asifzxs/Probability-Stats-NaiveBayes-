"""
SPAMSHIELD - Probabilistic SMS Spam Detection (web app)
Run:  streamlit run app.py
"""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import spam_core as core

st.set_page_config(page_title="SpamShield · Bayesian SMS Detector", page_icon="🛡️", layout="wide")

PINK, CYAN, LIME, AMBER, VIOLET = "#ff2e88", "#00e5ff", "#7CFF6B", "#ffb020", "#a66bff"
BG, PANEL = "#0a0b1a", "#141633"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Inter:wght@400;600&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: radial-gradient(1200px 600px at 10% -10%, #2a0d4a 0%, transparent 60%),
                      radial-gradient(900px 500px at 110% 10%, #06324a 0%, transparent 55%), {BG}; }}
.hero {{ text-align:center; padding: 18px 0 6px; }}
.hero h1 {{ font-family:'Orbitron',sans-serif; font-weight:800; font-size: 3.1rem; margin:0;
  background: linear-gradient(90deg,{PINK},{VIOLET},{CYAN}); -webkit-background-clip:text; color:transparent;
  letter-spacing: 3px; filter: drop-shadow(0 0 18px rgba(255,46,136,.35)); }}
.hero p {{ color:#a9abd6; margin: 4px 0 0; font-size: 1.02rem; }}
.chip {{ display:inline-block; padding:3px 10px; margin:6px 4px 0; border-radius:999px; font-size:.78rem;
  border:1px solid #2c2f66; background:#121436; color:#cfd1ff; }}
.card {{ background: linear-gradient(180deg, #17193d, #10122c); border:1px solid #262a5e; border-radius:16px;
  padding:16px 18px; box-shadow: 0 0 0 1px rgba(255,255,255,.02), 0 10px 30px rgba(0,0,0,.35); }}
.kpi {{ font-family:'Orbitron',sans-serif; font-size:1.9rem; font-weight:800; }}
.kpi-l {{ color:#9497c9; font-size:.8rem; text-transform:uppercase; letter-spacing:1.5px; }}
.verdict {{ font-family:'Orbitron',sans-serif; font-size:2.6rem; font-weight:800; text-align:center;
  padding:14px; border-radius:16px; letter-spacing:4px; }}
.v-spam {{ color:{PINK}; border:2px solid {PINK}; box-shadow:0 0 30px rgba(255,46,136,.45), inset 0 0 30px rgba(255,46,136,.15);
  animation: pulse 1.4s ease-in-out infinite; }}
.v-ham {{ color:{LIME}; border:2px solid {LIME}; box-shadow:0 0 30px rgba(124,255,107,.35), inset 0 0 30px rgba(124,255,107,.12); }}
@keyframes pulse {{ 50% {{ box-shadow:0 0 55px rgba(255,46,136,.75), inset 0 0 40px rgba(255,46,136,.25); }} }}
.tok {{ display:inline-block; padding:3px 7px; margin:3px 2px; border-radius:7px; font-size:1.05rem; }}
.formula {{ font-family: 'Consolas', monospace; background:#0d0f26; border-left:3px solid {CYAN};
  padding:10px 14px; border-radius:8px; color:#dfe1ff; }}
div[data-baseweb="tab-list"] button p {{ font-size:1rem; font-weight:600; }}
.stTextArea textarea {{ font-size:1.1rem; background:#0e1030; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training the Bayesian brain…")
def load():
    return core.run_all()


R = load()
df, P, M, model, vec = R["df"], R["priors"], R["metrics"], R["model"], R["vec"]


def style(fig, h=360):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=40, b=10), font=dict(family="Inter", color="#e8e9ff"))
    return fig


def fmt_p(p):
    """4 decimals, but never show a misleading exact 1.0000 or 0.0000."""
    return ">0.9999" if p > 0.99995 else "<0.0001" if p < 0.00005 else f"{p:.4f}"


def kpi(label, value, color=CYAN):
    st.markdown(f'<div class="card"><div class="kpi-l">{label}</div>'
                f'<div class="kpi" style="color:{color}">{value}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ hero
st.markdown(f"""
<div class="hero"><h1>🛡️ SPAMSHIELD</h1>
<p>Probabilistic SMS Spam Detection with Bayes' Theorem &amp; Multinomial Naive Bayes</p>
<span class="chip">Mohammad Asif Shaik · 25BDS0107</span>
<span class="chip">BAMAT207 · Probability &amp; Statistics</span>
<span class="chip">UCI SMS Spam Collection · {P['n']:,} clean messages</span>
<span class="chip">Test accuracy {M['accuracy']*100:.2f}%</span></div>
""", unsafe_allow_html=True)

tabs = st.tabs(["⚡ Live Scanner", "📦 Dataset", "🔎 Word Probabilities", "🎲 Random Variable",
                "🧠 Model Evaluation", "🧪 Bayes Lab"])

# ------------------------------------------------------------------ 1. live scanner
EXAMPLES = {
    "🎁 Prize scam": "Congratulations! You have won a free prize. Claim now.",
    "🚨 Urgent": "URGENT! Your mobile number has won £2000. Call 09061701461 to claim before it expires",
    "📚 Friend": "Hey, are we still meeting at 4 for the project demo?",
    "🏠 Mom": "Reached home safely, will call you after dinner",
}

with tabs[0]:
    if "msg" not in st.session_state:
        st.session_state.msg = EXAMPLES["🎁 Prize scam"]

    def set_msg(t):
        st.session_state.msg = t

    cols = st.columns(len(EXAMPLES))
    for c, (k, v) in zip(cols, EXAMPLES.items()):
        c.button(k, on_click=set_msg, args=(v,), width="stretch")
    msg = st.text_area("Type or paste any SMS 👇", key="msg", height=100)

    if msg.strip():
        res = core.predict(model, vec, msg)
        spam = res["label"] == "Spam"
        left, right = st.columns([1.1, 1])
        with left:
            st.markdown(f'<div class="verdict {"v-spam" if spam else "v-ham"}">'
                        f'{"🚫 SPAM" if spam else "✅ GENUINE"}</div>', unsafe_allow_html=True)
            st.write("")
            a, b = st.columns(2)
            with a:
                kpi("P(Spam | Message)", fmt_p(res["p_spam"]), PINK)
            with b:
                kpi("P(Ham | Message)", fmt_p(res["p_ham"]), LIME)
            st.write("")
            st.markdown("**Word-by-word evidence** — red pushes toward spam, green toward genuine, grey = unseen word")
            toks = core.explain(model, vec, msg)
            spans = []
            for t, s, known in toks:
                if not known:
                    bg, fg = "#23264f", "#8f92c2"
                else:
                    a_ = min(abs(s) / 4, 1) * 0.85 + 0.15
                    bg = f"rgba(255,46,136,{a_:.2f})" if s > 0 else f"rgba(124,255,107,{a_*0.7:.2f})"
                    fg = "#fff" if s > 0 else "#0a0b1a"
                spans.append(f'<span class="tok" title="log-ratio {s:+.2f}" style="background:{bg};color:{fg}">{html.escape(t)}</span>')
            st.markdown('<div class="card">' + "".join(spans) + "</div>", unsafe_allow_html=True)
        with right:
            g = go.Figure(go.Indicator(
                mode="gauge+number", value=res["p_spam"] * 100,
                number={"suffix": "%", "font": {"size": 54, "family": "Orbitron", "color": PINK if spam else LIME}},
                title={"text": "Spam probability", "font": {"size": 16}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": PINK if spam else LIME, "thickness": .3},
                       "bgcolor": PANEL, "borderwidth": 0,
                       "steps": [{"range": [0, 50], "color": "#13324a"}, {"range": [50, 100], "color": "#3a1030"}],
                       "threshold": {"line": {"color": "white", "width": 3}, "value": 50}}))
            st.plotly_chart(style(g, 300), width="stretch")

        # Log-odds waterfall: prior + each word's contribution = posterior
        st.markdown("#### 🧮 How Naive Bayes reached this decision (log-odds)")
        x = vec.transform([msg])
        idx = x.nonzero()[1]
        names = vec.get_feature_names_out()
        lp = model.feature_log_prob_
        prior_lo = model.class_log_prior_[1] - model.class_log_prior_[0]
        contrib = sorted(((names[i], x[0, i] * (lp[1, i] - lp[0, i])) for i in idx), key=lambda z: -abs(z[1]))[:14]
        total = prior_lo + sum(v for _, v in contrib)
        labels = ["Prior log(P(S)/P(H))"] + [w for w, _ in contrib] + ["Posterior"]
        vals = [prior_lo] + [v for _, v in contrib] + [total]
        wf = go.Figure(go.Waterfall(x=labels, y=vals, measure=["relative"] * (len(vals) - 1) + ["total"],
                                    increasing={"marker": {"color": PINK}}, decreasing={"marker": {"color": LIME}},
                                    totals={"marker": {"color": VIOLET}}, connector={"line": {"color": "#444a8a"}}))
        style(wf, 340).update_layout(yaxis_title="log-odds of Spam")
        st.plotly_chart(wf, width="stretch")
        st.markdown(f'<div class="formula">log P(S|m)/P(H|m) = log P(S)/P(H) + Σ count(w)·log P(w|S)/P(w|H)'
                    f' = <b>{total:+.2f}</b> &nbsp;⇒&nbsp; P(Spam|m) = 1/(1+e<sup>−{total:.2f}</sup>) = <b>{fmt_p(res["p_spam"])}</b></div>',
                    unsafe_allow_html=True)

# ------------------------------------------------------------------ 2. dataset
with tabs[1]:
    S = R["summary"]
    c = st.columns(5)
    with c[0]: kpi("Raw messages", f"{S['raw_rows']:,}")
    with c[1]: kpi("Missing values", f"{sum(S['missing'].values())}", LIME)
    with c[2]: kpi("Duplicates removed", f"{S['duplicates']}", AMBER)
    with c[3]: kpi("Clean messages", f"{S['clean_rows']:,}", VIOLET)
    with c[4]: kpi("Labels: spam / ham", "1 / 0", PINK)
    st.write("")
    l, r = st.columns(2)
    with l:
        bar = go.Figure(go.Bar(x=["Ham (0)", "Spam (1)"], y=[P["n_ham"], P["n_spam"]], marker_color=[LIME, PINK],
                               text=[f"{P['n_ham']}<br>P = {P['p_ham']:.4f}", f"{P['n_spam']}<br>P = {P['p_spam']:.4f}"],
                               textposition="outside"))
        style(bar, 380).update_layout(title="Class distribution", yaxis_range=[0, P["n_ham"] * 1.2])
        st.plotly_chart(bar, width="stretch")
    with l:
        st.markdown(f'<div class="formula">P(Spam) = {P["n_spam"]} / {P["n"]} = <b>{P["p_spam"]:.4f}</b><br>'
                    f'P(Ham) &nbsp;= {P["n_ham"]} / {P["n"]} = <b>{P["p_ham"]:.4f}</b></div>', unsafe_allow_html=True)
    with r:
        h = go.Figure()
        for lab, col, name in [(0, LIME, "Ham"), (1, PINK, "Spam")]:
            h.add_histogram(x=df[df.label_num == lab]["length"], name=name, marker_color=col, opacity=.7,
                            xbins=dict(size=10), histnorm="probability")
        style(h, 380).update_layout(barmode="overlay", title="Message length distribution (characters)",
                                    xaxis_range=[0, 400], yaxis_title="proportion")
        st.plotly_chart(h, width="stretch")
        st.caption(f"Average length — Ham: {df[df.label_num==0].length.mean():.0f} chars · "
                   f"Spam: {df[df.label_num==1].length.mean():.0f} chars. Spam messages are much longer.")
    st.markdown("##### Sample of the cleaned dataset")
    st.dataframe(df[["label", "label_num", "text", "length"]].sample(8, random_state=1), width="stretch", hide_index=True)

# ------------------------------------------------------------------ 3. words
with tabs[2]:
    W = R["words"]
    st.markdown('<div class="formula">Direct: P(Spam|w) = #spam containing w / #messages containing w'
                '<br>Bayes&nbsp;: P(Spam|w) = P(w|Spam) · P(Spam) / P(w)</div>', unsafe_allow_html=True)
    st.write("")
    f = go.Figure()
    f.add_bar(x=W.word, y=W["P(Spam|word) direct"], name="Direct count", marker_color=PINK)
    f.add_bar(x=W.word, y=W["P(Spam|word) Bayes"], name="Bayes' theorem", marker_color=CYAN)
    f.add_hline(y=P["p_spam"], line_dash="dash", line_color=AMBER,
                annotation_text=f"Prior P(Spam) = {P['p_spam']:.3f}", annotation_font_color=AMBER)
    style(f, 380).update_layout(barmode="group", title="P(Spam | word) — both methods agree ✔", yaxis_range=[0, 1.1])
    st.plotly_chart(f, width="stretch")
    show = W.copy()
    st.dataframe(show.style.format({c: "{:.4f}" for c in show.columns if c.startswith("P(")} | {"lift": "{:.2f}×"}),
                 width="stretch", hide_index=True)
    st.info(f"Seeing **prize** raises the spam probability from {P['p_spam']:.1%} (prior) to "
            f"{W.set_index('word').loc['prize','P(Spam|word) direct']:.0%} — every one of its "
            f"{W.set_index('word').loc['prize','msgs_with_word']} messages is spam.")

    st.markdown("#### 🔬 Try any word")
    w = st.text_input("Word", "call").strip().lower()
    if w:
        one = core.word_probabilities(df, [w]).iloc[0]
        if one.msgs_with_word == 0:
            st.warning(f"'{w}' does not appear in the dataset.")
        else:
            c = st.columns(4)
            with c[0]: kpi("Messages with word", int(one.msgs_with_word))
            with c[1]: kpi("P(word | Spam)", f"{one['P(word|Spam)']:.4f}", VIOLET)
            with c[2]: kpi("P(Spam | word)", f"{one['P(Spam|word) direct']:.4f}", PINK)
            with c[3]: kpi("Lift over prior", f"{one['lift']:.2f}×", AMBER)

# ------------------------------------------------------------------ 4. random variable
with tabs[3]:
    rv = R["rv"]
    st.markdown('<div class="formula">Y = 1 if the message is spam, Y = 0 if ham &nbsp;→&nbsp; Y ~ Bernoulli(p), p = P(Spam)'
                '<br>E(Y) = 0·(1−p) + 1·p = p &nbsp;&nbsp;|&nbsp;&nbsp; Var(Y) = E(Y²) − E(Y)² = p − p² = p(1−p)</div>',
                unsafe_allow_html=True)
    st.write("")
    c = st.columns(4)
    with c[0]: kpi("p = P(Spam)", f"{rv['p']:.4f}", PINK)
    with c[1]: kpi("E(Y)", f"{rv['E(Y)']:.4f}", CYAN)
    with c[2]: kpi("Var(Y) = p(1−p)", f"{rv['Var(Y)']:.4f}", VIOLET)
    with c[3]: kpi("SD(Y)", f"{rv['SD(Y)']:.4f}", AMBER)
    st.caption(f"Check with numpy on the actual labels: mean = {rv['sample_mean']:.4f}, variance = {rv['sample_var']:.4f} ✔")
    l, r = st.columns(2)
    with l:
        pmf = go.Figure(go.Bar(x=["Y = 0 (Ham)", "Y = 1 (Spam)"], y=[1 - rv["p"], rv["p"]], marker_color=[LIME, PINK],
                               text=[f"{1-rv['p']:.4f}", f"{rv['p']:.4f}"], textposition="outside"))
        style(pmf, 380).update_layout(title="Probability mass function of Y", yaxis_range=[0, 1.05])
        st.plotly_chart(pmf, width="stretch")
    with r:
        n = st.slider("Simulate drawing random messages (Law of Large Numbers)", 50, 5000, 2000, 50)
        rng = np.random.default_rng(7)
        draws = rng.choice(df["label_num"].to_numpy(), size=n)
        run = np.cumsum(draws) / np.arange(1, n + 1)
        lln = go.Figure(go.Scatter(y=run, mode="lines", line=dict(color=CYAN, width=2), name="running mean"))
        lln.add_hline(y=rv["p"], line_dash="dash", line_color=PINK, annotation_text="E(Y) = p")
        style(lln, 380).update_layout(title="Sample mean of Y converges to E(Y)", xaxis_title="messages drawn",
                                      yaxis_title="running mean")
        st.plotly_chart(lln, width="stretch")

# ------------------------------------------------------------------ 5. evaluation
with tabs[4]:
    st.markdown(f"Split **80 / 20 stratified** → {M['n_train']:,} train · {M['n_test']:,} test "
                f"(spam ratio {M['train_spam_ratio']:.3f} vs {M['test_spam_ratio']:.3f} — proportion preserved). "
                f"CountVectorizer vocabulary: **{M['vocab']:,} words**. Model: `MultinomialNB()`.")
    c = st.columns(4)
    with c[0]: kpi("Accuracy", f"{M['accuracy']*100:.2f}%", CYAN)
    with c[1]: kpi("Precision (spam)", f"{M['precision']*100:.2f}%", VIOLET)
    with c[2]: kpi("Recall (spam)", f"{M['recall']*100:.2f}%", AMBER)
    with c[3]: kpi("F1-score (spam)", f"{M['f1']*100:.2f}%", PINK)
    st.write("")
    l, r = st.columns([1, 1.1])
    with l:
        cm = M["cm"]
        txt = [[f"TN<br><b>{cm[0,0]}</b>", f"FP<br><b>{cm[0,1]}</b>"], [f"FN<br><b>{cm[1,0]}</b>", f"TP<br><b>{cm[1,1]}</b>"]]
        hm = go.Figure(go.Heatmap(z=np.log1p(cm), x=["Pred Ham", "Pred Spam"], y=["Actual Ham", "Actual Spam"],
                                  text=txt, texttemplate="%{text}", textfont={"size": 22},
                                  colorscale=[[0, "#1a1240"], [.5, VIOLET], [1, PINK]], showscale=False))
        style(hm, 400).update_layout(title="Confusion matrix", yaxis_autorange="reversed")
        st.plotly_chart(hm, width="stretch")
    with r:
        st.markdown("##### Classification report")
        st.code(M["report"], language="text")
        st.markdown(f"- 📵 **Spam classified as genuine (FN): {M['fn']}** — slipped through the filter\n"
                    f"- 📨 **Genuine classified as spam (FP): {M['fp']}** — real messages wrongly blocked")
    T = R["test"]
    a, b = st.columns(2)
    with a:
        st.markdown(f"##### 📵 Spam → predicted Ham ({M['fn']})")
        st.dataframe(T[(T.actual == 1) & (T.pred == 0)][["text", "p_spam"]], width="stretch", hide_index=True)
    with b:
        st.markdown(f"##### 📨 Ham → predicted Spam ({M['fp']})")
        st.dataframe(T[(T.actual == 0) & (T.pred == 1)][["text", "p_spam"]], width="stretch", hide_index=True)
    sp, hm_ = core.top_words(model, vec, 15)
    tw = go.Figure()
    tw.add_bar(y=sp.word[::-1], x=sp.log_ratio[::-1], orientation="h", marker_color=PINK, name="spam words")
    style(tw, 460).update_layout(title="Top 15 spam-indicating words: log P(w|Spam) − log P(w|Ham)")
    st.plotly_chart(tw, width="stretch")

# ------------------------------------------------------------------ 6. bayes lab
with tabs[5]:
    st.markdown("Play with Bayes' theorem directly. Defaults are the real numbers for the word **free**.")
    fw = R["words"].set_index("word").loc["free"]
    c = st.columns(3)
    pri = c[0].slider("Prior P(Spam)", 0.01, 0.99, float(round(P["p_spam"], 3)), 0.005)
    lws = c[1].slider("P(word | Spam)", 0.0, 1.0, float(round(fw["P(word|Spam)"], 3)), 0.005)
    lwh = c[2].slider("P(word | Ham)", 0.0, 1.0, float(round(fw["P(word|Ham)"], 3)), 0.001)
    pw = lws * pri + lwh * (1 - pri)
    post = lws * pri / pw if pw else 0
    st.markdown(f'<div class="formula">P(word) = P(w|S)P(S) + P(w|H)P(H) = {lws:.3f}·{pri:.3f} + {lwh:.3f}·{1-pri:.3f} = <b>{pw:.4f}</b>'
                f'<br>P(Spam|word) = {lws:.3f}·{pri:.3f} / {pw:.4f} = <b style="color:{PINK}">{post:.4f}</b></div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_bar(x=["Before seeing the word", "After seeing the word"], y=[pri, post], marker_color=[AMBER, PINK],
                text=[f"{pri:.3f}", f"{post:.3f}"], textposition="outside")
    style(fig, 360).update_layout(title="Belief update: prior → posterior", yaxis_range=[0, 1.1])
    st.plotly_chart(fig, width="stretch")

st.markdown('<p style="text-align:center;color:#666a9e;margin-top:30px">SpamShield · BAMAT207 Probability & Statistics project · '
            'Dataset: UCI SMS Spam Collection (Almeida & Hidalgo, 2011)</p>', unsafe_allow_html=True)
