"""
IPL Winning Team Prediction System
====================================
A multi-page Streamlit app with two prediction systems:
  1. Pre-Match Winner Predictor
  2. Live Match Win Predictor
plus an Analytics Dashboard summarizing historical IPL data.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# ------------------------------------------------------------------
# Page config & global styling
# ------------------------------------------------------------------
st.set_page_config(
    page_title="IPL Winning Team Prediction System",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #1E3A8A 0%, #DC2626 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub-title {
    color: #6B7280;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: var(--card-bg, #161B22);
    border: 1px solid rgba(120,120,120,0.25);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.metric-card h3 {
    margin: 0;
    font-size: 0.85rem;
    font-weight: 600;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.metric-card p {
    margin: 0.3rem 0 0 0;
    font-size: 1.7rem;
    font-weight: 800;
}
.winner-banner {
    background: linear-gradient(135deg, #064E3B 0%, #047857 100%);
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    color: white;
    margin-bottom: 1rem;
}
.winner-banner h2 {
    margin: 0;
    font-size: 1.8rem;
    font-weight: 800;
}
.winner-banner p {
    margin: 0.2rem 0 0 0;
    opacity: 0.85;
    font-size: 0.95rem;
}
.prob-bar-outer {
    width: 100%;
    height: 38px;
    border-radius: 19px;
    background: #1F2937;
    display: flex;
    overflow: hidden;
    font-weight: 700;
    font-size: 0.95rem;
    color: white;
    margin: 0.6rem 0 0.3rem 0;
}
.prob-bar-left {
    background: linear-gradient(90deg, #2563EB, #3B82F6);
    display: flex;
    align-items: center;
    padding-left: 14px;
    white-space: nowrap;
}
.prob-bar-right {
    background: linear-gradient(90deg, #DC2626, #EF4444);
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 14px;
    white-space: nowrap;
}
.insight-box {
    background: rgba(59,130,246,0.08);
    border-left: 4px solid #3B82F6;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.92rem;
}
.section-label {
    font-weight: 700;
    font-size: 1.05rem;
    margin-top: 1.2rem;
    margin-bottom: 0.3rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Paths & cached loaders
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

PRE_MODEL_PATH = os.path.join(MODELS_DIR, "pre_match_model.pkl")
PRE_LOOKUPS_PATH = os.path.join(MODELS_DIR, "encoders", "pre_match_lookups.pkl")
LIVE_MODEL_PATH = os.path.join(MODELS_DIR, "live_match_model.pkl")
LIVE_LOOKUPS_PATH = os.path.join(MODELS_DIR, "encoders", "live_match_lookups.pkl")


@st.cache_resource
def load_pre_match_artifacts():
    pipeline = joblib.load(PRE_MODEL_PATH)
    lookups = joblib.load(PRE_LOOKUPS_PATH)
    return pipeline, lookups


@st.cache_resource
def load_live_match_artifacts():
    pipeline = joblib.load(LIVE_MODEL_PATH)
    lookups = joblib.load(LIVE_LOOKUPS_PATH)
    return pipeline, lookups


@st.cache_data
def load_raw_data():
    matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
    return matches


TEAM_COLORS = {
    "Chennai Super Kings": "#FBD323", "Mumbai Indians": "#2784D5",
    "Royal Challengers Bengaluru": "#F40D14", "Kolkata Knight Riders": "#4F3379",
    "Delhi Capitals": "#04183DE7", "Punjab Kings": "#F56A71FF",
    "Rajasthan Royals": "#FA1D8F", "Sunrisers Hyderabad": "#F66B2B",
    "Gujarat Titans": "#1B2133", "Lucknow Super Giants": "#A4DE02",
    "Gujarat Lions": "#8B5249", "Rising Pune Supergiants": "#C2A267",
    "Pune Warriors": "#A8D6F7", "Kochi Tuskers Kerala": "#F59051",
}


def team_color(team, fallback="#3B82F6"):
    return TEAM_COLORS.get(team, fallback)


def render_prob_bar(team_a, prob_a, team_b, prob_b):
    """Renders a horizontal split probability bar like a sports-analytics win meter."""
    pct_a = prob_a * 100
    pct_b = prob_b * 100
    color_a = team_color(team_a, "#2563EB")
    color_b = team_color(team_b, "#DC2626")
    html = f"""
    <div class="prob-bar-outer">
        <div class="prob-bar-left" style="width:{pct_a}%; background:{color_a};">
            {team_a} {pct_a:.1f}%
        </div>
        <div class="prob-bar-right" style="width:{pct_b}%; background:{color_b};">
            {team_b} {pct_b:.1f}%
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(col, label, value, suffix=""):
    col.markdown(
        f"""<div class="metric-card"><h3>{label}</h3><p>{value}{suffix}</p></div>""",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.image("assets/ipl_logo.png", width=140)
st.sidebar.markdown("## 🏏 IPL Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏏 Pre-Match Predictor", "📈 Live Match Predictor", "📊 Analytics Dashboard"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Built on 2008–2024 IPL ball-by-ball data. Pre-match prediction reflects the genuine "
    "difficulty of forecasting T20 outcomes before play begins; live predictions use real-time "
    "match state and are considerably more accurate — see the Analytics Dashboard for details."
)

if "pre_match_history" not in st.session_state:
    st.session_state.pre_match_history = []
if "live_match_history" not in st.session_state:
    st.session_state.live_match_history = []


# ============================================================
# PAGE 1: PRE-MATCH WINNER PREDICTOR
# ============================================================
def page_pre_match():
    st.markdown('<div class="main-title">🏏 IPL Pre-Match Winner Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Predict the winner before a single ball is bowled — '
        'using team form, head-to-head record, toss, and venue history.</div>',
        unsafe_allow_html=True,
    )

    try:
        pipeline, lookups = load_pre_match_artifacts()
    except FileNotFoundError:
        st.error("Pre-match model artifacts not found. Run the notebook first to generate `models/pre_match_model.pkl`.")
        return

    teams = lookups["teams"]
    venues = lookups["venues"]
    cities = lookups["cities"]

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams, index=teams.index("Mumbai Indians") if "Mumbai Indians" in teams else 0)
    with col2:
        team2_options = [t for t in teams if t != team1]
        team2 = st.selectbox("Team 2", team2_options, index=team2_options.index("Chennai Super Kings") if "Chennai Super Kings" in team2_options else 0)

    col3, col4 = st.columns(2)
    with col3:
        venue = st.selectbox("Venue", venues)
    with col4:
        city = st.selectbox("City", cities)

    col5, col6 = st.columns(2)
    with col5:
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
    with col6:
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

    predict_clicked = st.button("🔮 Predict Winner", type="primary", use_container_width=True)

    if predict_clicked:
        t1_winpct = lookups["team_win_pct"].get(team1, 0.5)
        t2_winpct = lookups["team_win_pct"].get(team2, 0.5)
        t1_venue = lookups["venue_team_win_pct"].get(f"{venue}||{team1}", 0.5)
        t2_venue = lookups["venue_team_win_pct"].get(f"{venue}||{team2}", 0.5)
        t1_city = lookups["city_team_win_pct"].get(f"{city}||{team1}", 0.5)
        t2_city = lookups["city_team_win_pct"].get(f"{city}||{team2}", 0.5)
        t1_h2h = lookups["h2h_win_pct"].get(f"{team1}||{team2}", 0.5)
        t2_h2h = lookups["h2h_win_pct"].get(f"{team2}||{team1}", 0.5)
        t1_form = lookups["recent_form"].get(team1, 0.5)
        t2_form = lookups["recent_form"].get(team2, 0.5)

        row = pd.DataFrame([{
            "team1": team1, "team2": team2, "venue": venue, "city": city,
            "toss_winner": toss_winner, "toss_decision": toss_decision,
            "team1_win_pct": t1_winpct, "team2_win_pct": t2_winpct,
            "team1_venue_win_pct": t1_venue, "team2_venue_win_pct": t2_venue,
            "team1_city_win_pct": t1_city, "team2_city_win_pct": t2_city,
            "team1_h2h_win_pct": t1_h2h, "team2_h2h_win_pct": t2_h2h,
            "team1_recent_form": t1_form, "team2_recent_form": t2_form,
            "toss_winner_is_team1": int(toss_winner == team1),
        }])

        proba = pipeline.predict_proba(row)[0]
        team1_prob, team2_prob = float(proba[1]), float(proba[0])
        winner = team1 if team1_prob > team2_prob else team2
        confidence = max(team1_prob, team2_prob)

        st.session_state.pre_match_history.insert(0, {
            "Team 1": team1, "Team 2": team2, "Venue": venue,
            "Predicted Winner": winner, "Confidence": f"{confidence*100:.1f}%",
        })

        st.markdown(
            f"""<div class="winner-banner"><h2>🏆 Predicted Winner: {winner}</h2>
            <p>Model Confidence: {confidence*100:.1f}%</p></div>""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label">Win Probability Bar</div>', unsafe_allow_html=True)
        render_prob_bar(team1, team1_prob, team2, team2_prob)

        c1, c2, c3 = st.columns(3)
        render_metric_card(c1, f"{team1} Win Prob", f"{team1_prob*100:.1f}", "%")
        render_metric_card(c2, f"{team2} Win Prob", f"{team2_prob*100:.1f}", "%")
        render_metric_card(c3, "Model Confidence", f"{confidence*100:.1f}", "%")

        st.markdown('<div class="section-label">Probability Comparison Chart</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=[team1_prob*100, team2_prob*100], y=[team1, team2], orientation="h",
            marker_color=[team_color(team1, "#2563EB"), team_color(team2, "#DC2626")],
            text=[f"{team1_prob*100:.1f}%", f"{team2_prob*100:.1f}%"], textposition="auto",
        ))
        fig.update_layout(title="Pre-Match Win Probability Comparison", xaxis_title="Win Probability (%)",
                           height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-label">Prediction Explanation</div>', unsafe_allow_html=True)
        reasons = []
        if abs(t1_h2h - t2_h2h) > 0.1:
            leader = team1 if t1_h2h > t2_h2h else team2
            reasons.append(f"**Head-to-head record favors {leader}** ({max(t1_h2h,t2_h2h)*100:.0f}% historical win rate in this matchup).")
        if abs(t1_winpct - t2_winpct) > 0.05:
            leader = team1 if t1_winpct > t2_winpct else team2
            reasons.append(f"**{leader} has a stronger overall win percentage** across IPL history.")
        if abs(t1_form - t2_form) > 0.1:
            leader = team1 if t1_form > t2_form else team2
            reasons.append(f"**{leader} is in better recent form** over their last 5 matches.")
        toss_team_winpct = t1_venue if toss_winner == team1 else t2_venue
        reasons.append(f"**Toss**: {toss_winner} won the toss and chose to {toss_decision} — historically a "
                        f"{toss_team_winpct*100:.0f}% win rate at this venue for that team.")
        if not reasons:
            reasons.append("Both teams are closely matched on the historical signals available — this prediction carries lower certainty.")
        for r in reasons:
            st.markdown(f'<div class="insight-box">{r}</div>', unsafe_allow_html=True)

        st.info(
            "ℹ️ **About this model's accuracy**: Pre-match prediction is genuinely difficult in T20 cricket — "
            "this model's test ROC-AUC is around 0.52–0.58, only modestly better than chance. Treat this as a "
            "directional signal, not a certainty. The Live Match Predictor (next page) is far more reliable once "
            "the match is underway."
        )

    if st.session_state.pre_match_history:
        st.markdown('<div class="section-label">📜 Prediction History</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.pre_match_history), use_container_width=True, hide_index=True)


# ============================================================
# PAGE 2: LIVE MATCH WINNER PREDICTOR
# ============================================================
def page_live_match():
    # st.markdown('<div class="main-title">📈 IPL Live Match Win Predictor</div>', unsafe_allow_html=True)
    col_logo, col_title = st.columns([1, 9.5])
    with col_logo:
        st.image("assets/ipl_logo.png", use_container_width=True)
    with col_title:
        st.markdown('<div class="main-title">IPL Live Match Win Predictor</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="sub-title">Enter the current chase situation to get a real-time win probability, '
        'powered by the same kind of model used in professional broadcast graphics.</div>',
        unsafe_allow_html=True,
    )

    try:
        pipeline, lookups = load_live_match_artifacts()
    except FileNotFoundError:
        st.error("Live-match model artifacts not found. Run the notebook first to generate `models/live_match_model.pkl`.")
        return

    teams = lookups["teams"]
    venues = lookups["venues"]
    cities = lookups["cities"]

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox("Batting Team (Chasing)", teams, index=teams.index("Mumbai Indians") if "Mumbai Indians" in teams else 0, key="live_bat")
    with col2:
        bowl_options = [t for t in teams if t != batting_team]
        bowling_team = st.selectbox("Bowling Team (Defending)", bowl_options, index=bowl_options.index("Chennai Super Kings") if "Chennai Super Kings" in bowl_options else 0, key="live_bowl")

    col3, col4 = st.columns(2)
    with col3:
        venue = st.selectbox("Venue", venues, key="live_venue")
    with col4:
        city = st.selectbox("City", cities, key="live_city")

    st.markdown('<div class="section-label">Match Situation</div>', unsafe_allow_html=True)
    col5, col6, col7, col8, col9 = st.columns(5)
    with col5:
        target_score = st.number_input("Target Score", min_value=1, max_value=350, value=180, step=1)
    with col6:
        current_score = st.number_input("Current Score", min_value=0, max_value=349, value=120, step=1)
    with col7:
        wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10, value=3, step=1)
    with col8:
        overs_part = st.number_input("Overs", min_value=0, max_value=19, value=14, step=1)
    with col9:
        balls_part = st.number_input("Balls (0-5)", min_value=0, max_value=5, value=0, step=1)
    overs_completed = overs_part + balls_part / 10  # cricket notation: e.g. 14 overs + 3 balls -> 14.3

    predict_clicked = st.button("🔮 Predict Live Win Probability", type="primary", use_container_width=True, key="live_predict_btn")
        
    if predict_clicked:
        balls_bowled = overs_part * 6 + balls_part
        runs_remaining = max(target_score - current_score, 0)
        balls_remaining = max(120 - balls_bowled, 0)
        wickets_remaining = max(10 - wickets_lost, 0)
        current_run_rate = (current_score / (balls_bowled / 6)) if balls_bowled > 0 else 0.0
        required_run_rate = (runs_remaining * 6 / balls_remaining) if balls_remaining > 0 else 0.0
        run_rate_diff = current_run_rate - required_run_rate

        # ------------------------------------------------------------------
        # VALIDATION GUARDS — order matters. Target-reached is checked FIRST,
        # since in real cricket a chase ends the instant the winning run is
        # scored, regardless of whether a wicket also fell on that same ball.
        # ------------------------------------------------------------------
        if current_score >= target_score:
            st.success(
                f"🏆 **Target already reached.** {batting_team} have scored {current_score}, "
                f"meeting or beating the target of {target_score}. **{batting_team} have already won** this chase."
            )
            st.session_state.live_match_history.insert(0, {
                "Batting": batting_team, "Bowling": bowling_team,
                "Score": f"{current_score}/{wickets_lost} ({overs_part}.{balls_part})", "Target": target_score,
                "Predicted Winner": batting_team, "Confidence": "100.0% (target reached)",
            })
            return

        if wickets_lost >= 10:
            st.error(
                f"🏏 **All out.** {batting_team} have lost all 10 wickets — the innings is over. "
                f"**{bowling_team} win** by default; there's no remaining chase to predict."
            )
            st.session_state.live_match_history.insert(0, {
                "Batting": batting_team, "Bowling": bowling_team,
                "Score": f"{current_score}/{wickets_lost} ({overs_part}.{balls_part})", "Target": target_score,
                "Predicted Winner": bowling_team, "Confidence": "100.0% (all out)",
            })
            return

        if balls_remaining <= 0 and current_score < target_score:
            st.error(
                f"🏏 **Overs completed with target not reached.** {batting_team} faced all 120 balls "
                f"and fell short by {runs_remaining} run(s). **{bowling_team} win** — there are no balls left to predict."
            )
            st.session_state.live_match_history.insert(0, {
                "Batting": batting_team, "Bowling": bowling_team,
                "Score": f"{current_score}/{wickets_lost} ({overs_part}.{balls_part})", "Target": target_score,
                "Predicted Winner": bowling_team, "Confidence": "100.0% (overs complete)",
            })
            return


        # ------------------------------------------------------------------
        # If we reach here, the match state is genuinely live and unresolved —
        # proceed with the model prediction as before.
        # ------------------------------------------------------------------
        venue_avg_1st = lookups["venue_avg_first_innings_score"].get(venue, lookups["venue_avg_first_innings_overall"])
        venue_chase_rate = lookups["venue_chase_success_rate"].get(venue, lookups["venue_chase_success_overall"])
        venue_avg_win = lookups["venue_avg_winning_score"].get(venue, lookups["venue_avg_winning_overall"])
        bat_chase_pct = lookups["team_chase_success_pct"].get(batting_team, 0.5)
        bowl_defend_pct = lookups["team_defend_success_pct"].get(bowling_team, 0.5)

        row = pd.DataFrame([{
            "batting_team": batting_team, "bowling_team": bowling_team, "venue": venue, "city": city,
            "current_score": current_score, "wickets_lost": wickets_lost, "overs_completed": overs_completed,
            "target_score": target_score, "runs_remaining": runs_remaining, "balls_remaining": balls_remaining,
            "wickets_remaining": wickets_remaining, "current_run_rate": current_run_rate,
            "required_run_rate": required_run_rate, "run_rate_diff": run_rate_diff,
            "venue_avg_first_innings_score": venue_avg_1st, "venue_chase_success_rate": venue_chase_rate,
            "venue_avg_winning_score": venue_avg_win, "batting_team_chase_success_pct": bat_chase_pct,
            "bowling_team_defend_success_pct": bowl_defend_pct,
        }])

        proba = pipeline.predict_proba(row)[0]
        batting_prob, bowling_prob = float(proba[1]), float(proba[0])
        # Mathematically-impossible-chase override: if the runs needed exceed what's achievable
        # even with a six off every remaining ball, override the model's probability rather than
        # trusting an extrapolation it was never trained to make. Layout/metrics stay unchanged —
        # only the probability split is clamped.
        MAX_RUNS_PER_BALL = 6
        if balls_remaining > 0 and runs_remaining > balls_remaining * MAX_RUNS_PER_BALL:
            bowling_prob = 0.99
            batting_prob = 0.01
        winner = batting_team if batting_prob > bowling_prob else bowling_team
        confidence = max(batting_prob, bowling_prob)

        st.session_state.live_match_history.insert(0, {
            "Batting": batting_team, "Bowling": bowling_team,
            "Score": f"{current_score}/{wickets_lost} ({overs_completed})", "Target": target_score,
            "Predicted Winner": winner, "Confidence": f"{confidence*100:.1f}%",
        })

        st.markdown(
            f"""<div class="winner-banner"><h2>🏆 Predicted Winner: {winner}</h2>
            <p>Model Confidence: {confidence*100:.1f}%</p></div>""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label">Final Win Probability</div>', unsafe_allow_html=True)
        render_prob_bar(batting_team, batting_prob, bowling_team, bowling_prob)

        c1, c2, c3, c4 = st.columns(4)
        render_metric_card(c1, "Runs Remaining", runs_remaining)
        render_metric_card(c2, "Balls Remaining", balls_remaining)

        # Required run rate is mathematically valid even at extreme values (e.g. needing 12 off the last
        # ball gives RRR = 72.0), but a raw multi-digit rate reads as a bug. Cap the *displayed* string at
        # the threshold once it's exceeded — the underlying value still feeds the model unchanged.
        RRR_DISPLAY_CAP = 36.0  # roughly "6 sixes an over" — a practical ceiling for a real run rate
        if balls_remaining > 0 and required_run_rate > RRR_DISPLAY_CAP:
            rrr_display = f">{RRR_DISPLAY_CAP:.1f}"
        else:
            rrr_display = f"{required_run_rate:.2f}"

        render_metric_card(c3, "Required RR", rrr_display)
        render_metric_card(c4, "Current RR", f"{current_run_rate:.2f}")

        st.progress(min(max(current_score / target_score, 0.0), 1.0), text=f"Chase progress: {current_score}/{target_score} runs")

        st.markdown('<div class="section-label">Live Win Probability Comparison</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=[batting_prob*100, bowling_prob*100], y=[batting_team, bowling_team], orientation="h",
            marker_color=[team_color(batting_team, "#2563EB"), team_color(bowling_team, "#DC2626")],
            text=[f"{batting_prob*100:.1f}%", f"{bowling_prob*100:.1f}%"], textposition="auto",
        ))
        fig.update_layout(xaxis_title="Win Probability (%)", height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-label">Match Situation Analysis</div>', unsafe_allow_html=True)
        insights = []
        if required_run_rate > current_run_rate + 2:
            insights.append(f"⚠️ **Required run rate ({required_run_rate:.2f}) is well above the current rate ({current_run_rate:.2f})** — "
                             f"{batting_team} need to accelerate significantly to stay in the chase.")
        elif required_run_rate < current_run_rate - 1:
            insights.append(f"✅ **{batting_team} are ahead of the required rate** ({current_run_rate:.2f} vs {required_run_rate:.2f}) — "
                             f"a comfortable position if wickets are preserved.")
        else:
            insights.append(f"⚖️ **Current rate and required rate are closely matched** ({current_run_rate:.2f} vs {required_run_rate:.2f}) — a tense, balanced chase.")

        if wickets_remaining <= 3:
            insights.append(f"⚠️ **Only {wickets_remaining} wickets remain** — batting depth is thinning, raising risk for {batting_team}.")
        else:
            insights.append(f"✅ **{wickets_remaining} wickets in hand** gives {batting_team} room to take calculated risks.")

        insights.append(f"📍 **Venue history**: chasing teams have won **{venue_chase_rate*100:.0f}%** of matches at {venue}, "
                         f"with an average first-innings score of **{venue_avg_1st:.0f}**.")
        insights.append(f"📊 **{batting_team}** have historically won **{bat_chase_pct*100:.0f}%** of matches when chasing; "
                         f"**{bowling_team}** have defended successfully **{bowl_defend_pct*100:.0f}%** of the time.")

        for i in insights:
            st.markdown(f'<div class="insight-box">{i}</div>', unsafe_allow_html=True)

    if st.session_state.live_match_history:
        st.markdown('<div class="section-label">📜 Prediction History</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.live_match_history), use_container_width=True, hide_index=True)


# ============================================================
# PAGE 3: ANALYTICS DASHBOARD
# ============================================================
def page_analytics():
    col_logo, col_title = st.columns([1, 9.7])
    with col_logo:
        st.image("assets/ipl_logo.png", use_container_width=True)
    with col_title:
        st.markdown('<div class="main-title">IPL Analytics Dashboard</div>', unsafe_allow_html=True)
   
    st.markdown(
        '<div class="sub-title">Historical insights from 2008–2024 IPL matches — '
        'team performance, venue trends, and toss impact.</div>',
        unsafe_allow_html=True,
    )

    matches = load_raw_data()
    _, pre_lookups = load_pre_match_artifacts()
    _, live_lookups = load_live_match_artifacts()

    valid_matches = matches[matches["result"] != "no result"].dropna(subset=["winner"]).copy()

    c1, c2, c3, c4 = st.columns(4)
    render_metric_card(c1, "Total Matches", f"{len(valid_matches):,}")
    render_metric_card(c2, "Seasons", valid_matches["season"].nunique())
    render_metric_card(c3, "Teams", len(pre_lookups["teams"]))
    render_metric_card(c4, "Venues", len(pre_lookups["venues"]))

    st.markdown('<div class="section-label">Team Win % (All-Time)</div>', unsafe_allow_html=True)
    team_summary = pd.DataFrame({
        "Team": list(pre_lookups["team_win_pct"].keys()),
        "Win %": [v * 100 for v in pre_lookups["team_win_pct"].values()],
    }).sort_values("Win %", ascending=False)
    fig = px.bar(team_summary, x="Win %", y="Team", orientation="h",
                 color="Win %", color_continuous_scale="Blues")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=450, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-label">Toss Decision Split</div>', unsafe_allow_html=True)
        toss_split = valid_matches["toss_decision"].value_counts()
        fig2 = px.pie(values=toss_split.values, names=toss_split.index, hole=0.45,
                       color_discrete_sequence=["#3B82F6", "#F59E0B"])
        fig2.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        st.markdown('<div class="section-label">Toss Winner = Match Winner?</div>', unsafe_allow_html=True)
        toss_match = (valid_matches["toss_winner"] == valid_matches["winner"]).value_counts()
        fig3 = px.pie(values=toss_match.values, names=["Toss Winner Won", "Toss Winner Lost"], hole=0.45,
                       color_discrete_sequence=["#10B981", "#EF4444"])
        fig3.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-label">Chase Success Rate by Venue (Top 15 by Matches Hosted)</div>', unsafe_allow_html=True)
    venue_chase_df = pd.DataFrame({
        "Venue": list(live_lookups["venue_chase_success_rate"].keys()),
        "Chase Success %": [v * 100 for v in live_lookups["venue_chase_success_rate"].values()],
    })
    venue_counts = valid_matches["venue"].value_counts()
    venue_chase_df["Matches"] = venue_chase_df["Venue"].map(venue_counts).fillna(0)
    venue_chase_df = venue_chase_df[venue_chase_df["Matches"] >= 5].sort_values("Matches", ascending=False).head(15)
    fig4 = px.bar(venue_chase_df.sort_values("Chase Success %"), x="Chase Success %", y="Venue", orientation="h",
                  color="Chase Success %", color_continuous_scale="RdYlGn", range_color=[20, 80])
    fig4.update_layout(height=500, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-label">Matches Played per Season</div>', unsafe_allow_html=True)
    season_counts = valid_matches["season"].value_counts().sort_index()
    fig5 = px.line(x=season_counts.index.astype(str), y=season_counts.values, markers=True,
                    labels={"x": "Season", "y": "Matches"})
    fig5.update_traces(line_color="#3B82F6", marker=dict(size=8))
    fig5.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig5, use_container_width=True)

    with st.expander("ℹ️ About model accuracy on this dashboard"):
        st.markdown(
            "- **Pre-Match Predictor**: test ROC-AUC ≈ 0.52–0.58. Cricket outcomes before a ball is "
            "bowled are inherently hard to call — toss, team form, and head-to-head help, but only modestly.\n"
            "- **Live-Match Predictor**: test ROC-AUC ≈ 0.85–0.92. Once the chase is underway, required run "
            "rate, wickets in hand, and run-rate trajectory are strong, genuine predictors.\n\n"
            "Both numbers come directly from the project notebook's held-out test evaluation — see "
            "`notebooks/IPL_Winning_Team_Prediction.ipynb` for the full methodology."
        )


# ============================================================
# ROUTER
# ============================================================
if page == "🏏 Pre-Match Predictor":
    page_pre_match()
elif page == "📈 Live Match Predictor":
    page_live_match()
else:
    page_analytics()
