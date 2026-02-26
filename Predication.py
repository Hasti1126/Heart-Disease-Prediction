import gradio as gr
import numpy as np
import pandas as pd
import joblib

# ── Load Pipeline ──────────────────────────────────────────────────────────────
try:
    pipeline = joblib.load('heart_disease_pipeline.joblib')
except Exception as e:
    print(f"Warning: Could not load model – {e}")
    pipeline = None


# ── Clinical Reasoning Engine ──────────────────────────────────────────────────

FEATURE_CLINICAL_INFO = {
    "Age": {
        "icon": "🎂",
        "label": "Age",
        "high_risk": lambda v: v >= 55,
        "explanation": lambda v: (
            f"Age {v} is a significant independent risk factor. Cardiovascular risk roughly doubles every decade after 55." 
            if v >= 55 else
            f"Age {v} carries lower baseline risk, though other factors remain critical."
        )
    },
    "Sex": {
        "icon": "⚧",
        "label": "Biological Sex",
        "high_risk": lambda v: v == 1,
        "explanation": lambda v: (
            "Male sex is associated with 2–3× higher CAD risk pre-menopause compared to females due to hormonal protection differences."
            if v == 1 else
            "Female sex provides some protective effect pre-menopause, though risk equalizes post-menopause."
        )
    },
    "ChestPainType": {
        "icon": "💢",
        "label": "Chest Pain Type",
        "high_risk": lambda v: v == 3,
        "explanation": lambda v: {
            0: "Typical angina (exertional, relieved by rest/nitrates) is the classic presentation of obstructive CAD.",
            1: "Atypical angina meets only some criteria; intermediate suspicion for ischemia.",
            2: "Non-anginal pain is less likely to reflect ischemia but not entirely excluded.",
            3: "Paradoxically, 'asymptomatic' in this context means silent ischemia — a high-risk pattern in diabetics/elderly."
        }[v]
    },
    "RestingBP": {
        "icon": "🩸",
        "label": "Resting BP",
        "high_risk": lambda v: v >= 140,
        "explanation": lambda v: (
            f"BP {v} mmHg indicates Stage 2 hypertension — doubles long-term MI risk through arterial wall stress."
            if v >= 140 else
            f"BP {v} mmHg is {'borderline elevated' if v >= 130 else 'within acceptable range'}."
        )
    },
    "Cholesterol": {
        "icon": "🫐",
        "label": "Cholesterol",
        "high_risk": lambda v: v >= 240,
        "explanation": lambda v: (
            f"Total cholesterol {v} mg/dL exceeds the high-risk threshold. Promotes atherosclerotic plaque formation."
            if v >= 240 else
            f"Cholesterol {v} mg/dL is {'borderline' if v >= 200 else 'favorable'}. LDL/HDL ratio matters more."
        )
    },
    "FastingBS": {
        "icon": "🍬",
        "label": "Fasting Blood Sugar",
        "high_risk": lambda v: v == 1,
        "explanation": lambda v: (
            "Fasting glucose >120 mg/dL suggests impaired glucose tolerance or diabetes — 2–4× higher CAD risk."
            if v == 1 else
            "Normal fasting glucose. Glycemic control is favorable."
        )
    },
    "RestingECG": {
        "icon": "📈",
        "label": "Resting ECG",
        "high_risk": lambda v: v >= 1,
        "explanation": lambda v: {
            0: "Normal resting ECG. No baseline electrical abnormality detected.",
            1: "ST-T wave abnormalities at rest indicate possible ischemia or LV strain pattern.",
            2: "Left ventricular hypertrophy (LVH) is a strong marker of pressure-overload and long-standing hypertension."
        }[int(v)]
    },
    "MaxHR": {
        "icon": "❤️",
        "label": "Max Heart Rate",
        "high_risk": lambda v: v < 120,
        "explanation": lambda v: (
            f"Max HR {v} bpm is notably low, suggesting chronotropic incompetence — the heart cannot adequately increase output under stress."
            if v < 120 else
            f"Max HR {v} bpm. {'Good' if v >= 150 else 'Adequate'} chronotropic response to exercise."
        )
    },
    "ExerciseAngina": {
        "icon": "🏃",
        "label": "Exercise-Induced Angina",
        "high_risk": lambda v: v == 1,
        "explanation": lambda v: (
            "Exercise-induced angina is a strong positive stress test finding, indicating inducible ischemia."
            if v == 1 else
            "No exercise-induced angina — a reassuring finding on functional assessment."
        )
    },
    "Oldpeak": {
        "icon": "📉",
        "label": "ST Depression (Oldpeak)",
        "high_risk": lambda v: v >= 2.0,
        "explanation": lambda v: (
            f"ST depression of {v}mm is clinically significant. Indicates myocardial ischemia during exertion — correlates with CAD severity."
            if v >= 2.0 else
            f"ST depression of {v}mm is {'mildly elevated' if v >= 1.0 else 'minimal'}."
        )
    },
    "ST_Slope": {
        "icon": "📊",
        "label": "ST Slope",
        "high_risk": lambda v: v == 0,
        "explanation": lambda v: {
            0: "Downsloping ST segment is the most ominous pattern — strongly associated with multivessel CAD.",
            1: "Flat ST slope is intermediate risk — warrants further investigation.",
            2: "Upsloping ST segment is the most favorable pattern, often considered a normal stress response."
        }[int(v)]
    },
}

RISK_THRESHOLDS = {
    "Low": (0, 30),
    "Moderate": (30, 60),
    "High": (60, 80),
    "Very High": (80, 101),
}

def get_risk_tier(prob):
    for tier, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= prob < hi:
            return tier
    return "Very High"

def build_clinical_reasoning(input_df, weights, cols, prob):
    """Generate structured clinical reasoning with per-feature analysis."""
    contributions = {}
    for i, col in enumerate(cols):
        val = input_df.iloc[0, i]
        contributions[col] = abs(weights[i] * val)

    total = sum(contributions.values()) or 1
    sorted_features = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

    # Build feature rows HTML
    rows_html = ""
    red_flags = []
    for feat, contrib in sorted_features[:6]:  # Top 6 contributors
        pct = (contrib / total) * 100
        val = input_df.iloc[0, cols.index(feat)]
        info = FEATURE_CLINICAL_INFO.get(feat, {})
        is_risk = info.get("high_risk", lambda v: False)(val)
        explanation = info.get("explanation", lambda v: f"Value: {v}")(val)
        icon = info.get("icon", "•")
        label = info.get("label", feat)
        bar_color = "#ef4444" if is_risk else "#22c55e"
        text_color = "#fca5a5" if is_risk else "#86efac"

        if is_risk:
            red_flags.append(label)

        rows_html += f"""
        <div class="feat-row">
            <div class="feat-header">
                <span class="feat-icon">{icon}</span>
                <span class="feat-name">{label}</span>
                <span class="feat-badge {'badge-risk' if is_risk else 'badge-ok'}">
                    {'⚠ Risk Factor' if is_risk else '✓ Normal'}
                </span>
                <span class="feat-pct" style="color:{text_color}">{pct:.1f}%</span>
            </div>
            <div class="feat-bar-bg">
                <div class="feat-bar" style="width:{min(pct*2.5, 100):.1f}%; background:{bar_color};"></div>
            </div>
            <div class="feat-note">{explanation}</div>
        </div>
        """

    # Clinical summary
    tier = get_risk_tier(prob)
    tier_colors = {
        "Low": ("#22c55e", "#052e16"),
        "Moderate": ("#f59e0b", "#1c1003"),
        "High": ("#f97316", "#1a0a00"),
        "Very High": ("#ef4444", "#1f0707"),
    }
    tier_color, tier_bg = tier_colors[tier]

    red_flag_text = ""
    if red_flags:
        flags_joined = " · ".join(red_flags)
        red_flag_text = f'<div class="red-flags">🚩 Active risk factors: {flags_joined}</div>'

    recommendations = {
        "Low": "Routine follow-up. Encourage continued healthy lifestyle. Rescreen in 3–5 years.",
        "Moderate": "Consider stress testing (treadmill/nuclear). Evaluate lipid profile and BP control. Follow-up in 6–12 months.",
        "High": "Cardiology referral recommended. Consider coronary CT angiography. Review medications and modifiable risk factors urgently.",
        "Very High": "Urgent cardiology referral. Consider inpatient evaluation. Do not delay further workup.",
    }

    html = f"""
    <div class="report-card">
        <div class="report-header" style="border-color:{tier_color}; background:{tier_bg}20;">
            <div class="verdict-left">
                <div class="tier-badge" style="background:{tier_color}20; color:{tier_color}; border:1px solid {tier_color}40;">
                    {tier.upper()} RISK
                </div>
                <div class="prob-display">
                    <span class="prob-num" style="color:{tier_color}">{prob:.1f}</span>
                    <span class="prob-suffix">% probability</span>
                </div>
            </div>
            <div class="gauge-container">
                <svg viewBox="0 0 100 55" class="gauge-svg">
                    <!-- Background arc -->
                    <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#2a2d36" stroke-width="8" stroke-linecap="round"/>
                    <!-- Risk arc -->
                    <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="{tier_color}" stroke-width="8"
                          stroke-linecap="round"
                          stroke-dasharray="{prob * 1.257:.1f} 125.7"
                          opacity="0.85"/>
                    <text x="50" y="48" text-anchor="middle" fill="#9ca3af" font-size="7" font-family="monospace">{prob:.0f}%</text>
                </svg>
            </div>
        </div>

        {red_flag_text}

        <div class="section-title">📋 Feature Contribution Analysis</div>
        <div class="features-list">
            {rows_html}
        </div>

        <div class="rec-box" style="border-color:{tier_color}40;">
            <div class="rec-title" style="color:{tier_color}">🩺 Clinical Recommendation</div>
            <div class="rec-text">{recommendations[tier]}</div>
        </div>

        <div class="disclaimer">
            ⚕️ This tool is for <strong>clinical decision support only</strong> and does not replace physician judgement, 
            complete history, examination, or validated diagnostic pathways.
        </div>
    </div>
    """
    return html


def clinical_diagnostic_tool(age, sex, cp, bp, chol, fbs, ecg, max_hr, angina, oldpeak, slope):
    if pipeline is None:
        return "<div style='color:#ef4444; padding:20px;'>⚠️ Model not loaded. Please ensure heart_disease_pipeline.joblib is present.</div>"

    sex_num = 1 if sex == "Male" else 0
    angina_num = 1 if angina == "Yes" else 0
    fbs_num = 1 if fbs else 0
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal": 2, "Asymptomatic": 3}
    cp_num = cp_map[cp]

    cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    input_df = pd.DataFrame(
        [[age, sex_num, cp_num, bp, chol, fbs_num, ecg, max_hr, angina_num, oldpeak, slope]],
        columns=cols
    )

    prob = pipeline.predict_proba(input_df)[0][1] * 100

    # Extract weights for reasoning
    model_step_name = pipeline.steps[-1][0]
    classifier = pipeline.named_steps[model_step_name]
    weights = classifier.coef_[0]

    return build_clinical_reasoning(input_df, weights, cols, prob)


# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #0d0f14 !important;
    font-family: 'DM Sans', sans-serif !important;
}

.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    padding: 24px 16px !important;
}

/* Header */
.app-header {
    text-align: center;
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid #1e2130;
}
.app-title {
    font-size: 26px;
    font-weight: 600;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    margin: 0 0 4px 0;
}
.app-subtitle {
    font-size: 13px;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
    margin: 0;
}

/* Panel */
.panel {
    background: #13161f;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
.panel-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #475569;
    margin: 0 0 14px 0;
    font-family: 'IBM Plex Mono', monospace;
}

/* Override Gradio inputs */
label { color: #94a3b8 !important; font-size: 12px !important; font-weight: 500 !important; }
input[type=number], .gr-input { background: #0d0f14 !important; border-color: #1e2130 !important; color: #e2e8f0 !important; border-radius: 8px !important; }
.gr-slider input { accent-color: #3b82f6; }

/* Analyze button */
#analyze-btn {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    height: 48px !important;
    letter-spacing: 0.3px;
    color: white !important;
    transition: all 0.2s;
    box-shadow: 0 4px 20px #2563eb30;
}
#analyze-btn:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    box-shadow: 0 6px 28px #2563eb50 !important;
}

/* Report card */
.report-card {
    background: #13161f;
    border: 1px solid #1e2130;
    border-radius: 12px;
    overflow: hidden;
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}

.report-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    border-bottom: 1px solid #1e2130;
    border-left: 4px solid;
}

.verdict-left { display: flex; flex-direction: column; gap: 6px; }

.tier-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    font-family: 'IBM Plex Mono', monospace;
    width: fit-content;
}

.prob-display { display: flex; align-items: baseline; gap: 4px; }
.prob-num { font-size: 40px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; line-height: 1; }
.prob-suffix { font-size: 14px; color: #64748b; }

.gauge-container { width: 110px; }
.gauge-svg { width: 100%; }

.red-flags {
    background: #1f0707;
    border-bottom: 1px solid #2d1010;
    padding: 10px 24px;
    font-size: 13px;
    color: #fca5a5;
    font-weight: 500;
}

.section-title {
    padding: 14px 24px 8px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #475569;
    font-family: 'IBM Plex Mono', monospace;
}

.features-list { padding: 0 24px 16px; display: flex; flex-direction: column; gap: 12px; }

.feat-row { padding: 12px; background: #0d0f14; border-radius: 8px; border: 1px solid #1e2130; }
.feat-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.feat-icon { font-size: 14px; }
.feat-name { font-size: 13px; font-weight: 600; flex: 1; }
.feat-badge {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-risk { background: #1f0707; color: #fca5a5; border: 1px solid #7f1d1d40; }
.badge-ok   { background: #052e16; color: #86efac; border: 1px solid #14532d40; }
.feat-pct { font-size: 12px; font-family: 'IBM Plex Mono', monospace; font-weight: 600; margin-left: auto; }

.feat-bar-bg { background: #1e2130; border-radius: 4px; height: 4px; margin-bottom: 8px; overflow: hidden; }
.feat-bar { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
.feat-note { font-size: 12px; color: #64748b; line-height: 1.5; }

.rec-box {
    margin: 0 24px 16px;
    padding: 14px 16px;
    background: #0d0f14;
    border: 1px solid;
    border-radius: 8px;
}
.rec-title { font-size: 12px; font-weight: 700; margin-bottom: 6px; letter-spacing: 0.5px; }
.rec-text { font-size: 13px; color: #94a3b8; line-height: 1.6; }

.disclaimer {
    padding: 12px 24px;
    font-size: 11px;
    color: #374151;
    background: #0a0c11;
    border-top: 1px solid #1e2130;
    line-height: 1.5;
}
"""

# ── UI Layout ──────────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Base(), css=CSS) as demo:

    gr.HTML("""
    <div class="app-header">
        <p class="app-title"> Cardiovascular Risk Assessment</p>
        <p class="app-subtitle">clinical decision support · logistic regression · v2.0</p>
    </div>
    """)

    with gr.Row():
        # Left column — Demographics & Vitals
        with gr.Column(scale=1):
            gr.HTML('<div class="panel-title">Patient Demographics & Vitals</div>')
            age   = gr.Slider(20, 100, value=52, label="Age (years)", step=1)
            sex   = gr.Radio(["Male", "Female"], value="Male", label="Biological Sex")
            bp    = gr.Number(value=135, label="Resting Blood Pressure (mmHg)")
            chol  = gr.Number(value=240, label="Total Cholesterol (mg/dL)")
            fbs   = gr.Checkbox(label="Fasting Blood Sugar > 120 mg/dL", value=False)

        # Right column — Cardiac Findings
        with gr.Column(scale=1):
            gr.HTML('<div class="panel-title">Cardiac Findings</div>')
            cp    = gr.Dropdown(
                ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"],
                value="Asymptomatic", label="Chest Pain Type"
            )
            hr    = gr.Number(value=150, label="Maximum Heart Rate Achieved (bpm)")
            ang   = gr.Radio(["Yes", "No"], value="No", label="Exercise-Induced Angina")
            old   = gr.Slider(0.0, 6.0, value=1.5, step=0.1, label="ST Depression / Oldpeak (mm)")
            ecg   = gr.Slider(0, 2, value=0, step=1, label="Resting ECG  (0=Normal · 1=ST abnormality · 2=LVH)")
            slp   = gr.Slider(0, 2, value=1, step=1, label="ST Slope  (0=Down · 1=Flat · 2=Up)")

    btn = gr.Button("⚕  Run Clinical Analysis", elem_id="analyze-btn")

    output_html = gr.HTML()

    btn.click(
        fn=clinical_diagnostic_tool,
        inputs=[age, sex, cp, bp, chol, fbs, ecg, hr, ang, old, slp],
        outputs=output_html
    )

demo.launch()