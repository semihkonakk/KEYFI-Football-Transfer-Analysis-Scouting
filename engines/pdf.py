# PDF
import numpy as np
from fpdf import YPos, XPos, FPDF
from matplotlib import pyplot as plt


def draw_progress_bar(pdf, label, value, x=10, y=None, width=120, height=6):
    """
    value: 0–1 veya 0–100 kabul eder
    """
    # ölçek düzelt
    if value <= 1:
        value = value * 100

    if y:
        pdf.set_xy(x, y)

    pdf.set_font("DejaVu", size=10)
    pdf.cell(40, height, label)

    # renk
    if value >= 75:
        pdf.set_fill_color(40, 167, 69)   # green
    elif value >= 60:
        pdf.set_fill_color(255, 193, 7)   # yellow
    else:
        pdf.set_fill_color(220, 53, 69)   # red

    fill_width = width * (value / 100)

    pdf.cell(fill_width, height, "", fill=True)
    pdf.rect(pdf.get_x() - fill_width, pdf.get_y(), width, height)

    pdf.ln(height + 2)

def risk_level(risk_score):
    if risk_score <= 30:
        return "LOW", (40, 167, 69)
    elif risk_score <= 50:
        return "MEDIUM", (255, 193, 7)
    else:
        return "HIGH", (220, 53, 69)

def transfer_decision(final_score, risk_score):
    if final_score >= 75 and risk_score <= 30:
        return "BUY"
    elif final_score >= 60 and risk_score <= 50:
        return "HOLD"
    else:
        return "PASS"

def decision_explanation(final_score, risk_score):
    reasons = []

    # normalize scale
    if final_score <= 1:
        fs = final_score * 100
    else:
        fs = final_score

    if fs < 60:
        reasons.append("Final performance score below acceptable level")

    if risk_score > 0.50:
        reasons.append("High injury risk")

    if not reasons:
        reasons.append("Decision driven by overall model constraints")

    return " | ".join(reasons)

feature_contributions = {
    "goals_per_90": 0.18,
    "goal_contribution_per90": 0.14,
    "minutes_played": 0.11,
    "injury_days_per_season": -0.09,
    "age": -0.04
}

def explainable_ai_text(contributions, top_n=3):
    sorted_feats = sorted(contributions.items(),
                           key=lambda x: abs(x[1]),
                           reverse=True)[:top_n]

    positives = [f.replace("_", " ") for f,v in sorted_feats if v > 0]
    negatives = [f.replace("_", " ") for f,v in sorted_feats if v < 0]

    text = "Model decision is mainly driven by "

    if positives:
        text += f"strong {', '.join(positives)}"
    if negatives:
        text += f", but negatively impacted by {', '.join(negatives)}"

    return text + "."

def generate_player_report(row, file_name="scout_report.pdf"):
    pdf = FPDF()
    pdf.add_page()

    # LOGO
    pdf.image("assets/keyfi.png", x=10, y=8, w=30)
    pdf.ln(25)

    # FONTLAR
    pdf.add_font("DejaVu", "", "Fonts/DejaVuSans.ttf")
    pdf.add_font("DejaVu", "B", "Fonts/DejaVuSans-Bold.ttf")

    # HEADER
    pdf.set_font("DejaVu", "B", 16)
    pdf.cell(0, 12, f"Scout Report – {row['player_name']}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("DejaVu", size=12)
    pdf.cell(0, 8, f"Position: {row['main_position']} | Age: {row['age']}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, f"Market Value: €{int(row['market_value']):,}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(5)

    # PERFORMANCE SCORES
    pdf.set_font("DejaVu", "B", 13)
    pdf.cell(0, 8, "Performance Scores",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    draw_progress_bar(pdf, "Final Score", row["final_score"])
    draw_progress_bar(pdf, "ML Prediction", row["ml_pred"])
    draw_progress_bar(pdf, "Ensemble Score", row["ensemble_score"])

    # RISK PROFILE
    level, color = risk_level(row["risk_score"])

    pdf.ln(3)
    pdf.set_font("DejaVu", "B", 13)
    pdf.cell(0, 8, "Risk Profile",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(40, 10, level, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(12)

    pdf.set_font("DejaVu", size=11)
    pdf.cell(0, 8, f"Risk Score: {row['risk_score']:.2f}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, f"Injury Days / Season: {row['injury_days_per_season']:.1f}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # TRANSFER DECISION
    decision = transfer_decision(row["final_score"], row["risk_score"])

    pdf.ln(4)
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, f"Transfer Recommendation: {decision}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # DECISION RATIONALE
    explanation = decision_explanation(row["final_score"], row["risk_score"])
    pdf.set_font("DejaVu", size=10)
    pdf.multi_cell(0, 6, f"Decision Rationale: {explanation}")

    # EXPLAINABLE AI
    ai_text = explainable_ai_text(feature_contributions)

    pdf.ln(2)
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 8, "Model Explanation (XAI)",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("DejaVu", size=10)
    pdf.multi_cell(0, 6, ai_text)

    # SCOUT COMMENT
    pdf.ln(2)
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 8, "Scout Comment",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("DejaVu", size=11)
    pdf.multi_cell(0, 8, row["scout_comment"])

    pdf.output(file_name)
