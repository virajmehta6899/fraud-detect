"""
LLM Explainer — Uses Groq to explain why a transaction was flagged
Combines SHAP values + LLM narrative explanation
"""
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def explain_prediction(
    prediction: str,
    confidence: float,
    amount: float,
    top_features: list,
) -> str:
    """
    Generate a human-readable explanation for fraud prediction.
    
    Args:
        prediction:   'FRAUD' or 'LEGITIMATE'
        confidence:   probability score 0-1
        amount:       transaction amount
        top_features: list of {feature, value, importance} dicts
    """

    features_text = "\n".join([
        f"  - {f['feature']}: value={f['value']:.4f}, importance={f['importance']:.4f}"
        for f in top_features
    ])

    prompt = f"""You are a fraud detection expert at a bank. Explain why this transaction 
was classified as {prediction} in simple, clear language for a bank analyst.

Transaction Details:
- Amount: €{amount:.2f}
- Prediction: {prediction}
- Confidence: {confidence*100:.1f}%

Top contributing features from the ML model:
{features_text}

Note: Features V1-V28 are PCA-transformed bank transaction features (anonymized).
High absolute values in features like V14, V4, V11 are known fraud indicators.

Provide a 3-4 sentence explanation that:
1. States the verdict clearly
2. Explains the key risk factors in plain English
3. Suggests what action the bank should take

Keep it professional and concise."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()