# AI Methodology and the CAPEXX Robot

## Positioning

The robot (CAPEXX AI AGENT) is the **communication layer** of the platform.
It never performs financial calculations. All numbers come from
`capexx_engine.run_analysis()`.

```
CALCULATIONS
     ↓
AI INTERPRETATION
     ↓
MANAGEMENT REPORT ─→ EMAIL
     ↓
ROBOT VOICE (audio)
```

One source of truth means the report, email and audio never disagree.

## Explainable AI

For every major result CAPEXX answers five questions with real numbers:

1. **WHAT HAPPENED?** - describes the result (e.g. NPV value).
2. **WHY?** - names the drivers (capex size, revenue, costs, discount rate,
   construction length, currency effects).
3. **SO WHAT?** - the financial meaning for management.
4. **WHAT IF?** - scenario and stress outcomes.
5. **WHAT SHOULD MANAGEMENT MONITOR?** - factual monitoring points.

Every conclusion is traceable:

```
Input → Calculation → Evidence → Interpretation
```

The **"Why did CAPEXX reach this result?"** button expands exactly those four
steps for the current project.

## Narration script

`build_narration_text()` dynamically assembles a spoken briefing from the
analysis results (project, capex, NPV, IRR, MIRR, payback, risks, currency,
scenarios, stress, decision, monitoring points, disclaimer). The final
decision is never hard-coded - it always comes from the decision engine.

## Text-to-speech

- gTTS (Google Text-to-Speech) is used when available; it produces an .mp3
  saved under `outputs/audio/`.
- `synthesize()` returns a status dict; on any failure the interface shows
  **⚠️ VOICE SERVICE UNAVAILABLE** and displays the narration script instead.
- The rest of the application continues to work regardless of TTS state.

## Failure handling

The robot gracefully degrades on: missing project data, invalid numbers,
missing currencies/rates, failed report/audio generation, missing robot
image, missing TTS package, and network/API failure. No optional feature can
crash the platform.

## Global behaviour

The robot works for any capital project in any country. If the current
project is the GZU demo it adds the standard disclaimer that the figures are
hypothetical, not actual GZU financial data.