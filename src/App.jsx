import { useCallback, useMemo, useState } from "react";
import styles from "./App.module.scss";

const API_ENDPOINT = "http://localhost:8000/predict";
const SHORT_TEXT_WORD_LIMIT = 35;

const EMOTION_DETAILS = {
  admiration: {
    title: "Admiration",
    description:
      "Warm approval and appreciation of someone’s qualities or achievements.",
  },
  amusement: {
    title: "Amusement",
    description:
      "Playful enjoyment and light-hearted delight in the situation.",
  },
  anger: {
    title: "Anger",
    description:
      "Intense displeasure or hostility toward something perceived as wrong.",
  },
  annoyance: {
    title: "Annoyance",
    description:
      "Irritation and mild frustration that something is getting in the way.",
  },
  approval: {
    title: "Approval",
    description:
      "Positive acknowledgement that something meets expectations or values.",
  },
  caring: {
    title: "Caring",
    description:
      "Supportive concern and willingness to help or comfort someone.",
  },
  confusion: {
    title: "Confusion",
    description:
      "Uncertainty about what is happening or what decision to make.",
  },
  curiosity: {
    title: "Curiosity",
    description: "A desire to explore, learn more, or understand the unknown.",
  },
  desire: {
    title: "Desire",
    description: "A strong wish or longing for an outcome or experience.",
  },
  disappointment: {
    title: "Disappointment",
    description: "Let-down feelings caused by outcomes falling short of hopes.",
  },
  disapproval: {
    title: "Disapproval",
    description:
      "Judgement that something is not acceptable or is misaligned with values.",
  },
  disgust: {
    title: "Disgust",
    description:
      "Revulsion and rejection triggered by something unpleasant or offensive.",
  },
  embarrassment: {
    title: "Embarrassment",
    description:
      "Awkwardness and self-consciousness about how something looks to others.",
  },
  excitement: {
    title: "Excitement",
    description:
      "High energy and anticipation focused on something positive ahead.",
  },
  fear: {
    title: "Fear",
    description:
      "Unease or alarm about potential danger or negative consequences.",
  },
  gratitude: {
    title: "Gratitude",
    description:
      "A thankful recognition of help, support, or kindness received.",
  },
  grief: {
    title: "Grief",
    description: "Deep sorrow that lingers after a meaningful loss or tragedy.",
  },
  joy: {
    title: "Joy",
    description:
      "Bright, uplifting happiness that energizes the entire message.",
  },
  love: {
    title: "Love",
    description:
      "Warm affection, care, and emotional closeness toward someone or something.",
  },
  nervousness: {
    title: "Nervousness",
    description: "Restless tension and worry about what might happen next.",
  },
  optimism: {
    title: "Optimism",
    description:
      "Confidence that things will work out well, even if challenges remain.",
  },
  pride: {
    title: "Pride",
    description:
      "Satisfaction and honor connected to achievements or identity.",
  },
  realization: {
    title: "Realization",
    description:
      "A new understanding or fresh insight that reshapes the perspective.",
  },
  relief: {
    title: "Relief",
    description: "Release of tension after worries ease or a risk passes.",
  },
  remorse: {
    title: "Remorse",
    description: "Regret and self-reproach about a past choice or consequence.",
  },
  sadness: {
    title: "Sadness",
    description:
      "Quiet heaviness and sorrow about a loss, hardship, or unmet need.",
  },
  surprise: {
    title: "Surprise",
    description:
      "A sudden reaction to something unexpected or out of the ordinary.",
  },
  neutral: {
    title: "Neutral",
    description:
      "Balanced, matter-of-fact delivery without strong emotional signals.",
  },
};

const KNOWN_LABELS = Object.keys(EMOTION_DETAILS);

const SUGGESTIONS = [
  "The launch went better than expected, and I feel genuinely proud of how everyone showed up.",
  "I'm worried we might miss the next milestone, and I can sense the pressure building across the team.",
  "I still can't believe the update landed that well — the reactions were honestly mind-blowing.",
  "My energy is drained today, and I’m fighting to stay focused no matter how hard I try.",
];

const toTitleCase = (value) =>
  value
    .toLowerCase()
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");

const slugifyLabel = (value) =>
  value
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

const clampScore = (score) => Math.max(0, Math.min(1, score));

const normalizePredictions = (payload) => {
  if (payload == null) {
    return [];
  }

  let list = [];

  if (Array.isArray(payload)) {
    list = payload;
  } else if (Array.isArray(payload.predictions)) {
    list = payload.predictions;
  } else if (Array.isArray(payload.results)) {
    list = payload.results;
  } else if (Array.isArray(payload.output)) {
    list = payload.output;
  } else if (Array.isArray(payload.emotions)) {
    list = payload.emotions;
  } else if (payload.scores && typeof payload.scores === "object") {
    list = Object.entries(payload.scores).map(([label, score]) => ({
      label,
      score,
    }));
  } else if (typeof payload === "object") {
    const fallbackEntries = Object.entries(payload).filter(([key, value]) => {
      if (typeof value !== "number" && typeof value !== "string") {
        return false;
      }
      const slug = slugifyLabel(key);
      return KNOWN_LABELS.includes(slug);
    });

    if (fallbackEntries.length > 0) {
      list = fallbackEntries.map(([label, score]) => ({
        label,
        score,
      }));
    }
  }

  return list
    .map((item) => {
      if (item == null) {
        return null;
      }

      const rawLabel =
        item.label ?? item.emotion ?? item.id ?? item.name ?? item.tag ?? "";
      const label = rawLabel.toString().trim();
      if (!label) {
        return null;
      }

      const slug = slugifyLabel(label);

      const numericScore =
        typeof item.score === "number"
          ? item.score
          : typeof item.confidence === "number"
          ? item.confidence
          : typeof item.probability === "number"
          ? item.probability
          : typeof item.value === "number"
          ? item.value
          : typeof item.score === "string"
          ? Number.parseFloat(item.score)
          : typeof item.confidence === "string"
          ? Number.parseFloat(item.confidence)
          : typeof item.probability === "string"
          ? Number.parseFloat(item.probability)
          : typeof item.value === "string"
          ? Number.parseFloat(item.value)
          : null;

      if (!Number.isFinite(numericScore)) {
        return null;
      }

      return {
        id: slug,
        rawLabel: label,
        score: clampScore(numericScore),
      };
    })
    .filter(Boolean)
    .sort((a, b) => b.score - a.score);
};

const determineThreshold = (value) => {
  const words = value.split(/\s+/).filter(Boolean);
  const lengthType = words.length <= SHORT_TEXT_WORD_LIMIT ? "short" : "long";

  return { threshold: 0.09, lengthType };
};

const formatPercent = (value) => `${Math.round(value * 100)}%`;

function App() {
  const [text, setText] = useState("");
  const [analysis, setAnalysis] = useState({
    predictions: [],
    threshold: null,
    lengthType: null,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasAttempt, setHasAttempt] = useState(false);

  const enrichedPredictions = useMemo(
    () =>
      analysis.predictions.map((prediction) => {
        const metadata = EMOTION_DETAILS[prediction.id] ?? {
          title: toTitleCase(prediction.rawLabel),
          description:
            "We do not have additional context for this label yet, but the score passed your threshold.",
        };

        return {
          ...prediction,
          title: metadata.title,
          description: metadata.description,
        };
      }),
    [analysis.predictions]
  );

  const primary = enrichedPredictions[0] ?? null;

  const hasResult = enrichedPredictions.length > 0;

  const handleChange = (event) => {
    setText(event.target.value);
  };

  const runPrediction = useCallback(async (inputText) => {
    const trimmed = inputText.trim();

    if (!trimmed) {
      setAnalysis({ predictions: [], threshold: null, lengthType: null });
      setHasAttempt(false);
      setError(null);
      return;
    }

    const { threshold, lengthType } = determineThreshold(trimmed);

    setIsLoading(true);
    setError(null);
    setHasAttempt(true);

    try {
      const response = await fetch(API_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: trimmed,
          threshold,
        }),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data = await response.json();
      const predictions = normalizePredictions(data);

      setAnalysis({
        predictions,
        threshold,
        lengthType,
      });
    } catch (fetchError) {
      setAnalysis({
        predictions: [],
        threshold,
        lengthType,
      });

      setError(
        "Не удалось получить ответ от сервиса. Убедитесь, что API доступен по http://localhost:8000 и попробуйте ещё раз."
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleSubmit = async (event) => {
    event.preventDefault();
    await runPrediction(text);
  };

  const handleSuggestion = async (suggestion) => {
    setText(suggestion);
    await runPrediction(suggestion);
  };

  const isSubmitDisabled = text.trim().length < 5 || isLoading;

  const thresholdHint =
    analysis.lengthType != null
      ? "Default threshold 0.09 applied automatically for this text length."
      : null;

  return (
    <div className={styles.wrapper}>
      <div className={styles.panel}>
        <header className={styles.header}>
          <span className={styles.badge}>Emotion AI Toolkit</span>
          <h1 className={styles.title}>Emotions Prediction</h1>
          <p className={styles.subtitle}>
            Describe any message, feedback, or scenario. We will highlight the
            dominant emotions and supporting tones so you can respond with
            confidence.
          </p>
        </header>

        <div className={styles.contentGrid}>
          <section className={styles.inputSection}>
            <form className={styles.form} onSubmit={handleSubmit}>
              <label className={styles.label} htmlFor="emotion-input">
                Enter text to analyse
              </label>
              <textarea
                id="emotion-input"
                className={styles.textarea}
                placeholder="For example: The team’s resilience under pressure today left me genuinely proud and inspired."
                value={text}
                onChange={handleChange}
                spellCheck="true"
                aria-label="Text to analyse for emotional tone"
              />
              <div className={styles.actions}>
                <button
                  className={styles.submitButton}
                  type="submit"
                  disabled={isSubmitDisabled}
                >
                  {isLoading ? "Analysing..." : "Run prediction"}
                </button>
              </div>
            </form>

            <div className={styles.suggestions}>
              <span className={styles.suggestionsLabel}>
                Try a quick sample
              </span>
              <div className={styles.suggestionsList}>
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    className={styles.suggestionButton}
                    onClick={() => handleSuggestion(suggestion)}
                    disabled={isLoading}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          </section>

          <section className={styles.resultSection}>
            <div className={styles.resultHeader}>
              <h2 className={styles.resultTitle}>Prediction insights</h2>
              <p className={styles.resultDescription}>
                We surface every emotion that clears the chosen threshold so you
                can see the full tone profile in one glance.
              </p>
            </div>

            {thresholdHint && (
              <div className={styles.metaStrip}>
                <div className={styles.metaItem}>
                  <span className={styles.metaLabel}>Threshold</span>
                  <span className={styles.metaValue}>
                    {analysis.threshold != null
                      ? analysis.threshold.toFixed(2)
                      : "—"}
                  </span>
                </div>
                <div className={styles.metaItem}>
                  <span className={styles.metaLabel}>Input length</span>
                  <span className={styles.metaValue}>
                    {analysis.lengthType === "short"
                      ? "Short text"
                      : "Long text"}
                  </span>
                </div>
                <div className={styles.metaHint}>{thresholdHint}</div>
              </div>
            )}

            {error && (
              <div className={`${styles.statusCard} ${styles.statusCardError}`}>
                {error}
              </div>
            )}

            {isLoading && (
              <div className={`${styles.statusCard} ${styles.statusCardInfo}`}>
                Fetching predictions from the model…
              </div>
            )}

            {!isLoading && !error && hasResult && primary && (
              <>
                <div className={styles.highlight}>
                  <span className={styles.highlightLabel}>Primary emotion</span>
                  <div className={styles.primaryLine}>
                    <span
                      className={`${styles.emotionChip} ${
                        styles[`chip-${primary.id}`] ?? styles["chip-default"]
                      }`}
                    >
                      {primary.title}
                    </span>
                    <span className={styles.primaryScore}>
                      {formatPercent(primary.score)}
                    </span>
                  </div>
                  <p className={styles.primaryDescription}>
                    {primary.description}
                  </p>
                </div>

                <ul className={styles.rankingList}>
                  {enrichedPredictions.slice(0, 8).map((emotion) => (
                    <li key={emotion.id} className={styles.rankingItem}>
                      <div className={styles.rankingHeader}>
                        <span
                          className={`${styles.emotionChip} ${
                            styles[`chip-${emotion.id}`] ??
                            styles["chip-default"]
                          }`}
                        >
                          {emotion.title}
                        </span>
                        <span className={styles.rankingValue}>
                          {formatPercent(emotion.score)}
                        </span>
                      </div>
                      <div className={styles.progress}>
                        <span
                          className={`${styles.progressFill} ${
                            styles[`barFill-${emotion.id}`] ??
                            styles["barFill-default"]
                          }`}
                          style={{ width: formatPercent(emotion.score) }}
                        />
                      </div>
                      <p className={styles.rankingDescription}>
                        {emotion.description}
                      </p>
                    </li>
                  ))}
                </ul>
              </>
            )}

            {!isLoading && !error && hasAttempt && !hasResult && (
              <div className={styles.placeholder}>
                No emotions passed the current threshold, but your text is
                captured. Try adding more detail or lowering the threshold to
                surface subtler tones.
              </div>
            )}

            {!hasAttempt && (
              <div className={styles.placeholder}>
                Paste or type any message to reveal its emotional fingerprint in
                real time.
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}

export default App;
