document.addEventListener("DOMContentLoaded", () => {
  const q1Input = document.getElementById("question1");
  const q2Input = document.getElementById("question2");
  const compareBtn = document.getElementById("compareBtn");
  const clearBtn = document.getElementById("clearBtn");
  const errorMsg = document.getElementById("errorMsg");

  const resultSection = document.getElementById("resultSection");
  const resultTag = document.getElementById("resultTag");
  const confidenceValue = document.getElementById("confidenceValue");
  const confidenceBar = document.getElementById("confidenceBar");
  const similarityValue = document.getElementById("similarityValue");
  const similarityBar = document.getElementById("similarityBar");
  const resultExplanation = document.getElementById("resultExplanation");

  const exampleChips = document.querySelectorAll(".example-chip");

  function showError(message) {
    errorMsg.textContent = message;
    errorMsg.hidden = false;
  }

  function hideError() {
    errorMsg.hidden = true;
    errorMsg.textContent = "";
  }

  function setLoading(isLoading) {
    compareBtn.classList.toggle("loading", isLoading);
    compareBtn.disabled = isLoading;
  }

  function explanationFor(similarity) {
    if (similarity >= 0.66) {
      return "These questions have strong similarity in their wording and features.";
    } else if (similarity >= 0.33) {
      return "These questions share some similarities but may have different meanings.";
    }
    return "These questions appear to discuss different topics or intentions.";
  }

  function renderResult(data) {
    resultSection.hidden = false;

    resultTag.textContent = data.is_duplicate ? "DUPLICATE" : "NOT DUPLICATE";
    resultTag.className = "result-tag " + (data.is_duplicate ? "is-duplicate" : "is-not-duplicate");

    const confidencePct = Math.round(data.confidence * 100);
    const similarityPct = Math.round(data.similarity * 100);

    confidenceValue.textContent = confidencePct + "%";
    similarityValue.textContent = similarityPct + "%";

    // Reset then animate on next frame so the transition is visible.
    confidenceBar.style.width = "0%";
    similarityBar.style.width = "0%";
    requestAnimationFrame(() => {
      confidenceBar.style.width = confidencePct + "%";
      similarityBar.style.width = similarityPct + "%";
    });

    resultExplanation.textContent = explanationFor(data.similarity);

    resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  async function compareQuestions() {
    const question1 = q1Input.value.trim();
    const question2 = q2Input.value.trim();

    hideError();

    if (!question1 || !question2) {
      showError("Please enter both questions.");
      return;
    }

    setLoading(true);
    resultSection.hidden = true;

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question1, question2 }),
      });

      const data = await response.json();

      if (!response.ok) {
        showError(data.error || "Something went wrong. Please try again.");
        return;
      }

      renderResult(data);
    } catch (err) {
      showError("Could not reach the server. Please check that the Flask app is running.");
    } finally {
      setLoading(false);
    }
  }

  function clearAll() {
    q1Input.value = "";
    q2Input.value = "";
    hideError();
    resultSection.hidden = true;
  }

  compareBtn.addEventListener("click", compareQuestions);
  clearBtn.addEventListener("click", clearAll);

  exampleChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      q1Input.value = chip.dataset.q1;
      q2Input.value = chip.dataset.q2;
      hideError();
      resultSection.hidden = true;
      q1Input.focus();
    });
  });
});
