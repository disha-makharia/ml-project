document.addEventListener("DOMContentLoaded", () => {
    // ============================================================
    // INPUT ELEMENTS
    // ============================================================

    const q1Input = document.getElementById("question1");
    const q2Input = document.getElementById("question2");

    const compareBtn = document.getElementById("compareBtn");
    const clearBtn = document.getElementById("clearBtn");

    const errorMsg = document.getElementById("errorMsg");

    // ============================================================
    // RESULT ELEMENTS
    // ============================================================

    const resultSection = document.getElementById("resultSection");

    const resultTag = document.getElementById("resultTag");

    const confidenceValue =
        document.getElementById("confidenceValue");

    const confidenceBar =
        document.getElementById("confidenceBar");

    const similarityValue =
        document.getElementById("similarityValue");

    const similarityBar =
        document.getElementById("similarityBar");

    const resultExplanation =
        document.getElementById("resultExplanation");

    // Example buttons
    const exampleChips =
        document.querySelectorAll(".example-chip");


    // ============================================================
    // ERROR HANDLING
    // ============================================================

    function showError(message) {
        errorMsg.textContent = message;
        errorMsg.hidden = false;
    }


    function hideError() {
        errorMsg.textContent = "";
        errorMsg.hidden = true;
    }


    // ============================================================
    // LOADING STATE
    // ============================================================

    function setLoading(isLoading) {

        compareBtn.classList.toggle(
            "loading",
            isLoading
        );

        compareBtn.disabled = isLoading;
    }


    // ============================================================
    // SAFE PERCENTAGE
    // ============================================================

    function toPercentage(value) {

        const number = Number(value);

        if (!Number.isFinite(number)) {
            return 0;
        }

        return Math.min(
            100,
            Math.max(
                0,
                Math.round(number * 100)
            )
        );
    }


    // ============================================================
    // RESULT EXPLANATION
    // ============================================================

    function explanationFor(
        isDuplicate,
        similarity
    ) {

        const percentage =
            toPercentage(similarity);


        if (isDuplicate) {

            if (percentage >= 75) {

                return (
                    "These questions have a high degree of similarity "
                    + "and are likely asking about the same intent."
                );

            } else if (percentage >= 50) {

                return (
                    "These questions have similar meaning and "
                    + "appear to ask about the same intent."
                );

            } else {

                return (
                    "The model detected enough similarity between "
                    + "the questions to classify them as duplicate."
                );
            }
        }


        // NOT DUPLICATE

        if (percentage >= 50) {

            return (
                "These questions share some similarities, "
                + "but the model detected differences in their intent."
            );

        } else if (percentage >= 30) {

            return (
                "These questions have some common features, "
                + "but are likely asking different things."
            );

        } else {

            return (
                "These questions appear to have different "
                + "topics or intentions."
            );
        }
    }


    // ============================================================
    // RENDER RESULT
    // ============================================================

    function renderResult(data) {

        resultSection.hidden = false;


        // --------------------------------------------------------
        // Duplicate / Not Duplicate
        // --------------------------------------------------------

        const isDuplicate =
            Boolean(data.is_duplicate);


        resultTag.textContent =
            isDuplicate
                ? "DUPLICATE"
                : "NOT DUPLICATE";


        resultTag.className =
            "result-tag " +
            (
                isDuplicate
                    ? "is-duplicate"
                    : "is-not-duplicate"
            );


        // --------------------------------------------------------
        // Confidence
        // --------------------------------------------------------

        const confidencePct =
            toPercentage(data.confidence);


        // --------------------------------------------------------
        // Similarity
        //
        // The backend now sends:
        //
        // data.similarity
        //
        // which represents the hybrid similarity score.
        // --------------------------------------------------------

        const similarityPct =
            toPercentage(data.similarity);


        // --------------------------------------------------------
        // Update text
        // --------------------------------------------------------

        confidenceValue.textContent =
            confidencePct + "%";


        similarityValue.textContent =
            similarityPct + "%";


        // --------------------------------------------------------
        // Reset progress bars
        // --------------------------------------------------------

        confidenceBar.style.width = "0%";
        similarityBar.style.width = "0%";


        // --------------------------------------------------------
        // Animate progress bars
        // --------------------------------------------------------

        requestAnimationFrame(() => {

            confidenceBar.style.width =
                confidencePct + "%";

            similarityBar.style.width =
                similarityPct + "%";
        });


        // --------------------------------------------------------
        // Explanation
        // --------------------------------------------------------

        resultExplanation.textContent =
            explanationFor(
                isDuplicate,
                data.similarity
            );


        // --------------------------------------------------------
        // Scroll to result
        // --------------------------------------------------------

        resultSection.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });
    }


    // ============================================================
    // COMPARE QUESTIONS
    // ============================================================

    async function compareQuestions() {

        const question1 =
            q1Input.value.trim();

        const question2 =
            q2Input.value.trim();


        hideError();


        // --------------------------------------------------------
        // Validation
        // --------------------------------------------------------

        if (!question1 || !question2) {

            showError(
                "Please enter both questions."
            );

            return;
        }


        // --------------------------------------------------------
        // Loading
        // --------------------------------------------------------

        setLoading(true);

        resultSection.hidden = true;


        try {

            // ----------------------------------------------------
            // Send request to Flask
            // ----------------------------------------------------

            const response =
                await fetch(
                    "/predict",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            question1: question1,
                            question2: question2
                        })
                    }
                );


            // ----------------------------------------------------
            // Read response
            // ----------------------------------------------------

            const data =
                await response.json();


            // ----------------------------------------------------
            // Server error
            // ----------------------------------------------------

            if (!response.ok) {

                showError(
                    data.error ||
                    "Something went wrong. Please try again."
                );

                return;
            }


            // ----------------------------------------------------
            // Validate similarity
            // ----------------------------------------------------

            if (
                data.similarity === undefined ||
                data.similarity === null
            ) {

                console.warn(
                    "Backend did not return similarity:",
                    data
                );

                showError(
                    "The server did not return a similarity score."
                );

                return;
            }


            // ----------------------------------------------------
            // Render
            // ----------------------------------------------------

            renderResult(data);

        }

        catch (error) {

            console.error(
                "Prediction error:",
                error
            );

            showError(
                "Could not reach the server. "
                + "Please check that the Flask app is running."
            );

        }

        finally {

            setLoading(false);
        }
    }


    // ============================================================
    // CLEAR
    // ============================================================

    function clearAll() {

        q1Input.value = "";
        q2Input.value = "";

        hideError();

        resultSection.hidden = true;

        q1Input.focus();
    }


    // ============================================================
    // BUTTON EVENTS
    // ============================================================

    compareBtn.addEventListener(
        "click",
        compareQuestions
    );


    clearBtn.addEventListener(
        "click",
        clearAll
    );


    // ============================================================
    // EXAMPLE CHIPS
    // ============================================================

    exampleChips.forEach((chip) => {

        chip.addEventListener(
            "click",
            () => {

                q1Input.value =
                    chip.dataset.q1;

                q2Input.value =
                    chip.dataset.q2;

                hideError();

                resultSection.hidden = true;

                q1Input.focus();
            }
        );
    });


    // ============================================================
    // ENTER KEY SUPPORT
    // ============================================================

    q1Input.addEventListener(
        "keydown",
        (event) => {

            if (
                event.key === "Enter" &&
                event.ctrlKey
            ) {

                compareQuestions();
            }
        }
    );


    q2Input.addEventListener(
        "keydown",
        (event) => {

            if (
                event.key === "Enter" &&
                event.ctrlKey
            ) {

                compareQuestions();
            }
        }
    );
});