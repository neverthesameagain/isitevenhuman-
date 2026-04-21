/* ═══════════════════════════════════════════════════════════
   isitEven Human? — Frontend Logic
   ═══════════════════════════════════════════════════════════ */

(function () {
    "use strict";

    // ── DOM refs ────────────────────────────────────────────
    const textInput = document.getElementById("textInput");
    const charCount = document.getElementById("charCount");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const resultsSection = document.getElementById("resultsSection");
    const verdictBadge = document.getElementById("verdictBadge");
    const verdictLabel = document.getElementById("verdictLabel");
    const verdictModel = document.getElementById("verdictModel");
    const gaugeFillAI = document.getElementById("gaugeFillAI");
    const gaugeFillHuman = document.getElementById("gaugeFillHuman");
    const gaugeValAI = document.getElementById("gaugeValAI");
    const gaugeValHuman = document.getElementById("gaugeValHuman");
    const featureGroupGrid = document.getElementById("featureGroupGrid");
    const selectAllBtn = document.getElementById("selectAll");
    const selectNoneBtn = document.getElementById("selectNone");

    let currentModel = "new";
    let enabledGroups = new Set();
    let featureGroups = [];

    const CIRCUMFERENCE = 2 * Math.PI * 52;

    // ── Model toggle ────────────────────────────────────────
    document.querySelectorAll(".toggle-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentModel = btn.dataset.model;
        });
    });

    // ── Char counter ────────────────────────────────────────
    textInput.addEventListener("input", () => {
        charCount.textContent = textInput.value.length + " chars";
    });

    // ── Load feature groups from backend ────────────────────
    async function loadFeatureGroups() {
        try {
            const res = await fetch("/api/feature-groups");
            const data = await res.json();
            featureGroups = data.groups;
            featureGroups.forEach(g => enabledGroups.add(g.key));
            renderGroupChips();
        } catch (e) {
            console.error("Failed to load feature groups:", e);
        }
    }

    function renderGroupChips() {
        featureGroupGrid.innerHTML = "";
        featureGroups.forEach(group => {
            const chip = document.createElement("div");
            chip.className = "fs-chip" + (enabledGroups.has(group.key) ? " active" : "");
            chip.dataset.group = group.key;
            chip.innerHTML = `
                <div class="fs-chip-toggle"></div>
                <div class="fs-chip-info">
                    <span class="fs-chip-label">${group.label}</span>
                    <span class="fs-chip-desc">${group.description}</span>
                </div>
            `;
            chip.addEventListener("click", () => {
                if (enabledGroups.has(group.key)) {
                    enabledGroups.delete(group.key);
                    chip.classList.remove("active");
                } else {
                    enabledGroups.add(group.key);
                    chip.classList.add("active");
                }
            });
            featureGroupGrid.appendChild(chip);
        });
    }

    // ── Select All / None ───────────────────────────────────
    selectAllBtn.addEventListener("click", () => {
        featureGroups.forEach(g => enabledGroups.add(g.key));
        document.querySelectorAll(".fs-chip").forEach(c => c.classList.add("active"));
    });
    selectNoneBtn.addEventListener("click", () => {
        enabledGroups.clear();
        document.querySelectorAll(".fs-chip").forEach(c => c.classList.remove("active"));
    });

    // ── Analyze ─────────────────────────────────────────────
    analyzeBtn.addEventListener("click", async () => {
        const text = textInput.value.trim();
        if (!text) return;
        if (enabledGroups.size === 0) {
            alert("Please enable at least one feature group.");
            return;
        }

        analyzeBtn.classList.add("loading");
        resultsSection.style.display = "none";

        try {
            const body = {
                text,
                model_variant: currentModel,
                enabled_groups: Array.from(enabledGroups),
            };
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });

            const responseText = await res.text();

            let data;
            try {
                data = JSON.parse(responseText);
            } catch {
                throw new Error("Invalid server response: " + responseText.slice(0, 200));
            }

            if (!res.ok) {
                throw new Error(data.detail || "Server error");
            }

            renderResults(data);
        } catch (e) {
            alert("Error: " + e.message);
        } finally {
            analyzeBtn.classList.remove("loading");
        }
    });

    textInput.addEventListener("keydown", e => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyzeBtn.click();
    });

    // ── Render results ──────────────────────────────────────
    function renderResults(data) {
        const isAI = data.prediction === "AI-generated";

        // Verdict
        verdictBadge.className = "verdict-badge " + (isAI ? "ai" : "human");
        verdictBadge.textContent = isAI ? "AI" : "H";
        verdictLabel.textContent = data.prediction;
        verdictLabel.style.color = isAI ? "var(--ai-color)" : "var(--human-color)";
        verdictModel.textContent = `Inference processed!`;

        // Gauges
        requestAnimationFrame(() => {
            gaugeFillAI.style.strokeDashoffset = CIRCUMFERENCE * (1 - data.probability_ai);
            gaugeFillHuman.style.strokeDashoffset = CIRCUMFERENCE * (1 - data.probability_human);
        });
        gaugeValAI.textContent = (data.probability_ai * 100).toFixed(1) + "%";
        gaugeValHuman.textContent = (data.probability_human * 100).toFixed(1) + "%";

        resultsSection.style.display = "block";
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // ── Init ────────────────────────────────────────────────
    loadFeatureGroups();
})();
