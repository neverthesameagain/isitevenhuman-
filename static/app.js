/* ═══════════════════════════════════════════════════════════
   TextScope v2 — Frontend Logic
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
    const featureCount = document.getElementById("featureCount");
    const featureActive = document.getElementById("featureActiveCount");
    const featuresGrid = document.getElementById("featuresGrid");
    const featureSearch = document.getElementById("featureSearch");
    const featureGroupFilter = document.getElementById("featureGroupFilter");
    const featureGroupGrid = document.getElementById("featureGroupGrid");
    const selectAllBtn = document.getElementById("selectAll");
    const selectNoneBtn = document.getElementById("selectNone");
    const sentenceArea = document.getElementById("sentenceHighlightArea");

    let currentModel = "new";
    let enabledGroups = new Set();
    let featureGroups = [];
    let lastFeatures = {};

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
            populateGroupFilter();
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

    function populateGroupFilter() {
        featureGroups.forEach(g => {
            const opt = document.createElement("option");
            opt.value = g.key;
            opt.textContent = g.label;
            featureGroupFilter.appendChild(opt);
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

            // ✅ Read response only once
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
        verdictModel.textContent = `Model: ${data.model_variant} · ${data.active_features}/${data.feature_count} features active`;

        // Gauges
        requestAnimationFrame(() => {
            gaugeFillAI.style.strokeDashoffset = CIRCUMFERENCE * (1 - data.probability_ai);
            gaugeFillHuman.style.strokeDashoffset = CIRCUMFERENCE * (1 - data.probability_human);
        });
        gaugeValAI.textContent = (data.probability_ai * 100).toFixed(1) + "%";
        gaugeValHuman.textContent = (data.probability_human * 100).toFixed(1) + "%";

        // ── Sentence-level highlighting ──
        renderSentences(data.sentences || []);

        // Feature counts
        featureCount.textContent = data.feature_count + " total";
        featureActive.textContent = data.active_features + " active";

        // Features
        lastFeatures = data.features;
        renderFeatureCards(lastFeatures);

        resultsSection.style.display = "block";
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // ── Sentence rendering ──────────────────────────────────
    function renderSentences(sentences) {
        sentenceArea.innerHTML = "";
        if (!sentences.length) {
            sentenceArea.textContent = "No sentences detected.";
            return;
        }
        sentences.forEach(s => {
            const span = document.createElement("span");
            span.className = "sent-span sent-" + s.prediction;
            span.textContent = s.text + " ";

            // Tooltip with probability
            const tip = document.createElement("span");
            tip.className = "sent-tooltip";
            const pct = (s.probability_ai * 100).toFixed(1);
            tip.textContent = s.prediction === "ai"
                ? `AI ${pct}%`
                : `Human ${(100 - pct).toFixed(1)}%`;
            span.appendChild(tip);

            sentenceArea.appendChild(span);
        });
    }

    // ── Feature cards ───────────────────────────────────────
    function renderFeatureCards(features, textFilter = "", groupFilter = "all") {
        featuresGrid.innerHTML = "";
        const entries = Object.entries(features);
        const maxVal = Math.max(...entries.map(([, f]) => Math.abs(f.value)), 1e-9);

        entries.forEach(([name, info]) => {
            if (textFilter && !name.toLowerCase().includes(textFilter.toLowerCase())) return;
            if (groupFilter !== "all" && info.group !== groupFilter) return;

            const card = document.createElement("div");
            card.className = "feat-card" + (info.active ? "" : " inactive");

            const barWidth = Math.min(100, (Math.abs(info.value) / maxVal) * 100);
            const groupLabel = featureGroups.find(g => g.key === info.group);

            card.innerHTML = `
                <div class="feat-card-header">
                    <span class="feat-name" title="${name}">${name}</span>
                    <span class="feat-group-tag">${groupLabel ? groupLabel.label.split(" ")[0] : info.group}</span>
                </div>
                <span class="feat-value">${formatValue(info.value)}</span>
                <div class="feat-bar">
                    <div class="feat-bar-fill" style="width:${barWidth}%"></div>
                </div>
            `;
            featuresGrid.appendChild(card);
        });
    }

    function formatValue(v) {
        if (Number.isInteger(v)) return v.toString();
        if (Math.abs(v) < 0.0001 && v !== 0) return v.toExponential(2);
        return v.toFixed(4);
    }

    // ── Filters ─────────────────────────────────────────────
    featureSearch.addEventListener("input", () => {
        renderFeatureCards(lastFeatures, featureSearch.value, featureGroupFilter.value);
    });
    featureGroupFilter.addEventListener("change", () => {
        renderFeatureCards(lastFeatures, featureSearch.value, featureGroupFilter.value);
    });

    // ── Init ────────────────────────────────────────────────
    loadFeatureGroups();
})();
