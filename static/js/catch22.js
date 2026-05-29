function copyBibTeX() {
  const bibtexElement = document.getElementById("bibtex-code");
  const button = document.querySelector(".copy-bibtex-btn");
  const copyText = button ? button.querySelector(".copy-text") : null;

  if (!bibtexElement || !button || !copyText) {
    return;
  }

  const setCopiedState = () => {
    button.classList.add("copied");
    copyText.textContent = "Copied";
    window.setTimeout(() => {
      button.classList.remove("copied");
      copyText.textContent = "Copy";
    }, 1800);
  };

  if (navigator.clipboard) {
    navigator.clipboard.writeText(bibtexElement.textContent).then(setCopiedState).catch(() => {
      fallbackCopy(bibtexElement.textContent);
      setCopiedState();
    });
  } else {
    fallbackCopy(bibtexElement.textContent);
    setCopiedState();
  }
}

function fallbackCopy(text) {
  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.setAttribute("readonly", "");
  textArea.style.position = "absolute";
  textArea.style.left = "-9999px";
  document.body.appendChild(textArea);
  textArea.select();
  document.execCommand("copy");
  document.body.removeChild(textArea);
}

function scrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: "smooth"
  });
}

window.addEventListener("scroll", () => {
  const scrollButton = document.querySelector(".scroll-to-top");
  if (!scrollButton) {
    return;
  }
  scrollButton.classList.toggle("visible", window.scrollY > 320);
});

const siteTabDefaults = {
  home: "home",
  watermarks: "watermark-types",
  catch22: "catch22-overview",
  hybrid: "selector-walkthrough",
  citation: "reproducibility"
};

function activeSiteTabForElement(element) {
  const panel = element ? element.closest("[data-site-tab-panel]") : null;
  return panel ? panel.dataset.siteTabPanel : null;
}

function activateSiteTab(tabKey, options = {}) {
  const activePanel = document.querySelector(`[data-site-tab-panel="${tabKey}"]`);
  if (!activePanel) {
    return false;
  }

  document.querySelectorAll(".site-tab").forEach((tab) => {
    const isActive = tab.dataset.siteTab === tabKey;
    tab.classList.toggle("is-active", isActive);
    tab.setAttribute("aria-selected", String(isActive));
  });

  document.querySelectorAll(".site-tab-panel").forEach((panel) => {
    const isActive = panel.dataset.siteTabPanel === tabKey;
    panel.classList.toggle("is-active", isActive);
    panel.hidden = !isActive;
  });

  if (options.scroll !== false) {
    const targetId = options.targetId || siteTabDefaults[tabKey];
    const target = document.getElementById(targetId) || activePanel;
    const behavior = options.behavior || "smooth";
    window.requestAnimationFrame(() => {
      target.scrollIntoView({ behavior, block: "start" });
    });
  }

  return true;
}

function activateSiteTabFromHash(hash, options = {}) {
  if (!hash || hash === "#") {
    return false;
  }

  const targetId = decodeURIComponent(hash.slice(1));
  const target = document.getElementById(targetId);
  const tabKey = activeSiteTabForElement(target);
  if (!tabKey) {
    return false;
  }

  return activateSiteTab(tabKey, {
    targetId,
    behavior: options.behavior || "auto",
    scroll: options.scroll
  });
}

const catch22Modes = {
  biased: {
    title: "Biased probability watermark",
    kicker: "High signal, high visibility",
    copy: "A strong sampling bias accumulates evidence quickly, but the same probability shift gives a keyless observer more statistical drift to exploit.",
    note: "Strong evidence is useful for verification, but it also moves the text farther from the unwatermarked distribution.",
    baseEvidence: 0.92,
    detectability: 0.86,
    editSensitivity: 0.55,
    pointColor: "#b85c48"
  },
  biasfree: {
    title: "Bias-free token watermark",
    kicker: "Less drift, less margin",
    copy: "Balanced token preferences reduce outsider-visible bias, but token substitutions still remove the evidence that the verifier expects to count.",
    note: "Bias-free token schemes move left on detectability, while robustness still depends on how many marked tokens survive editing.",
    baseEvidence: 0.66,
    detectability: 0.34,
    editSensitivity: 0.9,
    pointColor: "#1f6f68"
  },
  semantic: {
    title: "Semantic watermark",
    kicker: "Surface-edit tolerant",
    copy: "Sentence- or meaning-level signals are less tied to exact tokens, so paraphrasing hurts less unless the edit changes the semantic features.",
    note: "Semantic schemes trade some clean statistical sharpness for better survival under surface rewrites.",
    baseEvidence: 0.74,
    detectability: 0.46,
    editSensitivity: 0.35,
    pointColor: "#c6922c"
  },
  undetectable: {
    title: "Distribution-preserving watermark",
    kicker: "Stealth first",
    copy: "The output distribution is preserved against keyless tests, but the verifier has little slack when edits disturb the hidden alignment.",
    note: "Distribution-preserving schemes sit near zero keyless detectability, but edits can erase the fragile verifier evidence.",
    baseEvidence: 0.48,
    detectability: 0.03,
    editSensitivity: 1.25,
    pointColor: "#17212b"
  }
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function labelForScore(score) {
  if (score >= 0.72) {
    return "High";
  }
  if (score >= 0.38) {
    return "Moderate";
  }
  return "Low";
}

function updateCatch22(modeKey) {
  const mode = catch22Modes[modeKey];
  if (!mode) {
    return;
  }

  const editSlider = document.getElementById("catch22-edit-slider");
  const editRate = editSlider ? Number(editSlider.value) / 100 : 0.25;
  const survivingEvidence = clamp(mode.baseEvidence * Math.pow(1 - mode.editSensitivity * editRate, 2), 0.03, 1);
  const robustness = clamp(0.18 + survivingEvidence * 0.82, 0.05, 1);
  const detectability = clamp(mode.detectability * (0.92 + 0.08 * (1 - editRate)), 0, 1);

  const setText = (id, value) => {
    const element = document.getElementById(id);
    if (element) {
      element.textContent = value;
    }
  };

  setText("catch22-edit-label", `${Math.round(editRate * 100)}%`);
  setText("catch22-kicker", mode.kicker);
  setText("catch22-mode-title", mode.title);
  setText("catch22-mode-copy", mode.copy);
  setText("catch22-note", mode.note);
  setText("catch22-evidence-value", `${Math.round(survivingEvidence * 100)}%`);
  setText("catch22-detectability-value", labelForScore(detectability));

  const evidenceBar = document.getElementById("catch22-evidence-bar");
  if (evidenceBar) {
    evidenceBar.style.width = `${Math.round(survivingEvidence * 100)}%`;
  }

  const detectabilityBar = document.getElementById("catch22-detectability-bar");
  if (detectabilityBar) {
    detectabilityBar.style.width = `${Math.round(detectability * 100)}%`;
  }

  const frontierPoint = document.getElementById("catch22-frontier-point");
  if (frontierPoint) {
    frontierPoint.style.setProperty("--frontier-x", `${12 + detectability * 76}%`);
    frontierPoint.style.setProperty("--frontier-y", `${86 - robustness * 72}%`);
    frontierPoint.style.background = mode.pointColor;
  }

  document.querySelectorAll(".catch22-tab").forEach((tab) => {
    const isActive = tab.dataset.catchMode === modeKey;
    tab.classList.toggle("is-active", isActive);
    tab.setAttribute("aria-selected", String(isActive));
  });
}

const selectorRoutes = {
  archive: {
    title: "Verbatim archive release",
    copy: "The answer is stored or served nearly unchanged, so the selector can use a distribution-preserving watermark whose evidence depends on preserving alignment.",
    epsilon: "&epsilon; &asymp; 0",
    epsilonS: "&epsilon;<sub>s</sub> &asymp; 0",
    familyKey: "distribution",
    family: "Distribution-preserving watermark",
    method: "CGW",
    score: "1.990",
    auc: "0.99",
    z: "-5.80",
    note: "This is the low-edit corner of the diagram: distribution preservation gives the strongest stealth signal when the downstream channel is expected to leave the text nearly intact."
  },
  copyedit: {
    title: "Copy-edited LFQA answer",
    copy: "The answer goes through a human-style copy edit: surface tokens change, but most sentence-level watermark evidence survives. The selector moves to the token-based branch and picks the best candidate in that branch.",
    epsilon: "&epsilon; &asymp; 0.25",
    epsilonS: "&epsilon;<sub>s</sub> &asymp; 0.06",
    familyKey: "token",
    family: "Token-based watermark",
    method: "HCW",
    score: "1.137",
    auc: "0.910",
    z: "3.40",
    note: "This follows the token-based column in the diagram: HCW has the highest selector score among the token-level candidates under the moderate edit regime."
  },
  summary: {
    title: "Summarized explainer",
    copy: "The answer is rewritten into a shorter public explainer. Many tokens change, but the semantic route is still relatively stable, so the selector prefers a sentence-level semantic watermark.",
    epsilon: "&epsilon; &asymp; 0.42",
    epsilonS: "&epsilon;<sub>s</sub> &asymp; 0.10",
    familyKey: "semantic",
    family: "Semantic-based watermark",
    method: "PMark",
    score: "1.305",
    auc: "0.850",
    z: "1.20",
    note: "This follows the semantic-based column in the diagram: PMark is chosen because semantic evidence survives the high-token-edit, low-semantic-flip regime better than the token branch."
  }
};

function updateSelectorRoute(routeKey) {
  const route = selectorRoutes[routeKey];
  if (!route) {
    return;
  }

  const setText = (id, value) => {
    const element = document.getElementById(id);
    if (element) {
      element.textContent = value;
    }
  };

  const setHtml = (id, value) => {
    const element = document.getElementById(id);
    if (element) {
      element.innerHTML = value;
    }
  };

  setText("selector-route-title", route.title);
  setText("selector-route-copy", route.copy);
  setHtml("selector-epsilon", route.epsilon);
  setHtml("selector-epsilon-s", route.epsilonS);
  setText("selector-method", route.method);
  setText("selector-family", route.family);
  setText("selector-score", route.score);
  setText("selector-auc", route.auc);
  setText("selector-z", route.z);
  setText("selector-note", route.note);

  document.querySelectorAll(".route-tab").forEach((tab) => {
    const isActive = tab.dataset.route === routeKey;
    tab.classList.toggle("is-active", isActive);
    tab.setAttribute("aria-selected", String(isActive));
  });

  document.querySelectorAll(".family-column").forEach((column) => {
    column.classList.toggle("is-selected", column.dataset.family === route.familyKey);
  });

  document.querySelectorAll(".method-chip").forEach((chip) => {
    chip.classList.toggle("is-winner", chip.dataset.method === route.method);
  });
}

const implementationCodeBase = "https://github.com/K1015/Catch-22-Pareto-Frontier-Watermark-in-LLMs/blob/main/";

const implementationMethods = {
  kgw: {
    title: "KGW",
    family: "Biased token",
    familyKey: "biased-token",
    preset: "kgw",
    module: "llama2_KGW_inference_LFQA.py",
    paperTitle: "A Watermark for Large Language Models",
    paper: "https://openreview.net/pdf?id=aX8ig9X2a7",
    code: `${implementationCodeBase}Llama2-Watermark/llama2_KGW_inference_LFQA.py`,
    summary: "Context-dependent green-list watermark that adds logit mass to keyed token subsets and detects a surplus of green tokens."
  },
  unigram: {
    title: "Unigram",
    family: "Biased token",
    familyKey: "biased-token",
    preset: "unigram",
    module: "llama2_Unigram_inference_LFQA.py",
    paperTitle: "Provable Robust Watermarking for AI-Generated Text",
    paper: "https://openreview.net/pdf?id=SsmT8aO45L",
    code: `${implementationCodeBase}Llama2-Watermark/llama2_Unigram_inference_LFQA.py`,
    summary: "Fixed-green-list token watermark with a simple unigram detector and a robust token-count signal."
  },
  dipmark: {
    title: "DiPMark",
    family: "Bias-free token",
    familyKey: "bias-free-token",
    preset: "dipmark",
    module: "llama2_DiPMark_inference_LFQA.py",
    paperTitle: "A Resilient and Accessible Distribution-Preserving Watermark for Large Language Models",
    paper: "https://openreview.net/pdf?id=c8qWiNiqRY",
    code: `${implementationCodeBase}Llama2-Watermark/llama2_DiPMark_inference_LFQA.py`,
    summary: "Distribution-preserving p-alpha reweighting method that keeps the sampling interface token-level while reducing observable bias."
  },
  hcw: {
    title: "HCW",
    family: "Bias-free token",
    familyKey: "bias-free-token",
    preset: "hcw",
    module: "llama2_HCW_inference_LFQA.py",
    paperTitle: "Unbiased Watermark for Large Language Models",
    paper: "https://openreview.net/forum?id=uWVC5FVidc",
    code: `${implementationCodeBase}Llama2-Watermark/llama2_HCW_inference_LFQA.py`,
    summary: "Hu-style unbiased inverse-CDF reweighting that preserves marginal token probabilities while leaving keyed verifier evidence."
  },
  heavywater: {
    title: "HeavyWater",
    family: "Bias-free token",
    familyKey: "bias-free-token",
    preset: "heavywater",
    module: "llama2_HeavyWater_inference_LFQA.py",
    paperTitle: "HeavyWater and SimplexWater",
    paper: "https://openreview.net/forum?id=R5EBtNE2Y9",
    code: `${implementationCodeBase}Llama2-Watermark/llama2_HeavyWater_inference_LFQA.py`,
    summary: "Heavy-tailed PRF token preference watermark designed to work better in low-entropy decoding regimes."
  },
  simplexwater: {
    title: "SimplexWater",
    family: "Bias-free token",
    familyKey: "bias-free-token",
    preset: "simplexwater",
    module: "llama2_SimplexWater_inference_LFQA.py",
    paperTitle: "HeavyWater and SimplexWater",
    paper: "https://openreview.net/forum?id=R5EBtNE2Y9",
    code: `${implementationCodeBase}Llama2-Watermark/llama2_SimplexWater_inference_LFQA.py`,
    summary: "Simplex-direction PRF watermark from the HeavyWater/SimplexWater framework, evaluated through the shared token-level adapter."
  },
  kuditipudi: {
    title: "Kuditipudi",
    family: "Bias-free token",
    familyKey: "bias-free-token",
    preset: "kuditipudi",
    module: "llama2_Kuditipudi_inference_LFQA.py",
    paperTitle: "Robust Distortion-free Watermarks for Language Models",
    paper: "https://openreview.net/forum?id=FpaCL1MO2C",
    code: `${implementationCodeBase}Llama2-Watermark/llama2_Kuditipudi_inference_LFQA.py`,
    summary: "Gumbel-key randomized sampling watermark with detector-side alignment against a keyed random sequence."
  },
  semstamp: {
    title: "SemStamp",
    family: "Semantic",
    familyKey: "semantic",
    preset: "semstamp",
    module: "llama2_SemStamp_inference_LFQA.py",
    paperTitle: "SemStamp: A Semantic Watermark with Paraphrastic Robustness",
    paper: "https://aclanthology.org/2024.naacl-long.226/",
    code: `${implementationCodeBase}Llama2-Watermark/llama2_SemStamp_inference_LFQA.py`,
    summary: "Sentence-level semantic watermark using embedding-space locality-sensitive hashing and rejection sampling."
  },
  pmark: {
    title: "PMark",
    family: "Semantic",
    familyKey: "semantic",
    preset: "pmark",
    module: "llama2_PMark_inference_LFQA.py",
    paperTitle: "PMark: Robust and Distortion-free Semantic-level Watermarking",
    paper: "https://arxiv.org/abs/2509.21057",
    code: `${implementationCodeBase}Llama2-Watermark/llama2_PMark_inference_LFQA.py`,
    summary: "Semantic-level channel-constrained watermark implemented through the shared semantic adapter."
  },
  simmark: {
    title: "SimMark",
    family: "Semantic",
    familyKey: "semantic",
    preset: "simmark",
    module: "llama2_SimMark_inference_LFQA.py",
    paperTitle: "SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm",
    paper: "https://arxiv.org/abs/2502.02787",
    code: `${implementationCodeBase}Llama2-Watermark/llama2_SimMark_inference_LFQA.py`,
    summary: "Sentence-similarity watermark that detects whether adjacent semantic embeddings fall in keyed similarity intervals."
  },
  cgw: {
    title: "CGW",
    family: "Distribution-preserving",
    familyKey: "distribution",
    preset: "cgw",
    module: "llama2_CGW_inference_LFQA.py",
    paperTitle: "Undetectable Watermarks for Language Models",
    paper: "https://proceedings.mlr.press/v247/christ24a.html",
    code: `${implementationCodeBase}Llama2-Watermark/llama2_CGW_inference_LFQA.py`,
    summary: "Distribution-preserving inverse-CDF sampling watermark designed to remain undetectable to keyless statistical tests."
  },
  gaussmark: {
    title: "GaussMark",
    family: "Training-time",
    familyKey: "adaptive",
    preset: "gaussmark",
    module: "llama2_GaussMark_inference_LFQA.py",
    paperTitle: "GaussMark: A Practical Approach for Structural Watermarking of Language Models",
    paper: "https://openreview.net/pdf?id=YG3DbpAQBf",
    code: `${implementationCodeBase}Llama2-Watermark/llama2_GaussMark_inference_LFQA.py`,
    summary: "Structural watermark evaluated through an inference hook for GaussMark checkpoints in the common CLI."
  },
  dawa: {
    title: "DAWA",
    family: "Adaptive",
    familyKey: "adaptive",
    preset: "dawa",
    module: "dawa_watermark.py",
    paperTitle: "A Distribution-Adaptive Approach to LLM Watermarking",
    paper: "https://openreview.net/forum?id=Lzi8raVEQu",
    code: `${implementationCodeBase}common/dawa_watermark.py`,
    summary: "Distribution-aware adaptive watermark that adjusts watermark strength to the model's next-token distribution."
  },
  hybrid: {
    title: "Hybrid",
    family: "Hybrid selector",
    familyKey: "hybrid",
    preset: "hybrid",
    module: "llama2_Hybrid_inference_LFQA.py",
    paperTitle: "Catch-22",
    paper: "static/pdfs/catch22-paper.pdf",
    code: `${implementationCodeBase}Llama2-Watermark/llama2_Hybrid_inference_LFQA.py`,
    summary: "Catch-22 selector wrapper that combines token-level and semantic branches according to the anticipated edit regime."
  }
};

function updateImplementation(methodKey) {
  const method = implementationMethods[methodKey];
  if (!method) {
    return;
  }

  const setText = (id, value) => {
    const element = document.getElementById(id);
    if (element) {
      element.textContent = value;
    }
  };

  setText("implementation-family", method.family);
  setText("implementation-title", method.title);
  setText("implementation-summary", method.summary);
  setText("implementation-preset", method.preset);
  setText("implementation-module", method.module);
  setText("implementation-paper-title", method.paperTitle);

  const paperLink = document.getElementById("implementation-paper-link");
  if (paperLink) {
    paperLink.href = method.paper;
  }

  const codeLink = document.getElementById("implementation-code-link");
  if (codeLink) {
    codeLink.href = method.code;
  }

  document.querySelectorAll(".implementation-tab").forEach((tab) => {
    const isActive = tab.dataset.implementation === methodKey;
    tab.classList.toggle("is-active", isActive);
    tab.setAttribute("aria-selected", String(isActive));
  });

  document.querySelectorAll(".classification-item").forEach((item) => {
    item.classList.toggle("is-active", item.dataset.familyGuide === method.familyKey);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  let activeCatchMode = "biased";

  document.querySelectorAll(".site-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const targetId = tab.dataset.scrollTarget || siteTabDefaults[tab.dataset.siteTab];
      if (targetId) {
        history.pushState(null, "", `#${targetId}`);
      }
      activateSiteTab(tab.dataset.siteTab, { targetId });
    });
  });

  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener("click", (event) => {
      const targetId = link.dataset.scrollTarget || link.getAttribute("href").slice(1);
      const target = document.getElementById(targetId);
      const tabKey = link.dataset.siteTab || activeSiteTabForElement(target);

      if (!target || !tabKey) {
        return;
      }

      event.preventDefault();
      history.pushState(null, "", `#${targetId}`);
      activateSiteTab(tabKey, { targetId });
    });
  });

  window.addEventListener("hashchange", () => {
    activateSiteTabFromHash(window.location.hash, { behavior: "smooth" });
  });

  if (!activateSiteTabFromHash(window.location.hash, { behavior: "auto" })) {
    activateSiteTab("home", { scroll: false });
  }

  document.querySelectorAll(".catch22-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      activeCatchMode = tab.dataset.catchMode;
      updateCatch22(activeCatchMode);
    });
  });

  const catch22Slider = document.getElementById("catch22-edit-slider");
  if (catch22Slider) {
    catch22Slider.addEventListener("input", () => updateCatch22(activeCatchMode));
    updateCatch22(activeCatchMode);
  }

  document.querySelectorAll(".route-tab").forEach((tab) => {
    tab.addEventListener("click", () => updateSelectorRoute(tab.dataset.route));
  });

  document.querySelectorAll(".implementation-tab").forEach((tab) => {
    tab.addEventListener("click", () => updateImplementation(tab.dataset.implementation));
  });

  updateImplementation("kgw");
});
