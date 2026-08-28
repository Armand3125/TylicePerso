const pages = [
  {
    title: "Página 1 — Nosotros dos",
    intro: "Empezamos con frases sencillas sobre la pareja.",
    questions: [
      { instruction: "Completa la frase.", prompt: "1. Tu ___ ma copine.", options: ["est", "es", "suis"], answer: 1 },
      { instruction: "Elige la respuesta correcta.", prompt: "2. Comment s’appelle ton copain ?", options: ["Il habite français.", "Il a Toulouse.", "Il s’appelle Armand."], answer: 2 },
      { instruction: "Completa la frase.", prompt: "3. Nous ___ en couple.", options: ["sommes", "avons", "êtes"], answer: 0 }
    ]
  },
  {
    title: "Página 2 — La vida cotidiana",
    intro: "Verbos sencillos para hablar de todos los días.",
    questions: [
      { instruction: "Completa la frase.", prompt: "4. Le matin, je ___ un café.", options: ["mange", "bois", "parle"], answer: 1 },
      { instruction: "Completa la frase.", prompt: "5. Tu ___ à Buenos Aires.", options: ["habitons", "habite", "habites"], answer: 2 },
      { instruction: "Completa la frase.", prompt: "6. Nous ___ au téléphone le soir.", options: ["parlons", "parlez", "parle"], answer: 0 }
    ]
  },
  {
    title: "Página 3 — Amor y sentimientos",
    intro: "Decir lo que sientes con frases muy simples.",
    questions: [
      { instruction: "Elige la traducción correcta.", prompt: "7. « Te quiero. »", options: ["Je te regarde.", "Je t’aime.", "Je te téléphone."], answer: 1 },
      { instruction: "Completa la frase.", prompt: "8. Tu me ___ beaucoup.", options: ["manque", "manquez", "manques"], answer: 2 },
      { instruction: "Elige la frase correcta.", prompt: "9. ¿Cómo se dice « Me gusta cuando me besas »?", options: ["J’aime quand tu m’embrasses.", "Je suis quand tu embrasses.", "J’aime tu me regarde."], answer: 0 }
    ]
  },
  {
    title: "Página 4 — Una cita romántica",
    intro: "Preparar una cita y hablar de lo que llevas puesto.",
    questions: [
      { instruction: "Completa la frase.", prompt: "10. Notre rendez-vous est ___ soir.", options: ["cette", "ces", "ce"], answer: 2 },
      { instruction: "Elige la traducción correcta.", prompt: "11. « Vemos una película juntos. »", options: ["Nous regardons un film ensemble.", "Nous mangeons le film demain.", "Nous sommes un film."], answer: 0 },
      { instruction: "Completa la frase.", prompt: "12. Je porte ___ robe noire.", options: ["un", "une", "des"], answer: 1 }
    ]
  },
  {
    title: "Página 5 — El cuerpo y la atracción",
    intro: "Palabras sencillas para hablar de belleza y sensaciones.",
    questions: [
      { instruction: "Elige la respuesta correcta.", prompt: "13. Avec quoi est-ce qu’on embrasse ?", options: ["Les lèvres", "Les pieds", "Les oreilles"], answer: 0 },
      { instruction: "Completa la frase.", prompt: "14. Tu as de très beaux ___.", options: ["main", "cheveu", "yeux"], answer: 2 },
      { instruction: "Elige la traducción correcta.", prompt: "15. « Me gusta tu perfume. »", options: ["J’aime ton téléphone.", "J’aime ton parfum.", "Je porte ton parfum."], answer: 1 }
    ]
  },
  {
    title: "Página 6 — En casa",
    intro: "Acciones habituales para una noche juntos.",
    questions: [
      { instruction: "Completa la frase.", prompt: "16. Je prépare ___ dîner.", options: ["la", "le", "les"], answer: 1 },
      { instruction: "Elige la traducción correcta.", prompt: "17. « Vamos al restaurante. »", options: ["Nous sommes le restaurant.", "Nous parlons au restaurant.", "Nous allons au restaurant."], answer: 2 },
      { instruction: "Completa la frase.", prompt: "18. Après le travail, je ___ à la maison.", options: ["rentre", "rentrons", "rentrez"], answer: 0 }
    ]
  },
  {
    title: "Página 7 — Pequeñas instrucciones",
    intro: "Comprender algunas frases simples du professeur.",
    questions: [
      { instruction: "Elige la traducción correcta.", prompt: "19. « Cierra los ojos. »", options: ["Ouvre la bouche.", "Ferme les yeux.", "Tourne la tête."], answer: 1 },
      { instruction: "Elige la traducción correcta.", prompt: "20. « Espera aquí. »", options: ["Parle ici.", "Regarde demain.", "Attends ici."], answer: 2 },
      { instruction: "Elige la traducción correcta.", prompt: "21. « Dime lo que quieres. »", options: ["Dis-moi ce que tu veux.", "Je veux ton téléphone.", "Tu dis demain."], answer: 0 }
    ]
  },
  {
    title: "Página 8 — Deseo e intensidad",
    intro: "Expresar lo que quieres de manera sencilla.",
    questions: [
      { instruction: "Elige la traducción correcta.", prompt: "22. « Un poco más. »", options: ["Moins vite.", "Très loin.", "Encore un peu."], answer: 2 },
      { instruction: "Elige la traducción correcta.", prompt: "23. « Más despacio. »", options: ["Plus doucement.", "Plus grand.", "Moins près."], answer: 0 },
      { instruction: "Completa la frase.", prompt: "24. Ne t’arrête ___ maintenant.", options: ["rien", "pas", "jamais de"], answer: 1 }
    ]
  },
  {
    title: "Página 9 — Esta noche y mañana",
    intro: "Hablar de horarios y de planes sencillos.",
    questions: [
      { instruction: "Elige la pregunta correcta.", prompt: "25. La réponse est : « À vingt-deux heures. »", options: ["À quelle heure ?", "Quel âge as-tu ?", "Où habites-tu ?"], answer: 0 },
      { instruction: "Completa la frase.", prompt: "26. Demain, je ___ travailler.", options: ["va", "allez", "vais"], answer: 2 },
      { instruction: "Elige la frase correcta.", prompt: "27. ¿Cómo se dice « Te llamo esta noche »?", options: ["Je te vois hier.", "Je t’appelle ce soir.", "Je parle demain soir toi."], answer: 1 }
    ]
  },
  {
    title: "Página 10 — Comprensión final",
    intro: "Una última lectura antes de terminar el desafío.",
    reading: "Ce soir, Ailin et Armand ont rendez-vous en vidéo. Ils parlent de leur journée, rient et se disent des mots doux. Ensuite, ils jouent en français. Armand valide chaque page en direct avant de passer à la suivante. Quand Ailin réussit une page parfaite, la récompense alterne : une photo d’Armand, puis un ordre à lui donner.",
    questions: [
      { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "28. Comment est leur rendez-vous ?", options: ["Au restaurant", "À l’école", "En vidéo"], answer: 2 },
      { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "29. Qui valide chaque page ?", options: ["Armand", "Le téléphone", "Le restaurant"], answer: 0 },
      { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "30. Que peut gagner Ailin après une page parfaite ?", options: ["Une photo ou le droit de donner un ordre", "Un billet d’avion", "Un nouveau téléphone"], answer: 0 }
    ]
  }
];

const bonuses = [
  "J’aime quand tu décides quand je peux me toucher.",
  "J’aime quand tu me donnes des ordres.",
  "J’aimerais te donner des ordres.",
  "J’aime que tu m’attaches.",
  "Je veux m’asseoir sur ton visage."
];

const gages = [
  "Répète les trois bonnes réponses en disant « Monsieur le professeur ».",
  "Attends une minute sans te toucher, puis demande la permission en français.",
  "Dis trois fois : « Je vais être une élève très sage. »",
  "Décris en une phrase française la récompense que tu veux gagner.",
  "Reste immobile pendant trente secondes avant la prochaine question."
];

let currentPage = 0;
let activeBonus = null;
let pendingSubmission = null;
let pendingContinue = null;
let pendingAdjustment = 0;

const answers = {};
const bonusAnswers = {};
const pageScores = Array(pages.length).fill(null);
const bonusValidated = Array(bonuses.length).fill(false);
const professorAdjustments = [];
const quiz = document.getElementById("quiz");

function clampLevel(value) {
  return Math.max(0, Math.min(10, value));
}

function baseDeltaValue(kind, score) {
  if (kind === "bonus") return 0.5;
  if (score === 3) return 1;
  if (score === 2) return 0.5;
  if (score === 0) return -0.5;
  return 0;
}

function getGameLevel() {
  let level = 1;
  pageScores.forEach((score) => {
    if (score === 3) level += 1;
    else if (score === 2) level += 0.5;
    else if (score === 0) level -= 0.5;
  });
  bonusValidated.forEach((ok) => {
    if (ok) level += 0.5;
  });
  professorAdjustments.forEach((adjustment) => {
    level += adjustment;
  });
  return clampLevel(level);
}

function formatLevel(level) {
  return Number.isInteger(level) ? String(level) : String(level).replace(".", ",");
}

function formatSigned(value) {
  if (!value) return "0";
  return `${value > 0 ? "+" : "−"}${formatLevel(Math.abs(value))}`;
}

function getLevelStatus(level) {
  const band = Math.floor(level);
  if (band <= 2) return "Solo teasing";
  if (band <= 4) return "Teasing intenso";
  if (band <= 6) return "Vibración suave";
  if (band <= 8) return "Vibración media";
  return "Vibración fuerte";
}

function updateIntensity(deltaText = "") {
  const level = getGameLevel();
  const percent = Math.max(0, Math.min(100, level * 10));
  document.getElementById("levelValue").textContent = formatLevel(level);
  document.getElementById("levelStatus").textContent = getLevelStatus(level);
  const delta = document.getElementById("levelDelta");
  delta.textContent = deltaText;
  delta.classList.remove("flash");
  if (deltaText) {
    void delta.offsetWidth;
    delta.classList.add("flash");
  }
  const fill = document.getElementById("meterFill");
  const marker = document.getElementById("meterMarker");
  if (window.matchMedia("(max-width:900px)").matches) {
    fill.style.width = `${percent}%`;
    marker.style.left = `${percent}%`;
  } else {
    fill.style.height = `${percent}%`;
    marker.style.bottom = `${percent}%`;
  }
}

function quizStepIndex(pageIndex) {
  return pageIndex + Math.floor(pageIndex / 2);
}

function bonusStepIndex(bonusIndex) {
  return bonusIndex * 3 + 2;
}

function updateProgress(stepIndex, label) {
  document.getElementById("progress").style.width = `${((stepIndex + 1) / 15) * 100}%`;
  document.getElementById("pageCounter").textContent = label;
}

function renderPage() {
  activeBonus = null;
  const page = pages[currentPage];
  let html = `<div class="page"><h2>${page.title}</h2><p class="page-intro">${page.intro}</p>`;
  if (page.reading) html += `<div class="reading">${page.reading}</div>`;
  page.questions.forEach((question, localIndex) => {
    const globalIndex = currentPage * 3 + localIndex;
    const chosen = answers[globalIndex];
    html += `<div class="question"><div class="instruction">${question.instruction}</div><div class="prompt">${question.prompt}</div>`;
    question.options.forEach((option, optionIndex) => {
      html += `<label class="option"><input type="radio" name="q${globalIndex}" value="${optionIndex}" ${chosen === optionIndex ? "checked" : ""} onchange="saveAnswer(${globalIndex},${optionIndex})"><span>${String.fromCharCode(65 + optionIndex)}. ${option}</span></label>`;
    });
    html += `</div>`;
  });
  html += `<div class="validation-warning" id="validationWarning">Responde las tres preguntas antes de continuar.</div><div class="controls"><button class="primary" onclick="submitCurrentPage()">Valider</button></div></div>`;
  quiz.innerHTML = html;
  updateProgress(quizStepIndex(currentPage), `Página ${currentPage + 1} de ${pages.length}`);
  updateIntensity();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderBonus(bonusIndex) {
  activeBonus = bonusIndex;
  const choice = bonusAnswers[bonusIndex];
  quiz.innerHTML = `<div class="bonus-page"><span class="bonus-chip">Bonus intime ${bonusIndex + 1} / ${bonuses.length}</span><h2>Vrai ou faux ?</h2><p class="page-intro">Réponds simplement et honnêtement.</p><div class="bonus-question"><div class="bonus-fr">${bonuses[bonusIndex]}</div><p class="bonus-points">Réponse validée : +0,5 point</p><label class="option"><input type="radio" name="bonus${bonusIndex}" value="true" ${choice === true ? "checked" : ""} onchange="saveBonusAnswer(${bonusIndex},true)"><span>Vrai</span></label><label class="option"><input type="radio" name="bonus${bonusIndex}" value="false" ${choice === false ? "checked" : ""} onchange="saveBonusAnswer(${bonusIndex},false)"><span>Faux</span></label><div class="validation-warning" id="validationWarning">Choisis Vrai ou Faux avant de continuer.</div></div><div class="controls"><button class="primary" onclick="submitBonus()">Valider</button></div></div>`;
  updateProgress(bonusStepIndex(bonusIndex), `Bonus ${bonusIndex + 1} sur ${bonuses.length}`);
  updateIntensity();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function saveAnswer(index, value) {
  answers[index] = value;
  hideWarning();
}

function saveBonusAnswer(index, value) {
  bonusAnswers[index] = value;
  hideWarning();
}

function hideWarning() {
  const warning = document.getElementById("validationWarning");
  if (warning) warning.style.display = "none";
}

function perfectReward() {
  const previousPerfectPages = pageScores.filter((score) => score === 3).length;
  return previousPerfectPages % 2 === 0
    ? { type: "photo", text: "Tu as gagné une photo d’Armand 📸" }
    : { type: "order", text: "Tu peux donner un ordre à Armand 😈" };
}

function submitCurrentPage() {
  const start = currentPage * 3;
  if (![0, 1, 2].every((offset) => answers[start + offset] !== undefined)) {
    document.getElementById("validationWarning").style.display = "block";
    return;
  }

  let score = 0;
  const details = pages[currentPage].questions.map((question, index) => {
    const selectedIndex = answers[start + index];
    const isCorrect = selectedIndex === question.answer;
    if (isCorrect) score += 1;
    return {
      number: start + index + 1,
      prompt: question.prompt,
      selected: question.options[selectedIndex],
      correct: question.options[question.answer],
      isCorrect
    };
  });

  const reward = score === 3 ? perfectReward() : null;
  pendingSubmission = {
    kind: "quiz",
    pageIndex: currentPage,
    score,
    details,
    reward,
    gage: score === 0 ? gages[currentPage % gages.length] : null
  };

  pageScores[currentPage] = score;
  pendingAdjustment = 0;
  pendingContinue = () => continueAfterPage(currentPage);
  showQuizFeedback(pendingSubmission);
}

function submitBonus() {
  if (bonusAnswers[activeBonus] === undefined) {
    document.getElementById("validationWarning").style.display = "block";
    return;
  }

  pendingSubmission = {
    kind: "bonus",
    bonusIndex: activeBonus,
    statement: bonuses[activeBonus],
    answer: bonusAnswers[activeBonus]
  };

  bonusValidated[activeBonus] = true;
  pendingAdjustment = 0;
  pendingContinue = () => continueAfterBonus(activeBonus);
  showBonusFeedback(pendingSubmission);
}

function baseDeltaLabel(kind, score) {
  const base = baseDeltaValue(kind, score);
  if (!base) return "Sans changement";
  return `${formatSigned(base)} point${Math.abs(base) > 1 ? "s" : ""}`;
}

function adjustmentControlsHtml() {
  return `
    <div style="margin-top:18px;padding:14px;border:1px solid #dccde0;border-radius:14px;background:#fbf7fc">
      <div style="font-weight:850;margin-bottom:5px;color:var(--accent)">Ajustement du professeur</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px">
        <button type="button" class="secondary local-adjustment" data-adjustment="-1" onclick="setLocalAdjustment(-1)">−1</button>
        <button type="button" class="secondary local-adjustment active-adjustment" data-adjustment="0" onclick="setLocalAdjustment(0)">Aucun</button>
        <button type="button" class="secondary local-adjustment" data-adjustment="1" onclick="setLocalAdjustment(1)">+1</button>
      </div>
      <p id="adjustmentPreview" style="margin:10px 0 0;color:var(--muted)">Niveau après validation : <strong>${formatLevel(getGameLevel())}/10</strong></p>
    </div>`;
}

function setLocalAdjustment(value) {
  pendingAdjustment = [-1, 0, 1].includes(Number(value)) ? Number(value) : 0;
  document.querySelectorAll(".local-adjustment").forEach((button) => {
    const active = Number(button.dataset.adjustment) === pendingAdjustment;
    button.classList.toggle("active-adjustment", active);
    button.style.outline = active ? "2px solid var(--accent)" : "none";
    button.style.background = active ? "#f8e8f2" : "";
  });
  const preview = document.getElementById("adjustmentPreview");
  if (preview) {
    const previewLevel = clampLevel(getGameLevel() + pendingAdjustment);
    const label = pendingAdjustment === 0 ? "Aucun ajustement" : `Ajustement ${formatSigned(pendingAdjustment)}`;
    preview.innerHTML = `${label} — niveau après validation : <strong>${formatLevel(previewLevel)}/10</strong>`;
  }
}

function showQuizFeedback(submission) {
  const mistakes = submission.details.filter((detail) => !detail.isCorrect);
  let content = "";

  if (mistakes.length === 0) {
    content = `
      <h2 id="feedbackTitle">Parfait 💜</h2>
      <p class="perfect">Aucune faute sur cette page.</p>
      <p><strong>${submission.reward.text}</strong></p>`;
  } else {
    const mistakesHtml = mistakes.map((detail) => `
      <article style="margin-bottom:14px;padding:14px;border-radius:12px;background:var(--bad-soft);border:1px solid #f0c4cb">
        <div style="font-weight:850;margin-bottom:9px;color:var(--ink)">${detail.prompt}</div>
        <div style="color:var(--bad);margin-bottom:6px"><strong>Ta réponse :</strong> « ${detail.selected} »</div>
        <div style="color:var(--good);font-weight:800"><strong>Bonne réponse :</strong> « ${detail.correct} »</div>
      </article>`).join("");
    content = `
      <h2 id="feedbackTitle">À revoir avec le professeur</h2>
      <p>${submission.score} / 3 — ${baseDeltaLabel("quiz", submission.score)}</p>
      <div>${mistakesHtml}</div>
      ${submission.gage ? `<p><strong>Gage :</strong> ${submission.gage}</p>` : ""}`;
  }

  document.getElementById("modalContent").innerHTML = `${content}${adjustmentControlsHtml()}`;
  document.getElementById("feedbackModal").classList.add("open");
  setLocalAdjustment(0);
}

function showBonusFeedback(submission) {
  document.getElementById("modalContent").innerHTML = `
    <h2 id="feedbackTitle">💜 Bonus validé</h2>
    <p><strong>${submission.statement}</strong></p>
    <p>Ta réponse : <strong>${submission.answer ? "Vrai" : "Faux"}</strong></p>
    <div class="modal-delta">+0,5 point</div>
    ${adjustmentControlsHtml()}`;
  document.getElementById("feedbackModal").classList.add("open");
  setLocalAdjustment(0);
}

function closeFeedbackAndContinue() {
  document.getElementById("feedbackModal").classList.remove("open");
  professorAdjustments.push(pendingAdjustment);

  const base = pendingSubmission ? baseDeltaValue(pendingSubmission.kind, pendingSubmission.score) : 0;
  const total = base + pendingAdjustment;
  const deltaText = pendingAdjustment
    ? `${formatSigned(base)} auto · ${formatSigned(pendingAdjustment)} prof · total ${formatSigned(total)}`
    : baseDeltaLabel(pendingSubmission.kind, pendingSubmission.score);

  updateIntensity(deltaText);
  pendingAdjustment = 0;
  pendingSubmission = null;

  if (pendingContinue) {
    const continueFunction = pendingContinue;
    pendingContinue = null;
    continueFunction();
  }
}

function continueAfterPage(pageIndex) {
  if (pageIndex % 2 === 1) renderBonus(Math.floor(pageIndex / 2));
  else if (pageIndex < pages.length - 1) {
    currentPage = pageIndex + 1;
    renderPage();
  } else {
    finishQuiz();
  }
}

function continueAfterBonus(bonusIndex) {
  if (bonusIndex === bonuses.length - 1) {
    finishQuiz();
  } else {
    currentPage = (bonusIndex + 1) * 2;
    renderPage();
  }
}

function finishQuiz() {
  quiz.style.display = "none";
  document.getElementById("result").style.display = "block";
  document.getElementById("resultMessage").textContent = `Niveau final : ${formatLevel(getGameLevel())}/10 — ${getLevelStatus(getGameLevel())}.`;
  document.getElementById("pageCounter").textContent = "Défi terminé";
  document.getElementById("progress").style.width = "100%";
  updateIntensity();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

window.addEventListener("resize", () => updateIntensity());
renderPage();
