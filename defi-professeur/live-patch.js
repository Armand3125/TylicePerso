// Mode visio : correction réellement immédiate, question par question.

// Cinq bonus supplémentaires pour la seconde moitié du jeu.
bonuses.push(
  "J’aimerais que tu me regardes me toucher pendant notre appel vidéo.",
  "J’ai envie que tu me dises exactement où et comment me toucher.",
  "J’aimerais me montrer nue à la caméra pendant que tu me donnes des ordres.",
  "J’ai envie que tu me fasses attendre avant de me laisser jouir.",
  "Quand on se retrouvera, j’aimerais te dire exactement comment me faire jouir."
);

const bonusTranslations = [
  "Me gusta cuando decides cuándo puedo tocarme.",
  "Me gusta cuando me das órdenes.",
  "Me gustaría darte órdenes.",
  "Me gusta que me ates.",
  "Quiero sentarme en tu cara.",
  "Me gustaría que me miraras tocarme durante nuestra videollamada.",
  "Quiero que me digas exactamente dónde y cómo tocarme.",
  "Me gustaría mostrarme desnuda ante la cámara mientras me das órdenes.",
  "Quiero que me hagas esperar antes de dejarme llegar al orgasmo.",
  "Cuando volvamos a vernos, me gustaría decirte exactamente cómo hacerme llegar al orgasmo."
];

const immediateStates = {};

function pageState(pageIndex) {
  if (!immediateStates[pageIndex]) {
    immediateStates[pageIndex] = {
      attempts: [0, 0, 0],
      resolved: [false, false, false],
      firstCorrect: [false, false, false],
      awardedScore: 0,
      finished: false
    };
  }
  return immediateStates[pageIndex];
}

function liveTotalSteps() {
  return pages.length + bonuses.length;
}

updateProgress = function (stepIndex, label) {
  document.getElementById("progress").style.width = `${((stepIndex + 1) / liveTotalSteps()) * 100}%`;
  document.getElementById("pageCounter").textContent = label;
};

function paintOption(label, kind) {
  if (!label) return;
  if (kind === "good") {
    label.style.background = "#e9f7ef";
    label.style.borderColor = "#61b986";
    label.style.color = "#187847";
  } else if (kind === "bad") {
    label.style.background = "#fff0f2";
    label.style.borderColor = "#df7283";
    label.style.color = "#b52d42";
  }
}

function lockQuestion(localIndex) {
  document.querySelectorAll(`#question-${localIndex} input`).forEach((input) => {
    input.disabled = true;
  });
}

function removeQuestionNote(localIndex) {
  const old = document.getElementById(`question-note-${localIndex}`);
  if (old) old.remove();
}

function showQuestionNote(localIndex, text, kind) {
  removeQuestionNote(localIndex);
  const block = document.getElementById(`question-${localIndex}`);
  if (!block) return;

  const note = document.createElement("div");
  note.id = `question-note-${localIndex}`;
  note.textContent = text;
  note.style.marginTop = "9px";
  note.style.fontWeight = "850";
  note.style.color = kind === "good" ? "#187847" : "#b52d42";
  block.appendChild(note);
}

function showCorrectAnswer(localIndex, correctIndex) {
  const label = document.getElementById(`option-${localIndex}-${correctIndex}`);
  paintOption(label, "good");
}

function scrollToNextQuestion(localIndex) {
  const next = document.getElementById(`question-${localIndex + 1}`);
  if (next) {
    setTimeout(() => next.scrollIntoView({ behavior: "smooth", block: "center" }), 300);
  }
}

function updateLivePageScore(pageIndex) {
  const state = pageState(pageIndex);
  const firstTryCorrect = state.firstCorrect.filter(Boolean).length;

  // Le barème final reste identique : 0/1 = 0, 2 = +0,5, 3 = +1.
  // On crédite cependant la jauge dès que le seuil est atteint.
  let scoreForMeter = 0;
  if (firstTryCorrect >= 3) scoreForMeter = 3;
  else if (firstTryCorrect >= 2) scoreForMeter = 2;

  if (scoreForMeter === state.awardedScore) return;

  const previous = state.awardedScore;
  state.awardedScore = scoreForMeter;
  pageScores[pageIndex] = scoreForMeter;

  const previousDelta = previous === 3 ? 1 : previous === 2 ? 0.5 : 0;
  const newDelta = scoreForMeter === 3 ? 1 : scoreForMeter === 2 ? 0.5 : 0;
  const gained = newDelta - previousDelta;
  updateIntensity(gained > 0 ? `+${formatLevel(gained)} point` : "");
}

function renderNextButton(pageIndex) {
  let footer = document.getElementById("page-next-zone");
  if (footer) footer.remove();

  footer = document.createElement("div");
  footer.id = "page-next-zone";
  footer.style.marginTop = "20px";
  footer.style.display = "flex";
  footer.style.justifyContent = "center";

  const button = document.createElement("button");
  button.className = "primary";
  button.textContent = "Suivant";
  button.onclick = () => continueAfterPage(pageIndex);
  footer.appendChild(button);
  quiz.appendChild(footer);
}

function finalizeImmediatePage(pageIndex) {
  const state = pageState(pageIndex);
  if (state.finished || !state.resolved.every(Boolean)) return;
  state.finished = true;

  const firstTryCorrect = state.firstCorrect.filter(Boolean).length;
  pageScores[pageIndex] = firstTryCorrect;
  updateIntensity();
  renderNextButton(pageIndex);
}

window.chooseImmediateAnswer = function (localIndex, optionIndex) {
  const pageIndex = currentPage;
  const state = pageState(pageIndex);
  if (state.resolved[localIndex]) return;

  const question = pages[pageIndex].questions[localIndex];
  const globalIndex = pageIndex * 3 + localIndex;
  const label = document.getElementById(`option-${localIndex}-${optionIndex}`);
  const input = document.getElementById(`input-${localIndex}-${optionIndex}`);

  answers[globalIndex] = optionIndex;
  state.attempts[localIndex] += 1;

  // Bonne réponse : retour vert immédiatement, sans attendre aucune autre question.
  if (optionIndex === question.answer) {
    const firstTry = state.attempts[localIndex] === 1;
    if (firstTry) state.firstCorrect[localIndex] = true;

    paintOption(label, "good");
    showQuestionNote(localIndex, firstTry ? "Correct ✓" : "Correct ✓ — deuxième tentative", "good");
    state.resolved[localIndex] = true;
    lockQuestion(localIndex);
    updateLivePageScore(pageIndex);

    if (localIndex < 2) scrollToNextQuestion(localIndex);
    finalizeImmediatePage(pageIndex);
    return;
  }

  // Mauvaise réponse : rouge immédiatement et deuxième chance sur place.
  paintOption(label, "bad");
  if (input) input.disabled = true;

  if (state.attempts[localIndex] === 1) {
    showQuestionNote(localIndex, "Incorrect ✗ — essaie encore", "bad");
    return;
  }

  // Deuxième erreur : on révèle la bonne réponse directement dans cette question.
  showCorrectAnswer(localIndex, question.answer);
  showQuestionNote(localIndex, `Bonne réponse : ${question.options[question.answer]}`, "good");
  state.resolved[localIndex] = true;
  lockQuestion(localIndex);
  updateLivePageScore(pageIndex);

  if (localIndex < 2) scrollToNextQuestion(localIndex);
  finalizeImmediatePage(pageIndex);
};

renderPage = function () {
  activeBonus = null;
  const page = pages[currentPage];
  const state = pageState(currentPage);
  let html = `<div class="page"><h2>${page.title}</h2><p class="page-intro">${page.intro}</p>`;
  if (page.reading) html += `<div class="reading">${page.reading}</div>`;

  page.questions.forEach((question, localIndex) => {
    html += `<div class="question" id="question-${localIndex}"><div class="instruction">${question.instruction}</div><div class="prompt">${question.prompt}</div>`;
    question.options.forEach((option, optionIndex) => {
      html += `<label class="option" id="option-${localIndex}-${optionIndex}"><input id="input-${localIndex}-${optionIndex}" type="radio" name="q${currentPage}-${localIndex}" value="${optionIndex}" onchange="chooseImmediateAnswer(${localIndex},${optionIndex})"><span>${String.fromCharCode(65 + optionIndex)}. ${option}</span></label>`;
    });
    html += `</div>`;
  });

  html += `</div>`;
  quiz.innerHTML = html;
  updateProgress(quizStepIndex(currentPage), `Página ${currentPage + 1} de ${pages.length}`);
  updateIntensity();

  if (state.finished) renderNextButton(currentPage);
  window.scrollTo({ top: 0, behavior: "smooth" });
};

window.chooseBonusImmediate = function (bonusIndex, value) {
  if (bonusValidated[bonusIndex]) return;
  bonusAnswers[bonusIndex] = value;
  bonusValidated[bonusIndex] = true;
  updateIntensity("+0,5 point");

  document.querySelectorAll(`#bonus-page-${bonusIndex} input`).forEach((input) => {
    input.disabled = true;
  });
  const chosen = document.getElementById(`bonus-${bonusIndex}-${value ? "true" : "false"}`);
  paintOption(chosen, "good");

  const footer = document.createElement("div");
  footer.style.marginTop = "20px";
  footer.style.display = "flex";
  footer.style.justifyContent = "center";
  const button = document.createElement("button");
  button.className = "primary";
  button.textContent = "Suivant";
  button.onclick = () => continueAfterBonus(bonusIndex);
  footer.appendChild(button);
  quiz.appendChild(footer);
};

renderBonus = function (bonusIndex) {
  activeBonus = bonusIndex;
  const translation = bonusTranslations[bonusIndex] || "";
  quiz.innerHTML = `
    <div class="bonus-page" id="bonus-page-${bonusIndex}">
      <span class="bonus-chip">Bonus intime ${bonusIndex + 1} / ${bonuses.length}</span>
      <h2>Vrai ou faux ?</h2>
      <p class="page-intro">Réponds simplement et honnêtement.</p>
      <div class="bonus-question">
        <div class="bonus-fr">${bonuses[bonusIndex]}</div>
        <div style="margin-top:7px;color:#8a818b;font-size:.88rem;line-height:1.4;font-weight:500">${translation}</div>
        <p class="bonus-points">Réponse : +0,5 point</p>
        <label class="option" id="bonus-${bonusIndex}-true"><input type="radio" name="bonus${bonusIndex}" onchange="chooseBonusImmediate(${bonusIndex},true)"><span>Vrai</span></label>
        <label class="option" id="bonus-${bonusIndex}-false"><input type="radio" name="bonus${bonusIndex}" onchange="chooseBonusImmediate(${bonusIndex},false)"><span>Faux</span></label>
      </div>
    </div>`;
  updateProgress(bonusStepIndex(bonusIndex), `Bonus ${bonusIndex + 1} sur ${bonuses.length}`);
  updateIntensity();
  window.scrollTo({ top: 0, behavior: "smooth" });
};

// Le script principal a déjà rendu la première page avant le chargement de ce patch.
renderPage();
