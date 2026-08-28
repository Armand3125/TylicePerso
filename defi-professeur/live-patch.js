// Mode visio : correction immédiate, une seconde chance, puis bouton Suivant.

// Cinq bonus supplémentaires pour la seconde moitié du jeu.
bonuses.push(
  "J’aimerais que tu me regardes me toucher pendant notre appel vidéo.",
  "J’ai envie que tu me dises exactement où et comment me toucher.",
  "J’aimerais me montrer nue à la caméra pendant que tu me donnes des ordres.",
  "J’ai envie que tu me fasses attendre avant de me laisser jouir.",
  "Quand on se retrouvera, j’aimerais te dire exactement comment me faire jouir."
);

const immediateStates = {};

function pageState(pageIndex) {
  if (!immediateStates[pageIndex]) {
    immediateStates[pageIndex] = {
      attempts: [0, 0, 0],
      resolved: [false, false, false],
      firstCorrect: [false, false, false],
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

function removeTryAgain(localIndex) {
  const old = document.getElementById(`try-again-${localIndex}`);
  if (old) old.remove();
}

function showTryAgain(localIndex) {
  removeTryAgain(localIndex);
  const block = document.getElementById(`question-${localIndex}`);
  if (!block) return;
  const note = document.createElement("div");
  note.id = `try-again-${localIndex}`;
  note.textContent = "Essaie encore — une deuxième chance 💜";
  note.style.marginTop = "9px";
  note.style.color = "#b52d42";
  note.style.fontWeight = "800";
  block.appendChild(note);
}

function showCorrectAnswer(localIndex, correctIndex) {
  const label = document.getElementById(`option-${localIndex}-${correctIndex}`);
  paintOption(label, "good");
}

function renderNextButton(pageIndex, score) {
  let footer = document.getElementById("page-next-zone");
  if (footer) footer.remove();

  footer = document.createElement("div");
  footer.id = "page-next-zone";
  footer.style.marginTop = "22px";
  footer.style.display = "flex";
  footer.style.flexDirection = "column";
  footer.style.alignItems = "center";
  footer.style.gap = "12px";

  if (score === 3) {
    const perfect = document.createElement("div");
    perfect.textContent = "Parfait 💜";
    perfect.style.color = "#187847";
    perfect.style.fontSize = "1.2rem";
    perfect.style.fontWeight = "900";
    footer.appendChild(perfect);
  }

  const button = document.createElement("button");
  button.className = "primary";
  button.textContent = "Suivant";
  button.onclick = () => continueAfterPage(pageIndex);
  footer.appendChild(button);
  quiz.appendChild(footer);
  footer.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function finalizeImmediatePage(pageIndex) {
  const state = pageState(pageIndex);
  if (state.finished || !state.resolved.every(Boolean)) return;
  state.finished = true;

  const score = state.firstCorrect.filter(Boolean).length;
  pageScores[pageIndex] = score;
  updateIntensity(deltaLabel("quiz", score));
  renderNextButton(pageIndex, score);
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

  if (optionIndex === question.answer) {
    if (state.attempts[localIndex] === 1) state.firstCorrect[localIndex] = true;
    paintOption(label, "good");
    removeTryAgain(localIndex);
    state.resolved[localIndex] = true;
    lockQuestion(localIndex);
    finalizeImmediatePage(pageIndex);
    return;
  }

  paintOption(label, "bad");
  if (input) input.disabled = true;

  if (state.attempts[localIndex] === 1) {
    showTryAgain(localIndex);
    return;
  }

  removeTryAgain(localIndex);
  showCorrectAnswer(localIndex, question.answer);
  state.resolved[localIndex] = true;
  lockQuestion(localIndex);
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
      html += `<label class="option" id="option-${localIndex}-${optionIndex}"><input id="input-${localIndex}-${optionIndex}" type="radio" name="q${currentPage}-${localIndex}" value="${optionIndex}" onclick="chooseImmediateAnswer(${localIndex},${optionIndex})"><span>${String.fromCharCode(65 + optionIndex)}. ${option}</span></label>`;
    });
    html += `</div>`;
  });

  html += `</div>`;
  quiz.innerHTML = html;
  updateProgress(quizStepIndex(currentPage), `Página ${currentPage + 1} de ${pages.length}`);
  updateIntensity();

  // Une page déjà terminée n'est normalement jamais réaffichée, mais on garde un rendu cohérent si cela arrive.
  if (state.finished) renderNextButton(currentPage, state.firstCorrect.filter(Boolean).length);
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
  footer.style.marginTop = "22px";
  footer.style.display = "flex";
  footer.style.justifyContent = "center";
  const button = document.createElement("button");
  button.className = "primary";
  button.textContent = "Suivant";
  button.onclick = () => continueAfterBonus(bonusIndex);
  footer.appendChild(button);
  quiz.appendChild(footer);
  footer.scrollIntoView({ behavior: "smooth", block: "nearest" });
};

renderBonus = function (bonusIndex) {
  activeBonus = bonusIndex;
  quiz.innerHTML = `
    <div class="bonus-page" id="bonus-page-${bonusIndex}">
      <span class="bonus-chip">Bonus intime ${bonusIndex + 1} / ${bonuses.length}</span>
      <h2>Vrai ou faux ?</h2>
      <p class="page-intro">Réponds simplement et honnêtement.</p>
      <div class="bonus-question">
        <div class="bonus-fr">${bonuses[bonusIndex]}</div>
        <p class="bonus-points">Réponse : +0,5 point</p>
        <label class="option" id="bonus-${bonusIndex}-true"><input type="radio" name="bonus${bonusIndex}" onclick="chooseBonusImmediate(${bonusIndex},true)"><span>Vrai</span></label>
        <label class="option" id="bonus-${bonusIndex}-false"><input type="radio" name="bonus${bonusIndex}" onclick="chooseBonusImmediate(${bonusIndex},false)"><span>Faux</span></label>
      </div>
    </div>`;
  updateProgress(bonusStepIndex(bonusIndex), `Bonus ${bonusIndex + 1} sur ${bonuses.length}`);
  updateIntensity();
  window.scrollTo({ top: 0, behavior: "smooth" });
};

// Le script principal a déjà rendu la première page avant le chargement de ce patch.
renderPage();
