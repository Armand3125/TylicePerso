// Ajustements d'affichage pour le jeu en visio : aucune popup, correction directement sur la page.

// Cinq bonus supplémentaires pour la seconde moitié du jeu.
bonuses.push(
  "J’aimerais que tu me regardes me toucher pendant notre appel vidéo.",
  "J’ai envie que tu me dises exactement où et comment me toucher.",
  "J’aimerais me montrer nue à la caméra pendant que tu me donnes des ordres.",
  "J’ai envie que tu me fasses attendre avant de me laisser jouir.",
  "Quand on se retrouvera, j’aimerais te dire exactement comment me faire jouir."
);

// Le total initial était calculé avant l'ajout des cinq nouveaux bonus.
updateProgress = function (stepIndex, label) {
  const steps = pages.length + bonuses.length;
  document.getElementById("progress").style.width = `${((stepIndex + 1) / steps) * 100}%`;
  document.getElementById("pageCounter").textContent = label;
};

function removeOldInlineResult() {
  const old = document.getElementById("inlineFeedback");
  if (old) old.remove();
  const perfect = document.getElementById("pagePerfect");
  if (perfect) perfect.remove();
  document.querySelectorAll(".inline-question-correction").forEach((element) => element.remove());
}

function hideValidationControls() {
  const controls = document.querySelector("#quiz .controls");
  if (controls) controls.style.display = "none";
}

function highlightMistake(questionElement, correctAnswer) {
  if (!questionElement) return;
  questionElement.style.border = "2px solid #d83b52";
  questionElement.style.borderRadius = "14px";
  questionElement.style.padding = "14px";
  questionElement.style.background = "#fff4f6";

  const correction = document.createElement("div");
  correction.className = "inline-question-correction";
  correction.style.marginTop = "12px";
  correction.style.paddingTop = "10px";
  correction.style.borderTop = "1px solid #efbcc5";
  correction.style.color = "#b52d42";
  correction.style.fontWeight = "800";
  correction.innerHTML = `Erreur — bonne réponse : <strong>${correctAnswer}</strong>`;
  questionElement.appendChild(correction);
}

submitCurrentPage = function () {
  const submittedPage = currentPage;
  const start = submittedPage * 3;

  if (![0, 1, 2].every((offset) => answers[start + offset] !== undefined)) {
    document.getElementById("validationWarning").style.display = "block";
    return;
  }

  removeOldInlineResult();

  let score = 0;
  const details = pages[submittedPage].questions.map((question, localIndex) => {
    const selectedIndex = answers[start + localIndex];
    const isCorrect = selectedIndex === question.answer;
    if (isCorrect) score += 1;
    return {
      localIndex,
      correct: question.options[question.answer],
      isCorrect
    };
  });

  pageScores[submittedPage] = score;
  disableCurrentForm();
  hideValidationControls();
  updateIntensity(deltaLabel("quiz", score));

  const mistakes = details.filter((detail) => !detail.isCorrect);

  if (mistakes.length === 0) {
    const perfect = document.createElement("div");
    perfect.id = "pagePerfect";
    perfect.textContent = "Parfait 💜";
    perfect.style.marginTop = "20px";
    perfect.style.textAlign = "center";
    perfect.style.color = "#187847";
    perfect.style.fontSize = "1.25rem";
    perfect.style.fontWeight = "900";
    quiz.appendChild(perfect);
    perfect.scrollIntoView({ behavior: "smooth", block: "nearest" });
    setTimeout(() => continueAfterPage(submittedPage), 1800);
    return;
  }

  const questionElements = Array.from(document.querySelectorAll("#quiz .question"));
  mistakes.forEach((detail) => {
    highlightMistake(questionElements[detail.localIndex], detail.correct);
  });

  const firstError = questionElements[mistakes[0].localIndex];
  if (firstError) {
    setTimeout(() => firstError.scrollIntoView({ behavior: "smooth", block: "center" }), 80);
  }

  setTimeout(() => continueAfterPage(submittedPage), 5200);
};

submitBonus = function () {
  const submittedBonus = activeBonus;
  if (bonusAnswers[submittedBonus] === undefined) {
    document.getElementById("validationWarning").style.display = "block";
    return;
  }

  bonusValidated[submittedBonus] = true;
  disableCurrentForm();
  hideValidationControls();
  updateIntensity("+0,5 point");

  const note = document.createElement("div");
  note.id = "pagePerfect";
  note.textContent = "Bonus validé 💜";
  note.style.marginTop = "20px";
  note.style.textAlign = "center";
  note.style.color = "#187847";
  note.style.fontSize = "1.15rem";
  note.style.fontWeight = "900";
  quiz.appendChild(note);
  note.scrollIntoView({ behavior: "smooth", block: "nearest" });

  setTimeout(() => continueAfterBonus(submittedBonus), 1400);
};

// Recalcule immédiatement la progression du premier écran avec les 30 étapes (20 pages + 10 bonus).
updateProgress(quizStepIndex(currentPage), `Página ${currentPage + 1} de ${pages.length}`);
