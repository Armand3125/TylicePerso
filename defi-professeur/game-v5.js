// V6 : niveaux pédagogiques + malus immédiat de 0,25 par mauvaise réponse.

function levelForPageIndex(pageIndex) {
  if (pageIndex < 8) return "A1";
  if (pageIndex < 16) return "A2";
  return "B1";
}

pages.forEach((page, pageIndex) => {
  page.level = levelForPageIndex(pageIndex);
});

// Le niveau de jeu tient compte de chaque clic faux, pas seulement du score final de la page.
getGameLevel = function () {
  let level = 1;

  pageScores.forEach((score) => {
    if (score === 3) level += 0.5;
    else if (score === 2) level += 0.25;
  });

  bonusValidated.forEach((ok) => {
    if (ok) level += 0.25;
  });

  Object.values(immediateStates).forEach((state) => {
    level -= (state.wrongCount || 0) * 0.25;
  });

  return clampLevel(level);
};

// Chaque mauvaise tentative coûte immédiatement 0,25 point.
const v5OriginalChooseImmediateAnswer = window.chooseImmediateAnswer;
window.chooseImmediateAnswer = function (localIndex, optionIndex) {
  const pageIndex = currentPage;
  const state = pageState(pageIndex);
  if (state.resolved[localIndex]) return;

  const question = pages[pageIndex].questions[localIndex];
  const isWrong = optionIndex !== question.answer;

  if (isWrong) {
    state.wrongCount = (state.wrongCount || 0) + 1;
  }

  const result = v5OriginalChooseImmediateAnswer(localIndex, optionIndex);

  if (isWrong) {
    updateIntensity("−0,25 point");
  }

  return result;
};

function addLevelBadge(container, level) {
  if (!container || container.querySelector(".language-level-badge")) return;
  const badge = document.createElement("div");
  badge.className = "language-level-badge";
  badge.textContent = `Niveau ${level}`;
  badge.style.display = "inline-block";
  badge.style.marginBottom = "10px";
  badge.style.padding = "5px 10px";
  badge.style.borderRadius = "999px";
  badge.style.background = "#f1e8f4";
  badge.style.color = "#7b3f70";
  badge.style.fontSize = ".78rem";
  badge.style.fontWeight = "900";
  badge.style.letterSpacing = ".04em";
  badge.style.textTransform = "uppercase";
  container.insertBefore(badge, container.firstChild);
}

// Badge A1/A2/B1 en haut de chaque page de questions.
const v5OriginalRenderPage = renderPage;
renderPage = function (...args) {
  const result = v5OriginalRenderPage.apply(this, args);
  const pageElement = document.querySelector("#quiz .page");
  addLevelBadge(pageElement, levelForPageIndex(currentPage));
  return result;
};

// Les bonus indiquent aussi le niveau correspondant à l'étape où ils apparaissent.
const v5OriginalRenderBonus = renderBonus;
renderBonus = function (bonusIndex, ...args) {
  const result = v5OriginalRenderBonus.call(this, bonusIndex, ...args);
  const relatedPageIndex = Math.min((bonusIndex + 1) * 2 - 1, pages.length - 1);
  const bonusElement = document.querySelector("#quiz .bonus-page");
  addLevelBadge(bonusElement, levelForPageIndex(relatedPageIndex));
  return result;
};

// Le premier rendu a déjà eu lieu avant le chargement de ce fichier.
const initialPageElement = document.querySelector("#quiz .page");
addLevelBadge(initialPageElement, levelForPageIndex(currentPage));
updateIntensity();
