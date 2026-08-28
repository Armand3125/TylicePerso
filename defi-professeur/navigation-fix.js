// Navigation finale : descente vers « Suivant » puis retour réellement forcé en haut.

if ("scrollRestoration" in history) history.scrollRestoration = "manual";

// Le scroll automatique question -> question reste fluide car il demande explicitement
// behavior:"smooth". Pour les changements de page, on désactive le smooth global qui
// provoquait des conflits dans certains navigateurs.
document.documentElement.style.scrollBehavior = "auto";

function hardScrollTop() {
  const scrollingElement = document.scrollingElement || document.documentElement;
  if (scrollingElement) scrollingElement.scrollTop = 0;
  document.documentElement.scrollTop = 0;
  document.body.scrollTop = 0;
  window.scrollTo(0, 0);

  const header = document.querySelector(".app > header");
  if (header) header.scrollIntoView({ behavior: "auto", block: "start" });
}

function lockTopAfterRender() {
  hardScrollTop();
  requestAnimationFrame(() => {
    hardScrollTop();
    requestAnimationFrame(hardScrollTop);
  });
  setTimeout(hardScrollTop, 30);
  setTimeout(hardScrollTop, 100);
  setTimeout(hardScrollTop, 250);
  setTimeout(hardScrollTop, 500);
}

function scrollToNextButton(button) {
  if (!button) return;
  setTimeout(() => button.scrollIntoView({ behavior: "smooth", block: "center" }), 120);
}

function goToNextPage(pageIndex, button) {
  if (button) button.blur();
  // Le rendu suivant est synchrone : on navigue, puis on recale immédiatement le viewport.
  continueAfterPage(pageIndex);
  lockTopAfterRender();
}

function goToNextBonus(bonusIndex, button) {
  if (button) button.blur();
  continueAfterBonus(bonusIndex);
  lockTopAfterRender();
}

// Remplace la création du bouton de fin de page afin que sa navigation ne dépende
// d'aucun listener ou scroll hérité.
renderNextButton = function (pageIndex) {
  let footer = document.getElementById("page-next-zone");
  if (footer) footer.remove();

  footer = document.createElement("div");
  footer.id = "page-next-zone";
  footer.style.marginTop = "20px";
  footer.style.display = "flex";
  footer.style.justifyContent = "center";

  const button = document.createElement("button");
  button.className = "primary";
  button.type = "button";
  button.textContent = "Suivant";
  button.onclick = () => goToNextPage(pageIndex, button);

  footer.appendChild(button);
  quiz.appendChild(footer);
  scrollToNextButton(button);
};

// scoring-v2 crée le bouton des bonus. On garde sa logique de score puis on remplace
// immédiatement l'action du bouton par la navigation robuste ci-dessus.
const navigationBonusChoice = window.chooseBonusImmediate;
window.chooseBonusImmediate = function (bonusIndex, value) {
  const result = navigationBonusChoice.call(this, bonusIndex, value);
  const buttons = Array.from(document.querySelectorAll("#quiz button.primary"));
  const button = buttons.reverse().find((item) => item.textContent.trim() === "Suivant");
  if (button) {
    button.type = "button";
    button.onclick = () => goToNextBonus(bonusIndex, button);
    scrollToNextButton(button);
  }
  return result;
};

// Sécurité supplémentaire : tout nouveau rendu appelé par une autre partie du code
// revient également en haut.
const navigationRenderPage = renderPage;
renderPage = function (...args) {
  const result = navigationRenderPage.apply(this, args);
  lockTopAfterRender();
  return result;
};

const navigationRenderBonus = renderBonus;
renderBonus = function (...args) {
  const result = navigationRenderBonus.apply(this, args);
  lockTopAfterRender();
  return result;
};
