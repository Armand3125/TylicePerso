// Navigation robuste : descente vers « Suivant », puis retour forcé en haut après transition.

if ("scrollRestoration" in history) {
  history.scrollRestoration = "manual";
}

function forceGameTop() {
  const goTop = () => {
    const scrollingElement = document.scrollingElement || document.documentElement;
    if (scrollingElement) scrollingElement.scrollTop = 0;
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  };

  goTop();
  requestAnimationFrame(() => {
    goTop();
    requestAnimationFrame(goTop);
  });
  setTimeout(goTop, 40);
  setTimeout(goTop, 120);
  setTimeout(goTop, 300);
}

function scrollToNextControl() {
  const buttons = Array.from(document.querySelectorAll("#quiz button.primary"));
  const nextButton = buttons.reverse().find((button) => button.textContent.trim() === "Suivant");
  if (!nextButton) return;
  setTimeout(() => nextButton.scrollIntoView({ behavior: "smooth", block: "center" }), 120);
}

// À chaque rendu d'une nouvelle page ou d'un bonus, on revient réellement en haut.
const navigationOriginalRenderPage = renderPage;
renderPage = function (...args) {
  const result = navigationOriginalRenderPage.apply(this, args);
  forceGameTop();
  return result;
};

const navigationOriginalRenderBonus = renderBonus;
renderBonus = function (...args) {
  const result = navigationOriginalRenderBonus.apply(this, args);
  forceGameTop();
  return result;
};

// Quand les trois questions sont résolues, le bouton Suivant entre automatiquement dans la vue.
const navigationOriginalRenderNextButton = renderNextButton;
renderNextButton = function (...args) {
  const result = navigationOriginalRenderNextButton.apply(this, args);
  scrollToNextControl();
  return result;
};

// Même comportement pour les bonus Vrai/Faux.
const navigationOriginalChooseBonusImmediate = window.chooseBonusImmediate;
window.chooseBonusImmediate = function (...args) {
  const result = navigationOriginalChooseBonusImmediate.apply(this, args);
  scrollToNextControl();
  return result;
};

// Filet de sécurité : après n'importe quel clic sur « Suivant », on force le haut
// une fois le rendu suivant effectué et après que le bouton ait perdu le focus.
document.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button || button.textContent.trim() !== "Suivant") return;
  button.blur();
  setTimeout(forceGameTop, 0);
  setTimeout(forceGameTop, 80);
  setTimeout(forceGameTop, 220);
});
