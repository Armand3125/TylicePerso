// Barème V3 : avec 20 pages, les gains sont divisés par deux par rapport au jeu à 10 pages.
// 3/3 du premier coup = +0,5 ; 2/3 = +0,25 ; bonus = +0,25.

baseDeltaValue = function (kind, score) {
  if (kind === "bonus") return 0.25;
  if (score === 3) return 0.5;
  if (score === 2) return 0.25;
  return 0;
};

getGameLevel = function () {
  let level = 1;
  pageScores.forEach((score) => {
    if (score === 3) level += 0.5;
    else if (score === 2) level += 0.25;
  });
  bonusValidated.forEach((ok) => {
    if (ok) level += 0.25;
  });
  return clampLevel(level);
};

// Les bonus suivent la même montée que les pages : mignon -> couple -> hot -> sexuel.
bonuses.splice(0, bonuses.length,
  "J’aime recevoir un petit message tendre de toi avant de dormir.",
  "J’aime quand on prend le temps de préparer une soirée rien que tous les deux.",
  "J’aime quand tu me regardes longtemps avec un air provocateur.",
  "J’aime quand tu me dis clairement que tu me trouves sexy.",
  "J’aime quand tu me demandes ce que j’aimerais que tu me fasses.",
  "J’aime quand tu me regardes me toucher.",
  "J’aime quand tu me donnes des ordres pendant que je me touche.",
  "J’aimerais utiliser un vibromasseur pendant que tu me regardes.",
  "J’aime quand tu décides quand je peux jouir.",
  "J’aimerais te dire exactement comment me faire jouir."
);

bonusTranslations.splice(0, bonusTranslations.length,
  "Me gusta recibir un mensajito tierno tuyo antes de dormir.",
  "Me gusta cuando nos tomamos el tiempo de preparar una noche solo para nosotros dos.",
  "Me gusta cuando me miras durante mucho tiempo con una expresión provocadora.",
  "Me gusta cuando me dices claramente que te parezco sexy.",
  "Me gusta cuando me preguntas qué me gustaría que me hicieras.",
  "Me gusta cuando me miras tocarme.",
  "Me gusta cuando me das órdenes mientras me toco.",
  "Me gustaría usar un vibrador mientras me miras.",
  "Me gusta cuando decides cuándo puedo llegar al orgasmo.",
  "Me gustaría decirte exactamente cómo hacerme llegar al orgasmo."
);

updateLivePageScore = function (pageIndex) {
  const state = pageState(pageIndex);
  const firstTryCorrect = state.firstCorrect.filter(Boolean).length;

  let scoreForMeter = 0;
  if (firstTryCorrect >= 3) scoreForMeter = 3;
  else if (firstTryCorrect >= 2) scoreForMeter = 2;

  if (scoreForMeter === state.awardedScore) return;

  const previous = state.awardedScore;
  state.awardedScore = scoreForMeter;
  pageScores[pageIndex] = scoreForMeter;

  const gained = baseDeltaValue("quiz", scoreForMeter) - baseDeltaValue("quiz", previous);
  updateIntensity(gained > 0 ? `+${formatLevel(gained)} point` : "");
};

window.chooseBonusImmediate = function (bonusIndex, value) {
  if (bonusValidated[bonusIndex]) return;
  bonusAnswers[bonusIndex] = value;
  bonusValidated[bonusIndex] = true;
  const gain = baseDeltaValue("bonus");
  updateIntensity(`+${formatLevel(gain)} point`);

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
  const gain = baseDeltaValue("bonus");
  quiz.innerHTML = `
    <div class="bonus-page" id="bonus-page-${bonusIndex}">
      <span class="bonus-chip">Bonus intime ${bonusIndex + 1} / ${bonuses.length}</span>
      <h2>Vrai ou faux ?</h2>
      <p class="page-intro">Réponds simplement et honnêtement.</p>
      <div class="bonus-question">
        <div class="bonus-fr">${bonuses[bonusIndex]}</div>
        <div style="margin-top:7px;color:#8a818b;font-size:.88rem;line-height:1.4;font-weight:500">${translation}</div>
        <p class="bonus-points">Réponse : +${formatLevel(gain)} point</p>
        <label class="option" id="bonus-${bonusIndex}-true"><input type="radio" name="bonus${bonusIndex}" onchange="chooseBonusImmediate(${bonusIndex},true)"><span>Vrai</span></label>
        <label class="option" id="bonus-${bonusIndex}-false"><input type="radio" name="bonus${bonusIndex}" onchange="chooseBonusImmediate(${bonusIndex},false)"><span>Faux</span></label>
      </div>
    </div>`;
  updateProgress(bonusStepIndex(bonusIndex), `Bonus ${bonusIndex + 1} sur ${bonuses.length}`);
  updateIntensity();
  window.scrollTo({ top: 0, behavior: "auto" });
};
