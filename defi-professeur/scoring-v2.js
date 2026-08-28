// Barème V2 : tous les gains de teasing sont doublés.

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

  const previousDelta = baseDeltaValue("quiz", previous);
  const newDelta = baseDeltaValue("quiz", scoreForMeter);
  const gained = newDelta - previousDelta;
  updateIntensity(gained > 0 ? `+${formatLevel(gained)} point${gained > 1 ? "s" : ""}` : "");
};

window.chooseBonusImmediate = function (bonusIndex, value) {
  if (bonusValidated[bonusIndex]) return;
  bonusAnswers[bonusIndex] = value;
  bonusValidated[bonusIndex] = true;
  const gain = baseDeltaValue("bonus");
  updateIntensity(`+${formatLevel(gain)} point${gain > 1 ? "s" : ""}`);

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
        <p class="bonus-points">Réponse : +${formatLevel(gain)} point${gain > 1 ? "s" : ""}</p>
        <label class="option" id="bonus-${bonusIndex}-true"><input type="radio" name="bonus${bonusIndex}" onchange="chooseBonusImmediate(${bonusIndex},true)"><span>Vrai</span></label>
        <label class="option" id="bonus-${bonusIndex}-false"><input type="radio" name="bonus${bonusIndex}" onchange="chooseBonusImmediate(${bonusIndex},false)"><span>Faux</span></label>
      </div>
    </div>`;
  updateProgress(bonusStepIndex(bonusIndex), `Bonus ${bonusIndex + 1} sur ${bonuses.length}`);
  updateIntensity();
  window.scrollTo({ top: 0, behavior: "auto" });
};
