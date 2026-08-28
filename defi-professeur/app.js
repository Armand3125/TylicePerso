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
    title: "Página 10 — Comprensión",
    intro: "Una lectura antes de pasar al nivel siguiente.",
    reading: "Ce soir, Ailin et Armand ont rendez-vous en vidéo. Ils parlent de leur journée, rient et se disent des mots doux. Ensuite, ils jouent en français. Chaque page est corrigée en direct avant de passer à la suivante.",
    questions: [
      { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "28. Comment est leur rendez-vous ?", options: ["Au restaurant", "À l’école", "En vidéo"], answer: 2 },
      { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "29. Que font-ils avant de jouer ?", options: ["Ils parlent de leur journée", "Ils prennent l’avion", "Ils dorment"], answer: 0 },
      { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "30. Quand la page suivante apparaît-elle ?", options: ["Avant les réponses", "Après la correction", "Le lendemain"], answer: 1 }
    ]
  },
  {
    title: "Página 11 — Pronombres y cercanía",
    intro: "Ahora usamos pronombres para hablar de gestos y cercanía con más naturalidad.",
    questions: [
      { instruction: "Completa la frase.", prompt: "31. Quand tu m’embrasses dans le cou, j’___ pense encore après.", options: ["en", "y", "le"], answer: 1 },
      { instruction: "Elige la frase correcta.", prompt: "32. « Je veux te dire ce qui me plaît. »", options: ["Je veux le te dire.", "Je veux te le dire.", "Je te veux le dire."], answer: 1 },
      { instruction: "Completa la frase.", prompt: "33. Cette façon de me regarder, je ___ adore.", options: ["la", "lui", "y"], answer: 0 }
    ]
  },
  {
    title: "Página 12 — El condicional del deseo",
    intro: "Expresar deseos íntimos de forma más matizada con el conditionnel.",
    questions: [
      { instruction: "Completa la frase.", prompt: "34. J’___ que tu sois ici avec moi ce soir.", options: ["aimerais", "aimerai", "aimais"], answer: 0 },
      { instruction: "Elige la traducción correcta.", prompt: "35. « Me gustaría besarte lentamente. »", options: ["Je voudrais t’embrasser doucement.", "Je vais t’embrassais doucement.", "Je voudrais que je t’embrasse hier."], answer: 0 },
      { instruction: "Completa la frase.", prompt: "36. Tu ___ me dire ce que tu veux vraiment ?", options: ["pourrais", "pourras", "pouvais"], answer: 0 }
    ]
  },
  {
    title: "Página 13 — Recuerdos íntimos",
    intro: "Combinar passé composé e imparfait para contar un recuerdo cercano.",
    questions: [
      { instruction: "Completa la frase.", prompt: "37. Hier, pendant que nous ___ en vidéo, tu m’as fait rougir.", options: ["parlions", "avons parlé", "parlerons"], answer: 0 },
      { instruction: "Completa la frase.", prompt: "38. Quand tu ___ ma robe, j’ai souri.", options: ["as complimenté", "complimentais toujours", "complimenteras"], answer: 0 },
      { instruction: "Elige la frase más natural.", prompt: "39. Pour raconter une ambiance qui durait :", options: ["La lumière était douce.", "La lumière a été douce chaque seconde.", "La lumière sera douce hier."], answer: 0 }
    ]
  },
  {
    title: "Página 14 — Qui, que, dont",
    intro: "Describir a la otra persona y lo que te atrae con pronombres relativos.",
    questions: [
      { instruction: "Completa la frase.", prompt: "40. La façon ___ tu me regardes me déstabilise.", options: ["qui", "dont", "où"], answer: 1 },
      { instruction: "Completa la frase.", prompt: "41. Le baiser ___ tu m’as donné m’a surprise.", options: ["que", "qui", "dont"], answer: 0 },
      { instruction: "Completa la frase.", prompt: "42. J’aime les mots ___ me font rougir.", options: ["que", "qui", "dont"], answer: 1 }
    ]
  },
  {
    title: "Página 15 — Deseo y subjuntivo",
    intro: "Expresar deseos, preferencias y límites avec le subjonctif.",
    questions: [
      { instruction: "Completa la frase.", prompt: "43. Je veux que tu me ___ ce qui te plaît.", options: ["dises", "dis", "diras"], answer: 0 },
      { instruction: "Completa la frase.", prompt: "44. Je préfère que tu ___ plus doux avec moi.", options: ["es", "sois", "seras"], answer: 1 },
      { instruction: "Completa la frase.", prompt: "45. J’aime que tu me ___ avant d’aller plus loin.", options: ["demandes", "demanderas", "demandais"], answer: 0 }
    ]
  },
  {
    title: "Página 16 — Órdenes y pronombres",
    intro: "Comprender instrucciones directas y colocar bien los pronombres.",
    questions: [
      { instruction: "Elige la forma correcta.", prompt: "46. À l’impératif affirmatif :", options: ["Regarde-moi.", "Me regarde.", "Moi regarde."], answer: 0 },
      { instruction: "Elige la forma correcta.", prompt: "47. À l’impératif négatif :", options: ["Ne me quitte pas des yeux.", "Ne quitte-moi pas des yeux.", "Me ne quitte pas des yeux."], answer: 0 },
      { instruction: "Completa la frase.", prompt: "48. Dis-___ ce que tu veux que je fasse.", options: ["moi", "me", "je"], answer: 0 }
    ]
  },
  {
    title: "Página 17 — Si estuvieras aquí",
    intro: "Construir hipótesis íntimas con si + imparfait y conditionnel.",
    questions: [
      { instruction: "Completa la frase.", prompt: "49. Si tu ___ ici, je t’embrasserais tout de suite.", options: ["étais", "serais", "es"], answer: 0 },
      { instruction: "Completa la frase.", prompt: "50. Si nous étions seuls, je te ___ ce que j’ai en tête.", options: ["dirais", "disais", "dirai"], answer: 0 },
      { instruction: "Elige la frase correcta.", prompt: "51. Hipótesis irreal en el presente :", options: ["Si tu venais, je serais très heureuse.", "Si tu viendrais, je serais très heureuse.", "Si tu viens hier, je suis heureuse."], answer: 0 }
    ]
  },
  {
    title: "Página 18 — Límites y consentimiento",
    intro: "Hablar de deseo también significa saber expresar límites con precisión.",
    questions: [
      { instruction: "Elige la traducción correcta.", prompt: "52. « Si algo no te gusta, dímelo. »", options: ["Si quelque chose ne te plaît pas, dis-le-moi.", "Si quelque chose ne plaît, me dis.", "Si tu n’aimes pas, dis-moi le hier."], answer: 0 },
      { instruction: "Completa la frase.", prompt: "53. Je veux que tu t’arrêtes dès que je te le ___.", options: ["demande", "demander", "demandais de"], answer: 0 },
      { instruction: "Elige la frase más natural.", prompt: "54. Para preguntar antes de continuar :", options: ["Tu veux que je continue ?", "Tu veux je continue ?", "Est-ce tu veux que continuer ?"], answer: 0 }
    ]
  },
  {
    title: "Página 19 — Matices del deseo",
    intro: "Elegir expresiones más naturales para hablar de atracción y tensión.",
    questions: [
      { instruction: "Elige la frase más natural.", prompt: "55. « Me vuelves loca cuando me miras así. »", options: ["Tu me rends folle quand tu me regardes comme ça.", "Tu fais moi folle quand regardes ça.", "Je deviens toi folle comme regard."], answer: 0 },
      { instruction: "Completa la frase.", prompt: "56. Plus tu t’approches, plus j’___ envie de t’embrasser.", options: ["ai", "suis", "fais"], answer: 0 },
      { instruction: "Elige la traducción correcta.", prompt: "57. « No puedo dejar de pensar en ti. »", options: ["Je n’arrive pas à arrêter de penser à toi.", "Je ne peux penser toi jamais.", "Je n’arrête pas te penser de."], answer: 0 }
    ]
  },
  {
    title: "Página 20 — Comprensión avanzada",
    intro: "Última lectura : intención, deseo, confianza y comunicación.",
    reading: "Pendant leur appel, Ailin dit à Armand qu’elle aime quand il prend l’initiative, mais qu’elle veut toujours pouvoir lui dire de ralentir ou de s’arrêter. Armand lui répond qu’il préfère qu’elle dise clairement ce qu’elle veut. Ils décident alors de continuer le jeu en se donnant chacun leur tour une consigne en français, sans jamais oublier de vérifier que l’autre est à l’aise.",
    questions: [
      { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "58. Qu’est-ce qu’Ailin apprécie ?", options: ["Qu’Armand prenne parfois l’initiative", "Qu’Armand ne lui parle jamais", "Que le jeu s’arrête immédiatement"], answer: 0 },
      { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "59. Que veut Armand ?", options: ["Qu’elle dise clairement ce qu’elle veut", "Qu’elle ne réponde plus", "Qu’elle parle uniquement espagnol"], answer: 0 },
      { instruction: "Lee el texto y elige la réponse correcte.", prompt: "60. Quelle règle gardent-ils pendant le jeu ?", options: ["Vérifier que l’autre est à l’aise", "Ne jamais changer de sujet", "Toujours parler très vite"], answer: 0 }
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

let currentPage = 0;
let activeBonus = null;

const answers = {};
const bonusAnswers = {};
const pageScores = Array(pages.length).fill(null);
const bonusValidated = Array(bonuses.length).fill(false);
const quiz = document.getElementById("quiz");
const totalSteps = pages.length + bonuses.length;

function clampLevel(value) {
  return Math.max(0, Math.min(10, value));
}

function baseDeltaValue(kind, score) {
  if (kind === "bonus") return 0.5;
  if (score === 3) return 1;
  if (score === 2) return 0.5;
  return 0;
}

function getGameLevel() {
  let level = 1;
  pageScores.forEach((score) => {
    if (score === 3) level += 1;
    else if (score === 2) level += 0.5;
  });
  bonusValidated.forEach((ok) => {
    if (ok) level += 0.5;
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
  return pageIndex + Math.min(Math.floor(pageIndex / 2), bonuses.length);
}

function bonusStepIndex(bonusIndex) {
  return bonusIndex * 3 + 2;
}

function updateProgress(stepIndex, label) {
  document.getElementById("progress").style.width = `${((stepIndex + 1) / totalSteps) * 100}%`;
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

function disableCurrentForm() {
  document.querySelectorAll("#quiz input, #quiz button").forEach((element) => {
    element.disabled = true;
  });
}

function appendInlineFeedback(html, success = false) {
  const old = document.getElementById("inlineFeedback");
  if (old) old.remove();
  const feedback = document.createElement("div");
  feedback.id = "inlineFeedback";
  feedback.setAttribute("aria-live", "polite");
  feedback.style.marginTop = "18px";
  feedback.style.padding = "16px";
  feedback.style.borderRadius = "14px";
  feedback.style.border = success ? "1px solid #b9e2c8" : "1px solid #efb7c0";
  feedback.style.background = success ? "#edf9f1" : "#fff0f2";
  feedback.style.color = success ? "#187847" : "#b52d42";
  feedback.innerHTML = html;
  quiz.appendChild(feedback);
  feedback.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function deltaLabel(kind, score) {
  const value = baseDeltaValue(kind, score);
  if (!value) return "Pas d’augmentation";
  return `${formatSigned(value)} point${Math.abs(value) > 1 ? "s" : ""}`;
}

function submitCurrentPage() {
  const submittedPage = currentPage;
  const start = submittedPage * 3;
  if (![0, 1, 2].every((offset) => answers[start + offset] !== undefined)) {
    document.getElementById("validationWarning").style.display = "block";
    return;
  }

  let score = 0;
  const details = pages[submittedPage].questions.map((question, index) => {
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

  pageScores[submittedPage] = score;
  disableCurrentForm();
  const delta = deltaLabel("quiz", score);
  updateIntensity(delta);

  const mistakes = details.filter((detail) => !detail.isCorrect);
  if (mistakes.length === 0) {
    appendInlineFeedback(`<div style="font-size:1.15rem;font-weight:900">Félicitations 💜 — 3/3</div><div style="margin-top:5px">${delta}</div>`, true);
    setTimeout(() => continueAfterPage(submittedPage), 2200);
    return;
  }

  const mistakesHtml = mistakes.map((detail) => `
    <div style="margin-top:10px;padding-top:10px;border-top:1px solid rgba(181,45,66,.18)">
      <div style="font-weight:850">${detail.prompt}</div>
      <div style="margin-top:4px"><strong>Ta réponse :</strong> « ${detail.selected} »</div>
      <div style="margin-top:3px"><strong>Bonne réponse :</strong> « ${detail.correct} »</div>
    </div>`).join("");

  appendInlineFeedback(`<div style="font-size:1.05rem;font-weight:900">${mistakes.length} erreur${mistakes.length > 1 ? "s" : ""} — ${score}/3 · ${delta}</div>${mistakesHtml}`, false);
  setTimeout(() => continueAfterPage(submittedPage), 4200);
}

function submitBonus() {
  const submittedBonus = activeBonus;
  if (bonusAnswers[submittedBonus] === undefined) {
    document.getElementById("validationWarning").style.display = "block";
    return;
  }

  bonusValidated[submittedBonus] = true;
  disableCurrentForm();
  updateIntensity("+0,5 point");
  appendInlineFeedback(`<div style="font-size:1.1rem;font-weight:900">Bonus validé 💜 — +0,5 point</div>`, true);
  setTimeout(() => continueAfterBonus(submittedBonus), 1800);
}

function continueAfterPage(pageIndex) {
  if (pageIndex % 2 === 1) {
    const bonusIndex = Math.floor(pageIndex / 2);
    if (bonusIndex < bonuses.length) {
      renderBonus(bonusIndex);
      return;
    }
  }

  if (pageIndex < pages.length - 1) {
    currentPage = pageIndex + 1;
    renderPage();
  } else {
    finishQuiz();
  }
}

function continueAfterBonus(bonusIndex) {
  const nextPage = (bonusIndex + 1) * 2;
  if (nextPage < pages.length) {
    currentPage = nextPage;
    renderPage();
  } else {
    finishQuiz();
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