// Rééquilibrage V3 : première moitié plus légère, seconde moitié plus adulte et plus axée compréhension.

// Pages 8 à 10 : moins sexuelles, davantage orientées conversation et couple.
pages[7] = {
  title: "Página 8 — Organizar una cita",
  intro: "Hablar de horarios, lugares y pequeños planes juntos.",
  questions: [
    { instruction: "Completa la frase.", prompt: "22. On se retrouve ___ vingt heures ?", options: ["à", "en", "de"], answer: 0 },
    { instruction: "Elige la traducción correcta.", prompt: "23. « Podemos cenar juntos. »", options: ["Nous pouvons dîner ensemble.", "Nous pouvons dormir demain.", "Nous sommes dîner ensemble."], answer: 0 },
    { instruction: "Completa la frase.", prompt: "24. Après le dîner, nous ___ nous promener.", options: ["allons", "avons", "sommes"], answer: 0 }
  ]
};

pages[8] = {
  title: "Página 9 — Petites habitudes",
  intro: "Parler de ce qu’on fait souvent quand on est en couple.",
  questions: [
    { instruction: "Completa la frase.", prompt: "25. Le week-end, nous ___ souvent longtemps au téléphone.", options: ["parlons", "parlez", "parlent"], answer: 0 },
    { instruction: "Elige la frase correcta.", prompt: "26. ¿Cómo se dice « Me mandas un mensaje cuando llegas »?", options: ["Tu m’envoies un message quand tu arrives.", "Tu me message quand arrives.", "Tu envoies moi quand arrivé."], answer: 0 },
    { instruction: "Completa la frase.", prompt: "27. J’aime bien quand on ___ ensemble.", options: ["rit", "rions", "rire"], answer: 0 }
  ]
};

pages[9] = {
  title: "Página 10 — Comprensión de pareja",
  intro: "Lee una pequeña escena y responde sin traducir palabra por palabra.",
  reading: "Ce samedi, Léa et Hugo ont prévu une soirée tranquille. Ils commencent par cuisiner ensemble, puis ils regardent une série. Hugo propose ensuite de sortir boire un verre, mais Léa préfère rester à la maison parce qu’elle est fatiguée. Ils décident finalement de préparer un dessert et de continuer la soirée sur le canapé.",
  questions: [
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "28. Que font-ils d’abord ?", options: ["Ils cuisinent ensemble", "Ils vont au cinéma", "Ils prennent l’avion"], answer: 0 },
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "29. Pourquoi Léa ne veut-elle pas sortir ?", options: ["Elle est fatiguée", "Elle travaille", "Elle n’aime pas Hugo"], answer: 0 },
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "30. Que décident-ils finalement ?", options: ["Préparer un dessert", "Aller danser", "Se coucher immédiatement"], answer: 0 }
  ]
};

// Page 11 : correction de la question 33 + début de la montée en intensité.
pages[10] = {
  title: "Página 11 — Désir et vocabulaire intime",
  intro: "À partir d’ici, le vocabulaire devient plus adulte.",
  questions: [
    { instruction: "Completa la frase.", prompt: "31. Quand tu m’embrasses dans le cou, j’___ pense encore après.", options: ["en", "y", "le"], answer: 1 },
    { instruction: "Elige la frase correcta.", prompt: "32. « Je veux te dire ce qui me plaît. »", options: ["Je veux le te dire.", "Je veux te le dire.", "Je te veux le dire."], answer: 1 },
    { instruction: "Completa la frase.", prompt: "33. Cette façon de me regarder, je ___ adore.", options: ["l’", "lui", "y"], answer: 0 }
  ]
};

pages[11] = {
  title: "Página 12 — Comprensión : faire monter le désir",
  intro: "Comprendre une scène intime et distinguer désir, rythme et consentement.",
  reading: "Deux adultes sont en appel vidéo. Après quelques minutes, l’un demande à l’autre s’il a envie de continuer le jeu de manière plus intime. L’autre répond oui, mais précise qu’il veut pouvoir ralentir à tout moment. Ils commencent par se regarder, se complimenter et se décrire ce qu’ils aimeraient faire lorsqu’ils seront ensemble. Chacun vérifie régulièrement que l’autre est toujours à l’aise.",
  questions: [
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "34. Que demande l’un des deux avant de rendre le jeu plus intime ?", options: ["Si l’autre en a envie", "Son mot de passe", "Son adresse"], answer: 0 },
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "35. Que veut pouvoir faire l’autre à tout moment ?", options: ["Ralentir", "Quitter la ville", "Changer de langue"], answer: 0 },
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "36. Que font-ils régulièrement ?", options: ["Ils vérifient que l’autre est à l’aise", "Ils coupent la caméra", "Ils changent de sujet"], answer: 0 }
  ]
};

pages[12] = {
  title: "Página 13 — Parler de masturbation",
  intro: "Vocabulaire direct pour exprimer ce qu’on fait et ce qu’on veut.",
  questions: [
    { instruction: "Elige la traducción correcta.", prompt: "37. « J’ai envie de me toucher pendant que tu me regardes. »", options: ["Tengo ganas de tocarme mientras me miras.", "Tengo ganas de dormir mientras hablas.", "Quiero que cierres la cámara."], answer: 0 },
    { instruction: "Completa la frase.", prompt: "38. Je veux que tu me ___ quand ralentir.", options: ["dises", "dit", "diras"], answer: 0 },
    { instruction: "Elige la frase correcta.", prompt: "39. Pour demander clairement la permission :", options: ["Tu veux que je continue ?", "Je continue sans demander.", "Tu dois continuer."], answer: 0 }
  ]
};

pages[13] = {
  title: "Página 14 — Comprensión : se déshabiller",
  intro: "Comprendre une scène plus explicite sans perdre les nuances de langue.",
  reading: "Pendant leur jeu, Camille demande à Alex s’il souhaite qu’elle enlève son haut. Alex répond qu’il en a envie, mais lui dit de prendre son temps. Camille retire lentement quelques vêtements, puis demande à Alex ce qu’il aimerait voir ensuite. Alex répond qu’il préfère la regarder se toucher plutôt que lui donner immédiatement une nouvelle consigne.",
  questions: [
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "40. Que demande Camille au début ?", options: ["Si Alex veut qu’elle enlève son haut", "Si Alex veut dîner", "Si Alex veut partir"], answer: 0 },
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "41. Que lui répond Alex sur le rythme ?", options: ["De prendre son temps", "D’aller très vite", "D’arrêter l’appel"], answer: 0 },
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "42. Que préfère finalement Alex ?", options: ["La regarder se toucher", "Éteindre la caméra", "Mettre de la musique"], answer: 0 }
  ]
};

pages[14] = {
  title: "Página 15 — Plaisir et orgasme",
  intro: "Employer des mots adultes avec une grammaire plus précise.",
  questions: [
    { instruction: "Completa la frase.", prompt: "43. J’aimerais que tu me ___ ce qui te fait le plus de bien.", options: ["dises", "dis", "dirais"], answer: 0 },
    { instruction: "Elige la traducción correcta.", prompt: "44. « Je suis proche de l’orgasme. »", options: ["Estoy cerca del orgasmo.", "Estoy lejos de casa.", "Tengo sueño."], answer: 0 },
    { instruction: "Completa la frase.", prompt: "45. Ne me laisse pas jouir avant que je te le ___.", options: ["demande", "demandes", "demander"], answer: 0 }
  ]
};

pages[15] = {
  title: "Página 16 — Comprensión : jeu de contrôle",
  intro: "Lire une scène de contrôle consensuel et comprendre les consignes.",
  reading: "Deux partenaires adultes décident de jouer avec le contrôle. L’un demande à l’autre de garder les mains immobiles pendant trente secondes. Ensuite, il lui permet de se toucher mais seulement très lentement. Ils ont choisi le mot « rouge » pour arrêter immédiatement le jeu et « orange » pour demander de ralentir. Quand l’excitation devient trop forte, le partenaire qui reçoit les consignes dit « orange », et l’autre diminue aussitôt l’intensité.",
  questions: [
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "46. Que doit faire la personne pendant trente secondes ?", options: ["Garder les mains immobiles", "Courir", "Fermer l’ordinateur"], answer: 0 },
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "47. Que signifie « rouge » ?", options: ["Arrêter immédiatement", "Accélérer", "Changer de position"], answer: 0 },
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "48. Que fait l’autre quand il entend « orange » ?", options: ["Il diminue l’intensité", "Il continue pareil", "Il quitte la pièce"], answer: 0 }
  ]
};

pages[16] = {
  title: "Página 17 — Sexe oral et préférences",
  intro: "Comprendre et formuler des préférences sexuelles de manière directe.",
  questions: [
    { instruction: "Elige la traducción correcta.", prompt: "49. « J’aimerais que tu me fasses du sexe oral. »", options: ["Me gustaría que me hicieras sexo oral.", "Me gustaría que cocinaras.", "Quiero que cierres la puerta."], answer: 0 },
    { instruction: "Completa la frase.", prompt: "50. Dis-moi si tu préfères que je ___ plus doucement.", options: ["continue", "continues", "continuerai"], answer: 0 },
    { instruction: "Elige la frase más natural.", prompt: "51. Pour demander une préférence :", options: ["Tu préfères plus vite ou plus doucement ?", "Tu préfères je vite ?", "Plus doucement tu préférence ?"], answer: 0 }
  ]
};

pages[17] = {
  title: "Página 18 — Comprensión : utiliser un jouet",
  intro: "Comprendre une scène où deux adultes parlent d’un sextoy et de limites.",
  reading: "Avant de commencer, Nora montre à Sam un petit vibromasseur et lui demande s’il est d’accord pour qu’elle l’utilise pendant leur jeu. Sam répond oui et lui demande de commencer au niveau le plus faible. Après quelques minutes, Nora augmente légèrement l’intensité. Sam lui dit que c’est agréable mais qu’il ne veut pas encore aller plus fort. Nora garde donc le même réglage jusqu’à ce qu’il lui demande lui-même de l’augmenter.",
  questions: [
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "52. Que fait Nora avant d’utiliser le jouet ?", options: ["Elle demande si Sam est d’accord", "Elle l’utilise sans prévenir", "Elle le range"], answer: 0 },
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "53. Quel niveau Sam demande-t-il au début ?", options: ["Le plus faible", "Le plus fort", "Aucun"], answer: 0 },
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "54. Pourquoi Nora n’augmente-t-elle pas davantage ?", options: ["Sam lui dit qu’il ne veut pas encore plus fort", "Le jouet est cassé", "Elle est fatiguée"], answer: 0 }
  ]
};

pages[18] = {
  title: "Página 19 — Attacher, ordonner, arrêter",
  intro: "Vocabulaire explicite autour des consignes et du consentement.",
  questions: [
    { instruction: "Completa la frase.", prompt: "55. J’aime quand tu m’___ les poignets, si je suis d’accord.", options: ["attaches", "attache", "attacher"], answer: 0 },
    { instruction: "Elige la traducción correcta.", prompt: "56. « Donne-moi un ordre, mais arrête si je dis stop. »", options: ["Dame una orden, pero para si digo stop.", "Dame tu teléfono y vete.", "No me hables nunca."], answer: 0 },
    { instruction: "Completa la frase.", prompt: "57. Si je te demande d’arrêter, tu dois ___ immédiatement.", options: ["t’arrêter", "arrêtes", "arrêté"], answer: 0 }
  ]
};

pages[19] = {
  title: "Página 20 — Comprensión finale : plaisir, confiance et limites",
  intro: "Dernier texte, plus long et plus explicite.",
  reading: "Maya et Julien passent une soirée intime. Ils ont parlé avant de commencer de ce qu’ils aiment et de ce qu’ils ne veulent pas faire. Maya aime être guidée et recevoir des ordres, mais elle veut rester libre de changer d’avis. Julien aime prendre l’initiative, mais il lui demande régulièrement si le rythme lui convient. Plus tard, Maya lui demande de l’attacher légèrement et d’utiliser un vibromasseur. Julien vérifie les attaches, commence doucement et augmente seulement quand Maya le lui demande. Lorsqu’elle dit qu’elle est proche de l’orgasme, ils décident ensemble s’ils veulent continuer ou ralentir. Pour eux, le jeu fonctionne parce que le désir, la confiance et la communication restent présents du début à la fin.",
  questions: [
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "58. Qu’ont-ils fait avant de commencer ?", options: ["Ils ont parlé de leurs envies et limites", "Ils ont évité toute discussion", "Ils ont invité des amis"], answer: 0 },
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "59. Quand Julien augmente-t-il l’intensité du vibromasseur ?", options: ["Quand Maya le lui demande", "Immédiatement", "Jamais"], answer: 0 },
    { instruction: "Lee el texto y elige la respuesta correcta.", prompt: "60. Pourquoi leur jeu fonctionne-t-il ?", options: ["Parce qu’ils communiquent et se font confiance", "Parce qu’ils ne parlent jamais", "Parce qu’ils ignorent leurs limites"], answer: 0 }
  ]
};

// Double tous les gains de la jauge : page parfaite +2, deux bonnes du premier coup +1, bonus +1.
baseDeltaValue = function (kind, score) {
  if (kind === "bonus") return 1;
  if (score === 3) return 2;
  if (score === 2) return 1;
  return 0;
};

getGameLevel = function () {
  let level = 1;
  pageScores.forEach((score) => {
    if (score === 3) level += 2;
    else if (score === 2) level += 1;
  });
  bonusValidated.forEach((ok) => {
    if (ok) level += 1;
  });
  return clampLevel(level);
};
