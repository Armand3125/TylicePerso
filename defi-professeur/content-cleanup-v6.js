// V6 : le contexte de couple est implicite. On garde uniquement langue, scènes de vie et sexualité.

// Page 7 : formulations sensuelles sans détour par arrêt / ralentissement.
pages[6].questions[1] = {
  instruction: "Completa la frase.",
  prompt: "20. Quand Ailin se rapproche, Armand la ___ contre lui.",
  options: ["serre", "serres", "serrer"],
  answer: 0
};

pages[6].questions[2] = {
  instruction: "Elige la traducción correcta.",
  prompt: "21. « Bésame otra vez. »",
  options: ["Embrasse-moi encore.", "Regarde-moi demain.", "Laisse-moi partir."],
  answer: 0
};

// Page 15 : plaisir et orgasme, sans question de permission.
pages[14].questions[1] = {
  instruction: "Completa la frase.",
  prompt: "44. Ailin demande à Armand de ___ encore un peu.",
  options: ["continuer", "continue", "continué"],
  answer: 0
};

pages[14].questions[2] = {
  instruction: "Elige la traducción correcta.",
  prompt: "45. « Más fuerte, por favor. »",
  options: ["Plus fort, s’il te plaît.", "Plus tard, s’il te plaît.", "Plus loin, s’il te plaît."],
  answer: 0
};

// Page 20 entièrement recentrée sur une scène de vie de couple qui devient sexuelle.
pages[19] = {
  title: "Página 20 — Comprensión finale : une nuit ensemble",
  intro: "Dernier texte B1 : comprendre une scène complète et les détails de l’histoire.",
  reading: "Après une longue journée à Toulouse, Armand et Ailin rentrent chez eux vers minuit. Ils commencent par manger quelque chose dans la cuisine en racontant leur soirée. Ailin va ensuite se doucher pendant qu’Armand range les verres. Quand elle revient dans la chambre avec seulement une serviette, Armand la regarde en souriant. Elle s’approche de lui, laisse tomber la serviette et l’embrasse. Ils passent ensuite un long moment au lit. Ailin sort le vibromasseur du tiroir et le pose dans la main d’Armand. Plus tard, elle lui dit qu’elle est proche de l’orgasme et lui demande d’augmenter l’intensité. Après, ils restent allongés ensemble quelques minutes avant qu’Armand se lève pour aller chercher deux verres d’eau.",
  questions: [
    {
      instruction: "Lee el texto y elige la respuesta correcta.",
      prompt: "58. Que font Armand et Ailin en rentrant ?",
      options: ["Ils mangent quelque chose dans la cuisine", "Ils vont directement dormir", "Ils sortent à nouveau"],
      answer: 0
    },
    {
      instruction: "Lee el texto y elige la respuesta correcta.",
      prompt: "59. Qu’est-ce qu’Ailin donne à Armand dans la chambre ?",
      options: ["Le vibromasseur", "Son téléphone", "Une bouteille"],
      answer: 0
    },
    {
      instruction: "Lee el texto y elige la respuesta correcta.",
      prompt: "60. Que fait Armand à la fin ?",
      options: ["Il va chercher de l’eau", "Il part travailler", "Il prend une douche"],
      answer: 0
    }
  ]
};

// Nettoyage des derniers mots hérités d'anciennes versions susceptibles d'être encore affichés.
function stripConsentVocabulary(text) {
  if (typeof text !== "string") return text;
  return text
    .replace(/consentement/gi, "complicité")
    .replace(/leurs limites/gi, "leurs envies")
    .replace(/les limites/gi, "les envies")
    .replace(/limites/gi, "envies")
    .replace(/à l[’']aise/gi, "dans l’ambiance")
    .replace(/mots? de sécurité/gi, "mots du jeu");
}

pages.forEach((page) => {
  page.title = stripConsentVocabulary(page.title);
  page.intro = stripConsentVocabulary(page.intro);
  if (page.reading) page.reading = stripConsentVocabulary(page.reading);
  page.questions.forEach((question) => {
    question.instruction = stripConsentVocabulary(question.instruction);
    question.prompt = stripConsentVocabulary(question.prompt);
    question.options = question.options.map(stripConsentVocabulary);
  });
});
