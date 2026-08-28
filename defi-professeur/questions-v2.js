// Nouvelle série de questions pour les pages 1 à 7.
// Chargée après app.js afin de remplacer uniquement le début du quiz.

pages.splice(0, 7,
  {
    title: "Página 1 — Nuestra videollamada",
    intro: "Nuevas frases sencillas para empezar la cita en vídeo.",
    questions: [
      { instruction: "Completa la frase.", prompt: "1. Ce soir, je ___ très contente de te voir.", options: ["suis", "es", "ai"], answer: 0 },
      { instruction: "Elige la traducción correcta.", prompt: "2. « Te espero a las nueve. »", options: ["Je te regarde à neuf heures.", "Je t’attends à neuf heures.", "Je dors à neuf heures."], answer: 1 },
      { instruction: "Completa la frase.", prompt: "3. Quand la caméra s’allume, tu me ___.", options: ["souris", "sourit", "souriez"], answer: 0 }
    ]
  },
  {
    title: "Página 2 — Mensajes entre nosotros",
    intro: "Hablar de los pequeños mensajes que nos enviamos.",
    questions: [
      { instruction: "Completa la frase.", prompt: "4. Je t’envoie ___ message avant de dormir.", options: ["une", "un", "des la"], answer: 1 },
      { instruction: "Completa la frase.", prompt: "5. Tu me ___ souvent rire.", options: ["fais", "fait", "faisons"], answer: 0 },
      { instruction: "Elige la traducción correcta.", prompt: "6. « Tengo ganas de verte. »", options: ["J’ai envie de te voir.", "Je dois te voir hier.", "Je suis envie de toi voir."], answer: 0 }
    ]
  },
  {
    title: "Página 3 — Acercarse",
    intro: "Vocabulario sencillo para hablar de cercanía y gestos.",
    questions: [
      { instruction: "Completa la frase.", prompt: "7. Viens un peu plus ___ de moi.", options: ["près", "loin", "derrière de"], answer: 0 },
      { instruction: "Completa la frase.", prompt: "8. Je pose ma main ___ ton épaule.", options: ["sur", "chez", "avec"], answer: 0 },
      { instruction: "Elige la traducción correcta.", prompt: "9. « Acércate. »", options: ["Éloigne-toi.", "Approche-toi.", "Assieds-moi."], answer: 1 }
    ]
  },
  {
    title: "Página 4 — Coquetear",
    intro: "Frases nuevas para hablar de atracción sin complicar la gramática.",
    questions: [
      { instruction: "Completa la frase.", prompt: "10. Quand tu souris comme ça, tu me ___ rougir.", options: ["fais", "fait", "faire"], answer: 0 },
      { instruction: "Completa la frase.", prompt: "11. J’aime ___ tu me regardes comme ça.", options: ["quand", "de que", "où de"], answer: 0 },
      { instruction: "Elige la frase correcta.", prompt: "12. ¿Cómo se dice « Estás muy sexy esta noche »?", options: ["Tu as sexy ce soir.", "Tu es très sexy ce soir.", "Tu fais très sexy hier soir."], answer: 1 }
    ]
  },
  {
    title: "Página 5 — Besos y deseo",
    intro: "Expresar un deseo sencillo y comprender instrucciones cortas.",
    questions: [
      { instruction: "Completa la frase.", prompt: "13. J’ai envie de t’___.", options: ["embrasser", "embrasses", "embrassé"], answer: 0 },
      { instruction: "Completa la frase.", prompt: "14. Embrasse-___ doucement.", options: ["moi", "me", "je"], answer: 0 },
      { instruction: "Elige la traducción correcta.", prompt: "15. « No pares. »", options: ["Ne commence pas.", "Ne t’arrête pas.", "Ne me parle pas."], answer: 1 }
    ]
  },
  {
    title: "Página 6 — Decir lo que quieres",
    intro: "Pedir más, menos o continuar con frases naturales.",
    questions: [
      { instruction: "Completa la frase.", prompt: "16. Je veux ___ tu continues.", options: ["que", "qui", "de"], answer: 0 },
      { instruction: "Completa la frase.", prompt: "17. Tu peux aller un peu plus ___ ?", options: ["vite", "vitesse", "vites"], answer: 0 },
      { instruction: "Elige la traducción correcta.", prompt: "18. « Más despacio, por favor. »", options: ["Plus doucement, s’il te plaît.", "Plus loin, s’il te plaît.", "Plus demain, s’il te plaît."], answer: 0 }
    ]
  },
  {
    title: "Página 7 — Deseo y confianza",
    intro: "Comprender preguntas simples sobre lo que cada uno quiere.",
    questions: [
      { instruction: "Completa la frase.", prompt: "19. Est-ce que tu ___ que je continue ?", options: ["veux", "veut", "voulez"], answer: 0 },
      { instruction: "Completa la frase.", prompt: "20. Si je dis stop, tu t’___.", options: ["arrêtes", "arrête", "arrêter"], answer: 0 },
      { instruction: "Elige la traducción correcta.", prompt: "21. « Estoy de acuerdo. »", options: ["Je suis d’accord.", "J’ai d’accord.", "Je fais accord."], answer: 0 }
    ]
  }
);
