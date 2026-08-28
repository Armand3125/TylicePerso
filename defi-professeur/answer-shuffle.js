// Répartit les bonnes réponses sur A/B/C de façon équilibrée.
// Chaque page contient exactement une bonne réponse en A, une en B et une en C,
// dans un ordre aléatoire à chaque nouveau chargement du jeu.

function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i -= 1) {
    const values = new Uint32Array(1);
    if (window.crypto && window.crypto.getRandomValues) {
      window.crypto.getRandomValues(values);
      const j = values[0] % (i + 1);
      [array[i], array[j]] = [array[j], array[i]];
    } else {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }
  return array;
}

pages.forEach((page) => {
  const targetPositions = shuffleArray([0, 1, 2]);

  page.questions.forEach((question, localIndex) => {
    const correctOption = question.options[question.answer];
    const wrongOptions = question.options.filter((_, index) => index !== question.answer);
    shuffleArray(wrongOptions);

    const target = targetPositions[localIndex];
    const reordered = [];
    let wrongIndex = 0;

    for (let position = 0; position < 3; position += 1) {
      if (position === target) reordered.push(correctOption);
      else reordered.push(wrongOptions[wrongIndex++]);
    }

    question.options = reordered;
    question.answer = target;
  });
});
