window.showQuizFeedback = function (submission, adjustment) {
  const mistakes = submission.details.filter((detail) => !detail.isCorrect);

  if (mistakes.length === 0) {
    document.getElementById("modalContent").innerHTML = `
      <h2 id="feedbackTitle">Parfait 💜</h2>
      <p class="perfect">Aucune faute sur cette page.</p>`;
    document.getElementById("feedbackModal").classList.add("open");
    return;
  }

  const mistakesHtml = mistakes.map((detail) => `
    <article style="margin-bottom:14px;padding:14px;border-radius:12px;background:var(--bad-soft);border:1px solid #f0c4cb">
      <div style="font-weight:850;margin-bottom:9px;color:var(--ink)">${detail.prompt}</div>
      <div style="color:var(--bad);margin-bottom:6px"><strong>Ta réponse :</strong> « ${detail.selected} »</div>
      <div style="color:var(--good);font-weight:800"><strong>Bonne réponse :</strong> « ${detail.correct} »</div>
    </article>`).join("");

  document.getElementById("modalContent").innerHTML = `
    <h2 id="feedbackTitle">À revoir avec le professeur</h2>
    <p>Voici uniquement les questions où tu t’es trompée :</p>
    <div>${mistakesHtml}</div>`;
  document.getElementById("feedbackModal").classList.add("open");
};
