# Le Défi du professeur

Mini-jeu de français A1 à distance avec validation en direct par le professeur.

## Mise en ligne avec GitHub Pages

Dans le dépôt `Armand3125/TylicePerso` :

1. Ouvrir **Settings**.
2. Ouvrir **Pages** dans la rubrique **Code and automation**.
3. Choisir **Deploy from a branch**.
4. Sélectionner la branche **main** et le dossier **/(root)**.
5. Cliquer sur **Save**.

Après publication :

- Élève : `https://armand3125.github.io/TylicePerso/defi-professeur/`
- Professeur : `https://armand3125.github.io/TylicePerso/defi-professeur/professeur.html`

## Utilisation

1. Ouvrir la page `professeur.html` sur l’ordinateur du professeur.
2. Cliquer sur **Copier le lien élève** et envoyer ce lien.
3. Garder les deux pages ouvertes pendant la partie.
4. Les réponses arrivent sur le tableau du professeur ; la page suivante ne se débloque qu’après validation.

## Fonctionnement technique

Le site est statique et la synchronisation en direct utilise PeerJS/WebRTC. Les réponses ne sont pas enregistrées dans le dépôt GitHub : elles sont transmises pendant la session entre les deux navigateurs.
