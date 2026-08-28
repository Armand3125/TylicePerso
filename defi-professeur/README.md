# Le Défi du professeur

Mini-jeu de français pensé pour être joué en direct pendant une visio, avec une progression de A1 vers A2/B1.

## Mise en ligne avec GitHub Pages

Dans le dépôt `Armand3125/TylicePerso` :

1. Ouvrir **Settings**.
2. Ouvrir **Pages** dans la rubrique **Code and automation**.
3. Choisir **Deploy from a branch**.
4. Sélectionner la branche **main** et le dossier **/(root)**.
5. Cliquer sur **Save**.

Après publication :

- Jeu : `https://armand3125.github.io/TylicePerso/defi-professeur/`
- L’URL racine du dépôt redirige également vers le jeu.

## Utilisation

1. Ouvrir le jeu pendant la visio.
2. Répondre aux trois questions de la page.
3. Cliquer sur **Valider**.
4. Si la page contient des erreurs, seules les erreurs et les bonnes réponses apparaissent en rouge sous le questionnaire. Si tout est juste, un message de félicitations apparaît.
5. La jauge de teasing est mise à jour automatiquement : 3/3 = +1, 2/3 = +0,5, 0 ou 1/3 = pas d’augmentation. Les bonus valent +0,5.
6. Le jeu passe automatiquement à la suite sans popup ni bouton supplémentaire.

Le défi contient maintenant 20 pages : les 10 premières restent simples, puis les pages 11 à 20 utilisent une grammaire plus avancée dans un registre intime/adulte.

## Fonctionnement technique

Le site est entièrement statique. Il n’utilise pas de connexion PeerJS/WebRTC, de code de session ni de second appareil. Toute la partie se déroule dans une seule page du navigateur.
