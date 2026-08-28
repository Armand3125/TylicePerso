# Le Défi du professeur

Mini-jeu de français A1 pensé pour être joué en direct pendant une visio.

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
4. La correction apparaît immédiatement sur le même écran.
5. Le professeur peut appliquer un ajustement de −1, 0 ou +1.
6. Cliquer sur **Siguiente** pour passer à la suite.

## Fonctionnement technique

Le site est entièrement statique. Il n’utilise plus de connexion PeerJS/WebRTC, de code de session ni de second appareil. Toute la partie se déroule dans une seule page du navigateur.
