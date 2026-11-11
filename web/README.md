# SeismicAI Pulse, site (Firebase Hosting + Cloud Firestore)

Site estático (HTML/JS + Leaflet, modo escuro, mapa-múndi vetorial próprio sem tiles externos nem chave de API) que lê as previsões publicadas pelo pipeline (`src/publish_firebase.py`) no Firestore.

## Testar localmente sem Firebase

```bash
python src/publish_firebase.py --dry-run --backfill-test 20      # gera web/public/data/*.json
cd web/public && python -m http.server 8000                     # http://localhost:8000/?source=local
```

## Publicar no Firebase (uma vez)

1. Crie um projeto em https://console.firebase.google.com e, nele, ative o **Cloud Firestore** (modo de produção).
2. **Conta de serviço** (para o pipeline escrever): Configurações do projeto, Contas de serviço, *Gerar nova
   chave privada*, salve como `secrets/serviceAccount.json` (a pasta está no .gitignore).
3. **App Web** (para o site ler): Configurações do projeto, Seus apps, *Adicionar app*, Web, copie o objeto
   `firebaseConfig` para `web/public/firebase-config.js`.
4. Instale as ferramentas: `npm install -g firebase-tools`, depois `firebase login`.
5. Em `web/`, edite `.firebaserc` com o id do projeto e rode `firebase deploy --only firestore:rules,hosting`.

## Rotina de publicação (toda semana, ou quando quiser)

```bash
pip install firebase-admin
python src/publish_firebase.py --feed --backfill-test 52       # 1ª vez: semana corrente + 52 semanas do teste
python src/publish_firebase.py --feed                          # depois: só a semana corrente
```

O site em `https://SEU-PROJETO.web.app` passa a mostrar a nova semana sem novo deploy.
