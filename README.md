# Detectarea Emoțiilor din Text folosind un  model de NLP

## Descrierea generală a lucrării

### Obiectiv

Construirea unui model simplu de clasificare a emoțiilor pe baza textului introdus de utilizator.

### Tehnologii și tool-uri

1. Python 3.x
2. Pandas
3. scikit-learn
4. Jupyter Notebook (sau Google Colab)

### Set de date:

Folosește setul Emotion Dataset de pe HuggingFace (dair-ai/motion) sau un CSV simplu cu coloanele text și label.

### Pași

1. Importul bibliotecilor și citirea datasetului 
2. Preprocesarea datelor
3. Eliminarea duplicatelor
4. Curățarea textului (opțional)
5. Vectorizarea textului cu TF-IDF
6. Împărțirea în train/test (supervizat și nesupervizat)
7. Antrenarea modelului
8. Evaluarea modelului
9. Interfață simplă de testare (CLI sau Streamlit)

### Exemplu de funcționare

Un mic tool care returnează emoția probabilă asociată unui text (ex: „Mă simt foarte bine azi” ->  „joy”).

## Prelucrarea datelor