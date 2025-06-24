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

- [x] Importul bibliotecilor și citirea datasetului 
- [x] Preprocesarea datelor
- [x] Eliminarea duplicatelor
- [x] Curățarea textului (opțional)
- [x] Vectorizarea textului cu TF-IDF
- [ ] Împărțirea în train/test (supervizat și nesupervizat)
- [ ] Antrenarea modelului
- [ ] Evaluarea modelului
- [ ] Interfață simplă de testare (CLI sau Streamlit)

### Exemplu de funcționare

Un mic tool care returnează emoția probabilă asociată unui text (ex: „Mă simt foarte bine azi” -> „joy”).

## Diagramă de flux

![Diagrama_Flux](Diagrama_Flux.svg)

## Ierarhia de fișiere

```
.
├── Dataset
│   ├── split
│   │   ├── test-00000-of-00001.parquet
│   │   ├── train-00000-of-00001.parquet
│   │   └── validation-00000-of-00001.parquet
│   └── unsplit
│       └── train-00000-of-00001.parquet
├── Diagrama_Flux.svg
├── README.md
├── hist.py
├── main.py
├── out.txt
├── out2.txt
├── plots
│   ├── label_distribution_test.png
│   ├── label_distribution_train.png
│   └── label_distribution_validation.png
├── script.sh
└── venv
```

În folderul Dataset se găsesc fișierele .parquet cu setul de date, test, train și validation.

Fișierul hist.py a fost folosit pentru crearea folderului plots în care se găsesc histograme cu clasele de emoții surprinse în setul de date. Aceste histograme sunt utile pentru a avea o impresie de ansamblu asupra datelor ce urmează a fi prelucrate.

Fișierul main.py conține implementarea efectivă a pașilor prezentați anterior în secțiunea Descrierea generală a lucrării.

Fișierele de tip .txt sunt folosite pentru a salva output-ul din urma rulării fișierului main.py.

Fișierul script.sh are câteva comenzi pentru a șterge fișierele de output și a le crea la loc, pentru a nu exista probleme la suprascriere și rularea fișierului main.py.

## Descrierea funcționalităților implementate

## Tehnologii folosite

| Biblioteca | Descriere | Link documentatie |
| ---------- | --------- | ----------------- |
| scikit-learn | bibliotecă folosită pentru analiză de date, conține metode și obiecte pentru regresie, clustering, reducere dimensională, selecția modelului și preprocesare | [scikit-learn](https://scikit-learn.org/stable/) |
| pandas   | biblioteca ce conține metode, obiecte și implementări pentru structuri de date și analiză de date în python | [pandas](https://pandas.pydata.org/docs/)|
| nltk   | bibliotecă care conține metode, obiecte și implementări pentru procesarea textului scris de oameni pentru a oferi suport în cadrul dezvoltării NLP-urilor | [nltk](https://www.nltk.org/) |
| matplotlib   | bibliotecă cuprinzătoare pentru crearea de vizualizări statice, animate și interactive | [matplotlib](https://matplotlib.org/stable/index.html)|

## Surse

[Hugging Face](https://huggingface.co/datasets/dair-ai/emotion)

[Preprocessing Steps for Natural Language Processing (NLP): A Beginner’s Guide](https://medium.com/@maleeshadesilva21/preprocessing-steps-for-natural-language-processing-nlp-a-beginners-guide-d6d9bf7689c9)

[Geeks for Geeks](https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/)

[Scikit-learn documentation](https://scikit-learn.org/stable/)

[Emotion Detection NLP](https://github.com/Nishant2018/Emotion-Detection-NLP-ML-acc-95-/)