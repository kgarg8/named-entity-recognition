# Named Entity Recognition with Pytorch
The repository is cloned from [this](https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/) for own learning.

## Corpus
[Annotated Corpus for Named Entity Recognition](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data)

This Groningen Meaning Bank-based corpus is tagged, annotated and built specifically to train the classifier to predict named entities such as geographical entity, person, event, location, etc.

The corpus is included in the repository [here](https://github.com/KrishnaGarg/named-entity-recognition/blob/master/data/ner_dataset.csv).

## Main Requirements
- numpy
- Pillow
- torch>=1.2
- tabulate
- tqdm

Please refer to [requirements.txt](https://github.com/KrishnaGarg/named-entity-recognition/blob/master/requirements.txt) for refering to the versions I used.

**The code is GPU-compatible.**

## Get ready to run the code yourself

1. Create train, val, test splits using:
```
python build_dataset.py
```

2. Build the words and tags vocabularies and dataset parameters using:
```
python build_vocab.py --min_count_word=1 --min_count_tag=1
````

3. Train the model using:
```
python train.py --model_dir=experiments/base_model --restore_file=best
```
Feel free to change the training parameters like learning_rate, batch_size, num_epochs, etc. using [params.json](https://github.com/KrishnaGarg/named-entity-recognition/blob/master/experiments/base_model/params.json)

4. Evaluate the model using:
```
python evaluate.py --model_dir=experiments/base_model --restore_file=best
```

## Credits
Source code borrowed from [here](https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/nlp).