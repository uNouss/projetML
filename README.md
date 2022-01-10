---

---

# projet ML partie 2

## NNI

## Définition de l'espace de recherche
```json
{
    "optimizer":{"_type":"choice","_value":["adam", "SGD"]},
    "learning_rate":{"_type":"choice","_value":[0.0001, 0.001, 0.002, 0.005, 0.01]},
    "nb_nodes": {"_type": "choice", "_value": [128, 256]},
    "metrics": {"_type": "choice", "_value": ["accuracy", "auc"]}
}
```

## Fichier de configuration
```yml
authorName: default
experimentName: example_mnist-keras
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote, pai
trainingServicePlatform: local
searchSpacePath: search_space.json
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  command: python3 mnist-keras.py
  codeDir: .
  gpuNum: 0
```

## Code python
```python
```