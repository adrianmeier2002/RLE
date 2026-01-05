# RLE
## Einleitung
Ziel dieses Repositories ist es, verschiedene Reinforcement Learning Algorithmen zu implementieren und zu evaluieren. Dabei wird versucht das Spiel "Space Invaders" aus der OpenAI Gym Umgebung zu meistern bzw. einen möglichst hohen Score zu erzielen.

In diesem Repo wird ein Random Agent als vergleichsbasis implementiert sowie ein Deep Q-Network (DQN) Agent als grundlage, der dann vier weitere Verbesserungen erfährt:
- Double DQN
- Dueling DQN
- Prioritized Experience Replay
- Noisy Nets

In diesem README werden die Schritte zum Einrichten der Umgebung, Trainieren und Evaluieren der Modelle beschrieben.
Meine persönlichen Erfahrungen und eine Analyse der Ergebnisse sind im Bericht dokumentiert.

## Einrichten der Umgebung
Um die Umgebung einzurichten sollte dieses README.md gelesen werden. Danach sollten alle benötigten Bibliotheken aus dem requirements.txt installiert werden. Dies kann mit dem folgenden Befehl in der Kommandozeile gemacht werden:

```bash
pip install -r requirements.txt
```

Zudem hatte ich selbst noch zwei probleme, bevor ich die Programme ausführen konnte:
1. Bei der Installation von PyTorch gab es bei mir Probleme mit der CUDA Version. Ich konnte dies durch folgenden Befehl lösen:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2. Ich hatte zu beginn Probleme mit 'gymnasium' und 'ale-py'. Diese konnte ich durch eine manuelle Installation lösen:

```bash
pip install "gymnasium[other]"
```

## Trainieren der Modelle
Die Modelle können einzeln oder alle zusammen trainiert werden. Um ein einzelnes Modell zu trainieren, kann das entsprechende Skript im `dqn`-Ordner ausgeführt werden. Zum Beispiel, um den DQN-Agenten zu trainieren, kann der folgende Befehl verwendet werden:

```bash
python dqn.dqn_initial
```
Um alle Modelle zu trainieren und sicher zu gehen, dass alle die gleichen Trainingsbedingungen haben, kann das `train_all.py` Skript im Hauptverzeichnis ausgeführt werden:

```bash
python train_all
```

Die Modelle werden im `models`-Ordner gespeichert. Um den Trainingsfortschritt zu überwachen, können die TensorBoard Logs im `runs`-Ordner eingesehen werden. Dies kann mit dem folgenden Befehl gestartet werden:

```bash
tensorboard --logdir runs
```

WICHTIG: Die Programme benötigen viel Rechenleistung und Zeit. Die Trainingszeit wird stark von der Hardware abhängen. Es wird empfohlen und in den Skripten standardmässig versucht, eine GPU zu verwenden, wenn eine verfügbar ist.

## Evaluation der Modelle
Um die trainierten Modelle zu evaluieren, kann das `eval_utils.py` Skript im `utils`-Ordner verwendet werden. Es werden dort alle Modelle evaluiert und die Ergebnisse in einem `results`-Ordner gespeichert. 

```bash
python utils.eval_utils
```
