# LAB 0

## Tutorial 1 key points:

* overview of MG Nodes
* basic passes (analysis / transform)

## Tutorial 2:

* setup MG - selecting which inpuits of the forward pass function to use. 
* the topology of the graph with attention mask + labels vs without:

with:
![idk](mdassets/mg.bert-all-labels.svg)
without:
![idk](mdassets/mg.bert-no-attention_mask-no-labels.svg)

    * labels: adds nodes to the bottom of the graph that takes the labels and finds the cross entropy loss
    * attention mask: adds placeholder node to get attention mask (without it uses torch.ones), which is used to inform the attention layer.

* then did full supervised finetuneing (on layers that aren't embedding related) acheiveing 0.81 accuracy
* finally did PEFT achierveing 0.83 accuracy

# LAB 1

## Tutorial 3:

* quantization pass: accuracy went 0.83 -> 0.815
* PQT: accuracy 0.815 ->0.839

## Tutorial 4:

* Random pruneing: 0.839 -> 0.75
* After training: 0.75 -> 0.83

## Quantisation / pruning sweeps

* did a sweep of quantization bit-widths:
[quantisation sweep](mdassets/ptq_sweep.png)
* we can see that after 8 bits, accuracy post training pretty much tapers off. Interstingly, huge ammounts of accuracy are lost when quanitsing to 5/7 bits, which can be re-gained almost entirely with some training.

* next we did sweeps of various pruning sparsities using both l1-norm and random pruneing as well as post prune training. 
[pruning sweep](mdassets/prune_sweep.png)
* We can see that L1-pruning is far superior to randomm pruning. This makes sense as L1 pruning removes weight with small magnitured, meaning they likely don't have much impact on the output of the network regardless.
* random pruning appears to be very destructive without post training
* we can infer that roughly 70% of weights are not critical for high accuracy
* interestingly l1-pruneing seems to perform similarly to random pruning and training implying that the lost weights' behaviour are re-learned. 
* random pruning clearly removes some important paths that can't be re-learned.

# LAB 2



# LAB 3

## Question 1

After allowing difrerent layers to have the extra widths, and exposing this as a parameter to optuna, we got the following graph - optuna was able to an excellent set after a few issues and more trials didn't increase the accuracy.

![](mdassets/l3t6q1.png)

This brought the accuracy up to 87.7%

## Question 2

I've extended this to run searches in different precisions - we can see that BlockFP performs best, followed by Minifloat, Binary, and then Integer.

![](mdassets/l3t6q2.png)

Accuracy for Log should be better (above integer) but due to a bug in my code only performs random guesses - unfortunately i don't have enough time to run it all again :(

Again, more iterations didn't help