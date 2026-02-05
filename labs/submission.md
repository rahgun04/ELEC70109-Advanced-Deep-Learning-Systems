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

## Quantisation sweeps