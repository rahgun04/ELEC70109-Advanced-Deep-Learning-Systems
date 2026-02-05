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

* then did full super
