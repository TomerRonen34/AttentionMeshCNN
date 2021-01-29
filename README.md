## Presentation
**_docs/MeshCNN Attention.pptx_**
https://github.com/TomerRonen34/AttentionMeshCNN/blob/master/docs/MeshCNN%20Attention.pptx

## Our contributions
* Global self attention mechanism for meshes
* MeshPool prioritizing with attention values 
* Local self attention mechanism for meshes
    * Efficient Cython implementation of shortest paths algorithm
* Positional encoding for meshes

## Usage Instructions
For training our best network run the script **_scripts/cubes/get_data.sh_** and then **_scripts/cubes/rpr_global.sh_**

The flags which were added by us:

--attn_n_heads, type=int, default=4 \
number of heads for Multi Headed Attention

--prioritize_with_attention \
if given, the priority queue for the pool operation is calculated from the attention softmax.
default priority is l2 norm.

--attn_dropout, type=float, default=0.1 \
dropout fraction for attention layer

--attn_max_dist, type=int, default=None \
max distance for local attention. default (None) is global attention

--attn_use_values_as_is \
if given, attention layers learn a weighting of the input features.
default behavior is learning a weighting of a linear transformation
of the input features.

--double_attention \
if given, the edge priorities are calculated using the results of applying the attention layer to the
results of itself. default behavior is calculating the priorities from the results of applying the
attention to the convolutional features.
NOTE: attn_use_values_as_is must be True if you use this option, since the attention layer works on its own outputs.

--attn_use_positional_encoding \
use relative positional encodings to add positional meaning to attention.
relative position is determined by the number "hops" it takes to reach one edge from another,
where hops are only allowed through convolutional neighbors (edges that share the same triangle).
mathematically, this is shortest path in a graph where every edge is a node and adjacency
is determined in the same way as convolutional neighborhood.

--attn_max_relative_position, type=int, default=6 \
the maximal relative position for positional encoding. edges further aways than max_pos
are treated as if their position is max_pos. an 5-distance-neighborhood of an edge contains
about 60 edges (see doc/neighbors_vs_local.png or .csv)
