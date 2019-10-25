## Visualization
### Chemical Compound
#### *"mol"*
- Format: RDKit Mol object.
#### *"prediction_score"*
- Format: A float value of the prediction score for use in the visualization process.
#### *"target_label"*
- Format: A int value of the label to predict
#### *"check_score"*
- Format: A float value IG(1)-IG(0), an approximate value of a total value of integrated gradients
#### *"sum_of_IG"*
- Format: A total value of integrated gradients.
#### *"features"*
- Format: A matrix of features.
#### *"features_IG"*
- Format: A matrix of integrated gradients of features.
#### *"adjs"*
- Format: A adjacency matrix.
#### *"adjs_IG"*
- Format: A matrix of integrated gradients of a adjacency matrix.

### Note
where the lengths of "adj" ("dense_adj"), "feature", "label",and "node" need to be equal.

