# Experiment 019 - checking variance in triplet loss
Trying to discern whether there is significant variance depending on which triplets are selected at random

## Notes
- Run name: SN_triplet_005
- Sweep name: sweep_remote_grid_021,sweep_remote_grid_022
	- margin: 0.2
	- n_triplets: 10
	- triplet_loss_weight: 0.1


## Overall Summary
- generally, the models behave the same way regarding triplet_loss
- wide range of iou_score (0.18 - 0.60)
- wide range of val_acc (0.49 - 0.71)





	

