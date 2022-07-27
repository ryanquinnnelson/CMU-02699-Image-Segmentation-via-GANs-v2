# Experiment 017 - faster contrastive loss
Vectorize accessing triplet values and calculating triplet loss. Simplified triplet loss eulcidean distance calculation because this is 1d data

## Notes
- Run name: SN_triplet_002
- Sweep name: sweep_remote_grid_016
	- margin: 0.1
	- n_triplets: 20
	- triplet_loss_weight: 0.1
- Sweep name: sweep_remote_grid_017
	- margin: 0.1
	- n_triplets: 50
	- triplet_loss_weight: 0.1
- Sweep name: sweep_remote_grid_017 (2)
	- margin: 0.1, 0.2, 0.4
	- n_triplets: 50
	- triplet_loss_weight: 0.1, 0.2
- Sweep name: sweep_remote_grid_018
	- margin: 0.1
	- n_triplets: 50
	- triplet_loss_weight: 0.05

## Overall Summary
- Performance of triplet loss is still linear in the number of triplets, with only a slight improvement.
- sweep_remote_grid_016
	- There is huge performance variance in different runs of the same parameters for n_triplets=20
- sweep_remote_grid_017
	- val_acc is going down
	- loss is increasing after these changes, rather than decreasing, but accuracy and iou_score aren't hugely different
- sweep_remote_grid_017 (2)
	- all options results in mediocre performance
	- margin=0.1, triplet_loss_weight=0.2 result in best performance of this group
- sweep_remote_grid_018
	- tried reducing the importance weight of triplet_loss in overall loss calculations. Reduction did not result in improvement, but didn't hurt.



	

