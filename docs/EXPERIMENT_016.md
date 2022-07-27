# Experiment 016 - contrastive loss
Implemented triplet loss and negative hard mining.

## Notes
- Run name: SN_triplet_001
- Sweep name: sweep_remote_grid_011
	- search over margin, n_triplets, triplet_loss_weight
	- margin: 0.1
	- n_triplets: 10,20
	- triplet_loss_weight: 0.1,0.2,0.5,1.0
- Sweep name: sweep_remote_grid_012
	- search over triplet_loss_weight
	- margin: 0.1
	- n_triplets: 20
	- triplet_loss_weight: 0.5, 1
- Sweep name: sweep_remote_grid_013
	- margin: 0.2
	- n_triplets: 10,20,50
	- triplet_loss_weight: 0.1,0.2,0.5,1.0
- Sweep name: sweep_remote_grid_014
	- margin: 0.2
	- n_triplets: 100
	- triplet_loss_weight: 0.1,0.2
- Sweep name: sweep_remote_grid_015
	- margin: 0.1,0.2,0.4,0.8
	- n_triplets: 20
	- triplet_loss_weight: 0.1

## Overall Summary
- Performance of triplet loss is linear in the number of triplets.
- sweep_remote_grid_011
	- A number of combinations resulted in no improvement in val_acc or val_iou_score (n_triplets=10, triplet_loss_weight=0.2,0.5,1.0
	- A number of combinations resulted in worse val_acc, val_iou_score (n_triplets=20,triplet_loss_weight=0.1,0.2)
	- Only one combination resulted in improvement (n_triplets=10,triplet_loss_weight=0.1)
- sweep_remote_grid_012
	- both resulted in poor performance
- sweep_remote_grid_013
	- improvement only with loss_weight=0.1,0.2
	- n_triplets=10 did almost as well as n_triplets=50
- sweep_remote_grid_014
	- performance improvement only with loss_weight=0.1
- sweep_remote_grid_015
	- margin had a marginal effect on performance



	

