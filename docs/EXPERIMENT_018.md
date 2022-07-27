# Experiment 018 - checking n_triplet_changes
Trying to discern whether n_triplets impacts performance

## Notes
- Run name: SN_triplet_003
- Sweep name: sweep_remote_grid_019
	- margin: 0.1
	- n_triplets: 20
	- triplet_loss_weight: 0.1

- Run name: SN_triplet_004
- Sweep name: sweep_remote_grid_020
	- margin: 0.1
	- n_triplets: 5,10,20,40,80
	- triplet_loss_weight: 0.1



## Overall Summary
- found out there was a bug in triplet loss calculation that resulted in an incorrect loss calculation. will rerun with the fix
- rerun shows that iou_score goes down with additional triplets, but effect is relatively minor (0.53 to 0.48 spread)
- rerun shows that accuracy generally goes down with additional triplets 




	

