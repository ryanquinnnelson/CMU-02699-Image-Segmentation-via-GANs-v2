# Experiment 020 check if gan or triplet loss improves performance
Trying to discern whether adding the gan and/or triplet loss actually improves performance

## Notes
- Run name: SN_triplet_006
- Sweep name: sweep_remote_grid_023
	- margin: 0.2
	- n_triplets: 10
	- triplet_loss_weight: 0.1


## Overall Summary
- model performs worse when triplet_loss is added
- model performance doesn't change when SN is used by itself or when EN (i.e. full gan) is used




	

