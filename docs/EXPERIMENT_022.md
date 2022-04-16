# Experiment 022 pretraining
Determine whether using triplet loss as pretraining instead of during training results in higher performance

## Notes
- Run name: SN_pretrain_001,SN_pretrain_002
	- n_triplets: 50
	- n_pretraining_epochs: 50
	- note: did not reset g_lr and d_lr

- Run name: SN_pretrain_003
	- n_triplets: 50
	- n_pretraining_epochs: 50
	- note: reset schedulers and optimizers after pretraining

- Run name: SN_pretrain_004
	- n_triplets: 50
	- n_pretraining_epochs: 100
	- note: reset schedulers and optimizers after pretraining

## Overall Summary
- not resetting lr after pretraining results in no real improvement after pretraining
- resetting SN_pretrain_003 does result in slightly better performing model than using the GAN alone





	

