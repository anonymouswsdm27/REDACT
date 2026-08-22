"""Unlearning methods, all evaluated against the retrain floor / oracle (§11/§12).

Implemented & wired in experiments/run.py:
- retrain-from-scratch  — oracle / ground truth / residue floor (models.mf.train_mf).
- naive local-delete    — leak strawman, re-fit user only (models.mf.refit_user).
- gradient-ascent       — generic FU baseline / NegGrad (methods.gradient_ascent_unlearn).
- fine-tune unlearning  — warm-start continue on D\\{(A,X)} (methods.finetune_unlearn).

To add (most faithful with the FedAvg loop, P1.1/P1.4): fru.py (log-based rollback),
fedshare.py (snapshot contrastive unlearning), and ours.py (P3 residue suppression)."""
