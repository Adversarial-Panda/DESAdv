# CIFAR-10 and CIFAR-100

DES-Adv uses a 14,000-image selection set and a 1,000-image, jointly
benign-correct evaluation split for each dataset. The manuscript uses six
candidate surrogates, selects four, and evaluates seven unseen targets.
GCViT-Tiny replaces XCiT-Tiny in the CIFAR surrogate pool.

CIFAR checkpoints, the corresponding preprocessing transforms and the exact
sample manifest are required to execute these benchmarks. The ImageNet
entrypoint does not automatically support CIFAR class mapping.

Common C/G/S scoring and fixed-set utilities are in `src/desadv/`.
See `docs/REPRODUCIBILITY.md` for input artifact requirements.
