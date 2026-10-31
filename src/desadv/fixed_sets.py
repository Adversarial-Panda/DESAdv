"""Exact 15 unique 4-of-6 fixed-subset names/order from manuscript Table 8."""
import itertools
MODEL_ORDER=('ResNet-18','Inception-V3','DeiT-Tiny','ViT-Tiny','EfficientNet-B0','XCiT-Tiny')
FIXED_SETS={
 1:(0,1,2,3), 2:(1,2,3,4), 3:(1,2,5,4),
 4:(0,1,3,5), 5:(0,1,3,4), 6:(0,1,5,4),
 7:(0,2,3,5), 8:(0,2,3,4), 9:(0,2,5,4),
 10:(0,3,5,4), 11:(1,2,3,5), 12:(0,1,2,5),
 13:(0,1,2,4), 14:(1,3,5,4), 15:(2,3,5,4)
}
assert len({frozenset(x) for x in FIXED_SETS.values()})==15
assert {frozenset(x) for x in FIXED_SETS.values()} == {frozenset(x) for x in itertools.combinations(range(6),4)}
