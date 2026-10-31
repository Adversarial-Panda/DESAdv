from desadv.web.matching import matches

def test_lexical_overlap_and_exact_rules():
    assert matches('tabby','tabby cat',mode='lexical_overlap')
    assert not matches('tabby','tabby cat',mode='exact')
