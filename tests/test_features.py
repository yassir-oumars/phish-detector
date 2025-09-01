from phishdetector.features import url_lexical_features

def test_basic_features():
    feats = url_lexical_features("http://example.com/login/index.php?id=1")
    assert feats["url_len"] > 0
    assert feats["num_params"] >= 1
