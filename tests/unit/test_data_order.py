from rhs_rcmtr.utils.data_order import DeterministicEpochSampler


def test_sample_order_hash_is_reproducible_per_epoch() -> None:
    sample_ids = ["a", "b", "c", "d", "e"]
    first = DeterministicEpochSampler(sample_ids, seed=0)
    second = DeterministicEpochSampler(sample_ids, seed=0)
    hashes_first = []
    hashes_second = []
    for epoch in range(6):
        first.set_epoch(epoch)
        second.set_epoch(epoch)
        hashes_first.append(first.order_hash())
        hashes_second.append(second.order_hash())
        assert first.ordered_sample_ids() == second.ordered_sample_ids()
    assert hashes_first == hashes_second
    assert len(set(hashes_first)) > 1


def test_sample_order_seed_changes_order() -> None:
    sample_ids = ["a", "b", "c", "d", "e"]
    first = DeterministicEpochSampler(sample_ids, seed=0)
    second = DeterministicEpochSampler(sample_ids, seed=1)
    assert first.order_hash() != second.order_hash()
