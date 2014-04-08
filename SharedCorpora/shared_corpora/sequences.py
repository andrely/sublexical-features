from collections import Sequence, Iterable


class FilteredSequence(Sequence):
    def __init__(self, base_sequence, included_indices):
        self.base_sequence = base_sequence
        self.included_indices = included_indices

    def __len__(self):
        return len(self.included_indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._get_index(i) for i in xrange(*index.indices(len(self)))]
        elif isinstance(index, Iterable):
            return [self._get_index(i) for i in index]
        elif isinstance(index, int):
            return self._get_index(index)
        else:
            raise TypeError

    def _get_index(self, index):
        return self.base_sequence[self.included_indices[index]]
