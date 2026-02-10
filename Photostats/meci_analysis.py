from collections import Counter


def count_meci_classes(self):
    if self.meci_labels is None:
        raise RuntimeError("Dataset was created without MECI labels.")

    return Counter(self.meci_labels)