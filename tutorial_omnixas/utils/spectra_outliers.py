# %%
import numpy as np


class OutlierDetector:

    @staticmethod
    def non_outliers(array: np.ndarray, std_factor: float) -> np.ndarray:
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)
        upper_bound = mean + std_factor * std
        lower_bound = mean - std_factor * std
        mask = np.all((array <= upper_bound) & (array >= lower_bound), axis=1)
        return mask

    @staticmethod
    def outliers(array: np.ndarray, std_factor: float) -> np.array:
        return ~OutlierDetector.non_outliers(array, std_factor)


# %%


if __name__ == "__main__":
    import unittest

    class TestOutlierRemover(unittest.TestCase):  # TODO: move to test file

        def test_outliers(self):
            spectra = np.random.rand(100, 5)
            spectra[0][0] += 10
            mask = OutlierDetector().outliers(spectra, 2)
            self.assertTrue(mask[0])
            self.assertFalse(np.any(mask[1:]))

    unittest.main(argv=[""], exit=False)
# %%
