import unittest
from unittest.mock import patch
import tempfile
import os
from a0.train.alphazero import get_most_recent_model_path

class TestGetMostRecentModelPath(unittest.TestCase):
    def test_no_training_dir(self):
        """Test when config.training_dir is None."""
        with patch('a0.train.alphazero.config.training_dir', None):
            result = get_most_recent_model_path()
            self.assertIsNone(result)

    def test_training_dir_no_models(self):
        """Test when training_dir is set but empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('a0.train.alphazero.config.training_dir', temp_dir):
                result = get_most_recent_model_path()
                self.assertIsNone(result)

    def test_training_dir_with_multiple_models(self):
        """Test with multiple valid model files, should return the one with highest iteration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake model files
            open(os.path.join(temp_dir, 'model_1.pkl'), 'w').close()
            open(os.path.join(temp_dir, 'model_10.pkl'), 'w').close()
            open(os.path.join(temp_dir, 'model_2.pkl'), 'w').close()
            with patch('a0.train.alphazero.config.training_dir', temp_dir):
                result = get_most_recent_model_path()
                self.assertIsNotNone(result)
                path, iteration = result
                self.assertEqual(iteration, 10)
                self.assertEqual(os.path.basename(path), 'model_10.pkl')

    def test_training_dir_with_invalid_models(self):
        """Test with invalid files (wrong name or non-integer iteration), should ignore them."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid files
            open(os.path.join(temp_dir, 'not_model.pkl'), 'w').close()
            open(os.path.join(temp_dir, 'model_abc.pkl'), 'w').close()
            open(os.path.join(temp_dir, 'model_5.pkl'), 'w').close()  # One valid
            with patch('a0.train.alphazero.config.training_dir', temp_dir):
                result = get_most_recent_model_path()
                self.assertIsNotNone(result)
                path, iteration = result
                self.assertEqual(iteration, 5)
                self.assertEqual(os.path.basename(path), 'model_5.pkl')

    def test_training_dir_with_single_model(self):
        """Test with only one valid model file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            open(os.path.join(temp_dir, 'model_3.pkl'), 'w').close()
            with patch('a0.train.alphazero.config.training_dir', temp_dir):
                result = get_most_recent_model_path()
                self.assertIsNotNone(result)
                path, iteration = result
                self.assertEqual(iteration, 3)
                self.assertEqual(os.path.basename(path), 'model_3.pkl')

    def test_training_dir_with_model_zero(self):
        """Test with model_0.pkl, should be valid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            open(os.path.join(temp_dir, 'model_0.pkl'), 'w').close()
            with patch('a0.train.alphazero.config.training_dir', temp_dir):
                result = get_most_recent_model_path()
                self.assertIsNotNone(result)
                path, iteration = result
                self.assertEqual(iteration, 0)
                self.assertEqual(os.path.basename(path), 'model_0.pkl')

if __name__ == '__main__':
    unittest.main()
