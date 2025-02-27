import unittest
from gcp_library import load_variables
import os

class TestLoadVariables(unittest.TestCase):
    
    def test_get_config_file_path_exists(self):
        """Test that the config file path exists."""
        config_path = load_variables.get_config_file_path()
        self.assertTrue(os.path.exists(config_path))

    def test_get_vars_returns_dict(self):
        """Test that get_vars returns a dictionary."""
        vars = load_variables.get_vars()
        self.assertIsInstance(vars, dict)

if __name__ == '__main__':
    unittest.main()
