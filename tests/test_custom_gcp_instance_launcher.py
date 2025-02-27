import unittest
from unittest.mock import MagicMock
from gcp_library.custom_gcp_instance_launcher import CustomGCPInstanceLauncher

class TestCustomGCPInstanceLauncher(unittest.TestCase):

    def setUp(self):
        # Mock the necessary objects and parameters for the tests
        self.credentials = MagicMock()
        self.image_name = "test-image"
        self.zone = "us-central1-a"
        self.machine_type = "n1-standard-1"
        self.image_project = "test-project"
        self.project_id = "test-project-id"
        self.service_account_email = "test@example.com"
        self.instance_type = "PERSISTENT"
        self.launcher = CustomGCPInstanceLauncher(
            self.credentials, self.image_name, self.zone, self.machine_type,
            self.image_project, self.project_id, self.service_account_email,
            self.instance_type
        )
        self.launcher.compute_client = MagicMock()
        self.launcher.zone_operation_client = MagicMock()

    def test_instance_creation_success(self):
        # Mock the compute client's insert method to return a successful operation
        operation = MagicMock()
        operation.name = "test-operation"
        self.launcher.compute_client.insert.return_value = operation

        # Mock the zone operation client's wait method to return a successful result
        zone_operation = MagicMock()
        zone_operation.error = None
        self.launcher.zone_operation_client.wait.return_value = zone_operation

        # Mock the compute client's get method to return an instance with an external IP
        instance = MagicMock()
        instance.network_interfaces = [MagicMock()]
        instance.network_interfaces[0].access_configs = [MagicMock()]
        instance.network_interfaces[0].access_configs[0].nat_i_p = "1.2.3.4"
        self.launcher.compute_client.get.return_value = instance

        # Call the launcher with a test startup script and instance name
        startup_script = "echo Hello, world!"
        instance_name = "test-instance"
        result = self.launcher(startup_script, self.image_name, instance_name)

        # Assert that the instance was created successfully and the instance name is returned
        self.assertEqual(result, instance_name)
        self.launcher.compute_client.insert.assert_called_once()
        self.launcher.zone_operation_client.wait.assert_called_once()
        self.launcher.compute_client.get.assert_called_once()

    def test_instance_creation_failure(self):
        # Mock the compute client's insert method to return a failed operation
        operation = MagicMock()
        operation.name = "test-operation"
        self.launcher.compute_client.insert.return_value = operation

        # Mock the zone operation client's wait method to return an error
        zone_operation = MagicMock()
        zone_operation.error = "Test error"
        self.launcher.zone_operation_client.wait.return_value = zone_operation

        # Call the launcher with a test startup script and instance name
        startup_script = "echo Hello, world!"
        instance_name = "test-instance"
        result = self.launcher(startup_script, self.image_name, instance_name)

        # Assert that the instance creation failed and None is returned
        self.assertIsNone(result)
        self.launcher.compute_client.insert.assert_called_once()
        self.launcher.zone_operation_client.wait.assert_called_once()
        self.launcher.compute_client.get.assert_not_called()

    def test_instance_already_exists(self):
        # Mock the compute client's insert method to raise a Conflict exception
        self.launcher.compute_client.insert.side_effect = Exception("Conflict")

        # Call the launcher with a test startup script and instance name
        startup_script = "echo Hello, world!"
        instance_name = "test-instance"
        result = self.launcher(startup_script, self.image_name, instance_name)

        # Assert that the instance already exists and the instance name is returned
        self.assertEqual(result, instance_name)
        self.launcher.compute_client.insert.assert_called_once()
        self.launcher.zone_operation_client.wait.assert_not_called()
        self.launcher.compute_client.get.assert_not_called()

if __name__ == '__main__':
    unittest.main()
