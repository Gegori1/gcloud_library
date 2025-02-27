import unittest
from unittest.mock import MagicMock
from google.api_core import exceptions
from gcp_library.custom_gcp_instance_monitor import CustomGCPInstanceMonitor

class TestCustomGCPInstanceMonitor(unittest.TestCase):

    def setUp(self):
        # Initialize the monitor with mock credentials and parameters
        self.credentials = MagicMock()
        self.zone = "us-central1-a"
        self.project_id = "test-project-id"
        self.instance_name = "test-instance"
        self.monitor = CustomGCPInstanceMonitor(self.credentials, self.zone, self.project_id)
        self.monitor.compute_client = MagicMock()
        self.monitor.zone_operation_client = MagicMock()

    def test_get_instance_status_success(self):
        # Mock the compute client's get method to return a mock instance with a status
        instance = MagicMock()
        instance.status = "RUNNING"
        self.monitor.compute_client.get.return_value = instance

        # Call the get_instance_status method and assert that it returns the correct status
        status = self.monitor.get_instance_status(self.instance_name)
        self.assertEqual(status, "RUNNING")
        self.monitor.compute_client.get.assert_called_once_with(
            project=self.project_id, zone=self.zone, instance=self.instance_name
        )

    def test_get_instance_status_not_found(self):
        # Mock the compute client's get method to raise a NotFound exception
        self.monitor.compute_client.get.side_effect = exceptions.NotFound("Instance not found")

        # Call the get_instance_status method and assert that it returns None
        status = self.monitor.get_instance_status(self.instance_name)
        self.assertIsNone(status)
        self.monitor.compute_client.get.assert_called_once_with(
            project=self.project_id, zone=self.zone, instance=self.instance_name
        )

    def test_get_instance_status_error(self):
        # Mock the compute client's get method to raise a generic exception
        self.monitor.compute_client.get.side_effect = Exception("Generic error")

        # Call the get_instance_status method and assert that it returns None
        status = self.monitor.get_instance_status(self.instance_name)
        self.assertIsNone(status)
        self.monitor.compute_client.get.assert_called_once_with(
            project=self.project_id, zone=self.zone, instance=self.instance_name
        )

    def test_delete_instance_success(self):
        # Mock the compute client's delete method to return a successful operation
        operation = MagicMock()
        operation.name = "test-operation"
        self.monitor.compute_client.delete.return_value = operation

        # Mock the zone operation client's wait method to return a successful result
        zone_operation = MagicMock()
        zone_operation.error = None
        self.monitor.zone_operation_client.wait.return_value = zone_operation

        # Call the delete_instance method
        self.monitor.delete_instance(self.instance_name)

        # Assert that the delete method was called and the zone operation client's wait method was called
        self.monitor.compute_client.delete.assert_called_once_with(
            project=self.project_id, zone=self.zone, instance=self.instance_name
        )
        self.monitor.zone_operation_client.wait.assert_called_once_with(
            operation=operation.name, project=self.project_id, zone=self.zone
        )

    def test_delete_instance_not_found(self):
        # Mock the compute client's delete method to raise a NotFound exception
        self.monitor.compute_client.delete.side_effect = exceptions.NotFound("Instance not found")

        # Call the delete_instance method
        self.monitor.delete_instance(self.instance_name)

        # Assert that the delete method was called
        self.monitor.compute_client.delete.assert_called_once_with(
            project=self.project_id, zone=self.zone, instance=self.instance_name
        )
        self.monitor.zone_operation_client.wait.assert_not_called()

    def test_delete_instance_error(self):
        # Mock the compute client's delete method to raise a generic exception
        self.monitor.compute_client.delete.side_effect = Exception("Generic error")

        # Call the delete_instance method
        self.monitor.delete_instance(self.instance_name)

        # Assert that the delete method was called
        self.monitor.compute_client.delete.assert_called_once_with(
            project=self.project_id, zone=self.zone, instance=self.instance_name
        )
        self.monitor.zone_operation_client.wait.assert_not_called()

if __name__ == '__main__':
    unittest.main()
