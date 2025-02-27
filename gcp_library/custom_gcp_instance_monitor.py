from google.cloud import compute_v1
from google.api_core import exceptions
import time

class CustomGCPInstanceMonitor:
    def __init__(self, credentials, zone, project_id):
        """
        Initializes the CustomGCPInstanceMonitor with necessary parameters.

        Args:
            credentials: GCP credentials.
            zone (str): The zone where the instance is located.
            project_id (str): The project ID for the instance.
        """
        self.credentials = credentials
        self.zone = zone
        self.project_id = project_id
        self.compute_client = compute_v1.InstancesClient(credentials=self.credentials)
        self.zone_operation_client = compute_v1.ZoneOperationsClient(credentials=self.credentials)

    def get_instance_status(self, instance_name):
        """Retrieves the status of a GCP instance."""
        try:
            instance = self.compute_client.get(project=self.project_id, zone=self.zone, instance=instance_name)
            return instance.status
        except exceptions.NotFound:
            print(f"Instance {instance_name} not found.")
            return None
        except Exception as e:
            print(f"Error getting instance status for {instance_name}: {e}")
            return None

    def delete_instance(self, instance_name):
        """Deletes a GCP instance."""
        try:
            delete_operation = self.compute_client.delete(project=self.project_id, zone=self.zone, instance=instance_name)
            zone_operation = self.zone_operation_client.wait(operation=delete_operation.name, project=self.project_id, zone=self.zone)

            if zone_operation.error:
                print(f"Error deleting instance {instance_name}: {zone_operation.error}")
            else:
                print(f"Instance {instance_name} deleted successfully")
        except exceptions.NotFound:
            print(f"Instance {instance_name} not found. Unable to delete.")
        except Exception as e:
            print(f"Error deleting instance {instance_name}: {e}")

    def monitor_cpu_usage(self, instance_name, threshold=10.0, check_interval=60):
        """Monitors CPU usage and stops the instance if usage is below threshold."""
        print(f"Waiting for the metric to start on instance {instance_name}")
        time.sleep(60 * 4)  # Wait for metrics to start
        retry = 0
        while True:
            # max_retries times if CPU usage data is not immediately available.
            try:
                # Use gcloud command to get CPU utilization
                # cpu_usage = get_cpu_utilization(self.project_id, instance_name) # I need to install google-cloud-monitoring
                cpu_usage = 5 # temporal value
                if cpu_usage:
                    print(f"CPU Usage for {instance_name}: {cpu_usage:.2f}%")

                    if cpu_usage < threshold:
                        print(f"CPU usage for {instance_name} is below {threshold}%. Stopping instance...")
                        self.delete_instance(instance_name)
                        break
                else:
                    if retry < 3:
                        retry += 1
                        print(f"Could not retrieve CPU usage for {instance_name}. retrying...")
                        time.sleep(60)
                        continue
                    else:
                        print(f"Could not retrieve CPU for {instance_name} for the second time. Stopping metric.")
                        break

                time.sleep(check_interval)

            except Exception as e:
                print(f"Error monitoring CPU usage for {instance_name}: {e}")
                break
