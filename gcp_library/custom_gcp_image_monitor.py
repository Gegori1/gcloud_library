from google.cloud import compute_v1
from google.api_core import exceptions

class CustomGCPImageMonitor:
    def __init__(self, credentials, zone, project_id):
        """
        Initializes the CustomGCPImageMonitor with necessary parameters.

        Args:
            credentials: GCP credentials.
            project_id (str): The project ID for the instance.
        """
        self.credentials = credentials
        self.zone = zone
        self.project_id = project_id
        self.image_client = compute_v1.ImagesClient(credentials=self.credentials)
        self.project_operation_client = compute_v1.GlobalOperationsClient(credentials=self.credentials)

    def create_image_from_instance(self, instance_name, image_name, description):
        """Creates an image from a GCP instance."""
        image_details = compute_v1.Image(
            name=image_name,
            description=description,
            source_disk=f"zones/{self.zone}/disks/{instance_name}"
        )
        
        try:
            image_operation = self.image_client.insert(
            project=self.project_id,
            image_resource=image_details,
            )
            global_operation = self.project_operation_client.wait(operation=image_operation.name, project=self.project_id)

            if global_operation.error:
                print(f"Error creating image: {global_operation.error}")
            else:
                print(f"Image {image_name} created successfully")
        except exceptions.Conflict:
            print(f"Image {image_name} already exists. Continuing with the process.")
        except Exception as e:
            print(f"Error creating image {image_name}: {e}")

    def delete_image(self, image_name):
        """Deletes a GCP image."""
        try:
            delete_operation = self.image_client.delete(project=self.project_id, image=image_name)
            global_operation = self.project_operation_client.wait(operation=delete_operation.name, project=self.project_id)

            if global_operation.error:
                print(f"Error deleting image {image_name}: {global_operation.error}")
            else:
                print(f"Image {image_name} deleted successfully")
        except exceptions.NotFound:
            print(f"Image {image_name} not found. Unable to delete.")
        except Exception as e:
            print(f"Error deleting image {image_name}: {e}")
