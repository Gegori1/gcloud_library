from google.cloud import compute_v1
from google.api_core import exceptions

class CustomGCPInstanceLauncher:
    def __init__(self, credentials, image_name, zone, machine_type, image_project, project_id, service_account_email, instance_type="PERSISTENT"):
        """
        Initializes the CustomGCPInstanceLauncher with necessary parameters.

        Args:
            credentials: GCP credentials.
            image_name (str): The name of the image as saved.
            zone (str): The zone where the instance will be created.
            machine_type (str): The machine type for the instance.
            image_project (str): The project ID where the image is located.
            project_id (str): The project ID for the instance.
            service_account_email (str): The service account email.
            instance_type (str): Type of the instance ('PERSISTENT' or 'SPOT'). Defaults to 'PERSISTENT'.
        """
        self.credentials = credentials
        self.image_name = image_name
        self.zone = zone
        self.machine_type = machine_type
        self.image_project = image_project
        self.project_id = project_id
        self.service_account_email = service_account_email
        self.instance_type = instance_type
        self.compute_client = compute_v1.InstancesClient(credentials=self.credentials)
        self.zone_operation_client = compute_v1.ZoneOperationsClient(credentials=self.credentials)

    def __call__(self, startup_script, instance_name):
        """Launches a GCP instance with the given startup script and image name."""
        instance_details = self._create_instance_details(instance_name, startup_script)

        try:
            operation = self.compute_client.insert(project=self.project_id, zone=self.zone, instance_resource=instance_details)
            zone_operation = self.zone_operation_client.wait(operation=operation.name, project=self.project_id, zone=self.zone)

            if zone_operation.error:
                print(f"Error creating instance {instance_name}: {zone_operation.error}")
                return None
            else:
                instance = self.compute_client.get(project=self.project_id, zone=self.zone, instance=instance_name)
                external_ip = instance.network_interfaces[0].access_configs[0].nat_i_p
                print(f"Instance {instance_name} created successfully with external IP: {external_ip}")
                return instance_name
        except exceptions.Conflict:
            print(f"Instance {instance_name} already exists. Continuing with the process.")
            return instance_name
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def _create_instance_details(self, instance_name, startup_script):
        """Creates the instance details object for the GCP instance."""
        machine_type = f"zones/{self.zone}/machineTypes/{self.machine_type}"
        image_path = f"projects/{self.image_project}/global/images/{self.image_name}"

        instance_details = compute_v1.Instance(
            name=instance_name,
            machine_type=machine_type,
            disks=[
                compute_v1.AttachedDisk(
                    auto_delete=True,
                    boot=True,
                    initialize_params=compute_v1.AttachedDiskInitializeParams(
                        source_image=image_path,
                    ),
                    type_="PERSISTENT" if self.instance_type.upper() != "SPOT" else "PERSISTENT",  # Modify as needed
                )
            ],
            network_interfaces=[
                compute_v1.NetworkInterface(
                    network=f"projects/{self.project_id}/global/networks/default",
                    access_configs=[compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT")]
                )
            ],
            metadata=compute_v1.Metadata(
                items=[
                    compute_v1.Items(
                        key="startup-script",
                        value=startup_script
                    )
                ]
            ),
            service_accounts=[
                compute_v1.ServiceAccount(
                    email=self.service_account_email,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            ]
        )

        if self.instance_type.upper() == "SPOT":
            instance_details.scheduling = compute_v1.Scheduling(automatic_restart=False, preemptible=True)

        return instance_details
