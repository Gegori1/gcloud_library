# GCP Cloud Library

This library provides a set of tools for managing Google Cloud Platform (GCP) resources, including launching and monitoring instances.

## Installation

To install the library, use pip:

```bash
pip install git+https://github.com/Gegori1/gcloud_library
```

or

```bash
pip install -e git+https://github.com/Gegori1/gcloud_library
```

## Usage

### Load Variables

The `load_variables` module allows you to load configuration variables from a `config.yml` file.

```python
from gcp_library import load_variables

config = load_variables.get_vars()
print(config)
```

### Custom GCP Instance Launcher

The `custom_gcp_instance_launcher` module allows you to launch GCP instances with a custom startup script and image.

```python
from gcp_library.custom_gcp_instance_launcher import CustomGCPInstanceLauncher
from google.oauth2 import service_account

# Load credentials from a service account key file
credentials = service_account.Credentials.from_service_account_file('path/to/your/service_account_key.json')

launcher = CustomGCPInstanceLauncher(
    credentials=credentials,
    image_name="your-image-name",
    zone="your-zone",
    machine_type="your-machine-type",
    image_project="your-image-project",
    project_id="your-project-id",
    service_account_email="your-service-account-email",
    instance_type="PERSISTENT"  # or "SPOT"
)

instance_name = "your-instance-name"
startup_script = "echo Hello, world!"
instance_name = launcher(startup_script, "image-name", instance_name)

if instance_name:
    print(f"Instance {instance_name} launched successfully.")
else:
    print("Instance launch failed.")
```

### Custom GCP Instance Monitor

The `custom_gcp_instance_monitor` module allows you to monitor and delete GCP instances.

```python
from gcp_library.custom_gcp_instance_monitor import CustomGCPInstanceMonitor
from google.oauth2 import service_account

# Load credentials from a service account key file
credentials = service_account.Credentials.from_service_account_file('path/to/your/service_account_key.json')


monitor = CustomGCPInstanceMonitor(
    credentials=credentials,
    zone="your-zone",
    project_id="your-project-id"
)

instance_name = "your-instance-name"
status = monitor.get_instance_status(instance_name)

if status:
    print(f"Instance {instance_name} status: {status}")
    monitor.delete_instance(instance_name)
else:
    print(f"Instance {instance_name} not found.")
```

## Configuration

The library uses a `config.yml` file to store configuration variables. You can place this file in any parent directory of the script.  The library will traverse up the directory tree to find it.

Example `config.yml`:

```yaml
project_id: "your-project-id"
zone: "your-zone"
machine_type: "your-machine-type"
image_project: "your-image-project"
service_account_email: "your-service-account-email"
image_name: "your-image-name"
```

## Tests

To run the tests, navigate to the `tests` directory and run:

```bash
python -m unittest discover -v
```

## License

This project uses a Proprietary License.