"""MemoryStore CRD handler."""

import kopf
import kubernetes.client
from kubernetes.client.rest import ApiException
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

API_GROUP = "memory.artificialmemory.dev"
API_VERSION = "v1"
STORE_PLURAL = "memorystores"


def get_k8s_apps_v1():
    return kubernetes.client.AppsV1Api()


def get_k8s_core_v1():
    return kubernetes.client.CoreV1Api()


def get_k8s_custom_objects():
    return kubernetes.client.CustomObjectsApi()


def create_condition(condition_type: str, status: str, reason: str, message: str) -> Dict[str, Any]:
    return {
        "type": condition_type,
        "status": status,
        "reason": reason,
        "message": message,
        "lastTransitionTime": datetime.utcnow().isoformat() + "Z",
    }


def update_status(custom_api, namespace: str, name: str, status: Dict[str, Any]):
    try:
        custom_api.patch_namespaced_custom_object_status(
            group=API_GROUP,
            version=API_VERSION,
            namespace=namespace,
            plural=STORE_PLURAL,
            name=name,
            body={"status": status},
        )
    except ApiException as e:
        logger.error(f"Failed to update status for {name}: {e}")


@kopf.on.create(API_GROUP, API_VERSION, STORE_PLURAL)
@kopf.on.update(API_GROUP, API_VERSION, STORE_PLURAL)
def reconcile_store(spec: Dict[str, Any], name: str, namespace: str, status: Dict[str, Any], logger: logging.Logger, **_) -> Dict[str, Any]:
    logger.info(f"Reconciling MemoryStore {namespace}/{name}")

    custom_api = get_k8s_custom_objects()
    apps_api = get_k8s_apps_v1()
    core_api = get_k8s_core_v1()

    if not status:
        status = {
            "phase": "Pending",
            "conditions": [],
            "connectionInfo": {},
        }

    backend = spec.get("backend", "postgres")

    try:
        if backend == "postgres":
            # For PostgreSQL, we typically use an external managed service
            # or deploy via CloudNativePG/CrunchyData operator
            # Here we just verify connectivity or create a secret
            store_ready = reconcile_postgres_store(core_api, custom_api, namespace, name, spec, logger)
            status["connectionInfo"] = {
                "host": spec.get("postgres", {}).get("host", "postgres"),
                "port": spec.get("postgres", {}).get("port", 5432),
                "database": spec.get("postgres", {}).get("database", "artificial_memory"),
            }
        else:
            # SQLite - create PVC and deployment
            store_ready = reconcile_sqlite_store(apps_api, core_api, namespace, name, spec, logger)

        if store_ready:
            status["phase"] = "Ready"
            status["conditions"] = [
                create_condition("Ready", "True", "StoreReady", "Memory store is ready"),
            ]
        else:
            status["phase"] = "Creating"
            status["conditions"] = [
                create_condition("Ready", "False", "StoreCreating", "Memory store is being created"),
            ]

        update_status(custom_api, namespace, name, status)
        return {"status": status}

    except Exception as e:
        logger.error(f"Failed to reconcile store {namespace}/{name}: {e}")
        status["phase"] = "Failed"
        status["conditions"] = [create_condition("Ready", "False", "ReconcileError", str(e))]
        update_status(custom_api, namespace, name, status)
        raise


def reconcile_postgres_store(core_api, custom_api, namespace: str, name: str, spec: Dict[str, Any], logger: logging.Logger) -> bool:
    """Reconcile PostgreSQL store (typically external)."""
    pg_spec = spec.get("postgres", {})
    secret_ref = pg_spec.get("secretRef")

    if secret_ref:
        try:
            secret = core_api.read_namespaced_secret(name=secret_ref, namespace=namespace)
            logger.info(f"Found PostgreSQL secret {secret_ref}")
            return True
        except ApiException as e:
            if e.status == 404:
                logger.warning(f"PostgreSQL secret {secret_ref} not found")
                return False
            raise
    else:
        # Check if we can connect to default postgres
        host = pg_spec.get("host", "postgres")
        logger.info(f"Using PostgreSQL at {host}")
        return True


def reconcile_sqlite_store(apps_api, core_api, namespace: str, name: str, spec: Dict[str, Any], logger: logging.Logger) -> bool:
    """Reconcile SQLite store with PVC."""
    sqlite_spec = spec.get("sqlite", {})
    path = sqlite_spec.get("path", "/data/memory.db")

    # Create PVC
    pvc_name = f"{name}-data"
    pvc_body = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": pvc_name, "namespace": namespace},
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": "10Gi"}},
        },
    }

    try:
        core_api.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc_body)
        logger.info(f"Created PVC {pvc_name}")
    except ApiException as e:
        if e.status != 409:
            raise

    return True


@kopf.on.delete(API_GROUP, API_VERSION, STORE_PLURAL)
def delete_store(name: str, namespace: str, logger: logging.Logger, **_):
    logger.info(f"Deleting MemoryStore {namespace}/{name}")
    return {"message": f"Store {name} deleted"}