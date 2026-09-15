"""ArtificialMemoryCluster CRD handler."""

import kopf
import kubernetes.client
from kubernetes.client.rest import ApiException
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

API_GROUP = "memory.artificialmemory.dev"
API_VERSION = "v1"
CLUSTER_PLURAL = "artificialmemoryclusters"


def get_k8s_apps_v1():
    """Get Kubernetes AppsV1Api client."""
    return kubernetes.client.AppsV1Api()


def get_k8s_core_v1():
    """Get Kubernetes CoreV1Api client."""
    return kubernetes.client.CoreV1Api()


def get_k8s_custom_objects():
    """Get Kubernetes CustomObjectsApi client."""
    return kubernetes.client.CustomObjectsApi()


def create_condition(condition_type: str, status: str, reason: str, message: str) -> Dict[str, Any]:
    """Create a condition dict."""
    return {
        "type": condition_type,
        "status": status,
        "reason": reason,
        "message": message,
        "lastTransitionTime": datetime.utcnow().isoformat() + "Z",
    }


def update_status(
    custom_api: kubernetes.client.CustomObjectsApi,
    namespace: str,
    name: str,
    status: Dict[str, Any],
) -> None:
    """Update CRD status."""
    try:
        custom_api.patch_namespaced_custom_object_status(
            group=API_GROUP,
            version=API_VERSION,
            namespace=namespace,
            plural=CLUSTER_PLURAL,
            name=name,
            body={"status": status},
        )
    except ApiException as e:
        logger.error(f"Failed to update status for {name}: {e}")


@kopf.on.create(API_GROUP, API_VERSION, CLUSTER_PLURAL)
@kopf.on.update(API_GROUP, API_VERSION, CLUSTER_PLURAL)
def reconcile_cluster(
    spec: Dict[str, Any],
    name: str,
    namespace: str,
    status: Dict[str, Any],
    logger: logging.Logger,
    **_,
) -> Dict[str, Any]:
    """Reconcile ArtificialMemoryCluster."""
    logger.info(f"Reconciling ArtificialMemoryCluster {namespace}/{name}")

    custom_api = get_k8s_custom_objects()
    apps_api = get_k8s_apps_v1()
    core_api = get_k8s_core_v1()

    # Initialize status if needed
    if not status:
        status = {
            "phase": "Creating",
            "conditions": [],
            "replicas": 0,
            "recallWorkersReady": 0,
            "consolidationWorkersReady": 0,
            "compilerWorkersReady": 0,
            "storageReady": False,
            "vectorIndexReady": False,
        }

    try:
        # Get desired replicas
        desired_replicas = spec.get("replicas", 3)

        # Reconcile Storage (MemoryStore)
        storage_ready = reconcile_storage(custom_api, namespace, name, spec, logger)
        status["storageReady"] = storage_ready

        # Reconcile VectorIndex
        vector_ready = reconcile_vector_index(custom_api, namespace, name, spec, logger)
        status["vectorIndexReady"] = vector_ready

        # Reconcile Runtime Deployments
        runtime_status = reconcile_runtime(
            apps_api, core_api, custom_api, namespace, name, spec, logger
        )
        status.update(runtime_status)

        # Update phase based on conditions
        all_ready = (
            storage_ready
            and vector_ready
            and runtime_status.get("replicas", 0) >= desired_replicas
            and runtime_status.get("recallWorkersReady", 0) >= spec.get("runtime", {}).get("recallWorkers", 4)
            and runtime_status.get("consolidationWorkersReady", 0) >= spec.get("runtime", {}).get("consolidationWorkers", 2)
        )

        if all_ready:
            status["phase"] = "Running"
        elif status.get("phase") == "Creating":
            status["phase"] = "Running"
        else:
            status["phase"] = "Running"

        # Update conditions
        status["conditions"] = [
            create_condition("Ready", "True" if all_ready else "False", "Reconciled", "Cluster reconciled"),
            create_condition("StorageReady", "True" if storage_ready else "False", "StorageReconciled", "Storage reconciled"),
            create_condition("VectorIndexReady", "True" if vector_ready else "False", "VectorIndexReconciled", "Vector index reconciled"),
        ]
        status["lastUpdateTime"] = datetime.utcnow().isoformat() + "Z"

        # Update status
        update_status(custom_api, namespace, name, status)

        logger.info(f"Reconciled ArtificialMemoryCluster {namespace}/{name}: phase={status['phase']}")

        return {"status": status}

    except Exception as e:
        logger.error(f"Failed to reconcile cluster {namespace}/{name}: {e}")
        status["phase"] = "Failed"
        status["conditions"] = [
            create_condition("Ready", "False", "ReconcileError", str(e)),
        ]
        update_status(custom_api, namespace, name, status)
        raise


def reconcile_storage(
    custom_api: kubernetes.client.CustomObjectsApi,
    namespace: str,
    cluster_name: str,
    spec: Dict[str, Any],
    logger: logging.Logger,
) -> bool:
    """Reconcile MemoryStore for the cluster."""
    storage_spec = spec.get("storage", {})
    backend = storage_spec.get("backend", "postgres")

    store_name = f"{cluster_name}-store"

    try:
        # Check if MemoryStore exists
        custom_api.get_namespaced_custom_object(
            group=API_GROUP,
            version=API_VERSION,
            namespace=namespace,
            plural="memorystores",
            name=store_name,
        )
        logger.info(f"MemoryStore {store_name} already exists")
        return True
    except ApiException as e:
        if e.status == 404:
            # Create MemoryStore
            logger.info(f"Creating MemoryStore {store_name}")
            store_body = {
                "apiVersion": f"{API_GROUP}/{API_VERSION}",
                "kind": "MemoryStore",
                "metadata": {
                    "name": store_name,
                    "namespace": namespace,
                    "ownerReferences": [{
                        "apiVersion": f"{API_GROUP}/{API_VERSION}",
                        "kind": "ArtificialMemoryCluster",
                        "name": cluster_name,
                        "uid": "",  # Will be filled by owner reference controller
                        "controller": True,
                    }],
                },
                "spec": {
                    "backend": backend,
                    "postgres": storage_spec.get("postgres", {}),
                    "resources": storage_spec.get("resources", {}),
                    "backup": storage_spec.get("backup", {}),
                },
            }
            custom_api.create_namespaced_custom_object(
                group=API_GROUP,
                version=API_VERSION,
                namespace=namespace,
                plural="memorystores",
                body=store_body,
            )
            return False  # Not ready yet
        else:
            raise


def reconcile_vector_index(
    custom_api: kubernetes.client.CustomObjectsApi,
    namespace: str,
    cluster_name: str,
    spec: Dict[str, Any],
    logger: logging.Logger,
) -> bool:
    """Reconcile VectorIndex for the cluster."""
    index_name = f"{cluster_name}-vector-index"

    try:
        custom_api.get_namespaced_custom_object(
            group=API_GROUP,
            version=API_VERSION,
            namespace=namespace,
            plural="vectorindexes",
            name=index_name,
        )
        logger.info(f"VectorIndex {index_name} already exists")
        return True
    except ApiException as e:
        if e.status == 404:
            logger.info(f"Creating VectorIndex {index_name}")
            index_body = {
                "apiVersion": f"{API_GROUP}/{API_VERSION}",
                "kind": "VectorIndex",
                "metadata": {
                    "name": index_name,
                    "namespace": namespace,
                    "ownerReferences": [{
                        "apiVersion": f"{API_GROUP}/{API_VERSION}",
                        "kind": "ArtificialMemoryCluster",
                        "name": cluster_name,
                        "uid": "",
                        "controller": True,
                    }],
                },
                "spec": {
                    "clusterRef": cluster_name,
                    "dimension": 384,
                    "indexType": "hnsw",
                    "hnsw": {"m": 16, "efConstruction": 64},
                    "storage": {"backend": "pgvector"},
                    "sync": {"enabled": True, "intervalSeconds": 60},
                },
            }
            custom_api.create_namespaced_custom_object(
                group=API_GROUP,
                version=API_VERSION,
                namespace=namespace,
                plural="vectorindexes",
                body=index_body,
            )
            return False
        else:
            raise


def reconcile_runtime(
    apps_api: kubernetes.client.AppsV1Api,
    core_api: kubernetes.client.CoreV1Api,
    custom_api: kubernetes.client.CustomObjectsApi,
    namespace: str,
    cluster_name: str,
    spec: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Reconcile runtime deployments (recall, consolidation, compiler workers)."""
    runtime_spec = spec.get("runtime", {})
    image_spec = spec.get("image", {})
    image = f"{image_spec.get('repository', 'artificial-memory')}:{image_spec.get('tag', 'latest')}"

    results = {
        "replicas": 0,
        "recallWorkersReady": 0,
        "consolidationWorkersReady": 0,
        "compilerWorkersReady": 0,
    }

    # Reconcile Recall Workers
    recall_replicas = runtime_spec.get("recallWorkers", 4)
    recall_ready = reconcile_deployment(
        apps_api,
        namespace,
        f"{cluster_name}-recall",
        recall_replicas,
        image,
        "recall",
        spec,
        logger,
    )
    results["recallWorkersReady"] = recall_ready

    # Reconcile Consolidation Workers
    consolidation_replicas = runtime_spec.get("consolidationWorkers", 2)
    consolidation_ready = reconcile_deployment(
        apps_api,
        namespace,
        f"{cluster_name}-consolidation",
        consolidation_replicas,
        image,
        "consolidation",
        spec,
        logger,
    )
    results["consolidationWorkersReady"] = consolidation_ready

    # Reconcile Compiler Workers
    compiler_replicas = runtime_spec.get("compilerWorkers", 2)
    compiler_ready = reconcile_deployment(
        apps_api,
        namespace,
        f"{cluster_name}-compiler",
        compiler_replicas,
        image,
        "compiler",
        spec,
        logger,
    )
    results["compilerWorkersReady"] = compiler_ready

    results["replicas"] = recall_ready + consolidation_ready + compiler_ready

    return results


def reconcile_deployment(
    apps_api: kubernetes.client.AppsV1Api,
    namespace: str,
    name: str,
    replicas: int,
    image: str,
    worker_type: str,
    cluster_spec: Dict[str, Any],
    logger: logging.Logger,
) -> int:
    """Reconcile a single deployment."""
    runtime_spec = cluster_spec.get("runtime", {})
    resources = runtime_spec.get("resources", {})

    deployment_body = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "artificial-memory",
                "app.kubernetes.io/component": worker_type,
                "app.kubernetes.io/part-of": "artificial-memory-cluster",
            },
        },
        "spec": {
            "replicas": replicas,
            "selector": {
                "matchLabels": {
                    "app.kubernetes.io/name": "artificial-memory",
                    "app.kubernetes.io/component": worker_type,
                },
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/name": "artificial-memory",
                        "app.kubernetes.io/component": worker_type,
                    },
                },
                "spec": {
                    "containers": [{
                        "name": "worker",
                        "image": image,
                        "command": ["python", "-m", "artificial_memory.worker", worker_type],
                        "env": [
                            {"name": "WORKER_TYPE", "value": worker_type},
                            {"name": "DATABASE_URL", "valueFrom": {"secretKeyRef": {"name": f"{name}-store", "key": "url"}}},
                        ],
                        "resources": {
                            "requests": resources.get("requests", {"memory": "256Mi", "cpu": "100m"}),
                            "limits": resources.get("limits", {"memory": "1Gi", "cpu": "500m"}),
                        },
                        "ports": [{"containerPort": 8000}],
                        "livenessProbe": {
                            "httpGet": {"path": "/health", "port": 8000},
                            "initialDelaySeconds": 30,
                            "periodSeconds": 10,
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/ready", "port": 8000},
                            "initialDelaySeconds": 10,
                            "periodSeconds": 5,
                        },
                    }],
                },
            },
        },
    }

    try:
        # Try to update existing deployment
        apps_api.patch_namespaced_deployment(
            name=name,
            namespace=namespace,
            body=deployment_body,
        )
        logger.info(f"Updated deployment {name}")
    except ApiException as e:
        if e.status == 404:
            # Create new deployment
            apps_api.create_namespaced_deployment(
                namespace=namespace,
                body=deployment_body,
            )
            logger.info(f"Created deployment {name}")
        else:
            raise

    # Get ready replicas
    try:
        deployment = apps_api.read_namespaced_deployment(name=name, namespace=namespace)
        return deployment.status.ready_replicas or 0
    except ApiException:
        return 0


@kopf.on.delete(API_GROUP, API_VERSION, CLUSTER_PLURAL)
def delete_cluster(name: str, namespace: str, logger: logging.Logger, **_):
    """Handle cluster deletion."""
    logger.info(f"Deleting ArtificialMemoryCluster {namespace}/{name}")
    # Child resources will be garbage collected via owner references
    return {"message": f"Cluster {name} deleted"}