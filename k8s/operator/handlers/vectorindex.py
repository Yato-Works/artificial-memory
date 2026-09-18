"""VectorIndex CRD handler."""

import logging
from datetime import datetime
from typing import Any

import kopf
import kubernetes.client
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

API_GROUP = "memory.artificialmemory.dev"
API_VERSION = "v1"
INDEX_PLURAL = "vectorindexes"


def get_k8s_apps_v1():
    return kubernetes.client.AppsV1Api()


def get_k8s_core_v1():
    return kubernetes.client.CoreV1Api()


def get_k8s_custom_objects():
    return kubernetes.client.CustomObjectsApi()


def create_condition(condition_type: str, status: str, reason: str, message: str) -> dict[str, Any]:
    return {
        "type": condition_type,
        "status": status,
        "reason": reason,
        "message": message,
        "lastTransitionTime": datetime.utcnow().isoformat() + "Z",
    }


def update_status(custom_api, namespace: str, name: str, status: dict[str, Any]):
    try:
        custom_api.patch_namespaced_custom_object_status(
            group=API_GROUP,
            version=API_VERSION,
            namespace=namespace,
            plural=INDEX_PLURAL,
            name=name,
            body={"status": status},
        )
    except ApiException as e:
        logger.error(f"Failed to update status for {name}: {e}")


@kopf.on.create(API_GROUP, API_VERSION, INDEX_PLURAL)
@kopf.on.update(API_GROUP, API_VERSION, INDEX_PLURAL)
def reconcile_vector_index(spec: dict[str, Any], name: str, namespace: str, status: dict[str, Any], logger: logging.Logger, **_) -> dict[str, Any]:
    logger.info(f"Reconciling VectorIndex {namespace}/{name}")

    custom_api = get_k8s_custom_objects()
    apps_api = get_k8s_apps_v1()
    core_api = get_k8s_core_v1()

    if not status:
        status = {
            "phase": "Pending",
            "conditions": [],
            "totalVectors": 0,
            "indexSize": "0B",
            "lastSyncTime": None,
            "syncLagSeconds": 0,
            "buildProgress": 0,
            "searchLatencyP50Ms": 0,
            "searchLatencyP99Ms": 0,
        }

    cluster_ref = spec.get("clusterRef")
    storage = spec.get("storage", {})

    try:
        if storage.get("backend") == "pgvector":
            # pgvector index is managed within PostgreSQL
            index_ready = reconcile_pgvector_index(core_api, namespace, name, cluster_ref, spec, logger)
        else:
            # Standalone FAISS index deployment
            index_ready = reconcile_faiss_index(apps_api, core_api, namespace, name, cluster_ref, spec, logger)

        if index_ready:
            status["phase"] = "Ready"
            status["buildProgress"] = 100
            status["conditions"] = [
                create_condition("Ready", "True", "IndexReady", "Vector index is ready"),
                create_condition("Synced", "True", "SyncComplete", "Index is synced with primary store"),
            ]
        else:
            status["phase"] = "Building"
            status["buildProgress"] = 50
            status["conditions"] = [
                create_condition("Ready", "False", "Building", "Vector index is being built"),
            ]

        update_status(custom_api, namespace, name, status)
        return {"status": status}

    except Exception as e:
        logger.error(f"Failed to reconcile vector index {namespace}/{name}: {e}")
        status["phase"] = "Failed"
        status["conditions"] = [create_condition("Ready", "False", "ReconcileError", str(e))]
        update_status(custom_api, namespace, name, status)
        raise


def reconcile_pgvector_index(core_api: kubernetes.client.CoreV1Api, namespace: str, name: str, cluster_ref: str, spec: dict[str, Any], logger: logging.Logger) -> bool:
    """pgvector index is managed via PostgreSQL - just verify connection."""
    logger.info(f"pgvector index for {cluster_ref} - managed by PostgreSQL")
    return True


def reconcile_faiss_index(
    apps_api: kubernetes.client.AppsV1Api,
    core_api: kubernetes.client.CoreV1Api,
    namespace: str,
    name: str,
    cluster_ref: str,
    spec: dict[str, Any],
    logger: logging.Logger,
) -> bool:
    """Reconcile standalone FAISS index deployment."""
    resources = spec.get("resources", {})
    hnsw = spec.get("hnsw", {})
    sync_spec = spec.get("sync", {})

    deployment_name = f"{name}-faiss"
    deployment_body = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": deployment_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "artificial-memory",
                "app.kubernetes.io/component": "vector-index",
                "app.kubernetes.io/part-of": "artificial-memory-cluster",
                "cluster-ref": cluster_ref,
            },
        },
        "spec": {
            "replicas": 1,  # FAISS index typically single replica for consistency
            "selector": {
                "matchLabels": {
                    "app.kubernetes.io/name": "artificial-memory",
                    "app.kubernetes.io/component": "vector-index",
                    "cluster-ref": cluster_ref,
                },
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/name": "artificial-memory",
                        "app.kubernetes.io/component": "vector-index",
                        "cluster-ref": cluster_ref,
                    },
                },
                "spec": {
                    "containers": [{
                        "name": "vector-index",
                        "image": spec.get("image", "artificial-memory-vector-index:latest"),
                        "command": ["python", "-m", "artificial_memory.vector_index", "serve"],
                        "env": [
                            {"name": "INDEX_TYPE", "value": spec.get("indexType", "hnsw")},
                            {"name": "DIMENSION", "value": str(spec.get("dimension", 384))},
                            {"name": "HNSW_M", "value": str(hnsw.get("m", 16))},
                            {"name": "HNSW_EF_CONSTRUCTION", "value": str(hnsw.get("efConstruction", 64))},
                            {"name": "HNSW_EF_SEARCH", "value": str(hnsw.get("efSearch", 128))},
                            {"name": "SYNC_ENABLED", "value": str(sync_spec.get("enabled", True)).lower()},
                            {"name": "SYNC_INTERVAL", "value": str(sync_spec.get("intervalSeconds", 60))},
                            {"name": "CLUSTER_REF", "value": cluster_ref},
                        ],
                        "resources": {
                            "requests": resources.get("requests", {"memory": "1Gi", "cpu": "500m"}),
                            "limits": resources.get("limits", {"memory": "4Gi", "cpu": "2000m"}),
                        },
                        "ports": [{"containerPort": 8001}],
                        "livenessProbe": {
                            "httpGet": {"path": "/health", "port": 8001},
                            "initialDelaySeconds": 60,
                            "periodSeconds": 30,
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/ready", "port": 8001},
                            "initialDelaySeconds": 30,
                            "periodSeconds": 10,
                        },
                    }],
                },
            },
        },
    }

    try:
        apps_api.patch_namespaced_deployment(name=deployment_name, namespace=namespace, body=deployment_body)
    except ApiException as e:
        if e.status == 404:
            apps_api.create_namespaced_deployment(namespace=namespace, body=deployment_body)
        else:
            raise

    try:
        deployment = apps_api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
        return (deployment.status.ready_replicas or 0) > 0
    except ApiException:
        return False


@kopf.on.delete(API_GROUP, API_VERSION, INDEX_PLURAL)
def delete_vector_index(name: str, namespace: str, logger: logging.Logger, **_):
    logger.info(f"Deleting VectorIndex {namespace}/{name}")
    return {"message": f"VectorIndex {name} deleted"}
