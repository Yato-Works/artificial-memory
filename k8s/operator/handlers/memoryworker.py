"""MemoryWorker CRD handler."""

import kopf
import kubernetes.client
from kubernetes.client.rest import ApiException
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

API_GROUP = "memory.artificialmemory.dev"
API_VERSION = "v1"
WORKER_PLURAL = "memoryworkers"


def get_k8s_apps_v1():
    return kubernetes.client.AppsV1Api()


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
            plural=WORKER_PLURAL,
            name=name,
            body={"status": status},
        )
    except ApiException as e:
        logger.error(f"Failed to update status for {name}: {e}")


@kopf.on.create(API_GROUP, API_VERSION, WORKER_PLURAL)
@kopf.on.update(API_GROUP, API_VERSION, WORKER_PLURAL)
def reconcile_worker(spec: Dict[str, Any], name: str, namespace: str, status: Dict[str, Any], logger: logging.Logger, **_) -> Dict[str, Any]:
    logger.info(f"Reconciling MemoryWorker {namespace}/{name}")

    custom_api = get_k8s_custom_objects()
    apps_api = get_k8s_apps_v1()

    if not status:
        status = {
            "phase": "Pending",
            "conditions": [],
            "readyReplicas": 0,
            "desiredReplicas": 0,
            "currentReplicas": 0,
            "processedJobs": 0,
            "failedJobs": 0,
            "avgProcessingTimeMs": 0,
        }

    worker_type = spec.get("workerType", "compiler")
    cluster_ref = spec.get("clusterRef")
    replicas = spec.get("replicas", 2)
    image = spec.get("image", "artificial-memory:latest")

    try:
        # Reconcile deployment
        ready_replicas = reconcile_worker_deployment(
            apps_api, namespace, name, worker_type, cluster_ref, replicas, image, spec, logger
        )

        status["readyReplicas"] = ready_replicas
        status["desiredReplicas"] = replicas
        status["currentReplicas"] = replicas  # Simplified

        if ready_replicas >= replicas:
            status["phase"] = "Running"
            status["conditions"] = [
                create_condition("Ready", "True", "WorkersReady", f"All {replicas} {worker_type} workers ready"),
            ]
        elif ready_replicas > 0:
            status["phase"] = "Scaling"
            status["conditions"] = [
                create_condition("Ready", "False", "Scaling", f"{ready_replicas}/{replicas} workers ready"),
            ]
        else:
            status["phase"] = "Creating"
            status["conditions"] = [
                create_condition("Ready", "False", "Creating", "Workers being created"),
            ]

        update_status(custom_api, namespace, name, status)
        return {"status": status}

    except Exception as e:
        logger.error(f"Failed to reconcile worker {namespace}/{name}: {e}")
        status["phase"] = "Failed"
        status["conditions"] = [create_condition("Ready", "False", "ReconcileError", str(e))]
        update_status(custom_api, namespace, name, status)
        raise


def reconcile_worker_deployment(
    apps_api: kubernetes.client.AppsV1Api,
    namespace: str,
    name: str,
    worker_type: str,
    cluster_ref: str,
    replicas: int,
    image: str,
    spec: Dict[str, Any],
    logger: logging.Logger,
) -> int:
    resources = spec.get("resources", {})
    config = spec.get("config", {})

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
                "cluster-ref": cluster_ref,
            },
        },
        "spec": {
            "replicas": replicas,
            "selector": {
                "matchLabels": {
                    "app.kubernetes.io/name": "artificial-memory",
                    "app.kubernetes.io/component": worker_type,
                    "cluster-ref": cluster_ref,
                },
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/name": "artificial-memory",
                        "app.kubernetes.io/component": worker_type,
                        "cluster-ref": cluster_ref,
                    },
                },
                "spec": {
                    "containers": [{
                        "name": "worker",
                        "image": image,
                        "command": ["python", "-m", "artificial_memory.worker", worker_type],
                        "env": [
                            {"name": "WORKER_TYPE", "value": worker_type},
                            {"name": "CLUSTER_REF", "value": cluster_ref},
                            {"name": "DATABASE_URL", "valueFrom": {"secretKeyRef": {"name": f"{cluster_ref}-store", "key": "url", "optional": True}}},
                        ] + [{"name": k.upper(), "value": v} for k, v in config.items()],
                        "resources": {
                            "requests": resources.get("requests", {"memory": "512Mi", "cpu": "250m"}),
                            "limits": resources.get("limits", {"memory": "2Gi", "cpu": "1000m"}),
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
        apps_api.patch_namespaced_deployment(name=name, namespace=namespace, body=deployment_body)
    except ApiException as e:
        if e.status == 404:
            apps_api.create_namespaced_deployment(namespace=namespace, body=deployment_body)
        else:
            raise

    try:
        deployment = apps_api.read_namespaced_deployment(name=name, namespace=namespace)
        return deployment.status.ready_replicas or 0
    except ApiException:
        return 0


@kopf.on.delete(API_GROUP, API_VERSION, WORKER_PLURAL)
def delete_worker(name: str, namespace: str, logger: logging.Logger, **_):
    logger.info(f"Deleting MemoryWorker {namespace}/{name}")
    return {"message": f"Worker {name} deleted"}