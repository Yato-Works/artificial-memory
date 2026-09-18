"""RecallWorker CRD handler."""

import logging
from datetime import datetime
from typing import Any

import kopf
import kubernetes.client
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

API_GROUP = "memory.artificialmemory.dev"
API_VERSION = "v1"
RECALL_PLURAL = "recallworkers"


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
            plural=RECALL_PLURAL,
            name=name,
            body={"status": status},
        )
    except ApiException as e:
        logger.error(f"Failed to update status for {name}: {e}")


@kopf.on.create(API_GROUP, API_VERSION, RECALL_PLURAL)
@kopf.on.update(API_GROUP, API_VERSION, RECALL_PLURAL)
def reconcile_recall_worker(spec: dict[str, Any], name: str, namespace: str, status: dict[str, Any], logger: logging.Logger, **_) -> dict[str, Any]:
    logger.info(f"Reconciling RecallWorker {namespace}/{name}")

    custom_api = get_k8s_custom_objects()
    apps_api = get_k8s_apps_v1()
    core_api = get_k8s_core_v1()

    if not status:
        status = {
            "phase": "Pending",
            "conditions": [],
            "readyReplicas": 0,
            "desiredReplicas": 0,
            "currentReplicas": 0,
            "totalQueries": 0,
            "avgLatencyMs": 0,
            "p99LatencyMs": 0,
            "cacheHitRate": 0.0,
            "serviceEndpoint": "",
        }

    cluster_ref = spec.get("clusterRef")
    replicas = spec.get("replicas", 4)
    image = spec.get("image", "artificial-memory:latest")
    autoscaling = spec.get("autoscaling", {})

    try:
        # Reconcile deployment
        ready_replicas = reconcile_recall_deployment(
            apps_api, core_api, namespace, name, cluster_ref, replicas, image, spec, logger
        )

        status["readyReplicas"] = ready_replicas
        status["desiredReplicas"] = replicas
        status["currentReplicas"] = replicas

        # Create service
        service_endpoint = reconcile_recall_service(core_api, namespace, name, cluster_ref, spec, logger)
        status["serviceEndpoint"] = service_endpoint

        # Create HPA if autoscaling enabled
        if autoscaling.get("enabled", True):
            reconcile_hpa(custom_api, namespace, name, autoscaling, replicas, logger)

        if ready_replicas >= replicas:
            status["phase"] = "Running"
            status["conditions"] = [
                create_condition("Ready", "True", "WorkersReady", f"All {replicas} recall workers ready"),
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
        logger.error(f"Failed to reconcile recall worker {namespace}/{name}: {e}")
        status["phase"] = "Failed"
        status["conditions"] = [create_condition("Ready", "False", "ReconcileError", str(e))]
        update_status(custom_api, namespace, name, status)
        raise


def reconcile_recall_deployment(
    apps_api: kubernetes.client.AppsV1Api,
    core_api: kubernetes.client.CoreV1Api,
    namespace: str,
    name: str,
    cluster_ref: str,
    replicas: int,
    image: str,
    spec: dict[str, Any],
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
                "app.kubernetes.io/component": "recall",
                "app.kubernetes.io/part-of": "artificial-memory-cluster",
                "cluster-ref": cluster_ref,
            },
        },
        "spec": {
            "replicas": replicas,
            "selector": {
                "matchLabels": {
                    "app.kubernetes.io/name": "artificial-memory",
                    "app.kubernetes.io/component": "recall",
                    "cluster-ref": cluster_ref,
                },
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/name": "artificial-memory",
                        "app.kubernetes.io/component": "recall",
                        "cluster-ref": cluster_ref,
                    },
                },
                "spec": {
                    "containers": [{
                        "name": "worker",
                        "image": image,
                        "command": ["python", "-m", "artificial_memory.worker", "recall"],
                        "env": [
                            {"name": "WORKER_TYPE", "value": "recall"},
                            {"name": "CLUSTER_REF", "value": cluster_ref},
                            {"name": "DATABASE_URL", "valueFrom": {"secretKeyRef": {"name": f"{cluster_ref}-store", "key": "url", "optional": True}}},
                        ] + [{"name": k.upper(), "value": v} for k, v in config.items()],
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


def reconcile_recall_service(
    core_api: kubernetes.client.CoreV1Api,
    namespace: str,
    name: str,
    cluster_ref: str,
    spec: dict[str, Any],
    logger: logging.Logger,
) -> str:
    service_spec = spec.get("service", {})
    port = service_spec.get("port", 8000)
    service_type = service_spec.get("type", "ClusterIP")

    service_name = f"{name}-service"
    service_body = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": service_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "artificial-memory",
                "app.kubernetes.io/component": "recall",
                "cluster-ref": cluster_ref,
            },
        },
        "spec": {
            "type": service_type,
            "ports": [{
                "port": port,
                "targetPort": 8000,
                "protocol": "TCP",
                "name": "http",
            }],
            "selector": {
                "app.kubernetes.io/name": "artificial-memory",
                "app.kubernetes.io/component": "recall",
                "cluster-ref": cluster_ref,
            },
        },
    }

    try:
        core_api.patch_namespaced_service(name=service_name, namespace=namespace, body=service_body)
    except ApiException as e:
        if e.status == 404:
            core_api.create_namespaced_service(namespace=namespace, body=service_body)
        else:
            raise

    if service_type == "ClusterIP":
        return f"{service_name}.{namespace}.svc.cluster.local:{port}"
    else:
        return f"{service_name}.{namespace}:{port}"


def reconcile_hpa(
    custom_api: kubernetes.client.CustomObjectsApi,
    namespace: str,
    name: str,
    autoscaling: dict[str, Any],
    replicas: int,
    logger: logging.Logger,
):
    """Create HorizontalPodAutoscaler for recall workers."""
    hpa_name = f"{name}-hpa"
    hpa_body = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": hpa_name, "namespace": namespace},
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": name,
            },
            "minReplicas": autoscaling.get("minReplicas", 2),
            "maxReplicas": autoscaling.get("maxReplicas", 20),
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": autoscaling.get("targetCPUUtilization", 60),
                        },
                    },
                },
                {
                    "type": "Resource",
                    "resource": {
                        "name": "memory",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": autoscaling.get("targetMemoryUtilization", 70),
                        },
                    },
                },
            ],
            "behavior": {
                "scaleUp": {
                    "stabilizationWindowSeconds": 60,
                },
                "scaleDown": {
                    "stabilizationWindowSeconds": 300,
                },
            },
        },
    }

    try:
        custom_api.patch_namespaced_custom_object(
            group="autoscaling",
            version="v2",
            namespace=namespace,
            plural="horizontalpodautoscalers",
            name=hpa_name,
            body=hpa_body,
        )
    except ApiException as e:
        if e.status == 404:
            custom_api.create_namespaced_custom_object(
                group="autoscaling",
                version="v2",
                namespace=namespace,
                plural="horizontalpodautoscalers",
                body=hpa_body,
            )
        else:
            raise


@kopf.on.delete(API_GROUP, API_VERSION, RECALL_PLURAL)
def delete_recall_worker(name: str, namespace: str, logger: logging.Logger, **_):
    logger.info(f"Deleting RecallWorker {namespace}/{name}")
    return {"message": f"RecallWorker {name} deleted"}
