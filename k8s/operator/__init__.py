"""Artificial Memory Operator - Main entry point."""

import logging

import kopf
import kubernetes.client
import kubernetes.config

# Import handlers
from k8s.operator.handlers import (
    artificialmemorycluster,
    memorystore,
    memoryworker,
    recallworker,
    vectorindex,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load in-cluster config or kubeconfig
try:
    kubernetes.config.load_incluster_config()
    logger.info("Loaded in-cluster Kubernetes config")
except kubernetes.config.ConfigException:
    try:
        kubernetes.config.load_kube_config()
        logger.info("Loaded local kubeconfig")
    except kubernetes.config.ConfigException:
        logger.warning("Could not load Kubernetes config")


@kopf.on.startup()
def configure(settings: kopf.OperatorSettings, **_):
    """Configure operator settings."""
    settings.posting.level = logging.INFO
    settings.watching.server_timeout = 60
    settings.watching.client_timeout = 60
    settings.persistence.finalizer = "memory.artificialmemory.dev/operator-finalizer"
    settings.persistence.diffbase_storage = kopf.AnnotationsDiffBaseStorage(
        prefix="memory.artificialmemory.dev",
        key="last-config",
    )
    settings.persistence.progress_storage = kopf.AnnotationsProgressStorage(
        prefix="memory.artificialmemory.dev",
        key="progress",
    )
    settings.networking.connect_timeout = 10
    settings.networking.read_timeout = 60
    logger.info("Operator settings configured")


@kopf.on.cleanup()
def cleanup(**_):
    """Cleanup on operator shutdown."""
    logger.info("Operator shutting down")


__all__ = [
    "configure",
    "cleanup",
    "artificialmemorycluster",
    "memorystore",
    "memoryworker",
    "recallworker",
    "vectorindex",
]
