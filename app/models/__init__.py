from app.models.gcn import StaticGCN
from app.models.gcn_gru import GcnGruHybrid
from app.models.tgat import TGAT

GCNEncoder = StaticGCN
GCNClassifier = StaticGCN
GCNGRUHybrid = GcnGruHybrid

__all__ = [
    "StaticGCN", "GcnGruHybrid", "TGAT",
    "GCNEncoder", "GCNClassifier", "GCNGRUHybrid",
]
