"""Verify ROCm device access and the project's GRU/PyG training operations.

Run with .venv/bin/python scripts/check_rocm.py from the repository root.
Exits nonzero if ROCm or any GPU operation is unavailable; never falls back to CPU.
"""

from pathlib import Path
import sys


def check_workload(device: str) -> None:
    import torch
    from torch_geometric.nn import GATv2Conv

    from epgnn.models.encoders import HistoryGRU
    from epgnn.train import masked_loss

    # Compare an actual GPU matrix multiplication with a known result.
    x = torch.arange(64, dtype=torch.float32, device=device).reshape(8, 8)
    expected = x.cpu() @ x.cpu().T
    torch.testing.assert_close((x @ x.T).cpu(), expected)

    encoder = HistoryGRU(n_in=3, hidden=8).to(device)
    graph = GATv2Conv(8, 24, heads=2, concat=False, edge_dim=2).to(device)
    parameters = list(encoder.parameters()) + list(graph.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    history = torch.randn(4, 12, 3, device=device)
    edges = torch.tensor([[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0]], device=device)
    edge_features = torch.randn(edges.shape[1], 2, device=device)
    target = torch.randn(4, 24, device=device)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[0, 0] = False
    pairs = edges[:, :2]

    optimizer.zero_grad()
    prediction = graph(encoder(history), edges, edge_features)
    loss = masked_loss(prediction, target, mask, pairs, spread_weight=0.1)
    if not torch.isfinite(loss).item():
        raise RuntimeError("GRU/GAT loss is not finite")
    loss.backward()
    for parameter in parameters:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError("GRU/GAT gradients are missing or not finite")
    optimizer.step()
    for parameter in parameters:
        if not torch.isfinite(parameter).all().item():
            raise RuntimeError("Adam produced non-finite parameters")


def main() -> int:
    print(f"Python: {sys.executable}", flush=True)
    try:
        import torch
        import torch_geometric

        print(f"PyTorch: {torch.__version__}; HIP: {torch.version.hip}")
        print(f"PyTorch Geometric: {torch_geometric.__version__}")
        if torch.version.hip is None:
            raise RuntimeError(
                "This is not a ROCm build of PyTorch. Install requirements-rocm.txt."
            )
        if not Path("/dev/kfd").exists() or not list(Path("/dev/dri").glob("renderD*")):
            raise RuntimeError(
                "ROCm device nodes are missing (/dev/kfd or /dev/dri/renderD*). "
                "Run in a host terminal or an environment that exposes the GPU devices."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ROCm PyTorch cannot initialize the GPU. Check rocminfo, device "
                "permissions, and GPU visibility environment variables."
            )
        device = "cuda:0"  # ROCm deliberately uses PyTorch's torch.cuda API.
        properties = torch.cuda.get_device_properties(device)
        print(f"GPU: {properties.name}; VRAM: {properties.total_memory / 2**30:.1f} GiB")
        print(f"Architecture: {properties.gcnArchName}", flush=True)
        check_workload(device)
        torch.cuda.synchronize(device)
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("PASS: ROCm matmul, HistoryGRU, GATv2, masked loss, backward, and Adam.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
