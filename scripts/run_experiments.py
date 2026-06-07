import json
import random
import sys
import time

MOCK_METRICS = {
    "custom_engine": {"avg_latency_ms": None, "peak_ram_mb": None, "recall@10": None},
    "pgvector": {"avg_latency_ms": None, "peak_ram_mb": None, "recall@10": None},
}

def simulate_custom_search(query_id: int) -> tuple[float, float]:
    latency = random.uniform(5.0, 50.0)
    ram = random.uniform(100, 600)
    time.sleep(latency / 1000)
    return latency, ram

def simulate_pgvector_search(query_id: int) -> tuple[float, float]:
    latency = random.uniform(2.0, 30.0)
    ram = random.uniform(200, 800)
    time.sleep(latency / 1000)
    return latency, ram

def compute_recall(k: int = 10) -> float:
    return random.uniform(0.65, 0.95)

def run_benchmark(num_queries: int = 100) -> dict:
    print(f"[Benchmark] Ejecutando {num_queries} consultas simuladas...\n")

    custom_times, custom_rams = [], []
    pgv_times, pgv_rams = [], []

    for qid in range(1, num_queries + 1):
        lt, rm = simulate_custom_search(qid)
        custom_times.append(lt)
        custom_rams.append(rm)

        lt2, rm2 = simulate_pgvector_search(qid)
        pgv_times.append(lt2)
        pgv_rams.append(rm2)

        if qid % 25 == 0:
            print(f"  Procesadas {qid}/{num_queries} consultas...")

    metrics = {
        "custom_engine": {
            "avg_latency_ms": round(sum(custom_times) / len(custom_times), 2),
            "peak_ram_mb": round(max(custom_rams), 2),
            "recall@10": round(compute_recall(10), 4),
        },
        "pgvector": {
            "avg_latency_ms": round(sum(pgv_times) / len(pgv_times), 2),
            "peak_ram_mb": round(max(pgv_rams), 2),
            "recall@10": round(compute_recall(10), 4),
        },
    }

    return metrics

def print_report(metrics: dict) -> None:
    print("\n" + "=" * 55)
    print("  REPORTE DE BENCHMARK (MOCK — Fase 4)")
    print("=" * 55)

    for engine, vals in metrics.items():
        print(f"\n  Motor: {engine}")
        print(f"    Latencia promedio : {vals['avg_latency_ms']:>8.2f} ms")
        print(f"    Pico de RAM       : {vals['peak_ram_mb']:>8.2f} MB")
        print(f"    Recall@10         : {vals['recall@10']:>8.4f}")

    print("\n" + "=" * 55)
    print("[Benchmark] Completado (modo mock).\n")

if __name__ == "__main__":
    n_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    metrics = run_benchmark(n_queries)
    print_report(metrics)

    with open("benchmark_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[Benchmark] Resultados guardados en benchmark_results.json")
