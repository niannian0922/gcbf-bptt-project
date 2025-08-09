#!/usr/bin/env python3
import re
import argparse
from pathlib import Path

# Simple parser to extract KPIs from evaluate_with_logging console output
# It supports both legacy 'eval/...' lines and Champion KPI section lines.

def parse_metrics(text: str) -> dict:
    metrics = {}
    # Legacy lines
    m = re.search(r"eval/success_rate[:\s]+([0-9\.]+)", text)
    if m:
        metrics['success_rate'] = float(m.group(1))
    m = re.search(r"eval/collision_rate[:\s]+([0-9\.]+)", text)
    if m:
        metrics['collision_rate'] = float(m.group(1))
    m = re.search(r"eval/avg_completion_time[:\s]+([0-9\.]+)", text)
    if m:
        metrics['avg_completion_time'] = float(m.group(1))

    # Champion KPI localized prints
    m = re.search(r"成功率[:\s]+([0-9\.]+)%", text)
    if m:
        metrics['success_rate'] = float(m.group(1)) / 100.0
    m = re.search(r"碰撞率[:\s]+([0-9\.]+)%", text)
    if m:
        metrics['collision_rate'] = float(m.group(1)) / 100.0
    m = re.search(r"最佳完成时间[:\s]+([0-9\.]+)\s*步", text)
    if m:
        metrics['best_completion_time'] = float(m.group(1))

    # Fallback: any 'avg_completion_time' in Champion KPIs
    m = re.search(r"平均完成时间[:\s]+([0-9\.]+)\s*步", text)
    if m:
        metrics['avg_completion_time'] = float(m.group(1))

    return metrics


def format_table(baseline: dict, candidate: dict) -> str:
    def fmt(x, pct=False):
        if x is None:
            return "-"
        return (f"{x*100:.1f}%" if pct else f"{x:.1f}")

    lines = []
    lines.append("Decisive Comparison (Baseline vs Candidate)")
    lines.append("")
    lines.append("Metric                 | Baseline | Candidate")
    lines.append("-----------------------|----------|-----------")
    lines.append(f"Success Rate           | {fmt(baseline.get('success_rate'), True):>8} | {fmt(candidate.get('success_rate'), True):>9}")
    lines.append(f"Collision Rate         | {fmt(baseline.get('collision_rate'), True):>8} | {fmt(candidate.get('collision_rate'), True):>9}")
    lines.append(f"Avg Completion Time    | {fmt(baseline.get('avg_completion_time')):>8} | {fmt(candidate.get('avg_completion_time')):>9}")
    if 'best_completion_time' in baseline or 'best_completion_time' in candidate:
        lines.append(f"Best Completion Time   | {fmt(baseline.get('best_completion_time')):>8} | {fmt(candidate.get('best_completion_time')):>9}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    base_txt = Path(args.baseline).read_text(encoding='utf-8', errors='ignore')
    cand_txt = Path(args.candidate).read_text(encoding='utf-8', errors='ignore')

    base_metrics = parse_metrics(base_txt)
    cand_metrics = parse_metrics(cand_txt)

    table = format_table(base_metrics, cand_metrics)
    Path(args.out).write_text(table, encoding='utf-8')
    print(table)

if __name__ == '__main__':
    main()


