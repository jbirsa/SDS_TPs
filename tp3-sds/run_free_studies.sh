#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

JAVA="java -Dframes.enabled=false -cp out simulation.Main"
SEED=42

run() {  # t1 t2 k mod simTime
  echo "→ mod=$4 t1=$1 t2=$2 k=$3 simTime=$5"
  $JAVA "$1" "$2" "$3" "$4" FREE "$5" "$SEED" >/dev/null
}

# Study 2.1 — t1=1, k=5, vary t2. Unstable for t2>=5 (ρ=t2/5)
for MOD in A B; do
  for T2 in 1.0 2.0 3.0 4.0 4.5 4.9 5.0 6.0 7.0 8.0; do
    run 1.0 "$T2" 5 "$MOD" 1000
  done
done

# Study 2.2 — t2=3, k=5, vary t1. Unstable for t1<=0.6 (ρ=3/(5·t1))
for MOD in A B; do
  for T1 in 0.3 0.4 0.5 0.6 0.7 1.0 1.5 2.0 2.5 3.0; do
    run "$T1" 3.0 5 "$MOD" 1000
  done
done

# Study 2.3 — t1=1, t2=3, vary k. Unstable for k<=3 (ρ=3/k)
for MOD in A B; do
  for K in 1 2 3 4 5 6 7 8; do
    run 1.0 3.0 "$K" "$MOD" 1000
  done
done

echo "Done."
