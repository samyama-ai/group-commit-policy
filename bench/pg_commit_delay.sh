#!/usr/bin/env bash
# H5b (real engine): PostgreSQL commit_delay sweep under closed-loop pgbench clients.
# Tests the thesis on a real engine: commit_delay=0 (parameter-free; PG's natural pipelined group
# commit self-clocks) is competitive with the best-tuned commit_delay. WAL on the slow device (EBS)
# is where group commit matters. No mocks.
#
#   bench/pg_commit_delay.sh <out_json> [clients] [seconds] [scale]
set -euo pipefail
OUT=${1:-results/pg_commit_delay.json}; CLIENTS=${2:-32}; SECS=${3:-20}; SCALE=${4:-50}
PGBIN=$(pg_config --bindir 2>/dev/null || echo /usr/lib/postgresql/14/bin)
export PGUSER=postgres PGDATABASE=pgbench_db
mkdir -p "$(dirname "$OUT")"

sudo systemctl start postgresql
sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='pgbench_db'" | grep -q 1 || \
  sudo -u postgres createdb pgbench_db
# ensure durable commits (so flushes actually fsync); init once
sudo -u postgres psql -q -c "ALTER SYSTEM SET fsync=on;" -c "ALTER SYSTEM SET synchronous_commit=on;" \
  -c "ALTER SYSTEM SET commit_siblings=1;" -c "SELECT pg_reload_conf();" >/dev/null
sudo -u postgres "$PGBIN/pgbench" -i -q -s "$SCALE" pgbench_db 2>/dev/null

echo "[" > "$OUT"; first=1
for cd in 0 10 25 50 100 200 500 1000 2000; do
  sudo -u postgres psql -q -c "ALTER SYSTEM SET commit_delay=$cd;" -c "SELECT pg_reload_conf();" >/dev/null
  sleep 1
  res=$(sudo -u postgres "$PGBIN/pgbench" -c "$CLIENTS" -j 8 -T "$SECS" -r -N pgbench_db 2>/dev/null)
  tps=$(echo "$res"   | grep -oE 'tps = [0-9.]+' | head -1 | grep -oE '[0-9.]+')
  lat=$(echo "$res"   | grep -oE 'latency average = [0-9.]+' | grep -oE '[0-9.]+')
  echo "commit_delay=$cd us -> tps=$tps lat_avg=${lat}ms"
  [ $first -eq 0 ] && echo "," >> "$OUT"; first=0
  printf '  {"commit_delay_us":%s,"tps":%s,"lat_avg_ms":%s,"clients":%s}' "$cd" "${tps:-0}" "${lat:-0}" "$CLIENTS" >> "$OUT"
done
echo "" >> "$OUT"; echo "]" >> "$OUT"
echo "wrote $OUT"
