Place Waymo `*.tfrecord*` shards here (too large for git).

Example:

```bash
cp /path/to/training.tfrecord-* ./data/
python3 ../tools/parse_data_scenarios.py --limit 5
```
