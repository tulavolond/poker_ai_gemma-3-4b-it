cd train_2
python -c "
from deep_dataset_v2 import generate_unified_dataset
from collections import Counter
examples = generate_unified_dataset(count_per_stage=500)
actions = Counter()
for x in examples:
    for line in x['completion'].split('\n'):
        if line.startswith('ДЕЙСТВИЕ:'):
            actions[line.split(':')[1].strip()] += 1
print(actions)
"
cd ..

