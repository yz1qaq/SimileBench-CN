# Boundary revision (2026-10-01)

The paper's boundary rule keeps the core tenor or vehicle and removes non-essential location, quantity, color, material, and other surface modifiers. This revision applies that rule to all 5,000 canonical records and the Task 1–4 data views. The machine-readable [`boundary_changes.json`](boundary_changes.json) records every changed retained span, its sentence ID, side, original text, revised text, and sentence context. [`pair_deduplications.json`](pair_deduplications.json) records redundant relations removed after comparing pairs within the same sentence.

- 95 retained label spans revised: 79 in train and 16 in test.
- 48 distinct original label forms were identified for contraction; one occurrence of `白云` was removed as a duplicate instead of retained as a shortened pair.
- Four redundant pairs were removed: three in train and one in test. The revision affects 81 sentences in total.
- Sentence count, train/test membership, and emotion labels remain 4,000/1,000. Pair counts change from 7,205/1,844 to **7,202/1,843** (9,045 total).
- Across both splits, distinct tenors change from 1,873 to 1,861, distinct vehicles from 3,528 to 3,500, and distinct pair types from 6,765 to 6,748.
- Task 1–3 gold views were rebuilt from the revised canonical pairs. Task 4 emotion records are unchanged.
- Unique tenor, vehicle, and pair-type counts changed, so any pair-matching or extraction evaluation reported on the previous annotation version must be recalculated before it is attributed to this revision.

Examples: `屋顶上的雪 → 雪`, `两个小瓣 → 瓣`, `白色的大花 → 花`, `黑色巨蟒 → 巨蟒`, `白云 → 云`. A sentence containing both `脸膛 → 白云` and `脸膛 → 云` now has one canonical `脸膛 → 云` pair. The paper's clouds/mirrors example is the same kind of boundary contraction; that exact sentence does not appear in these files.

Modifiers that materially define the image were kept. Examples include `腐烂的苹果`, `妈妈的手`, `醉酒的大汉`, `蓝宝石`, and the idiom `一团糟`. Two annotations such as `乌云 → 心头上的乌云` and `人们脸上的冬天 → 冬天` would become self-pairs if contracted mechanically. They need a separate judgment about the pair relation rather than a boundary-only edit, so this revision leaves them unchanged.

**Paper consistency:** Table I in the supplied manuscript reports the previous 9,049 pair total. Its pair counts, pair-number distribution, and pair-level experimental results refer to the older annotation version. This repository revision should be treated as a new dataset version until those manuscript numbers and experiments are updated.
