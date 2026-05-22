#!/usr/bin/env bash
# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

set -e
cd "$(dirname "$0")/.."

EXPECTED="9fe062e5b52e1212bf89437d24423cfdd34211e15a270a175d6263fc7789f25d *third_party/rck/self_model.py
4e8bc0b9a9919f25dbcaf236f5dc3ee6c3d8c20b5b6177aabb5d8ca535114734 *third_party/rck/theory_of_mind.py
a277d43b2cab9e979d604620f9d94aee5c9b91a635ea8a01da0b897c6b0bfd58 *third_party/rck/metacog.py
be46a30f303662d1c26b855dd523a3545e8b9cd220c3eb5f1c8aa07ddd0b47c6 *third_party/rck/introspect.py
39ad507b06c03960d69b8e802f1dcf5788404c45e48f2a185e1218b18fe935f7 *third_party/rck/dialogue.py
3db6c5d06b47a1877964ed28e7641147c6d40e0afa624815ee6e066a2379a5e9 *third_party/rck/inference.py
5151c903a02600992a7b9e60215f6bcaae11a5a854d232ab7e7b8737e298a23c *third_party/rck/compose.py
f30e9632db6b0f68ec4fa92b1915698e901049f4e5a9b2187c4ded3dbd1eada0 *third_party/rck/explain.py
8a6df92b7fbd481fd29764b9c0b6c6cb0e194a0945daec86fbdec6b85bab82a6 *third_party/rck/think_aloud.py
3b4357abe0aff1496cddfb1fd9aebdef1f77927db78f6fb6bc1701e26da7f625 *third_party/rck/compose_answer.py
d5f4d8596dcfe7a98333c20295d7292968b6e4a44c1039372adffd8919271df1 *third_party/rck/conscious_agent.py
462ae5b796ecc1cd9cde6e075305b48559defd6ca6f9fd5db91d37eef6989933 *third_party/rck/bigram.py
458a33d53cad06ad445c530d83001ccbc65fea75202675dfed3cdedf769f368c *third_party/rck/nlg.py
7ac0de0f89a27fba698fccb19848c7e065d697c315cfb3cefdf7da88ad75ba63 *third_party/rck/bulk_ingest.py
5e9ad3364deaa899593b21cff73f72ca08e891b60f47bebd9b5c9e0af4a02fc3 *third_party/rck/synonyms.py
b935e04171f294c2705e6017cba21d34f95c5bccde90b2c0190a2b0eae613e5d *third_party/rck/self_verify.py
6c2397da58dd15e1ef4828e80f458338b2fb09a31f9d7eee1a389cd4b130c8e9 *third_party/rck/federated_merge.py
fd31cd434e49cdd1380bd6121f4045c58f85cdc1f9adef9b06ca85c2854967e1 *third_party/rck/dreaming.py
43b7f6fa3375180dee072edb0a68b3e43b8e61107046f3e6be14fe9cc8a8152f *third_party/rck/personality.py
b068dc23d8415fc92f1a8fc0e59c4689e1823f53232bdd9a27f2c1b7adab818f *third_party/rck/idk_detection.py
0ebc8e31314c4378b0e047a1ce22b49873abf78cd36c5cb8f6cb21a08cd189ae *third_party/rck/provenance.py
df3ebf0411c84d9d6629b2c6610905ab6adcee6daaed36889e4e920c531ddfd5 *third_party/rck/episodic_consolidate.py"

CURRENT=$(for f in self_model.py theory_of_mind.py metacog.py introspect.py dialogue.py \
         inference.py compose.py explain.py think_aloud.py compose_answer.py \
         conscious_agent.py bigram.py nlg.py bulk_ingest.py synonyms.py \
         self_verify.py federated_merge.py dreaming.py personality.py \
         idk_detection.py provenance.py episodic_consolidate.py; do
    if [ -f "third_party/rck/$f" ]; then
        sha256sum "third_party/rck/$f"
    fi
done)

if [ "$EXPECTED" = "$CURRENT" ]; then
    echo "OK: RCK cognitive sources match recorded provenance"
    exit 0
else
    echo "MISMATCH:"
    diff <(echo "$EXPECTED") <(echo "$CURRENT") || true
    exit 1
fi
