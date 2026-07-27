# Evaluation results — Multi-Agent AI Analyst (F11)

Generated 2026-07-27 18:48. Question set: `documents/eval_questions.json`.
Every run used `use_memory=False` so the comparison is reproducible.

## Metrics: with critic vs without critic

| Metric | With critic | Without critic | Delta |
| --- | --- | --- | --- |
| Questions evaluated | 12 | 12 | +0.000 |
| Exact match rate | 0.833 | 1.000 | -0.167 |
| LLM judge mean (1-5) | 4.500 | 5.000 | -0.500 |
| RAGAS faithfulness | 0.917 | 1.000 | -0.083 |
| RAGAS answer_relevancy | 0.878 | 0.983 | -0.105 |
| RAGAS context_precision | 0.750 | 0.808 | -0.058 |
| RAGAS context_recall | 0.833 | 1.000 | -0.167 |
| Mean revisions | 0.000 | 0.000 | +0.000 |
| Mean seconds per question | 15.230 | 16.580 | -1.350 |
| Critic-verified rate | 1.000 | n/a | n/a |

## Per-question results

| ID | Category | Agents used | Exact (critic) | Judge (critic) | Exact (no critic) | Judge (no critic) | Revisions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q01 | doc | retriever | yes | 5 | yes | 5 | 0 |
| q02 | doc | retriever, data | yes | 5 | yes | 5 | 0 |
| q03 | doc | retriever, data | NO | 2 | yes | 5 | 0 |
| q04 | doc | retriever | yes | 4 | yes | 5 | 0 |
| q05 | sql | data | yes | 5 | yes | 5 | 0 |
| q06 | sql | data | yes | 5 | yes | 5 | 0 |
| q07 | sql | data | yes | 5 | yes | 5 | 0 |
| q08 | sql | data | yes | 5 | yes | 5 | 0 |
| q09 | code | code | yes | 5 | yes | 5 | 0 |
| q10 | code | code | yes | 5 | yes | 5 | 0 |
| q11 | mixed | data, retriever | yes | 5 | yes | 5 | 0 |
| q12 | mixed | data | NO | 3 | yes | 5 | 0 |

## Failures

### With critic — 2 failure(s)

**q03** (doc) — judge 2, exact match no
- Question: What is the first response target for a severity 1 support ticket?
- Expected: A severity 1 ticket, meaning a production outage, gets a first response within 1 hour, 24/7.
- Got: The first response target for a severity 1 support ticket is 40.0 minutes.
- Agents used: retriever, data
- Trace: supervisor→retriever -> retriever(failed: GoogleGenerativeAIError) -> supervisor→data -> data(sql) -> supervisor→finish -> generate -> critic(approved)
- Critic said: The drafted answer accurately reflects the average first response time calculated from the provided evidence.

**q12** (mixed) — judge 3, exact match no
- Question: How many employees are in the Engineering department, and how many annual leave days do they get?
- Expected: Four employees work in the Engineering department, and every full-time employee receives 24 days of paid annual leave per year.
- Got: There are 4 employees in the Engineering department. They have a total of 29 annual leave days.
- Agents used: data
- Trace: supervisor→data -> data(sql) -> supervisor→finish -> generate -> critic(approved)
- Critic said: The answer accurately reflects the counts and sums provided in the SQL query result.

### Without critic — 0 failure(s)

None.
