# Benchmark Inputs

The loaders convert each benchmark into a question, evidence passages, reference answers, and a stable question identifier. Reference answers are used by the evaluator after generation. Agent prompts contain the question and passages.

## Native Formats

HotpotQA and 2WikiMultihopQA use `question`, `answer`, and `context`, where each context entry contains a title and its sentences. The HotpotQA reader also accepts the column-oriented context representation used by Parquet exports.

MuSiQue-Ans uses `question`, `answer`, `answer_aliases`, and `paragraphs`. Each paragraph supplies its title and paragraph text. The reader verifies the answerable condition.

FEVEROUS uses `claim` and `label`. Retrieved evidence is supplied as passages on each record, or through a JSON object indexed by claim ID. For example,

```json
{"1": [{"title": "Retrieved page", "text": "Retrieved textual or table evidence."}]}
```

The evidence input is the output of retrieval. The evaluator reads the claim label after the model answers. The `load_wiki_page` helper retrieves a page from the released FEVEROUS SQLite database when constructing the retrieval pipeline.

## Shared Format

```json
{
  "id": "question-id",
  "dataset": "hotpotqa",
  "question": "The question text",
  "passages": [{"title": "Passage title", "text": "Passage text"}],
  "answers": ["Reference answer", "Accepted alias"]
}
```

Every development question appears once. IDs are checked for uniqueness, and the same input files are used across methods and seeds. The evaluator records dataset and answer-alias digests in the run metadata.

## Exact Match

The normalization routine lowercases the prediction, removes punctuation and articles, and collapses whitespace. Matching compares the complete normalized prediction with each accepted reference answer. FEVEROUS is evaluated on its three labels.

The default `whole_message` rule scores the final agent message, as stated in the paper. `last_line` is an explicit configuration option for deployments whose messages include reasoning followed by an answer line. The run signature records this choice.
