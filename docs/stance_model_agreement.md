# Cross-encoder vs Qwen judge — agreement

Stance: support/neutral/oppose (judge refusals/parse-errors excluded). `liberal_acc` is exact agreement on the signed liberal score (the thesis quantity).

- **llama** — n= 26168/26695   stance: acc=0.814 κ=0.671 κw=0.739  support_acc=0.814  liberal_acc=0.814  ρ(stance,support)=0.710
- **gemma** — n= 26695/26695   stance: acc=0.789 κ=0.621 κw=0.691  support_acc=0.789  liberal_acc=0.789  ρ(stance,support)=0.649
- **qwen** — n= 26695/26695   stance: acc=0.835 κ=0.723 κw=0.803  support_acc=0.835  liberal_acc=0.835  ρ(stance,support)=0.757
- **pooled** — n= 79558/80085   stance: acc=0.813 κ=0.674 κw=0.748  support_acc=0.813  liberal_acc=0.813  ρ(stance,support)=0.703

## Pooled confusion (rows=Qwen, cols=BERT)

| Qwen \ BERT | support | neutral | oppose |
|---|---|---|---|
| support | 24756 | 3037 | 92 |
| neutral | 7004 | 34527 | 1243 |
| oppose | 274 | 3238 | 5387 |
