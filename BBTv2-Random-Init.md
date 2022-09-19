# BBTv2 Random Initialization

## SNLI

| Strategy                 | Budget | Loss Type | Perforamnce      |
| ------------------------ | ------ | --------- | ---------------- |
| Original BBTv2           | 6000   | CE        | Test acc: 0.5759 |
| Xavier Normal            | 6000   | CE        | Test acc: 0.6066 |
| Kaiming Normal           | 6000   | CE        | Test acc: 0.4235 |
| Truncated Normal (-2, 2) | 6000   | CE        | Test acc: 0.5866 |
| Original BBTv2           | 6000   | Hinge     | Test acc: 0.566  |
| Xavier Normal            | 6000   | Hinge     | Test acc: 0.5923 |
| Kaiming Normal           | 6000   | Hinge     | Test acc: 0.3575 |
| Truncated Normal (-2, 2) | 6000   | Hinge     | Test acc: 0.6072 |

## MRPC

| Strategy                 | Budget | Loss Type | Perforamnce      |
| ------------------------ | ------ | --------- | ---------------- |
| Original BBTv2           | 6000   | CE        | Test acc: 0.7514 |
| Xavier Normal            | 6000   | CE        | Test acc: 0.7085 |
| Kaiming Normal           | 6000   | CE        | Test acc: 0.7439 |
| Truncated Normal (-2, 2) | 6000   | CE        |                  |
| Original BBTv2           | 6000   | Hinge     |                  |
| Xavier Normal            | 6000   | Hinge     |                  |
| Kaiming Normal           | 6000   | Hinge     |                  |
| Truncated Normal (-2, 2) | 6000   | Hinge     |                  |

## SST-2

| Strategy                 | Budget | Loss Type | Perforamnce      |
| ------------------------ | ------ | --------- | ---------------- |
| Original BBTv2           | 6000   | CE        | Test acc: 0.883  |
| Xavier Normal            | 6000   | CE        | Test acc: 0.8739 |
| Kaiming Normal           | 6000   | CE        | Test acc: 0.8624 |
| Truncated Normal (-2, 2) | 6000   | CE        |                  |
| Original BBTv2           | 6000   |           |                  |
| Xavier Normal            | 6000   |           |                  |
| Kaiming Normal           | 6000   |           |                  |
| Truncated Normal         | 6000   |           |                  |