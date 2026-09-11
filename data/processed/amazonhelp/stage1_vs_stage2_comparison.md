# Stage 1 vs. Stage 2 (LLM Distillation) Label Comparison

## Summary
- **Total Rows Evaluated by LLM**: 29
- **Agreements with Heuristic Proposal**: 17 (58.6%)
- **Disagreements Resolved by LLM**: 12 (41.4%)

## Stage 2 Dataset Composition
| Label Source | Count |
| :--- | :--- |
| `stage1_rule` | 824 |
| `stage1_sampled_other` | 400 |
| `llm_gemini` | 29 |

## Final Stage 2 Label Distribution
| Intent | Count |
| :--- | :--- |
| `delivery_order_status` | 478 |
| `other` | 401 |
| `service_complaint` | 116 |
| `device_digital_content` | 73 |
| `refund_return` | 68 |
| `account_access` | 33 |
| `seller_marketplace` | 28 |
| `payment_billing` | 24 |
| `item_quality_issue` | 20 |
| `prime_membership` | 12 |

## Sample Disagreements (LLM Correction vs. Priority Heuristic)
| Tweet ID | Category | Stage 1 Heuristic | Stage 2 LLM Label | Customer Message |
| :--- | :--- | :--- | :--- | :--- |
| `13157` | multi_match | `delivery_order_status` | **`service_complaint`** | @AmazonHelp I have asked the customer care exec on phone to cancel the order. Do... |
| `13159` | multi_match | `prime_membership` | **`delivery_order_status`** | @AmazonHelp I am now cancelling this order because your delivery service just su... |
| `14855` | multi_match | `seller_marketplace` | **`refund_return`** | @115821 407-8406190-7752345. i want refund amount -1399/- amazon seller was lyin... |
| `15279` | multi_match | `refund_return` | **`delivery_order_status`** | @AmazonHelp 408-9273159-6369903.replacement ordered from 24th Oct.no1 contact me... |
| `15672` | multi_match | `device_digital_content` | **`delivery_order_status`** | Fool me twice!  Another kindle release day, another missed "guaranteed release d... |
| `16428` | multi_match | `refund_return` | **`delivery_order_status`** | @AmazonHelp Not impressed with Amazon Logistics. Computer parts delayed a week a... |
| `16433` | multi_match | `delivery_order_status` | **`service_complaint`** | @119356 @AmazonHelp     horrible service, lousy customer care, non existent cour... |
| `1699` | multi_match | `refund_return` | **`prime_membership`** | @AmazonHelp I called customer service and was told my membership wouldn't be ren... |
| `17655` | multi_match | `seller_marketplace` | **`refund_return`** | @AmazonHelp I need a solution why u r not helping consumers on self return thing... |
| `17682` | multi_match | `delivery_order_status` | **`service_complaint`** | @AmazonHelp Oh you are so considerate, then why can't you treat the orders you t... |
| `18202` | multi_match | `device_digital_content` | **`payment_billing`** | @AmazonHelp #AmazonIndia ordered echo plus on 6th October, till now have not rec... |
| `18209` | multi_match | `delivery_order_status` | **`other`** | @115850. Thanks Joyeeta(customer service) for AMAZON service. Didn't opted for f... |