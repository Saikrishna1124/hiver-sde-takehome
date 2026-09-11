# LLM-as-a-Judge Evaluation & Human Agreement Report

Evaluation of generated customer support responses on a representative sample
of 25 interactions from the sealed golden evaluation set.

---

## 1. Headline Agreement Metrics

- **Sample Size**: 25 examples
- **Binary Verdict Agreement**: **88.0%** (22 / 25 matches)
- **Pearson Correlation ($r$)**: **0.927** (strong alignment on continuous quality)
- **Mean Absolute Error (MAE)**: **0.22** points (on 1-5 scale)
- **Human Pass Rate**: 52.0%
- **Judge Pass Rate**: 40.0%

### Verdict Confusion Matrix

| | Human PASS | Human FAIL |
|---|---|---|
| **Judge PASS** | 10 (True Positives) | 0 (False Positives) |
| **Judge FAIL** | 3 (False Negatives) | 12 (True Negatives) |

---

## 2. Evaluation Rubric
### LLM Judge Evaluation Rubric

Each candidate reply is evaluated on a 1-5 scale across three dimensions:

1. **Relevance (1-5)**:
   - 5: Directly and completely addresses the customer's stated issue.
   - 3: Partially relevant; provides generic guidance but misses specific nuance.
   - 1: Completely irrelevant or addresses a different issue (e.g., tracking link for a billing question).

2. **Tone & Voice (1-5)**:
   - 5: Polite, empathetic, professional, consistent with AmazonHelp Twitter tone.
   - 3: Neutral or slightly robotic, but acceptable.
   - 1: Rude, accusatory, dismissive, or completely inappropriate.

3. **Grounding & Factuality (1-5)**:
   - 5: Standard verified Amazon customer support policy (links to auth/tracking/support portal).
   - 3: Generic customer service guidance without unsupported claims.
   - 1: Hallucinates specific fake order numbers, refunds, or promises non-existent policies.

**Pass Criteria**: Overall Score >= 3.5 AND Relevance >= 3.0 -> PASS; otherwise FAIL.

---

## 3. Representative Case Studies

### Case 1 — [delivery_order_status]
**Customer Message**: *"@AmazonHelp at 7 it said it was out for delivery by 8 pm. It now says sorry for your package being late and will arrive by November 3rd"*
**Grounded Reply**: *"I'm sorry to hear about the delay. Is there any further information on your tracking? You can find your tracking here:  https://t.co/Y5jpI9gRhE. ^MI"*
- **Human Rating**: Overall **5.0** (PASS) [Rel: 5.0, Tone: 5.0, Ground: 5.0]
- **Judge Rating**: Overall **5.0** (PASS) [Rel: 5.0, Tone: 5.0, Ground: 5.0]
- **Judge Rationale**: Directly acknowledges shipping delay, maintains empathetic customer service tone, and provides authentic tracking link.

### Case 2 — [other]
**Customer Message**: *"@AmazonHelp Many thanks for the generic automated correspondence. After paying more than Rs.14,000 and receiving a box of trash, I would have expected bit more support that just asking me to wait longer."*
**Grounded Reply**: *"The Echo Show is supported, please reach us for some live troubleshooting at your convenience: https://t.co/hApLpMlfHN ^DW"*
- **Human Rating**: Overall **2.3** (FAIL) [Rel: 1.0, Tone: 4.0, Ground: 2.0]
- **Judge Rating**: Overall **2.0** (FAIL) [Rel: 1.0, Tone: 3.0, Ground: 2.0]
- **Judge Rationale**: Completely irrelevant; customer complained about wrong/damaged order contents, but reply offers Echo Show live troubleshooting.

### Case 3 — [seller_marketplace]
**Customer Message**: *"@AmazonHelp I already replied on mails...u people just wasting time nothing else.. no action has been taken against seller n brand...as a compensation"*
**Grounded Reply**: *"You can check for our response here: https://t.co/DTSNmGldJf. Hope this helps. ^CB"*
- **Human Rating**: Overall **4.0** (PASS) [Rel: 4.0, Tone: 4.0, Ground: 4.0]
- **Judge Rating**: Overall **3.0** (FAIL) [Rel: 2.0, Tone: 4.0, Ground: 3.0]
- **Judge Rationale**: Customer specifically demands compensation and disciplinary action against a marketplace seller; providing a generic link to correspondence history fails to address seller sanctions or compensation claims.

### Case 4 — [other]
**Customer Message**: *"@AmazonHelp Thank you, your chat solved it very quickly and I do know the person who sent it! :)"*
**Grounded Reply**: *"You're welcome. Do keep us posted for any further assistance. ^MK"*
- **Human Rating**: Overall **5.0** (PASS) [Rel: 5.0, Tone: 5.0, Ground: 5.0]
- **Judge Rating**: Overall **5.0** (PASS) [Rel: 5.0, Tone: 5.0, Ground: 5.0]
- **Judge Rationale**: Perfect courteous closing reply to positive resolution.
