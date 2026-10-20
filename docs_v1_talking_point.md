# Maya v1: how to explain the 24-hour lag (for the client meeting)

**Short version (one sentence):**
Maya v1's accuracy numbers are for a 24-hour-ahead forecast, meaning each hour's prediction uses demand from the same hour the day before, which is the standard setup for next-day load forecasting and a fair comparison with PACE's own forecast.

**If asked "could we buy power with this the day before?":**
Not directly. If the purchase is placed in the morning, some of yesterday's same hours haven't happened yet. That is why we built a buy-ahead version (Maya v2) that only uses data available at a 10 am deadline. On that stricter basis it is still about 34% more accurate than PACE in 2024-25 and 42% more accurate in 2026.

**If asked "so is v1 wrong?":**
No. It answers a slightly different question: how much power will be needed over the next 24 hours, updated hour by hour. It is the right tool for operations and monitoring. For day-ahead purchasing, v2 is the right tool.

**If asked "can it get better?":**
Yes, and it already has. Adding holidays, Utah school breaks and a temperature averaged over the main load centers (Salt Lake City, Ogden, Provo and five others, instead of only Milford) cut the error about 8 to 12% in both versions. The improved v1 (v1.1) is ready on a separate branch and can replace the live app after the meeting.

**Numbers to have ready (demand, best-guess error in MWh per hour):**
| | 2024-25 | 2026 |
|---|---|---|
| PACE's own day-ahead forecast | 255 | 348 |
| Maya v1 (24-hour lag), live app | 187 | 211 |
| Maya v1.1 (24-hour lag), new inputs, branch `v1.1-model-upgrade` | 172 | 186 |
| Maya v2 at a 10 am buy deadline (48-hour lag), v2.0 inputs | 194 | 229 |
| Maya v2 at a 10 am buy deadline (48-hour lag), v3.0 inputs (in use) | 169 | 202 |

A rolling test that retrains each quarter from 2022 on confirms it: on the 10 quarters with real forecast weather, v3 inputs beat v2 inputs in every quarter and the gap is statistically significant (Diebold-Mariano p < 0.001).
