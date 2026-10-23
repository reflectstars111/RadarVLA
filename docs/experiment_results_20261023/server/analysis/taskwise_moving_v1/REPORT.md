# Motion dataset acceptance

```json
{
  "status": "PASS",
  "policy": "moving_vehicle_01_background_v1",
  "counts": {
    "samples": 9841,
    "agents": 21502,
    "frames_over_8_agents": 111,
    "recovered_positive_instances": 0,
    "mode/LONG": 9500,
    "sft_available": 9831,
    "background/cyclist": 5104,
    "background/stationary": 21185,
    "mode/SHORT": 331,
    "mode/UNKNOWN": 10,
    "background/pedestrian": 122
  },
  "splits": {
    "test": 1522,
    "train": 6847,
    "val": 1472
  },
  "episodes": 64,
  "manifest_sha256": "6f2fdedd3cee3abd1fdc6ab9952b267ce5008aa2994aa014241d6b37a867672e",
  "source_sha256": "083532761bf1dd11c39342b6b1f7e044a1ac533c00275481797db7dee2676cb5",
  "raw_evidence_checked": true,
  "representative_gradients": [
    {
      "case": "empty",
      "sample_id": "c04ba6810ce847248c8f3e58f9227b68_5025661",
      "loss": 16.018646240234375
    },
    {
      "case": "long",
      "sample_id": "c04ba6810ce847248c8f3e58f9227b68_5025661",
      "loss": 16.018646240234375
    },
    {
      "case": "background",
      "sample_id": "c04ba6810ce847248c8f3e58f9227b68_5025661",
      "loss": 16.018646240234375
    },
    {
      "case": "many",
      "sample_id": "28dcaff613894b5e8e4253bd29e2076e_5347723",
      "loss": 48.83403015136719
    },
    {
      "case": "short",
      "sample_id": "310b01d3e46e4109b92e9f26db586bfe_5288500",
      "loss": 30.433528900146484
    },
    {
      "case": "unknown",
      "sample_id": "48d5eed039ab40a7b13320103273843c_321508",
      "loss": 5.369263172149658
    }
  ],
  "optimizer_steps": 0,
  "training_started": false
}
```
