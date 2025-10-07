package com.pulse.agg;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.pulse.common.dto.LogEvent;
import com.pulse.common.dto.PolicySnippet;
import com.pulse.common.dto.ScoreResponse;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
@RequestMapping("/v1/aggregate")
public class AggregateController {

    private final ObjectMapper om;
    private final PolicyKbClient kb;
    private final MlScorerClient scorer;

    public AggregateController(ObjectMapper om, PolicyKbClient kb, MlScorerClient scorer) {
        this.om = om;
        this.kb = kb;
        this.scorer = scorer;
    }

    @PostMapping
    public LogEvent aggregate(@RequestBody Map<String, Object> body) {
        LogEvent ev = om.convertValue(body, LogEvent.class);

        if (ev.getFeatures() == null)
            ev.setFeatures(new HashMap<>());
        if (ev.getRuleHits() == null)
            ev.setRuleHits(new ArrayList<>());

        String msg = ev.getMessage() == null ? "" : ev.getMessage();
        boolean ruleHit = !ev.getRuleHits().isEmpty();

        // Defaults
        double anomaly = 0.0;
        double zeroShotSerious = 0.2;
        double policySimMax = 0.0;

        // Policy search (best-effort)
        List<PolicySnippet> snippets = List.of();
        try {
            snippets = kb.search(msg, 3);
            policySimMax = snippets.stream().mapToDouble(PolicySnippet::getSimilarity).max().orElse(0.0);
        } catch (Exception ignored) {
        }

        // ML score (best-effort)
        try {
            ScoreResponse sr = scorer.score(msg, snippets, ev.getFeatures(), ruleHit);
            if (sr != null) {
                anomaly = sr.getAnomaly();
                zeroShotSerious = sr.getZeroShot().getOrDefault("serious", zeroShotSerious);
                policySimMax = Math.max(policySimMax, sr.getPolicySimMax());
            }
        } catch (Exception ignored) {
        }

        // Final risk formula (non-null)
        double finalRisk = 0.45 * (ruleHit ? 1 : 0) + 0.20 * anomaly + 0.25 * zeroShotSerious + 0.10 * policySimMax;

        ev.getFeatures().put("final_risk", finalRisk);
        ev.getFeatures().put("zero_shot_serious", zeroShotSerious);
        ev.getFeatures().put("policy_sim_max", policySimMax);

        return ev;
    }
}
