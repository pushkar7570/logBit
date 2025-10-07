package com.pulse.agg;

import com.pulse.common.dto.*;
import com.pulse.common.port.MessageBus;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.util.*;

@Component
@RequiredArgsConstructor
public class Aggregator {
    private final MessageBus<LogEvent> bus;

    @Value("${policy.kb.url:http://localhost:8084}")
    String policyKbUrl;
    @Value("${ml.scorer.url:http://localhost:8090}")
    String mlUrl;

    private final RestClient http = RestClient.create();

    @PostConstruct
    void init() {
        bus.subscribe("logs.scored.input", this::scoreAndPublish);
    }

    void scoreAndPublish(LogEvent ev) {
        // fetch top-k policy snippets
        String q = ev.getMessage();
        ResponseEntity<PolicySnippet[]> resp = http.get()
                .uri(policyKbUrl + "/v1/policies/search?q=" + q + "&k=3")
                .retrieve().toEntity(PolicySnippet[].class);
        List<PolicySnippet> snippets = resp.getBody() == null ? List.of() : Arrays.asList(resp.getBody());

        // detect if any rule hit
        boolean ruleHit = !ev.getRuleHits().isEmpty();

        // call ML scorer (or fallback)
        ScoreResponse sr;
        try {
            ScoreRequest req = ScoreRequest.builder()
                    .alertText(ev.getMessage())
                    .policySnippets(snippets)
                    .features(ev.getFeatures())
                    .build();
            sr = http.post().uri(mlUrl + "/score").contentType(MediaType.APPLICATION_JSON)
                    .body(req).retrieve().toEntity(ScoreResponse.class).getBody();
        } catch (Exception e) {
            sr = ScoreResponse.builder()
                    .zeroShot(Map.of("serious", ruleHit ? 0.6 : 0.2))
                    .anomaly(0.0)
                    .policySimMax(snippets.stream().mapToDouble(PolicySnippet::getSimilarity).max().orElse(0))
                    .policyRefs(snippets.stream().map(PolicySnippet::getSectionRef).toList())
                    .build();
        }

        double z = sr.getZeroShot().getOrDefault("serious", 0.0);
        double finalRisk = 0.45 * (ruleHit ? 1 : 0) + 0.20 * sr.getAnomaly() + 0.25 * z + 0.10 * sr.getPolicySimMax();

        // attach as evidence (in a real impl, persist to DB and publish to
        // alerts.scored)
        ev.getFeatures().put("final_risk", finalRisk);
        bus.publish("alerts.scored", ev);
    }
}