package com.pulse.agg;

import com.pulse.common.dto.PolicySnippet;
import com.pulse.common.dto.ScoreRequest;
import com.pulse.common.dto.ScoreResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.util.List;
import java.util.Map;

@Component
public class MlScorerClient {
    private final RestClient http;
    private final String base;

    public MlScorerClient(RestClient http,
            @Value("${ml.scorer.url}") String base) {
        this.http = http;
        this.base = base;
    }

    public ScoreResponse score(String alertText, List<PolicySnippet> snippets, Map<String, Object> features,
            boolean ruleHit) {
        ScoreRequest req = ScoreRequest.builder()
                .alertText(alertText)
                .policySnippets(snippets)
                .features(features)
                .ruleHit(ruleHit)
                .build();
        return http.post()
                .uri(base + "/score")
                .contentType(MediaType.APPLICATION_JSON)
                .body(req)
                .retrieve()
                .toEntity(ScoreResponse.class)
                .getBody();
    }
}
