package com.pulse.agg;

import com.pulse.common.dto.PolicySnippet;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.util.Arrays;
import java.util.List;

@Component
public class PolicyKbClient {
    private final RestClient http;
    private final String base;

    public PolicyKbClient(RestClient http,
            @Value("${policy.kb.url}") String base) {
        this.http = http;
        this.base = base;
    }

    public List<PolicySnippet> search(String q, int k) {
        ResponseEntity<PolicySnippet[]> resp = http.get()
                .uri(base + "/v1/policies/search?q=" + q + "&k=" + k)
                .retrieve().toEntity(PolicySnippet[].class);
        return resp.getBody() == null ? List.of() : Arrays.asList(resp.getBody());
    }
}
